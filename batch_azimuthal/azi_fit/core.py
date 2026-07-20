"""
Core azimuthal-fit routines.

Everything in this module is *pure* (no GUI, no global scan state) so it can be
reused from the Inspection tab, the parallel Batch worker, and any script. The
fitting logic is a verbatim port of the three notebooks:

    - periodic Gaussian model + fully-automatic fit_chi_peaks   (inspection / batch)
    - pair_into_fibers                                          (batch)
    - process_chunk                                             (batch, parallel worker)
    - sort_into_vertical_horizontal + angle utilities           (visualization)

Keeping it in one place means the fit you see interactively is exactly the fit
the batch runs and exactly the geometry the visualization assumes.
"""

from __future__ import annotations

import numpy as np

from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.signal import find_peaks
from scipy.optimize import least_squares


# =====================================================================
# Periodic Gaussian model (constant background + n wrapped Gaussians)
# =====================================================================

CHI_PERIOD = 360.0          # azimuthal data is periodic over 360 deg
CHI_WRAP_LO = -180.0        # report centres in [-180, 180)


def _wrap(angle):
    """Wrap angle(s) into [CHI_WRAP_LO, CHI_WRAP_LO + 360)."""
    return (np.asarray(angle, dtype=float) - CHI_WRAP_LO) % CHI_PERIOD + CHI_WRAP_LO


def _ang_diff(x, c):
    """Signed nearest angular difference x - c, in (-180, 180]."""
    return (x - c + 0.5 * CHI_PERIOD) % CHI_PERIOD - 0.5 * CHI_PERIOD


def periodic_model(x, params, n_peaks):
    """
    Constant background + n periodic (wrapped) Gaussians.

    params = [bg0, amp1, c1, s1, amp2, c2, s2, ...]
    """
    y = params[0] + 0.0 * x
    for i in range(n_peaks):
        amp = params[1 + 3 * i]
        c = params[2 + 3 * i]
        s = params[3 + 3 * i]
        d = _ang_diff(x, c)
        y = y + amp * np.exp(-0.5 * (d / s) ** 2)
    return y


# =====================================================================
# Cleaning / detection / fixed-n fit helpers
# =====================================================================

def _clean_profile(chi, y, despike_window=7, despike_factor=6.0):
    """
    Auto cleaning, every threshold derived from THIS pattern:
      - drop non-finite
      - drop detector-gap / beamstop points (intensity ~ 0)
      - drop sharp single-crystal spikes (median + MAD, adaptive)
    Returns x_clean, y_clean, baseline, robust_sigma, low_cut.
    """
    x = np.asarray(chi, dtype=float)
    y = np.asarray(y, dtype=float)

    order = np.argsort(x)
    x, y = x[order], y[order]

    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 20:
        raise ValueError("Not enough finite points.")

    # --- gap / beamstop: auto threshold from the median signal level ---
    pos = y[y > 0]
    med_pos = np.nanmedian(pos) if pos.size else 0.0
    low_cut = max(1e-6, 0.20 * med_pos)
    valid = y > low_cut
    x, y = x[valid], y[valid]
    if x.size < 20:
        raise ValueError("Too few points after gap removal.")

    # --- despike: threshold = factor * (robust noise of THIS pattern) ---
    w = despike_window + (1 - despike_window % 2)
    w = min(max(3, w), x.size if x.size % 2 == 1 else x.size - 1)
    y_med = median_filter(y, size=w)
    res = y - y_med
    mad = np.nanmedian(np.abs(res - np.nanmedian(res)))
    robust_sigma = 1.4826 * mad
    if robust_sigma > 0 and np.isfinite(robust_sigma):
        keep = res < despike_factor * robust_sigma   # only sharp UP spikes
    else:
        keep = np.ones_like(y, dtype=bool)
        std_res = np.nanstd(res)
        robust_sigma = std_res if np.isfinite(std_res) and std_res > 0 else 1.0
    x, y = x[keep], y[keep]
    if x.size < 20:
        raise ValueError("Too few points after despiking.")

    baseline = float(np.nanpercentile(y, 15))
    return x, y, baseline, max(robust_sigma, 1e-6), low_cut


def _detect_centers(x_clean, y_clean, baseline, robust_sigma,
                    min_distance_deg, n_grid, smooth_deg=5.0):
    """
    Periodic peak detection on a uniform circular grid (period 360),
    tiled x3 so a peak on the seam is found once. Returns centres (deg)
    strongest-first, plus their prominences.
    """
    start = float(np.min(x_clean))
    grid = start + np.arange(n_grid) * (CHI_PERIOD / n_grid)
    dgrid = CHI_PERIOD / n_grid

    yg = np.interp(grid, x_clean, y_clean, period=CHI_PERIOD)

    s_samp = max(1.0, smooth_deg / dgrid)
    yg = gaussian_filter1d(yg, sigma=s_samp, mode="wrap")

    p99 = np.nanpercentile(y_clean, 99)
    prom = max(2.0, 1.5 * robust_sigma, 0.10 * (p99 - baseline))
    dist = max(1, int(round(min_distance_deg / dgrid)))

    yt = np.concatenate([yg, yg, yg])
    idx, props = find_peaks(yt, prominence=prom, distance=dist)

    mid = (idx >= n_grid) & (idx < 2 * n_grid)
    idx = idx[mid] - n_grid
    proms = props["prominences"][mid]

    if idx.size == 0:
        return np.array([]), np.array([])

    order = np.argsort(proms)[::-1]
    return _wrap(grid[idx[order]]), proms[order]


def _seed_centers(detected, n, x_clean):
    """Return exactly n seed centres: top-n detected, padded if needed."""
    centers = list(detected[:n])
    if len(centers) < n:
        lo, hi = float(np.min(x_clean)), float(np.max(x_clean))
        extra = np.linspace(lo, hi, n + 2)[1:-1]
        for e in extra:
            if len(centers) >= n:
                break
            centers.append(float(e))
    return _wrap(np.array(centers[:n], dtype=float))


def _fit_fixed_n(x_clean, y_clean, centers, baseline, robust_sigma,
                 dx, sigma_max_deg):
    """Fit constant bg + n periodic Gaussians. Returns params, ssr, bic, ok."""
    n = len(centers)
    sigma_guess = 12.0
    sigma_upper = min(sigma_max_deg, 0.5 * CHI_PERIOD)

    p0 = [baseline]
    lo = [-np.inf]
    hi = [np.inf]

    for c in centers:
        d = np.abs(_ang_diff(x_clean, c))
        near = d < max(4.0, 3.0 * dx)
        if np.any(near):
            amp0 = max(np.nanmax(y_clean[near]) - baseline, 1.0)
        else:
            amp0 = max(np.nanpercentile(y_clean, 90) - baseline, 1.0)

        p0 += [amp0, c, sigma_guess]
        lo += [0.0, c - 180.0, max(0.5, dx)]   # center free over half circle
        hi += [np.inf, c + 180.0, sigma_upper]

    p0 = np.clip(np.array(p0, float), lo, hi)
    lo = np.array(lo, float)
    hi = np.array(hi, float)

    def resid(p):
        return periodic_model(x_clean, p, n) - y_clean

    fit = least_squares(
        resid, p0, bounds=(lo, hi),
        loss="soft_l1", f_scale=max(robust_sigma, 1.0), max_nfev=2000
    )
    p = fit.x

    r = periodic_model(x_clean, p, n) - y_clean
    ssr = float(np.sum(r ** 2))
    N = x_clean.size
    k = 1 + 3 * n
    bic = N * np.log(ssr / N + 1e-12) + k * np.log(N)
    return p, ssr, bic, bool(fit.success)


def _fit_flat(y_clean):
    """No-texture reference: a single constant (LS optimum is the mean)."""
    bg = float(np.mean(y_clean))
    r = y_clean - bg
    ssr = float(np.sum(r ** 2))
    N = y_clean.size
    bic = N * np.log(ssr / N + 1e-12) + 1.0 * np.log(N)
    return np.array([bg], dtype=float), ssr, bic


# =====================================================================
# The main automatic fit
# =====================================================================

def fit_chi_peaks(
    chi,
    intensity_vs_chi,
    candidates=(2, 4),
    despike_window=7,
    despike_factor=6.0,
    sigma_max_deg=70.0,
    snr_min=3.5,
    bic_margin=6.0,
):
    """
    Fully automatic, periodic fit of I(chi), with no-texture rejection.

    Returns a dict with keys:
      textured, n_peaks, peaks, center_chi, fwhm, params,
      x_clean, y_clean, baseline, low_cut, robust_sigma, snr, bic_by_n, fit_success
    """
    x_clean, y_clean, baseline, robust_sigma, low_cut = _clean_profile(
        chi, intensity_vs_chi,
        despike_window=despike_window,
        despike_factor=despike_factor,
    )

    dx = np.median(np.diff(x_clean))
    if not np.isfinite(dx) or dx <= 0:
        dx = (x_clean.max() - x_clean.min()) / max(1, x_clean.size - 1)

    n_grid = int(max(360, len(np.atleast_1d(chi))))
    max_n = max(candidates)

    min_distance_deg = CHI_PERIOD / (2.0 * max_n)

    detected, det_proms = _detect_centers(
        x_clean, y_clean, baseline, robust_sigma,
        min_distance_deg=min_distance_deg, n_grid=n_grid
    )

    flat_params, flat_ssr, flat_bic = _fit_flat(y_clean)

    # --- cheap no-texture GATE (before any least-squares fitting) ---
    strongest_prom = float(det_proms[0]) if det_proms.size else 0.0
    quick_snr = strongest_prom / robust_sigma if robust_sigma > 0 else np.inf

    if detected.size == 0 or quick_snr < 0.6 * snr_min:
        return {
            "textured": False, "n_peaks": 0, "peaks": [],
            "center_chi": np.nan, "fwhm": np.nan,
            "params": flat_params, "x_clean": x_clean, "y_clean": y_clean,
            "baseline": baseline, "low_cut": low_cut,
            "robust_sigma": robust_sigma, "snr": float(quick_snr),
            "bic_by_n": {0: flat_bic}, "fit_success": True,
        }

    # --- candidate fits ---
    fits = {}
    bic_by_n = {0: flat_bic}
    for n in candidates:
        try:
            seeds = _seed_centers(detected, n, x_clean)
            p, ssr, bic, ok = _fit_fixed_n(
                x_clean, y_clean, seeds, baseline, robust_sigma,
                dx, sigma_max_deg
            )
            fits[n] = (p, ok)
            bic_by_n[n] = bic
        except Exception:
            bic_by_n[n] = np.inf

    # --- best peak model, then texture significance test ---
    textured = False
    best_n = 0
    params = flat_params
    success = True
    snr = 0.0

    if fits:
        best_peak_n = min(fits.keys(), key=lambda n: bic_by_n[n])
        params_peak, ok_peak = fits[best_peak_n]

        amps = [params_peak[1 + 3 * i] for i in range(best_peak_n)]
        max_amp = max(amps) if amps else 0.0
        snr = max_amp / robust_sigma if robust_sigma > 0 else np.inf

        beats_flat = (bic_by_n[best_peak_n] + bic_margin) < flat_bic
        if beats_flat and snr >= snr_min:
            textured = True
            best_n = best_peak_n
            params = params_peak
            success = ok_peak

    if not textured:
        return {
            "textured": False, "n_peaks": 0, "peaks": [],
            "center_chi": np.nan, "fwhm": np.nan,
            "params": flat_params, "x_clean": x_clean, "y_clean": y_clean,
            "baseline": baseline, "low_cut": low_cut,
            "robust_sigma": robust_sigma, "snr": float(snr),
            "bic_by_n": bic_by_n, "fit_success": True,
        }

    # --- TEXTURED: collect peaks (centres wrapped to [-180, 180)) ---
    peaks = []
    for i in range(best_n):
        amp = params[1 + 3 * i]
        c = float(_wrap(params[2 + 3 * i]))
        s = abs(params[3 + 3 * i])
        peaks.append({
            "peak_number": i + 1,
            "center_chi": c,
            "amplitude": float(amp),
            "sigma": float(s),
            "fwhm": float(2.35482 * s),
            "area": float(amp * s * np.sqrt(2 * np.pi)),
        })
    peaks.sort(key=lambda d: d["center_chi"])
    for i, pk in enumerate(peaks):
        pk["peak_number"] = i + 1

    strongest = max(peaks, key=lambda d: d["amplitude"])

    return {
        "textured": True, "n_peaks": best_n, "peaks": peaks,
        "center_chi": strongest["center_chi"], "fwhm": strongest["fwhm"],
        "params": params, "x_clean": x_clean, "y_clean": y_clean,
        "baseline": baseline, "low_cut": low_cut,
        "robust_sigma": robust_sigma, "snr": float(snr),
        "bic_by_n": bic_by_n, "fit_success": success,
    }


# =====================================================================
# Fibre pairing (batch)
# =====================================================================

def _fold180(a):
    return (np.asarray(a, float) + 90.0) % 180.0 - 90.0      # -> [-90, 90)


def _circ_sep(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)              # period 360


def _circ_mean_180(angles):
    th = np.deg2rad(2.0 * np.asarray(angles, float))
    m = np.arctan2(np.nanmean(np.sin(th)), np.nanmean(np.cos(th)))
    return _fold180(np.rad2deg(m) / 2.0)


def pair_into_fibers(centers, fwhms, areas):
    """
    Pair fitted peaks into fibres (peaks ~180 deg apart = one fibre).
    Returns a list of dicts {orient, fwhm, area}, strongest first.
    """
    n = len(centers)
    if n == 0:
        return []
    if n == 1:
        return [dict(orient=float(_fold180(centers[0])), fwhm=fwhms[0], area=areas[0])]
    if n == 2:
        pairs = [(0, 1)]
    elif n == 4:
        parts = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
        cost = lambda p: sum(abs(_circ_sep(centers[i], centers[j]) - 180) for i, j in p)
        pairs = list(min(parts, key=cost))
    else:
        idx = list(range(n)); pairs = []
        while len(idx) >= 2:
            i = idx.pop(0)
            j = min(idx, key=lambda k: abs(_circ_sep(centers[i], centers[k]) - 180))
            idx.remove(j); pairs.append((i, j))
    fibers = []
    for i, j in pairs:
        fibers.append(dict(
            orient=float(_circ_mean_180([_fold180(centers[i]), _fold180(centers[j])])),
            fwhm=float(np.nanmean([fwhms[i], fwhms[j]])),
            area=float(np.nansum([areas[i], areas[j]])),
        ))
    fibers.sort(key=lambda d: -d["area"])
    return fibers


# =====================================================================
# Parallel worker: one chunk, all ROIs
# =====================================================================
# NOTE: signature takes everything as explicit arguments (no module globals)
# so it pickles cleanly to loky worker processes.

def process_chunk(start, stop, scan_path, intensity_path, chi,
                  roi_slices, fit_kwargs):
    """
    Fit every pattern in [start, stop) for every ROI.
    Returns (start, {roi_index: (chi1, dchi1, chi2, dchi2, npeaks)}).
    """
    import h5py
    nb = stop - start
    out = {}
    with h5py.File(scan_path, "r") as f:
        dset = f[intensity_path]
        for ri, (j0, j1) in enumerate(roi_slices):
            block = np.asarray(dset[start:stop, :, j0:j1], dtype=float)
            I_chi_block = np.nansum(block, axis=2)            # (nb, n_chi)

            chi1 = np.full(nb, np.nan); dchi1 = np.full(nb, np.nan)
            chi2 = np.full(nb, np.nan); dchi2 = np.full(nb, np.nan)
            npk = np.zeros(nb)

            for k in range(nb):
                try:
                    res = fit_chi_peaks(chi, I_chi_block[k], **fit_kwargs)
                    npk[k] = res["n_peaks"]
                    if res["textured"] and res["peaks"]:
                        fibers = pair_into_fibers(
                            [p["center_chi"] for p in res["peaks"]],
                            [p["fwhm"] for p in res["peaks"]],
                            [p["area"] for p in res["peaks"]],
                        )
                        if len(fibers) >= 1:
                            chi1[k] = fibers[0]["orient"]; dchi1[k] = fibers[0]["fwhm"]
                        if len(fibers) >= 2:
                            chi2[k] = fibers[1]["orient"]; dchi2[k] = fibers[1]["fwhm"]
                except Exception:
                    pass
            out[ri] = (chi1, dchi1, chi2, dchi2, npk)
    return start, out


# =====================================================================
# Visualization: axial angle utilities + family sorting
# =====================================================================

def wrap_axial_chi(angle):
    """Axial wrapping: chi and chi + 180 deg are equivalent. -> [-90, 90)."""
    return (np.asarray(angle, dtype=float) + 90.0) % 180.0 - 90.0


def axial_diff(a, b):
    """Axial angular difference. -> [-90, 90)."""
    return (np.asarray(a, dtype=float) - np.asarray(b, dtype=float) + 90.0) % 180.0 - 90.0


def axial_abs_diff(a, b):
    """Absolute axial angular separation. -> [0, 90]."""
    return np.abs(axial_diff(a, b))


def robust_vlim(arr, low=2, high=98):
    """Robust colorbar limits using percentiles."""
    vals = arr[np.isfinite(arr)]
    if len(vals) == 0:
        return None, None
    vmin = np.nanpercentile(vals, low)
    vmax = np.nanpercentile(vals, high)
    if np.isclose(vmin, vmax):
        vmin = np.nanmin(vals); vmax = np.nanmax(vals)
    if np.isclose(vmin, vmax):
        vmin = vmin - 1; vmax = vmax + 1
    return vmin, vmax


def sort_into_vertical_horizontal(
    chi1_map, dchi1_map, chi2_map, dchi2_map,
    vertical_ref=0.0, horizontal_ref=90.0, tol_deg=35.0
):
    """
    Sort fitted families into vertical / horizontal. A crossing pixel is
    included in BOTH maps if both fitted orientations exist and each one
    belongs to a family. Returns
    (chi_vertical, dchi_vertical, chi_horizontal, dchi_horizontal).
    """
    vertical_ref = wrap_axial_chi(vertical_ref)
    horizontal_ref = wrap_axial_chi(horizontal_ref)

    nrows, ncols = chi1_map.shape
    chi_vertical = np.full((nrows, ncols), np.nan)
    dchi_vertical = np.full((nrows, ncols), np.nan)
    chi_horizontal = np.full((nrows, ncols), np.nan)
    dchi_horizontal = np.full((nrows, ncols), np.nan)

    for i in range(nrows):
        for j in range(ncols):
            candidates = []
            if np.isfinite(chi1_map[i, j]):
                candidates.append({"chi": chi1_map[i, j], "dchi": dchi1_map[i, j]})
            if np.isfinite(chi2_map[i, j]):
                candidates.append({"chi": chi2_map[i, j], "dchi": dchi2_map[i, j]})
            if not candidates:
                continue

            best_v, best_v_dist = None, np.inf
            for cand in candidates:
                d = axial_abs_diff(cand["chi"], vertical_ref)
                if d < best_v_dist:
                    best_v_dist, best_v = d, cand

            best_h, best_h_dist = None, np.inf
            for cand in candidates:
                d = axial_abs_diff(cand["chi"], horizontal_ref)
                if d < best_h_dist:
                    best_h_dist, best_h = d, cand

            if best_v is not None and best_v_dist <= tol_deg:
                chi_vertical[i, j] = best_v["chi"]
                dchi_vertical[i, j] = best_v["dchi"]
            if best_h is not None and best_h_dist <= tol_deg:
                chi_horizontal[i, j] = best_h["chi"]
                dchi_horizontal[i, j] = best_h["dchi"]

    return chi_vertical, dchi_vertical, chi_horizontal, dchi_horizontal


def to_orientation(dchi_map):
    """Degree of orientation = (180 - FWHM) / 180, elementwise on a dchi map."""
    out = np.full_like(dchi_map, np.nan, dtype=float)
    mask = np.isfinite(dchi_map)
    out[mask] = (180.0 - dchi_map[mask]) / 180.0
    return out
