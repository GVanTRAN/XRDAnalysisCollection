# Azimuthal Fit

A desktop app (PyQt6) that replaces the three azimuthal-fit notebooks with one
window and four tabs. It fits periodic Gaussians to the azimuthal (χ) intensity
of integrated 2θ–χ scans, batch-fits every pixel in parallel, and visualizes the
resulting fibre-orientation maps.

The fitting math is a verbatim port of your notebooks
(`azi_inspection`, `batchaziparallel`, `azivisu`) — same cleaning, same
automatic peak-count selection, same no-texture rejection, same fibre pairing —
so results are identical. What changed is only the packaging and interactivity.

## Install

```bash
cd azimuthal_app
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
```

## Run

```bash
python run_app.py
```

## The four tabs

**1 · Dataset.** Browse to an HDF5 scan, click *Inspect paths* to list its
contents, and choose the internal paths for `intensity`, `2θ`, and `χ` (the
notebook defaults are pre-filled). Set the scan geometry `n_rows × n_cols`
(used to reshape batch maps) and the serpentine toggle, then *Load dataset*.
The file is opened once and kept open — unlike the notebooks, which reopened it
on every redraw — so slider scrubbing only reads a single 2D slice.

**2 · Inspection.** Scrub the index slider to browse patterns. Click two 2θ
positions on the 2D map to define a ROI, then *Submit ROI*. Auto-fit overlays
the periodic Gaussian fit on I(χ) and lists each peak's centre and FWHM.
Submitted ROIs are shared with the Batch tab. Zoom/pan (toolbar) is respected —
clicks only define ROIs when no zoom/pan mode is active.

**3 · Batch fit.** Runs `fit_chi_peaks` over every pattern for every submitted
ROI, in parallel, inside a background thread with a live progress bar and an
Abort button. Two fibre families per pixel (chi1/chi2) are kept so crossing
pixels retain both orientations. Save the maps to a `.npz`, or let the result
flow straight into the Visualization tab.

**4 · Visualization.** Load a `.npz` (or click *Use current batch result*).
Sorts the two families into vertical / horizontal by proximity to reference
angles and shows χ and Δχ for each. Live controls: ROI, tolerance, colormaps,
and a *degree of orientation* toggle `(180 − Δχ)/180`. Export the figure to
PNG/PDF/SVG.

## Typical flow

Dataset → Inspection (define ROIs, sanity-check the fit on a few patterns) →
Batch fit (run + save) → Visualization (maps). ROIs and the batch result carry
over between tabs automatically.

## Files

```
azimuthal_app/
├── run_app.py              entry point
├── requirements.txt
└── azi_fit/
    ├── core.py             fitting math (pure; verbatim port of the notebooks)
    ├── dataset.py          open-once HDF5 wrapper
    ├── mpl_canvas.py       embedded matplotlib canvas
    ├── tab_dataset.py      tab 1
    ├── tab_inspection.py   tab 2
    ├── tab_batch.py        tab 3 (+ background worker)
    ├── tab_visualization.py tab 4
    └── app.py              main window / shared state
```

## Notes

- `n_jobs = -1` uses all cores. Lower it if the machine also serves other users.
- If `n_rows × n_cols` doesn't equal the pattern count, the app warns before the
  batch (reshaping would otherwise fail).
- The parallel worker prefers joblib's streaming generator API for per-chunk
  progress; on older joblib it falls back to a blocking run automatically.
