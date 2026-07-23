# 1D Simulation — TOPAS Cellulose Diffraction Pattern Simulator

Interactive TOPAS-based simulation of the 1D powder-diffraction pattern of a two-phase cellulose Iβ system, with live sliders for crystallite size, preferred orientation, and phase fraction — useful for building physical intuition about how these parameters shape a pattern before running a real Rietveld refinement.

## Concept

The pattern shape of a crystalline sample depends on more than just its unit cell: crystallite size and shape (peak broadening) and preferred orientation (texture, which reflections are enhanced/suppressed) both leave a strong fingerprint. This is especially true for materials like **cellulose**, whose fibril-like crystallites are highly anisotropic — they aren't the same size in every direction, so peak broadening also depends on the reflection's direction (hkl), not just 2θ.

This folder combines:
1. A reusable **TOPAS macro library** implementing the anisotropic crystallite-size model of Ectors *et al.* (2015, *J. Appl. Cryst.* 48, 189–194) — crystallites modeled as ellipsoids, elliptic cylinders, or cuboids, so the apparent size (and hence the Lorentzian/Gaussian peak width) varies with reflection direction.
2. A **two-phase cellulose Iβ TOPAS input** (both phases share space group `P1121` and near-identical atomic coordinates, differing mainly in unit cell, crystallite-size model, and preferred orientation).
3. A **Python/matplotlib driver** that turns the key parameters of that input into live sliders, re-running TOPAS as a pure forward calculation (`iters 0` — no refinement, just "calculate the pattern for these parameter values") every time you move one.

## Files in this folder

| File | Role |
|---|---|
| `aniso.inc` | TOPAS macro library: `AnisoCS` (ellipsoid / elliptic-cylinder / cuboid anisotropic crystallite-size broadening), `AnisoCSg` (adds a lognormal size distribution on top), and `AnisoCSout` / `AnisoCSgout` (write a text analysis report per phase: metric tensors, rotation matrix, apparent size per reflection). `#include`d by both `.inp` files below. |
| `cellulose_template.inp` | The parametrized TOPAS input. Both cellulose phases are defined here, but the values that matter for exploration are left as `{Placeholder}` tokens (e.g. `{D_1}`, `{PO_CA1}`) instead of fixed numbers. |
| *(fixed-value example `.inp`)* | A non-templated version of the same two-phase model with literal numbers in place of the placeholders — runnable directly in TOPAS on its own, useful as a static reference. *(Exact filename wasn't in what you pasted — check your repo and let me know if it's not `cellulose_simulation.inp`.)* |
| *(Python slider script)* | The interactive driver described below. *(Same caveat — filename not given; I've referred to it as `cellulose_slider_simulation.py` below.)* |

## How the slider script works

1. Reads `cellulose_template.inp` and auto-detects every `{Placeholder}` token in it, **except** `{YOBS_XY}` and `{OUTPUT_XY}` (those two are just local temp-file names, not physical parameters).
2. Builds one Matplotlib `Slider` per detected placeholder. Known parameters get a friendly label and sensible min/max/step from `PARAM_CONFIG`; any placeholder you add to the template that *isn't* in `PARAM_CONFIG` still gets a slider automatically, with a generic 0–100 default range.
3. On every slider move: substitutes the current slider values into the template text, writes the result to `cellulose_current_run.inp`, and runs it through TOPAS (`tc.exe`, called as a subprocess) with `iters 0` — a pure forward calculation, not a refinement.
4. Loads the resulting `temp_simulated.xy`, normalizes it to its maximum, and updates the plotted curve and title live.

### Parameters (sliders)

| Placeholder | Label | Meaning |
|---|---|---|
| `Phase_1_WP` | P1 wt% | `Known_Weight_Percent` of phase 1 (Cellulose Iβ, allomorph 1) |
| `D_1` | P1 D | Isotropic crystallite size (`CS_G` / `CS_L`) of phase 1, nm |
| `PO_CA1` | P1 texture | Phase 1 preferred-orientation coefficient (single direction, `0 0 1`) |
| `D_2` | P2 D | Anisotropic ellipsoid semi-axes rx = ry (`AnisoCS`) of phase 2, nm |
| `Z_2` | P2 Z | Anisotropic ellipsoid semi-axis rz (`AnisoCS`) of phase 2, nm |
| `PO_CA2_001` | P2 Texture 001 | Phase 2 preferred-orientation coefficient along `0 0 1` |
| `PO_CA2_100` | P2 Texture 100 | Phase 2 preferred-orientation coefficient along `1 0 0` |
| `PO_WEIGHT2` | P2 001/100 weight | Blend weight between the two phase-2 preferred-orientation directions |

`{YOBS_XY}` / `{OUTPUT_XY}` are not sliders — they're substituted with fixed local filenames (`temp_calc.xy`, `temp_simulated.xy`) so TOPAS has something to read/write on every run. `temp_calc.xy` only defines the simulation's 2θ grid (1.5°–15°, 0.01° step by default via `yobs_eqn`); there's no real observed data involved, since this is a forward simulation, not a fit.

## Requirements

- A local install of **TOPAS** (Bruker TOPAS or TOPAS-Academic) with `tc.exe` available — this is commercial/licensed software and is **not** included in this repo. Edit the `topas_exe` path in the Python script to point to your own install (it defaults to `C:\Topas-7\tc.exe`).
- Python 3 with `numpy` and `matplotlib`

```bash
pip install numpy matplotlib
```

## Usage

1. Point `topas_exe` in the Python script at your local `tc.exe`.
2. Run the script from this folder (it locates `cellulose_template.inp` relative to itself via `Path(__file__)`).
3. Drag any slider — the plot updates with the newly simulated, normalized pattern after each TOPAS run.

## Data flow

```
cellulose_template.inp  (has {Placeholder} tokens)
        │  slider values substituted in
        ▼
cellulose_current_run.inp  (one real TOPAS input per slider move)
        │  tc.exe, iters 0 (forward calculation only)
        ▼
temp_simulated.xy  (2θ, calculated intensity)
        │  normalized to max intensity
        ▼
live matplotlib plot
```
