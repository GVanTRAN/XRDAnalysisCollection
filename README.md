# XRD Analysis Collection

A collection of standalone Python tools for processing X-ray diffraction (XRD) data — 2D/1D data viewing, ROI imaging, azimuthal fitting, 1D simulation, and batch Rietveld/profile refinement with TOPAS. Built around scanning-XRD datasets stored as HDF5 (e.g. ESRF ID13-style files with an integrated 1D pattern and a raw 2D Eiger detector stack per scan point).

## Repository structure

| Path | Description |
|---|---|
| [`roi2d.py`](./roi2d.py) | **ROI-from-2D Viewer** — build a real-space image from a region of the raw 2D detector. |
| [`sxrd-viewer.py`](./sxrd-viewer.py) | **Scanning XRD Viewer (v1)** — explore a scan pixel-by-pixel or with a single polygon ROI. |
| [`sxrd_viewer2.py`](./sxrd_viewer2.py) | **Scanning XRD Viewer (v2)** — adds multiple simultaneous polygon ROIs and background subtraction. |
| [`1Dsimulation/`](./1Dsimulation) | Simulating / modeling 1D diffraction patterns. |
| [`batch_azimuthal/`](./batch_azimuthal) | Batch azimuthal integration and fitting across many frames. |
| [`batchrefinement/`](./batchrefinement) | Batch profile / Rietveld refinement automation driven by TOPAS. |
| [`preprocessing/`](./preprocessing) | Preprocessing utilities for raw scan files ahead of analysis. |

> Each subfolder groups related scripts/notebooks for that stage of the workflow — open the folder for the specific tools it contains.

## Tools

### ROI-from-2D Viewer — `roi2d.py`

A focused PyQt5 GUI for one task: turning a region of the **2D diffraction pattern** into a real-space (scan-position) image.

**Workflow**
1. Load the scan `.h5` file — set the 1D and Eiger HDF5 paths and the map shape (rows × cols), then click **Load**.
2. The mean 1D pattern is shown on top. Drag a horizontal span (or use the min/max boxes) to pick a 2θ range — this builds the navigator map.
3. Click a few pixels on the navigator map. Their raw 2D detector frames are summed live and shown in the detector panel.
4. Draw a polygon on the summed detector frame to select a feature.
5. Click **Build image from detector ROI** — every scan position's 2D frame is integrated inside that polygon to produce a new real-space image.
6. Save the result as `.npy` + `.png` + `.json`.

```bash
python roi2d.py
```

### Scanning XRD Viewer — `sxrd-viewer.py` (v1) and `sxrd_viewer2.py` (v2)

A desktop GUI for exploring scanning XRD datasets. Both viewers share the same core workflow; **v2** adds multi-polygon ROI management and background subtraction on top of v1.

**Workflow**
1. Pick the scan `.h5` file, choose the scan entry (e.g. `1.1`), confirm the internal HDF5 paths, and click **Load**.
2. The mean 1D pattern is plotted on top — drag a span (or set 2θ min/max) to define the ROI used to build the spatial intensity map.
3. **Click-pixel mode**: click any pixel on the map to see its 1D pattern and raw 2D Eiger frame simultaneously.
4. **Polygon mode**: draw a polygon on the map. The 1D pattern (mean or sum over the polygon) appears immediately; click **Average 2D over polygon** to also build the averaged detector frame.
5. *(v2 only)* Add several polygons, step through them with `←`/`→`, place a matching background region per polygon, and optionally subtract it from the 1D pattern.
6. Save the current 1D pattern (`.xy`), everything (`.npz`), or just the polygon vertices (`.json`/`.npy`).

```bash
python sxrd-viewer.py     # v1 — single polygon
python sxrd_viewer2.py    # v2 — multi-polygon + background subtraction
```

## Requirements

All three viewers share the same dependencies:

- Python ≥ 3.8
- PyQt5
- matplotlib ≥ 3.5
- numpy
- h5py
- hdf5plugin (required to decode compressed/bitshuffled Eiger frames)

```bash
pip install PyQt5 "matplotlib>=3.5" numpy h5py hdf5plugin
```

## Data format

The viewers expect an HDF5 scan file containing, per scan entry (e.g. `1.1`):

- a 2θ (or q) axis, e.g. `/1.1/eiger_integrate/integrated/2th`
- an integrated intensity array, e.g. `/1.1/eiger_integrate/integrated/intensity`
- (optional) the raw 2D detector stack, e.g. `/1.1/measurement/eiger`

These paths and the scan map shape (rows × cols) are editable fields in each tool's UI — adjust them to match your beamline's file layout if it differs from the ESRF ID13-style defaults shown above.

## License

No license file is currently included in this repository, which means all rights are reserved by default. Add a `LICENSE` file if you want to permit reuse.

## Author

[GVanTRAN](https://github.com/GVanTRAN)
