"""
Visualization tab — the 4-map viewer of azivisu.ipynb.

Loads fibre maps from a .npz (or receives the just-computed maps from the Batch
tab), sorts the two fitted families into vertical / horizontal by proximity to
reference angles, and shows χ and Δχ (or degree of orientation) for each family.
Controls update the figure live.
"""

from __future__ import annotations

import os
import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QPushButton, QFileDialog, QMessageBox, QGroupBox,
)

from .mpl_canvas import MplCanvas
from .core import sort_into_vertical_horizontal, to_orientation, robust_vlim

CMAP_OPTIONS = ["twilight_shifted", "hsv", "viridis", "plasma", "coolwarm",
                "magma", "cividis", "turbo", "jet"]


class VisualizationTab(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.batch_result = None
        self._sort_cache = {}
        self._build()

    def _build(self):
        root = QVBoxLayout(self)

        # --- source row ---
        src = QHBoxLayout()
        self.load_btn = QPushButton("Load .npz…"); self.load_btn.clicked.connect(self._load_npz)
        self.use_batch_btn = QPushButton("Use current batch result"); self.use_batch_btn.clicked.connect(self._use_batch)
        self.source_lbl = QLabel("No maps loaded.")
        src.addWidget(self.load_btn)
        src.addWidget(self.use_batch_btn)
        src.addWidget(self.source_lbl, 1)
        root.addLayout(src)

        # --- canvas ---
        self.canvas = MplCanvas(figsize=(8.5, 8.5))
        self.axes = self.canvas.fig.subplots(2, 2).ravel()
        self._cbars = [None, None, None, None]
        root.addWidget(self.canvas, 1)

        # --- controls ---
        ctrl_box = QGroupBox("Controls")
        cg = QHBoxLayout(ctrl_box)

        cg.addWidget(QLabel("ROI"))
        self.roi_spin = QSpinBox(); self.roi_spin.setRange(1, 1); self.roi_spin.valueChanged.connect(self._update)
        cg.addWidget(self.roi_spin)

        cg.addWidget(QLabel("tol (°)"))
        self.tol_spin = QDoubleSpinBox(); self.tol_spin.setRange(5, 90); self.tol_spin.setValue(35); self.tol_spin.setSingleStep(2.5)
        self.tol_spin.valueChanged.connect(self._update)
        cg.addWidget(self.tol_spin)

        cg.addWidget(QLabel("χ cmap"))
        self.chi_cmap = QComboBox(); self.chi_cmap.addItems(CMAP_OPTIONS); self.chi_cmap.setCurrentText("twilight_shifted")
        self.chi_cmap.currentTextChanged.connect(self._update)
        cg.addWidget(self.chi_cmap)

        cg.addWidget(QLabel("Δχ cmap"))
        self.dchi_cmap = QComboBox(); self.dchi_cmap.addItems(CMAP_OPTIONS); self.dchi_cmap.setCurrentText("viridis")
        self.dchi_cmap.currentTextChanged.connect(self._update)
        cg.addWidget(self.dchi_cmap)

        self.orient_cb = QCheckBox("Show orientation (180−Δχ)/180")
        self.orient_cb.stateChanged.connect(self._update)
        cg.addWidget(self.orient_cb)

        self.export_btn = QPushButton("Export figure…"); self.export_btn.clicked.connect(self._export)
        cg.addWidget(self.export_btn)
        cg.addStretch(1)
        root.addWidget(ctrl_box)

        self._set_controls_enabled(False)

    def _set_controls_enabled(self, on):
        for w in (self.roi_spin, self.tol_spin, self.chi_cmap, self.dchi_cmap,
                  self.orient_cb, self.export_btn):
            w.setEnabled(on)

    # ---- data sources ---------------------------------------------------
    def _load_npz(self):
        start = ""
        if self.controller.dataset is not None:
            start = os.path.dirname(self.controller.dataset.scan_path)
        fn, _ = QFileDialog.getOpenFileName(self, "Load fibre maps", start, "NumPy (*.npz)")
        if not fn:
            return
        try:
            d = np.load(fn, allow_pickle=True)
            br = {
                "chi1_maps": d["chi1_maps"], "dchi1_maps": d["dchi1_maps"],
                "chi2_maps": d["chi2_maps"], "dchi2_maps": d["dchi2_maps"],
                "npeaks_maps": d["npeaks_maps"] if "npeaks_maps" in d.files else None,
                "saved_rois": [tuple(r) for r in np.atleast_2d(d["saved_rois"])],
                "n_rows": int(d["n_rows"]), "n_cols": int(d["n_cols"]),
            }
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return
        self._set_batch_result(br, source=os.path.basename(fn))

    def _use_batch(self):
        if self.controller.batch_result is None:
            QMessageBox.information(self, "No result", "Run a batch fit first (Batch tab).")
            return
        self._set_batch_result(self.controller.batch_result, source="current batch result")

    def on_batch_result(self):
        """Called by controller when a batch just finished."""
        if self.controller.batch_result is not None:
            self._set_batch_result(self.controller.batch_result, source="current batch result")

    def _set_batch_result(self, br, source):
        self.batch_result = br
        self._sort_cache.clear()
        n_roi = len(br["saved_rois"])
        self.roi_spin.blockSignals(True)
        self.roi_spin.setRange(1, max(1, n_roi))
        self.roi_spin.setValue(1)
        self.roi_spin.blockSignals(False)
        self.source_lbl.setText(
            f"{source} — {n_roi} ROI(s), maps {br['chi1_maps'].shape}")
        self._set_controls_enabled(True)
        self._update()

    # ---- plotting -------------------------------------------------------
    def _get_sorted(self, roi_id, tol):
        key = (roi_id, tol)
        if key not in self._sort_cache:
            self._sort_cache.clear()   # cache size 1
            iroi = roi_id - 1
            self._sort_cache[key] = sort_into_vertical_horizontal(
                self.batch_result["chi1_maps"][iroi],
                self.batch_result["dchi1_maps"][iroi],
                self.batch_result["chi2_maps"][iroi],
                self.batch_result["dchi2_maps"][iroi],
                vertical_ref=0.0, horizontal_ref=90.0, tol_deg=tol,
            )
        return self._sort_cache[key]

    def _update(self):
        if self.batch_result is None:
            return
        roi_id = self.roi_spin.value()
        tol = float(self.tol_spin.value())
        chi_cmap = self.chi_cmap.currentText()
        dchi_cmap = self.dchi_cmap.currentText()
        show_orient = self.orient_cb.isChecked()

        rmn, rmx = self.batch_result["saved_rois"][roi_id - 1]
        chi_v_raw, dchi_v, chi_h_raw, dchi_h = self._get_sorted(roi_id, tol)

        chi_v = chi_v_raw.copy()                                    # [-90, 90)
        chi_h = np.where(np.isfinite(chi_h_raw), chi_h_raw % 180.0, np.nan)  # [0, 180)

        if show_orient:
            panel_v = to_orientation(dchi_v); panel_h = to_orientation(dchi_h)
            dlabel = "orientation (0–1)"
            tv, th = "Vertical: orientation", "Horizontal: orientation"
            vlim_v = (0.0, 1.0); vlim_h = (0.0, 1.0)
        else:
            panel_v, panel_h = dchi_v, dchi_h
            dlabel = "deg"
            tv, th = "Vertical: Δχ", "Horizontal: Δχ"
            vlim_v = robust_vlim(panel_v); vlim_h = robust_vlim(panel_h)

        vlim_chi_v = robust_vlim(chi_v)
        vlim_chi_h = robust_vlim(chi_h)

        # clear axes + old colorbars
        for i, ax in enumerate(self.axes):
            ax.clear()
            if self._cbars[i] is not None:
                try:
                    self._cbars[i].remove()
                except Exception:
                    pass
                self._cbars[i] = None

        specs = [
            (chi_v, vlim_chi_v, chi_cmap, "Vertical: χ", "deg"),
            (panel_v, vlim_v, dchi_cmap, tv, dlabel),
            (chi_h, vlim_chi_h, chi_cmap, "Horizontal: χ", "deg"),
            (panel_h, vlim_h, dchi_cmap, th, dlabel),
        ]
        for i, (data, vlim, cmap, title, label) in enumerate(specs):
            ax = self.axes[i]
            im = ax.imshow(data, origin="upper", aspect="equal",
                           vmin=vlim[0], vmax=vlim[1], cmap=cmap)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            self._cbars[i] = self.canvas.fig.colorbar(im, ax=ax, label=label, fraction=0.046, pad=0.04)

        self.canvas.fig.suptitle(
            f"ROI {roi_id}: 2θ = {rmn:.4f}–{rmx:.4f}  (tol = {tol:.1f}°)", fontsize=12)
        self.canvas.fig.tight_layout(rect=(0, 0, 1, 0.96))
        self.canvas.draw()

    def _export(self):
        fn, _ = QFileDialog.getSaveFileName(
            self, "Export figure", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if not fn:
            return
        self.canvas.fig.savefig(fn, dpi=200, bbox_inches="tight")
