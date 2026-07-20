"""
Inspection tab — the interactive equivalent of azi_inspection.ipynb.

Browse patterns with the index slider, click two 2θ positions on the 2D map to
define a ROI, submit it, and auto-fit periodic Gaussians in χ. Submitted ROIs
are stored on the controller so the Batch tab can reuse them.
"""

from __future__ import annotations

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QSpinBox, QComboBox,
    QDoubleSpinBox, QPushButton, QCheckBox, QListWidget, QGroupBox,
)
from PyQt6.QtCore import Qt

from matplotlib.colors import LogNorm, PowerNorm, Normalize
from matplotlib.gridspec import GridSpec

from .mpl_canvas import MplCanvas
from .core import periodic_model, fit_chi_peaks


class InspectionTab(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller

        self.current_clicks = []       # in-progress ROI boundary clicks
        self.fit_results = []          # aligned with controller.saved_rois
        self.fit_for_index = None

        self._build()

    # ---- UI -------------------------------------------------------------
    def _build(self):
        root = QVBoxLayout(self)

        # canvas
        self.canvas = MplCanvas(figsize=(13, 5.5))
        gs = GridSpec(1, 3, figure=self.canvas.fig,
                      width_ratios=[1.0, 0.05, 1.1], wspace=0.35)
        self.ax_map = self.canvas.fig.add_subplot(gs[0, 0])
        self.cax = self.canvas.fig.add_subplot(gs[0, 1])
        self.ax_chi = self.canvas.fig.add_subplot(gs[0, 2])
        self.canvas.fig.subplots_adjust(left=0.07, right=0.97, bottom=0.12, top=0.9)
        self._im = None
        self._cbar = None
        self.canvas.canvas.mpl_connect("button_press_event", self._onclick)
        root.addWidget(self.canvas, 1)

        # --- index + contrast controls ---
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Index"))
        self.index_slider = QSlider(Qt.Orientation.Horizontal)
        self.index_slider.setMinimum(0); self.index_slider.setMaximum(0)
        self.index_slider.valueChanged.connect(self._index_changed)
        self.index_spin = QSpinBox(); self.index_spin.setMinimum(0); self.index_spin.setMaximum(0)
        self.index_spin.valueChanged.connect(self.index_slider.setValue)
        self.index_slider.valueChanged.connect(self.index_spin.setValue)
        row1.addWidget(self.index_slider, 1)
        row1.addWidget(self.index_spin)

        self.contrast_cb = QComboBox()
        self.contrast_cb.addItems(["linear", "sqrt", "log"])
        self.contrast_cb.setCurrentText("sqrt")
        self.contrast_cb.currentTextChanged.connect(self._redraw)
        row1.addWidget(QLabel("Contrast"))
        row1.addWidget(self.contrast_cb)

        self.vmin_spin = QDoubleSpinBox(); self.vmin_spin.setRange(-1e12, 1e12); self.vmin_spin.setValue(1.0)
        self.vmax_spin = QDoubleSpinBox(); self.vmax_spin.setRange(-1e12, 1e12); self.vmax_spin.setValue(10.0)
        self.vmin_spin.valueChanged.connect(self._redraw)
        self.vmax_spin.valueChanged.connect(self._redraw)
        row1.addWidget(QLabel("Min")); row1.addWidget(self.vmin_spin)
        row1.addWidget(QLabel("Max")); row1.addWidget(self.vmax_spin)
        root.addLayout(row1)

        # --- ROI controls ---
        row2 = QHBoxLayout()
        self.autofit_cb = QCheckBox("Auto-fit on change"); self.autofit_cb.setChecked(True)
        self.submit_btn = QPushButton("Submit ROI"); self.submit_btn.clicked.connect(self._submit_roi)
        self.fit_btn = QPushButton("Fit (auto)"); self.fit_btn.clicked.connect(self._do_fit)
        self.clear_cur_btn = QPushButton("Clear current"); self.clear_cur_btn.clicked.connect(self._clear_current)
        self.clear_all_btn = QPushButton("Clear all ROIs"); self.clear_all_btn.clicked.connect(self._clear_all)
        row2.addWidget(self.autofit_cb)
        row2.addWidget(self.submit_btn)
        row2.addWidget(self.fit_btn)
        row2.addWidget(self.clear_cur_btn)
        row2.addWidget(self.clear_all_btn)
        row2.addStretch(1)
        root.addLayout(row2)

        # --- ROI list + results ---
        row3 = QHBoxLayout()
        roi_box = QGroupBox("Submitted ROIs (shared with Batch tab)")
        rb = QVBoxLayout(roi_box)
        self.roi_list = QListWidget()
        rb.addWidget(self.roi_list)
        row3.addWidget(roi_box, 1)

        res_box = QGroupBox("Fit result")
        rl = QVBoxLayout(res_box)
        self.result_label = QLabel("—"); self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        rl.addWidget(self.result_label)
        row3.addWidget(res_box, 2)
        root.addLayout(row3)

        self.status = QLabel("Load a dataset first, then click two 2θ positions to define a ROI.")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self._set_enabled(False)

    def _set_enabled(self, on):
        for w in (self.index_slider, self.index_spin, self.contrast_cb,
                  self.vmin_spin, self.vmax_spin, self.autofit_cb,
                  self.submit_btn, self.fit_btn, self.clear_cur_btn, self.clear_all_btn):
            w.setEnabled(on)

    # ---- called by controller when a dataset loads ----------------------
    def on_dataset_loaded(self):
        ds = self.controller.dataset
        self._set_enabled(True)
        self.index_slider.setMaximum(ds.n_patterns - 1)
        self.index_spin.setMaximum(ds.n_patterns - 1)
        default_idx = min(70399, ds.n_patterns - 1)
        self.index_slider.setValue(default_idx)
        self.fit_results = []
        self.fit_for_index = None
        self._refresh_roi_list()
        self._redraw()

    # ---- helpers --------------------------------------------------------
    def _refresh_roi_list(self):
        self.roi_list.clear()
        for i, (a, b) in enumerate(self.controller.saved_rois):
            self.roi_list.addItem(f"ROI {i+1}:  2θ = {a:.4f} – {b:.4f}")

    def _make_norm(self, intensity, contrast, vmin, vmax):
        if vmax <= vmin:
            vmin_p = np.nanpercentile(intensity, 1)
            vmax_p = np.nanpercentile(intensity, 99.5)
        else:
            vmin_p, vmax_p = vmin, vmax
        if contrast == "linear":
            return Normalize(vmin=vmin_p, vmax=vmax_p)
        if contrast == "sqrt":
            return PowerNorm(gamma=0.5, vmin=max(vmin_p, 0), vmax=vmax_p)
        # log
        positive = intensity[intensity > 0]
        if positive.size == 0:
            return Normalize(vmin=vmin_p, vmax=vmax_p)
        vmin_log = max(vmin_p, float(np.nanmin(positive)))
        vmax_log = vmax_p if vmax_p > vmin_log else float(np.nanmax(positive))
        return LogNorm(vmin=vmin_log, vmax=vmax_log)

    # ---- redraw ---------------------------------------------------------
    def _redraw(self):
        ds = self.controller.dataset
        if ds is None:
            return
        index = self.index_slider.value()
        contrast = self.contrast_cb.currentText()
        intensity = ds.intensity(index)
        norm = self._make_norm(intensity, contrast, self.vmin_spin.value(), self.vmax_spin.value())

        self.ax_map.clear()
        self.ax_chi.clear()
        self.cax.clear()

        # --- 2D map ---
        im = self.ax_map.pcolormesh(ds.two_theta, ds.chi, intensity, shading="auto", norm=norm)
        self.canvas.fig.colorbar(im, cax=self.cax, label="Intensity")
        self.ax_map.set_xlim(ds.x_min, ds.x_max)
        self.ax_map.set_ylim(ds.y_min, ds.y_max)
        self.ax_map.set_box_aspect(1)
        self.ax_map.set_xlabel(r"2$\theta$")
        self.ax_map.set_ylabel(r"$\chi$")
        self.ax_map.set_title(f"index = {index} | contrast = {contrast}")

        for x in self.current_clicks:
            self.ax_map.axvline(x, color="red", lw=2, ls="--")
        if len(self.current_clicks) == 2:
            x1, x2 = sorted(self.current_clicks)
            self.ax_map.axvspan(x1, x2, color="red", alpha=0.15)

        for i, (rmn, rmx) in enumerate(self.controller.saved_rois):
            self.ax_map.axvline(rmn, color="red", lw=1.5)
            self.ax_map.axvline(rmx, color="red", lw=1.5)
            self.ax_map.axvspan(rmn, rmx, color="red", alpha=0.08)
            self.ax_map.text(0.5 * (rmn + rmx), ds.y_max, f"ROI {i+1}",
                             color="red", ha="center", va="bottom")

        # --- I(chi) + fit overlay ---
        rois = self.controller.saved_rois
        fits_valid = (self.fit_for_index == index
                      and len(self.fit_results) == len(rois) and len(rois) > 0)

        if not rois:
            self.ax_chi.text(0.5, 0.5, "No submitted ROI yet",
                             ha="center", va="center", transform=self.ax_chi.transAxes)
        else:
            x_fit = np.linspace(ds.y_min, ds.y_max, 2000)
            for i, (rmn, rmx) in enumerate(rois):
                ivc = ds.i_vs_chi(index, rmn, rmx)
                if ivc is None:
                    continue
                line, = self.ax_chi.plot(ds.chi, ivc, ".", ms=3, alpha=0.35,
                                         label=f"ROI {i+1}: {rmn:.3f}–{rmx:.3f}")
                if fits_valid and self.fit_results[i] is not None:
                    res = self.fit_results[i]
                    color = line.get_color()
                    y_fit = periodic_model(x_fit, res["params"], res["n_peaks"])
                    self.ax_chi.plot(x_fit, y_fit, "-", lw=2, color=color)
                    for pk in res["peaks"]:
                        self.ax_chi.axvline(pk["center_chi"], color=color, ls="--", alpha=0.6)
            self.ax_chi.legend(fontsize=9)

        self.ax_chi.set_xlabel(r"$\chi$")
        self.ax_chi.set_ylabel("Integrated intensity")
        self.ax_chi.set_title(r"Integrated intensity vs $\chi$")
        self.ax_chi.set_xlim(ds.y_min, ds.y_max)
        self.ax_chi.grid(True)

        self.canvas.draw()

    # ---- interaction ----------------------------------------------------
    def _onclick(self, event):
        # ignore clicks while zoom/pan is active
        tb = self.canvas.toolbar
        if tb is not None and tb.mode:
            return
        if event.inaxes != self.ax_map or event.xdata is None:
            return
        x = float(event.xdata)
        if len(self.current_clicks) >= 2:
            self.current_clicks.clear()
        self.current_clicks.append(x)
        if len(self.current_clicks) == 1:
            self.status.setText(f"First 2θ boundary: {x:.4f}. Click the second boundary.")
        else:
            a, b = sorted(self.current_clicks)
            self.status.setText(f"Current ROI: 2θ = {a:.4f} to {b:.4f}. Click “Submit ROI”.")
        self._redraw()

    def _index_changed(self, _val):
        self.fit_results = []
        self.fit_for_index = None
        self.result_label.setText("—")
        self._redraw()
        if self.autofit_cb.isChecked() and self.controller.saved_rois:
            self._do_fit()

    def _submit_roi(self):
        if len(self.current_clicks) != 2:
            self.status.setText("Select two 2θ positions first.")
            return
        a, b = sorted(self.current_clicks)
        self.controller.saved_rois.append((a, b))
        self.current_clicks.clear()
        self.fit_results = []
        self.fit_for_index = None
        self._refresh_roi_list()
        self.status.setText(f"Submitted ROI {len(self.controller.saved_rois)}: 2θ = {a:.4f} to {b:.4f}.")
        self._redraw()
        if self.autofit_cb.isChecked():
            self._do_fit()

    def _do_fit(self):
        rois = self.controller.saved_rois
        if not rois:
            self.status.setText("Submit at least one ROI first.")
            return
        ds = self.controller.dataset
        index = self.index_slider.value()
        self.fit_results = []
        lines = []
        for i, (rmn, rmx) in enumerate(rois):
            ivc = ds.i_vs_chi(index, rmn, rmx)
            if ivc is None:
                self.fit_results.append(None)
                lines.append(f"ROI {i+1}: empty (no 2θ points)")
                continue
            try:
                res = fit_chi_peaks(ds.chi, ivc, candidates=(2, 4))
                self.fit_results.append(res)
                if not res["textured"]:
                    lines.append(f"ROI {i+1}: NA — no texture (SNR={res['snr']:.1f})")
                else:
                    peak_txt = " | ".join(
                        f"χ={pk['center_chi']:.1f}°, FWHM={pk['fwhm']:.1f}°"
                        for pk in res["peaks"]
                    )
                    lines.append(f"ROI {i+1} [n={res['n_peaks']}]: {peak_txt}")
            except Exception as e:
                self.fit_results.append(None)
                lines.append(f"ROI {i+1}: FIT FAILED ({e})")
        self.fit_for_index = index
        self.status.setText(f"Auto-fitted index {index}.")
        self.result_label.setText("\n".join(lines))
        self._redraw()

    def _clear_current(self):
        self.current_clicks.clear()
        self.status.setText("Current ROI cleared.")
        self._redraw()

    def _clear_all(self):
        self.current_clicks.clear()
        self.controller.saved_rois.clear()
        self.fit_results = []
        self.fit_for_index = None
        self.result_label.setText("—")
        self._refresh_roi_list()
        self.status.setText("All ROIs cleared.")
        self._redraw()
