#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROI-from-2D Viewer
==================

A small GUI focused on ONE task: building a real-space image from a region of
the 2D diffraction pattern.

Workflow
--------
1. Load the scan .h5 (set the 1D / Eiger paths and the map shape, click Load).
2. The mean 1D pattern is shown on top. Drag a horizontal span (or use the
   min/max boxes) to pick a 2theta range. That builds the navigator map.
3. Click a few pixels on the navigator map. Their raw 2D detector frames are
   summed on the fly and shown in the detector panel.
4. Draw a polygon on the summed detector frame to pick a feature.
5. Click "Build image from detector ROI". Every scan position's 2D frame is
   integrated inside that polygon, giving a new real-space image.
6. Save the result (.npy + .png + .json).

Requirements: python>=3.8, PyQt5, matplotlib>=3.5, numpy, h5py, hdf5plugin.
Run:          python roi2d_viewer.py
"""

import os
import sys
import json
import traceback

import numpy as np

# hdf5plugin must be imported before h5py opens compressed Eiger data.
try:
    import hdf5plugin  # noqa: F401
    os.environ["HDF5_PLUGIN_PATH"] = hdf5plugin.PLUGIN_PATH
    _HAS_HDF5PLUGIN = True
except Exception:
    _HAS_HDF5PLUGIN = False

import h5py

from PyQt5 import QtCore, QtWidgets

from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector, PolygonSelector
from matplotlib.path import Path as MplPath
from matplotlib.colors import Normalize

try:
    from matplotlib.backends.backend_qtagg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )
except Exception:
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg as FigureCanvas,
        NavigationToolbar2QT as NavigationToolbar,
    )

CMAP_CHOICES = ["viridis", "magma", "inferno", "plasma", "gray"]


# ======================================================================
# Data
# ======================================================================
class ScanData:
    """One loaded scan: the integrated 1D patterns plus the raw 2D frames."""

    def __init__(self):
        self.scan_path = None
        self.h5file = None
        self.x = None                 # 2theta axis, shape (n_points,)
        self.patterns_map = None      # (n_rows, n_cols, n_points) float32
        self.n_rows = 0
        self.n_cols = 0
        self.n_points = 0
        self.eiger_data = None        # lazy h5py dataset, not loaded into RAM
        self.eiger_ndim = None
        self.eiger_path = None

    def close(self):
        if self.h5file is not None:
            try:
                self.h5file.close()
            except Exception:
                pass
        self.h5file = None
        self.eiger_data = None

    def load(self, scan_path, twotheta_path, intensity_path,
             eiger_path, n_rows, n_cols):
        self.close()
        self.scan_path = scan_path

        with h5py.File(scan_path, "r") as f:
            if twotheta_path not in f:
                raise KeyError("2theta path not found:\n  %s" % twotheta_path)
            if intensity_path not in f:
                raise KeyError("intensity path not found:\n  %s" % intensity_path)
            x = np.asarray(f[twotheta_path][:])
            patterns = np.asarray(f[intensity_path][:])

        if patterns.ndim == 2:
            n_pixels, n_points = patterns.shape
            if n_pixels != n_rows * n_cols:
                raise ValueError(
                    "Map shape mismatch: %d x %d = %d, but intensity has %d rows."
                    % (n_rows, n_cols, n_rows * n_cols, n_pixels))
            patterns_map = patterns.reshape(n_rows, n_cols, n_points)
        elif patterns.ndim == 3:
            n_rows, n_cols = patterns.shape[0], patterns.shape[1]
            n_points = patterns.shape[2]
            patterns_map = patterns
        else:
            raise ValueError("intensity must be 2D or 3D, got %dD." % patterns.ndim)

        self.x = x.astype(np.float64)
        self.patterns_map = patterns_map.astype(np.float32)
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.n_points = int(n_points)

        self.eiger_data = None
        self.eiger_ndim = None
        self.eiger_path = None
        if eiger_path:
            self.h5file = h5py.File(scan_path, "r")
            if eiger_path in self.h5file:
                ds = self.h5file[eiger_path]
                if ds.ndim in (3, 4):
                    self.eiger_data = ds
                    self.eiger_ndim = ds.ndim
                    self.eiger_path = eiger_path

    def get_raw_frame(self, row, col):
        """Raw 2D detector frame for scan pixel (row, col), or None."""
        if self.eiger_data is None:
            return None
        try:
            if self.eiger_ndim == 3:
                idx = row * self.n_cols + col
                img = self.eiger_data[idx, :, :]
            else:
                img = self.eiger_data[row, col, :, :]
            return np.asarray(img, dtype=float)
        except Exception:
            return None


# ======================================================================
# Main window
# ======================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROI-from-2D Viewer")

        self.data = ScanData()

        # workflow state
        self.points = []          # list of (row, col) picked scan pixels
        self.sum2d = None         # running sum of their raw 2D frames
        self.det_verts = None     # detector polygon vertices (x, y)
        self.det_mask = None      # boolean detector mask (det_y, det_x)
        self.vimage = None        # resulting real-space image (n_rows, n_cols)

        self.map_xmin = None
        self.map_xmax = None
        self._warned = False

        # matplotlib handles
        self.fig_avg = self.fig_map = self.fig_det = self.fig_res = None
        self.canvas_avg = self.canvas_map = self.canvas_det = self.canvas_res = None
        self.ax_avg = self.ax_map = self.ax_det = self.ax_res = None
        self.im_map = self.im_det = self.im_res = None
        self.line_avg = None
        self.marker = None
        self.span = None
        self.det_poly = None
        self.cbar_map = self.cbar_det = self.cbar_res = None

        self._build_ui()
        self.resize(1300, 820)

    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        panel = QtWidgets.QWidget()
        panel.setFixedWidth(330)
        pv = QtWidgets.QVBoxLayout(panel)
        pv.setAlignment(QtCore.Qt.AlignTop)
        pv.setSpacing(6)
        root.addWidget(panel)

        # --- 1. file ---
        gb_file = QtWidgets.QGroupBox("1. Data file")
        fl = QtWidgets.QFormLayout(gb_file)
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText("/path/to/scan.h5")
        browse = QtWidgets.QPushButton("Browse...")
        browse.clicked.connect(self.on_browse)
        roww = QtWidgets.QHBoxLayout()
        roww.addWidget(self.path_edit); roww.addWidget(browse)
        w = QtWidgets.QWidget(); w.setLayout(roww)
        fl.addRow("File:", w)

        self.scan_combo = QtWidgets.QComboBox()
        self.scan_combo.setEditable(True)
        self.scan_combo.currentTextChanged.connect(self.on_scan_changed)
        fl.addRow("Scan entry:", self.scan_combo)

        self.twotheta_edit = QtWidgets.QLineEdit("/1.1/eiger_integrate/integrated/2th")
        self.intensity_edit = QtWidgets.QLineEdit("/1.1/eiger_integrate/integrated/intensity")
        self.eiger_edit = QtWidgets.QLineEdit("/1.1/measurement/eiger")
        fl.addRow("2theta path:", self.twotheta_edit)
        fl.addRow("intensity path:", self.intensity_edit)
        fl.addRow("eiger path:", self.eiger_edit)

        self.rows_spin = QtWidgets.QSpinBox()
        self.rows_spin.setRange(1, 1000000); self.rows_spin.setValue(440)
        self.cols_spin = QtWidgets.QSpinBox()
        self.cols_spin.setRange(1, 1000000); self.cols_spin.setValue(160)
        rc = QtWidgets.QHBoxLayout()
        rc.addWidget(QtWidgets.QLabel("rows")); rc.addWidget(self.rows_spin)
        rc.addWidget(QtWidgets.QLabel("cols")); rc.addWidget(self.cols_spin)
        wrc = QtWidgets.QWidget(); wrc.setLayout(rc)
        fl.addRow("Map shape:", wrc)

        load = QtWidgets.QPushButton("Load")
        load.setStyleSheet("font-weight: bold;")
        load.clicked.connect(self.on_load)
        fl.addRow(load)
        pv.addWidget(gb_file)

        # --- 2. 2theta ROI for the navigator map ---
        gb_roi = QtWidgets.QGroupBox("2. 2theta ROI (navigator map)")
        rl = QtWidgets.QFormLayout(gb_roi)
        self.xmin_spin = QtWidgets.QDoubleSpinBox()
        self.xmax_spin = QtWidgets.QDoubleSpinBox()
        for s in (self.xmin_spin, self.xmax_spin):
            s.setDecimals(4); s.setRange(-1e6, 1e6); s.setSingleStep(0.1)
        self.xmin_spin.setValue(9.5); self.xmax_spin.setValue(10.5)
        self.xmin_spin.editingFinished.connect(self.on_xrange_spin)
        self.xmax_spin.editingFinished.connect(self.on_xrange_spin)
        rl.addRow("2theta min:", self.xmin_spin)
        rl.addRow("2theta max:", self.xmax_spin)
        rl.addRow(QtWidgets.QLabel("(or drag a span on the top plot)"))
        pv.addWidget(gb_roi)

        # --- 3. pick points + build ---
        gb_pick = QtWidgets.QGroupBox("3. 2D-diffraction ROI")
        kl = QtWidgets.QVBoxLayout(gb_pick)
        hint = QtWidgets.QLabel(
            "a) click a few pixels on the map\n"
            "   (their raw 2D frames are summed)\n"
            "b) draw a polygon on the summed detector\n"
            "c) build the real-space image")
        hint.setWordWrap(True)
        kl.addWidget(hint)

        self.pick_lbl = QtWidgets.QLabel("Picked points: 0")
        kl.addWidget(self.pick_lbl)
        self.clear_picks_btn = QtWidgets.QPushButton("Clear picked points")
        self.clear_picks_btn.clicked.connect(self.on_clear_picks)
        kl.addWidget(self.clear_picks_btn)

        self.det_lbl = QtWidgets.QLabel("Detector ROI: none")
        kl.addWidget(self.det_lbl)
        self.clear_poly_btn = QtWidgets.QPushButton("Clear detector polygon")
        self.clear_poly_btn.clicked.connect(self.on_clear_det_polygon)
        kl.addWidget(self.clear_poly_btn)

        red_row = QtWidgets.QHBoxLayout()
        red_row.addWidget(QtWidgets.QLabel("Reduction (faster):"))
        self.reduce_combo = QtWidgets.QComboBox()
        self.reduce_combo.addItems(["1 (full)", "2", "4", "8"])
        red_row.addWidget(self.reduce_combo)
        wred = QtWidgets.QWidget(); wred.setLayout(red_row)
        kl.addWidget(wred)
        red_hint = QtWidgets.QLabel(
            "factor F builds an (rows/F) x (cols/F) image by\n"
            "reading only every F-th scan frame.")
        red_hint.setWordWrap(True)
        kl.addWidget(red_hint)

        self.build_btn = QtWidgets.QPushButton("Build image from detector ROI")
        self.build_btn.setStyleSheet("font-weight: bold;")
        self.build_btn.clicked.connect(self.build_image)
        self.build_btn.setEnabled(False)
        kl.addWidget(self.build_btn)
        pv.addWidget(gb_pick)

        # --- 4. detector display ---
        gb_det = QtWidgets.QGroupBox("4. Detector display")
        dl = QtWidgets.QFormLayout(gb_det)
        self.vmin_spin = QtWidgets.QDoubleSpinBox()
        self.vmax_spin = QtWidgets.QDoubleSpinBox()
        for s in (self.vmin_spin, self.vmax_spin):
            s.setDecimals(2); s.setRange(-1e9, 1e9)
        self.vmin_spin.setValue(0); self.vmax_spin.setValue(10)
        self.vmin_spin.valueChanged.connect(self.update_det_norm)
        self.vmax_spin.valueChanged.connect(self.update_det_norm)
        dl.addRow("count min:", self.vmin_spin)
        dl.addRow("count max:", self.vmax_spin)
        auto = QtWidgets.QPushButton("Auto max from frame")
        auto.clicked.connect(self.on_auto_vmax)
        dl.addRow(auto)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(CMAP_CHOICES)
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)
        dl.addRow("colormap:", self.cmap_combo)
        pv.addWidget(gb_det)

        # --- 5. save ---
        gb_save = QtWidgets.QGroupBox("5. Save")
        svl = QtWidgets.QFormLayout(gb_save)
        self.savename_edit = QtWidgets.QLineEdit("roi_image")
        svl.addRow("Base name:", self.savename_edit)
        save_btn = QtWidgets.QPushButton("Save ROI image (.npy/.png/.json)")
        save_btn.clicked.connect(self.save_image)
        svl.addRow(save_btn)
        pv.addWidget(gb_save)

        pv.addStretch(1)

        # --- right: four plot panels ---
        self.fig_avg = self._new_fig()
        self.fig_map = self._new_fig()
        self.fig_det = self._new_fig()
        self.fig_res = self._new_fig()

        pane_avg, self.canvas_avg = self._make_pane(self.fig_avg)
        pane_map, self.canvas_map = self._make_pane(self.fig_map)
        pane_det, self.canvas_det = self._make_pane(self.fig_det)
        pane_res, self.canvas_res = self._make_pane(self.fig_res)

        self.canvas_map.mpl_connect("button_press_event", self.on_click_map)

        hsplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        hsplit.setChildrenCollapsible(False)
        hsplit.addWidget(pane_map)
        hsplit.addWidget(pane_det)
        hsplit.addWidget(pane_res)
        hsplit.setSizes([300, 350, 350])

        vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        vsplit.setChildrenCollapsible(False)
        vsplit.addWidget(pane_avg)
        vsplit.addWidget(hsplit)
        vsplit.setSizes([260, 560])
        root.addWidget(vsplit, 1)

        self.statusBar().showMessage(
            "Ready." if _HAS_HDF5PLUGIN else
            "WARNING: hdf5plugin not installed - compressed Eiger frames may fail.")
        self._set_enabled(False)

    def _new_fig(self):
        f = Figure()
        try:
            f.set_layout_engine("constrained")
        except Exception:
            pass
        return f

    def _make_pane(self, fig):
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                             QtWidgets.QSizePolicy.Expanding)
        canvas.setMinimumSize(120, 120)
        tb = NavigationToolbar(canvas, self)
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        lay.addWidget(tb); lay.addWidget(canvas, 1)
        return w, canvas

    def _set_enabled(self, on):
        for w in (self.xmin_spin, self.xmax_spin, self.clear_picks_btn,
                  self.clear_poly_btn, self.reduce_combo,
                  self.vmin_spin, self.vmax_spin, self.cmap_combo):
            w.setEnabled(on)

    def _draw_all(self):
        for c in (self.canvas_avg, self.canvas_map, self.canvas_det,
                  self.canvas_res):
            if c is not None:
                c.draw_idle()

    # ==================================================================
    # File handling
    # ==================================================================
    def on_browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select scan .h5 file", "",
            "HDF5 files (*.h5 *.hdf5 *.nxs);;All files (*)")
        if path:
            self.path_edit.setText(path)
            self.populate_scan_entries(path)

    def populate_scan_entries(self, path):
        self.scan_combo.blockSignals(True)
        self.scan_combo.clear()
        try:
            with h5py.File(path, "r") as f:
                keys = list(f.keys())
            scan_keys = [k for k in keys if any(c.isdigit() for c in k)] or keys
            self.scan_combo.addItems(scan_keys)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Open failed", str(e))
        self.scan_combo.blockSignals(False)
        if self.scan_combo.count():
            self.scan_combo.setCurrentIndex(0)
            self.on_scan_changed(self.scan_combo.currentText())

    def on_scan_changed(self, key):
        key = (key or "").strip().strip("/")
        if not key:
            return
        self.twotheta_edit.setText("/%s/eiger_integrate/integrated/2th" % key)
        self.intensity_edit.setText("/%s/eiger_integrate/integrated/intensity" % key)
        self.eiger_edit.setText("/%s/measurement/eiger" % key)

    def on_load(self):
        path = self.path_edit.text().strip()
        if not path or not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "No file", "Pick a valid .h5 file first.")
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            self.data.load(
                path, self.twotheta_edit.text().strip(),
                self.intensity_edit.text().strip(),
                self.eiger_edit.text().strip(),
                self.rows_spin.value(), self.cols_spin.value())
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.critical(
                self, "Load error", "%s\n\n%s" % (e, traceback.format_exc()))
            return
        QtWidgets.QApplication.restoreOverrideCursor()

        xlo, xhi = float(self.data.x.min()), float(self.data.x.max())
        for s in (self.xmin_spin, self.xmax_spin):
            s.setRange(xlo, xhi)
        if not (xlo <= self.xmin_spin.value() < self.xmax_spin.value() <= xhi):
            self.xmin_spin.setValue(xlo + 0.45 * (xhi - xlo))
            self.xmax_spin.setValue(xlo + 0.55 * (xhi - xlo))
        self.map_xmin = self.xmin_spin.value()
        self.map_xmax = self.xmax_spin.value()

        self._reset_state()
        self._build_plots()
        self._set_enabled(True)

        det_msg = (self.data.eiger_path if self.data.eiger_data is not None
                   else "NO 2D detector data found")
        self.statusBar().showMessage(
            "Loaded %s  |  map %d x %d x %d  |  detector: %s"
            % (os.path.basename(path), self.data.n_rows, self.data.n_cols,
               self.data.n_points, det_msg))

    # ==================================================================
    # Plot setup
    # ==================================================================
    def _build_plots(self):
        for f in (self.fig_avg, self.fig_map, self.fig_det, self.fig_res):
            f.clear()
        self.ax_avg = self.fig_avg.add_subplot(111)
        self.ax_map = self.fig_map.add_subplot(111)
        self.ax_det = self.fig_det.add_subplot(111)
        self.ax_res = self.fig_res.add_subplot(111)
        cmap = self.cmap_combo.currentText()

        # mean 1D pattern + span selector
        avg = self.data.patterns_map.mean(axis=(0, 1))
        (self.line_avg,) = self.ax_avg.plot(self.data.x, avg, color="#222222")
        self.ax_avg.set_title("Mean 1D pattern  (drag to set 2theta ROI)")
        self.ax_avg.set_xlabel("2theta"); self.ax_avg.set_ylabel("Mean intensity")
        self.ax_avg.grid(alpha=0.3)

        # navigator map
        self.im_map = self.ax_map.imshow(
            self._compute_map(), origin="upper", aspect="equal", cmap=cmap)
        self._set_map_clim()
        self.marker = self.ax_map.scatter([], [], s=80, marker="x",
                                          linewidths=2.0, color="red")
        self.ax_map.set_title("Navigator map (click to pick points)")
        self.ax_map.set_xticks([]); self.ax_map.set_yticks([])
        self.cbar_map = self.fig_map.colorbar(self.im_map, ax=self.ax_map,
                                              fraction=0.046, pad=0.04)

        # detector frame
        self.im_det = self.ax_det.imshow(
            np.zeros((10, 10)), origin="upper", aspect="equal", cmap=cmap,
            norm=Normalize(self.vmin_spin.value(), self.vmax_spin.value()))
        if self.data.eiger_data is None:
            self.ax_det.set_title("No 2D detector data")
        else:
            self.ax_det.set_title("Summed detector (draw polygon ROI)")
        self.ax_det.set_xticks([]); self.ax_det.set_yticks([])
        self.cbar_det = self.fig_det.colorbar(self.im_det, ax=self.ax_det,
                                              fraction=0.046, pad=0.04)

        # result image
        self.im_res = self.ax_res.imshow(
            np.zeros((10, 10)), origin="upper", aspect="equal", cmap=cmap)
        self.ax_res.set_title("Detector-ROI image")
        self.ax_res.set_xticks([]); self.ax_res.set_yticks([])
        self.cbar_res = self.fig_res.colorbar(self.im_res, ax=self.ax_res,
                                              fraction=0.046, pad=0.04)

        # selectors
        self.span = SpanSelector(
            self.ax_avg, self.on_span_select, "horizontal",
            useblit=False, interactive=True,
            props=dict(alpha=0.2, facecolor="red"))
        try:
            self.span.extents = (self.map_xmin, self.map_xmax)
        except Exception:
            pass
        self._create_det_polygon()
        self._draw_all()

    def _create_det_polygon(self):
        if self.det_poly is not None:
            try:
                self.det_poly.set_active(False)
                self.det_poly.disconnect_events()
            except Exception:
                pass
        # drop any leftover polygon outline artists from a previous selection
        try:
            for ln in list(self.ax_det.lines):
                ln.remove()
        except Exception:
            pass
        self.det_poly = PolygonSelector(
            self.ax_det, self.on_det_polygon_complete, useblit=False,
            props=dict(color="red", linewidth=2, alpha=0.8))

    # ==================================================================
    # Navigator map
    # ==================================================================
    def _compute_map(self):
        x = self.data.x
        mask = (x >= self.map_xmin) & (x <= self.map_xmax)
        if not np.any(mask):
            mask = np.zeros_like(x, dtype=bool)
            mask[np.argmin(np.abs(x - 0.5 * (self.map_xmin + self.map_xmax)))] = True
        return self.data.patterns_map[:, :, mask].sum(axis=2)

    def _set_map_clim(self):
        m = self.im_map.get_array()
        try:
            vmin = np.nanpercentile(m, 2); vmax = np.nanpercentile(m, 98)
            if vmax <= vmin:
                vmax = vmin + 1
            self.im_map.set_clim(vmin, vmax)
        except Exception:
            pass

    def _refresh_map(self):
        self.map_xmin = min(self.xmin_spin.value(), self.xmax_spin.value())
        self.map_xmax = max(self.xmin_spin.value(), self.xmax_spin.value())
        self.im_map.set_data(self._compute_map())
        self._set_map_clim()
        self.ax_map.set_title("Navigator map  %.3f-%.3f" % (self.map_xmin, self.map_xmax))
        self.canvas_map.draw_idle()

    def on_span_select(self, xmin, xmax):
        if xmax <= xmin:
            return
        self.xmin_spin.blockSignals(True); self.xmax_spin.blockSignals(True)
        self.xmin_spin.setValue(xmin); self.xmax_spin.setValue(xmax)
        self.xmin_spin.blockSignals(False); self.xmax_spin.blockSignals(False)
        self._refresh_map()

    def on_xrange_spin(self):
        try:
            self.span.extents = (self.xmin_spin.value(), self.xmax_spin.value())
        except Exception:
            pass
        self._refresh_map()

    # ==================================================================
    # Point picking -> summed 2D
    # ==================================================================
    def on_click_map(self, event):
        if self.data.patterns_map is None:
            return
        if event.inaxes != self.ax_map or event.xdata is None or event.ydata is None:
            return
        col = int(round(event.xdata)); row = int(round(event.ydata))
        if not (0 <= row < self.data.n_rows and 0 <= col < self.data.n_cols):
            return
        self._add_point(row, col)

    def _add_point(self, row, col):
        if self.data.eiger_data is None:
            if not self._warned:
                QtWidgets.QMessageBox.information(
                    self, "No detector",
                    "This file has no raw 2D detector data, so there is nothing "
                    "to sum.")
                self._warned = True
            return
        frame = self.data.get_raw_frame(row, col)
        if frame is None:
            self.statusBar().showMessage("Could not read frame at row=%d col=%d." % (row, col))
            return

        if self.sum2d is None or self.sum2d.shape != frame.shape:
            self.sum2d = np.zeros_like(frame, dtype=np.float64)
            self.points = []
            self.det_verts = None
            self.det_mask = None
        self.sum2d += frame
        self.points.append((row, col))

        pts = np.array([[c, r] for (r, c) in self.points], dtype=float)
        self.marker.set_offsets(pts)
        self.pick_lbl.setText("Picked points: %d" % len(self.points))

        self._show_detector(self.sum2d,
                            "Summed detector (%d points) - draw polygon ROI"
                            % len(self.points))
        self.statusBar().showMessage(
            "Picked %d point(s). Draw a polygon on the summed detector." % len(self.points))
        self._draw_all()

    def on_clear_picks(self):
        self._reset_state()
        if self.marker is not None:
            self.marker.set_offsets(np.empty((0, 2)))
        if self.im_det is not None:
            self.im_det.set_data(np.zeros((10, 10)))
            self.im_det.set_extent([0, 10, 10, 0])
            self.ax_det.set_title("Summed detector (draw polygon ROI)")
        if self.im_res is not None:
            self.im_res.set_data(np.zeros((10, 10)))
            self.im_res.set_extent([0, 10, 10, 0])
            self.ax_res.set_title("Detector-ROI image")
        self._create_det_polygon()
        self.statusBar().showMessage("Picked points cleared.")
        self._draw_all()

    def _reset_state(self):
        self.points = []
        self.sum2d = None
        self.det_verts = None
        self.det_mask = None
        self.vimage = None
        self.pick_lbl.setText("Picked points: 0")
        self.det_lbl.setText("Detector ROI: none")
        self.build_btn.setEnabled(False)

    # ==================================================================
    # Detector polygon -> mask
    # ==================================================================
    def on_det_polygon_complete(self, verts):
        if verts is None or len(verts) < 3:
            return
        if self.sum2d is None:
            self.statusBar().showMessage("Pick a few scan points first.")
            return
        verts = np.asarray(verts)              # (N, 2) in detector coords (x, y)
        det_y, det_x = self.sum2d.shape
        xv, yv = np.meshgrid(np.arange(det_x) + 0.5, np.arange(det_y) + 0.5)
        pts = np.column_stack([xv.ravel(), yv.ravel()])
        mask = MplPath(verts).contains_points(pts).reshape(det_y, det_x)
        npix = int(mask.sum())
        if npix == 0:
            self.statusBar().showMessage("Detector polygon contains no pixels.")
            return
        self.det_verts = verts
        self.det_mask = mask
        self.det_lbl.setText("Detector ROI: %d px" % npix)
        self.build_btn.setEnabled(self.data.eiger_data is not None)
        self.statusBar().showMessage(
            "Detector ROI = %d pixels. Press 'Build image from detector ROI'." % npix)

    def on_clear_det_polygon(self):
        """Wipe the current detector polygon so a new one can be drawn."""
        self.det_verts = None
        self.det_mask = None
        self.det_lbl.setText("Detector ROI: none")
        self.build_btn.setEnabled(False)
        self._create_det_polygon()
        if self.sum2d is not None:
            self._show_detector(self.sum2d,
                                "Summed detector (%d points) - draw polygon ROI"
                                % len(self.points))
        self.canvas_det.draw_idle()
        self.statusBar().showMessage("Detector polygon cleared - draw a new one.")

    def _reduction_factor(self):
        try:
            return max(1, int(self.reduce_combo.currentText().split()[0]))
        except Exception:
            return 1

    # ==================================================================
    # Build the real-space image from the detector ROI
    # ==================================================================
    def build_image(self):
        if self.data.eiger_data is None:
            QtWidgets.QMessageBox.information(self, "No detector",
                                              "This file has no raw 2D detector data.")
            return
        mask = self.det_mask
        if mask is None:
            QtWidgets.QMessageBox.information(
                self, "No detector ROI",
                "Draw a polygon on the summed detector frame first.")
            return
        ys, xs = np.where(mask)
        if ys.size == 0:
            return

        # crop to the polygon bounding box so we read as little as possible
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        sub_flat = mask[y0:y1, x0:x1].ravel()

        nr, nc = self.data.n_rows, self.data.n_cols
        ed = self.data.eiger_data
        ndim = self.data.eiger_ndim
        F = self._reduction_factor()
        out_rows = max(1, nr // F)
        out_cols = max(1, nc // F)
        img = np.full((out_rows, out_cols), np.nan, dtype=np.float64)
        n_total = ed.shape[0] if ndim == 3 else nr

        progress = QtWidgets.QProgressDialog(
            "Integrating detector ROI over %d rows (reduction %dx)..."
            % (out_rows, F), "Cancel", 0, out_rows, self)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setMinimumDuration(0)

        try:
            for oi in range(out_rows):
                if progress.wasCanceled():
                    break
                r = oi * F
                try:
                    if ndim == 3:
                        start = r * nc
                        if start >= n_total:
                            break
                        # every F-th frame within this scan row
                        stop = min(start + out_cols * F, n_total)
                        block = ed[start:stop:F, y0:y1, x0:x1]
                    else:
                        block = ed[r, 0:out_cols * F:F, y0:y1, x0:x1]
                    block = np.asarray(block, dtype=np.float64)
                    m = block.shape[0]
                    img[oi, :m] = block.reshape(m, -1)[:, sub_flat].sum(axis=1)
                except Exception:
                    pass
                progress.setValue(oi + 1)
                if oi % 5 == 0:
                    QtWidgets.QApplication.processEvents()
        finally:
            progress.setValue(out_rows)

        if np.all(np.isnan(img)):
            self.statusBar().showMessage("Could not read any detector frames.")
            return
        self.vimage = img
        # stretch the reduced image across the full scan extent so it still
        # lines up spatially with the navigator map
        self._show_result(
            img, "Detector-ROI image (%d det px, reduction %dx)"
            % (int(mask.sum()), F), extent=[0, nc, nr, 0])
        finite = int(np.isfinite(img).sum())
        self.statusBar().showMessage(
            "Built %dx%d detector-ROI image (%d frames read, reduction %dx)."
            % (out_rows, out_cols, finite, F))

    # ==================================================================
    # Display helpers
    # ==================================================================
    def _show_detector(self, frame, title):
        if frame is None:
            self.ax_det.set_title("Detector frame unavailable")
            return
        det_y, det_x = frame.shape
        self.im_det.set_data(frame)
        self.im_det.set_extent([0, det_x, det_y, 0])
        self.ax_det.set_xlim(0, det_x); self.ax_det.set_ylim(det_y, 0)
        self.ax_det.set_title(title)
        self.update_det_norm()

    def _show_result(self, img, title, extent=None):
        ny, nx = img.shape
        if extent is None:
            extent = [0, nx, ny, 0]
        self.im_res.set_data(img)
        self.im_res.set_extent(extent)
        self.ax_res.set_xlim(extent[0], extent[1])
        self.ax_res.set_ylim(extent[2], extent[3])
        finite = img[np.isfinite(img)]
        if finite.size:
            try:
                vmin = np.nanpercentile(finite, 2); vmax = np.nanpercentile(finite, 98)
                if vmax <= vmin:
                    vmax = vmin + 1
                self.im_res.set_clim(vmin, vmax)
            except Exception:
                pass
        self.im_res.set_cmap(self.cmap_combo.currentText())
        self.ax_res.set_title(title)
        if self.cbar_res is not None:
            self.cbar_res.update_normal(self.im_res)
        self.canvas_res.draw_idle()

    def update_det_norm(self, *_):
        if self.im_det is None:
            return
        vmin = self.vmin_spin.value(); vmax = self.vmax_spin.value()
        if vmax <= vmin:
            vmax = vmin + 1
        self.im_det.set_norm(Normalize(vmin, vmax))
        if self.cbar_det is not None:
            self.cbar_det.update_normal(self.im_det)
        self.canvas_det.draw_idle()

    def on_auto_vmax(self):
        arr = self.sum2d
        if arr is None and self.im_det is not None:
            arr = self.im_det.get_array()
        if arr is None:
            return
        arr = np.asarray(arr, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return
        self.vmin_spin.blockSignals(True)
        self.vmin_spin.setValue(0.0)
        self.vmin_spin.blockSignals(False)
        self.vmax_spin.setValue(float(np.nanmax(finite)))

    def on_cmap_changed(self, name):
        for im in (self.im_map, self.im_det, self.im_res):
            if im is not None:
                im.set_cmap(name)
        self._draw_all()

    # ==================================================================
    # Save
    # ==================================================================
    def save_image(self):
        if self.vimage is None:
            QtWidgets.QMessageBox.information(self, "Nothing to save",
                                              "Build a detector-ROI image first.")
            return
        name = self.savename_edit.text().strip() or "roi_image"
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose save folder", os.getcwd())
        if not folder:
            return
        base = os.path.join(folder, name)
        np.save(base + ".npy", self.vimage)
        if self.det_mask is not None:
            np.save(base + "_det_mask.npy", self.det_mask)
        meta = {
            "scan_path": str(self.data.scan_path),
            "n_rows": self.data.n_rows, "n_cols": self.data.n_cols,
            "map_xmin": self.map_xmin, "map_xmax": self.map_xmax,
            "pick_points_row_col": self.points,
            "detector_polygon_xy": (np.asarray(self.det_verts).tolist()
                                    if self.det_verts is not None else []),
            "n_detector_pixels": int(self.det_mask.sum())
            if self.det_mask is not None else 0,
            "reduction_factor": self._reduction_factor(),
            "image_shape_rows_cols": list(self.vimage.shape),
        }
        with open(base + ".json", "w") as fh:
            json.dump(meta, fh, indent=2)
        try:
            self.fig_res.savefig(base + ".png", dpi=200, bbox_inches="tight")
        except Exception:
            pass
        self.statusBar().showMessage(
            "Saved %s.npy / .png / .json" % os.path.basename(base))

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self.data.close()
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
