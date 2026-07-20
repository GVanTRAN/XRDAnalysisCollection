"""
Dataset tab — load the HDF5 scan and configure the shared state that every
other tab reads: the three internal HDF5 paths (intensity / 2theta / chi) and
the scan geometry (n_rows x n_cols) used to reshape batch results into a map.
"""

from __future__ import annotations

import os

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QSpinBox, QCheckBox, QFileDialog, QGroupBox,
    QPlainTextEdit, QMessageBox,
)
from PyQt6.QtCore import Qt

from .dataset import Dataset, peek_tree

# Defaults lifted from the notebooks so an existing user recognises them.
DEFAULT_INTENSITY = "/1.1/eiger_integrate_test1/integrated/intensity"
DEFAULT_TWOTHETA = "/1.1/eiger_integrate_test1/integrated/2th"
DEFAULT_CHI = "/1.1/eiger_integrate_test1/integrated/chi"


class DatasetTab(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self._build()

    def _build(self):
        root = QVBoxLayout(self)

        # ---- file picker -------------------------------------------------
        file_box = QGroupBox("HDF5 scan file")
        fl = QHBoxLayout(file_box)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("/path/to/scan.h5")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        scan_btn = QPushButton("Inspect paths")
        scan_btn.clicked.connect(self._scan_paths)
        fl.addWidget(self.path_edit, 1)
        fl.addWidget(browse)
        fl.addWidget(scan_btn)
        root.addWidget(file_box)

        # ---- internal HDF5 paths ----------------------------------------
        paths_box = QGroupBox("Internal HDF5 dataset paths")
        pg = QGridLayout(paths_box)
        self.intensity_cb = QComboBox(); self.intensity_cb.setEditable(True)
        self.twotheta_cb = QComboBox(); self.twotheta_cb.setEditable(True)
        self.chi_cb = QComboBox(); self.chi_cb.setEditable(True)
        self.intensity_cb.setCurrentText(DEFAULT_INTENSITY)
        self.twotheta_cb.setCurrentText(DEFAULT_TWOTHETA)
        self.chi_cb.setCurrentText(DEFAULT_CHI)
        pg.addWidget(QLabel("intensity"), 0, 0)
        pg.addWidget(self.intensity_cb, 0, 1)
        pg.addWidget(QLabel("2θ (twotheta)"), 1, 0)
        pg.addWidget(self.twotheta_cb, 1, 1)
        pg.addWidget(QLabel("χ (chi)"), 2, 0)
        pg.addWidget(self.chi_cb, 2, 1)
        root.addWidget(paths_box)

        # ---- geometry ----------------------------------------------------
        geom_box = QGroupBox("Scan geometry (for reshaping batch maps)")
        gg = QGridLayout(geom_box)
        self.rows_spin = QSpinBox(); self.rows_spin.setRange(1, 1_000_000); self.rows_spin.setValue(440)
        self.cols_spin = QSpinBox(); self.cols_spin.setRange(1, 1_000_000); self.cols_spin.setValue(160)
        self.serpentine_cb = QCheckBox("Serpentine scan (flip every other row)")
        gg.addWidget(QLabel("n_rows"), 0, 0)
        gg.addWidget(self.rows_spin, 0, 1)
        gg.addWidget(QLabel("n_cols"), 0, 2)
        gg.addWidget(self.cols_spin, 0, 3)
        gg.addWidget(self.serpentine_cb, 1, 0, 1, 4)
        root.addWidget(geom_box)

        # ---- load button + status ---------------------------------------
        load_row = QHBoxLayout()
        self.load_btn = QPushButton("Load dataset")
        self.load_btn.setStyleSheet("font-weight: bold;")
        self.load_btn.clicked.connect(self._load)
        self.status = QLabel("No dataset loaded.")
        self.status.setWordWrap(True)
        load_row.addWidget(self.load_btn)
        load_row.addWidget(self.status, 1)
        root.addLayout(load_row)

        # ---- HDF5 tree preview ------------------------------------------
        tree_box = QGroupBox("HDF5 contents")
        tl = QVBoxLayout(tree_box)
        self.tree_view = QPlainTextEdit()
        self.tree_view.setReadOnly(True)
        self.tree_view.setPlaceholderText("Click “Inspect paths” to list datasets in the file.")
        tl.addWidget(self.tree_view)
        root.addWidget(tree_box, 1)

    # ---- actions --------------------------------------------------------
    def _browse(self):
        start = os.path.dirname(self.path_edit.text()) or os.path.expanduser("~")
        fn, _ = QFileDialog.getOpenFileName(
            self, "Select HDF5 scan", start, "HDF5 files (*.h5 *.hdf5 *.nxs);;All files (*)"
        )
        if fn:
            self.path_edit.setText(fn)
            self._scan_paths()

    def _scan_paths(self):
        path = self.path_edit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "File", "Pick an existing HDF5 file first.")
            return
        try:
            all_paths, dset_paths = peek_tree(path)
        except Exception as e:
            QMessageBox.critical(self, "HDF5 error", str(e))
            return

        self.tree_view.setPlainText("\n".join(all_paths))
        for cb, default in (
            (self.intensity_cb, DEFAULT_INTENSITY),
            (self.twotheta_cb, DEFAULT_TWOTHETA),
            (self.chi_cb, DEFAULT_CHI),
        ):
            cur = cb.currentText()
            cb.clear()
            cb.addItems(["/" + p for p in dset_paths] + [p for p in dset_paths])
            # restore selection / default if present
            if cur:
                cb.setCurrentText(cur)
            elif default in ["/" + p for p in dset_paths]:
                cb.setCurrentText(default)

    def _load(self):
        path = self.path_edit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "File", "Pick an existing HDF5 file first.")
            return
        ip = self.intensity_cb.currentText().strip()
        tp = self.twotheta_cb.currentText().strip()
        cp = self.chi_cb.currentText().strip()

        try:
            ds = Dataset(path, ip, tp, cp)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        n_rows = self.rows_spin.value()
        n_cols = self.cols_spin.value()
        if n_rows * n_cols != ds.n_patterns:
            resp = QMessageBox.question(
                self, "Geometry mismatch",
                f"n_rows × n_cols = {n_rows*n_cols} but the file has "
                f"{ds.n_patterns} patterns.\n\nLoad anyway? "
                f"(Batch reshaping will fail until the geometry matches.)"
            )
            if resp != QMessageBox.StandardButton.Yes:
                ds.close()
                return

        self.controller.set_dataset(
            ds, n_rows=n_rows, n_cols=n_cols,
            serpentine=self.serpentine_cb.isChecked(),
        )
        self.status.setText(
            f"Loaded {os.path.basename(path)} — {ds.n_patterns} patterns, "
            f"shape ({ds.n_patterns}, {ds.n_chi}, {ds.n_twotheta}); "
            f"2θ [{ds.x_min:.3f}, {ds.x_max:.3f}], χ [{ds.y_min:.1f}, {ds.y_max:.1f}]."
        )
