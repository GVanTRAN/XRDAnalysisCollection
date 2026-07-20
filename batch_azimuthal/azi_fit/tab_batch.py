"""
Batch tab — the parallel batch of batchaziparallel.ipynb, run in a background
QThread so the window never freezes and a real progress bar advances.

It fits every pattern for every submitted ROI, pairs peaks into two fibre
families per pixel, reshapes to (n_roi, n_rows, n_cols), and can save the maps
to a compressed .npz that the Visualization tab reads. The just-computed result
is also handed straight to the Visualization tab so you can skip save+reload.
"""

from __future__ import annotations

import os
import time
import contextlib

import numpy as np

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QSpinBox,
    QDoubleSpinBox, QPushButton, QProgressBar, QGroupBox, QPlainTextEdit,
    QFileDialog, QMessageBox, QCheckBox,
)
from PyQt6.QtCore import QThread, pyqtSignal

from .core import process_chunk


# =====================================================================
# Worker thread
# =====================================================================

class BatchWorker(QThread):
    progress = pyqtSignal(int, int)     # done_chunks, total_chunks
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(dict, float)
    failed = pyqtSignal(str)

    def __init__(self, scan_path, intensity_path, chi, roi_slices, saved_rois,
                 n_patterns, n_rows, n_cols, serpentine,
                 fit_kwargs, n_jobs, chunk):
        super().__init__()
        self.scan_path = scan_path
        self.intensity_path = intensity_path
        self.chi = chi
        self.roi_slices = roi_slices
        self.saved_rois = saved_rois
        self.n_patterns = n_patterns
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.serpentine = serpentine
        self.fit_kwargs = fit_kwargs
        self.n_jobs = n_jobs
        self.chunk = chunk
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            from joblib import Parallel, delayed
        except Exception as e:
            self.failed.emit(f"joblib is required for batch fitting: {e}")
            return

        n_roi = len(self.roi_slices)
        chunks = [(s, min(s + self.chunk, self.n_patterns))
                  for s in range(0, self.n_patterns, self.chunk)]
        total = len(chunks)

        chi1 = np.full((n_roi, self.n_patterns), np.nan)
        dchi1 = np.full((n_roi, self.n_patterns), np.nan)
        chi2 = np.full((n_roi, self.n_patterns), np.nan)
        dchi2 = np.full((n_roi, self.n_patterns), np.nan)
        npk = np.zeros((n_roi, self.n_patterns))

        self.log.emit(f"Fitting {self.n_patterns} patterns × {n_roi} ROI(s) "
                      f"in {total} chunks (n_jobs={self.n_jobs})…")
        t0 = time.time()
        done = 0
        self.progress.emit(0, total)

        # Consume results as they complete so we can (a) advance the progress
        # bar per chunk and (b) honour an abort request promptly.
        try:
            parallel = Parallel(n_jobs=self.n_jobs, return_as="generator")
            gen = parallel(
                delayed(process_chunk)(
                    s, e, self.scan_path, self.intensity_path, self.chi,
                    self.roi_slices, self.fit_kwargs
                )
                for (s, e) in chunks
            )
        except TypeError:
            # Older joblib without return_as="generator": fall back to a list.
            gen = Parallel(n_jobs=self.n_jobs)(
                delayed(process_chunk)(
                    s, e, self.scan_path, self.intensity_path, self.chi,
                    self.roi_slices, self.fit_kwargs
                )
                for (s, e) in chunks
            )

        try:
            for start, out in gen:
                if self._abort:
                    self.failed.emit("Aborted by user.")
                    return
                for ri, (c1, d1, c2, d2, np_) in out.items():
                    stop = start + c1.shape[0]
                    chi1[ri, start:stop] = c1
                    dchi1[ri, start:stop] = d1
                    chi2[ri, start:stop] = c2
                    dchi2[ri, start:stop] = d2
                    npk[ri, start:stop] = np_
                done += 1
                self.progress.emit(done, total)
        except Exception as e:
            self.failed.emit(f"Fit failed: {e}")
            return

        elapsed = time.time() - t0

        def to_maps(flat2d):
            m = flat2d.reshape(n_roi, self.n_rows, self.n_cols)
            if self.serpentine:
                m[:, 1::2, :] = m[:, 1::2, ::-1]
            return m

        try:
            batch_result = {
                "saved_rois": [tuple(r) for r in self.saved_rois],
                "chi1_maps": to_maps(chi1),
                "dchi1_maps": to_maps(dchi1),
                "chi2_maps": to_maps(chi2),
                "dchi2_maps": to_maps(dchi2),
                "npeaks_maps": to_maps(npk),
                "n_rows": self.n_rows,
                "n_cols": self.n_cols,
            }
        except Exception as e:
            self.failed.emit(f"Reshape to ({self.n_rows}×{self.n_cols}) failed: {e}")
            return

        self.finished_ok.emit(batch_result, elapsed)


# =====================================================================
# Batch tab
# =====================================================================

class BatchTab(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.worker = None
        self._build()

    def _build(self):
        root = QVBoxLayout(self)

        info = QLabel("Batch-fits every pattern using the ROIs submitted in the "
                      "Inspection tab. Set geometry in the Dataset tab.")
        info.setWordWrap(True)
        root.addWidget(info)

        # --- fit settings ---
        fit_box = QGroupBox("Fit settings")
        fg = QGridLayout(fit_box)
        self.snr_spin = QDoubleSpinBox(); self.snr_spin.setRange(0.1, 50); self.snr_spin.setValue(3.5); self.snr_spin.setSingleStep(0.5)
        self.bic_spin = QDoubleSpinBox(); self.bic_spin.setRange(0.0, 100); self.bic_spin.setValue(2.0); self.bic_spin.setSingleStep(1.0)
        self.sigmax_spin = QDoubleSpinBox(); self.sigmax_spin.setRange(1.0, 180); self.sigmax_spin.setValue(70.0)
        fg.addWidget(QLabel("snr_min"), 0, 0); fg.addWidget(self.snr_spin, 0, 1)
        fg.addWidget(QLabel("bic_margin"), 0, 2); fg.addWidget(self.bic_spin, 0, 3)
        fg.addWidget(QLabel("sigma_max (°)"), 0, 4); fg.addWidget(self.sigmax_spin, 0, 5)
        self.cand4_cb = QCheckBox("Allow up to 4 peaks (crossings)"); self.cand4_cb.setChecked(True)
        fg.addWidget(self.cand4_cb, 1, 0, 1, 3)
        root.addWidget(fit_box)

        # --- parallel settings ---
        par_box = QGroupBox("Parallel settings")
        pg = QGridLayout(par_box)
        self.njobs_spin = QSpinBox(); self.njobs_spin.setRange(-1, 256); self.njobs_spin.setValue(-1)
        self.chunk_spin = QSpinBox(); self.chunk_spin.setRange(1, 100000); self.chunk_spin.setValue(100)
        pg.addWidget(QLabel("n_jobs (-1 = all cores)"), 0, 0); pg.addWidget(self.njobs_spin, 0, 1)
        pg.addWidget(QLabel("chunk size"), 0, 2); pg.addWidget(self.chunk_spin, 0, 3)
        root.addWidget(par_box)

        # --- run controls ---
        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run batch fit"); self.run_btn.setStyleSheet("font-weight: bold;")
        self.run_btn.clicked.connect(self._run)
        self.abort_btn = QPushButton("Abort"); self.abort_btn.setEnabled(False)
        self.abort_btn.clicked.connect(self._abort)
        self.save_btn = QPushButton("Save maps (.npz)…"); self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save)
        run_row.addWidget(self.run_btn)
        run_row.addWidget(self.abort_btn)
        run_row.addWidget(self.save_btn)
        run_row.addStretch(1)
        root.addLayout(run_row)

        self.progress = QProgressBar(); self.progress.setValue(0)
        root.addWidget(self.progress)

        log_box = QGroupBox("Log")
        ll = QVBoxLayout(log_box)
        self.log_view = QPlainTextEdit(); self.log_view.setReadOnly(True)
        ll.addWidget(self.log_view)
        root.addWidget(log_box, 1)

        self.setEnabled(True)
        self.run_btn.setEnabled(False)

    def on_dataset_loaded(self):
        self.run_btn.setEnabled(True)
        self._log(f"Dataset ready: {self.controller.dataset.n_patterns} patterns, "
                  f"geometry {self.controller.n_rows}×{self.controller.n_cols}.")

    # ---- actions --------------------------------------------------------
    def _log(self, msg):
        self.log_view.appendPlainText(msg)

    def _run(self):
        ds = self.controller.dataset
        if ds is None:
            QMessageBox.warning(self, "Dataset", "Load a dataset first.")
            return
        rois = self.controller.saved_rois
        if not rois:
            QMessageBox.warning(self, "ROIs", "Submit at least one ROI in the Inspection tab first.")
            return

        n_rows, n_cols = self.controller.n_rows, self.controller.n_cols
        if n_rows * n_cols != ds.n_patterns:
            resp = QMessageBox.question(
                self, "Geometry mismatch",
                f"n_rows × n_cols = {n_rows*n_cols} ≠ {ds.n_patterns} patterns.\n"
                "Reshaping will fail. Run anyway?"
            )
            if resp != QMessageBox.StandardButton.Yes:
                return

        roi_slices = []
        for (rmn, rmx) in rois:
            sl = ds.roi_slice(rmn, rmx)
            if sl is None:
                QMessageBox.critical(self, "ROI", f"ROI {rmn:.3f}–{rmx:.3f} has no 2θ points.")
                return
            roi_slices.append(sl)

        candidates = (2, 4) if self.cand4_cb.isChecked() else (2,)
        fit_kwargs = dict(
            candidates=candidates,
            snr_min=float(self.snr_spin.value()),
            bic_margin=float(self.bic_spin.value()),
            sigma_max_deg=float(self.sigmax_spin.value()),
        )

        self.worker = BatchWorker(
            scan_path=ds.scan_path, intensity_path=ds.intensity_path,
            chi=ds.chi, roi_slices=roi_slices, saved_rois=list(rois),
            n_patterns=ds.n_patterns, n_rows=n_rows, n_cols=n_cols,
            serpentine=self.controller.serpentine, fit_kwargs=fit_kwargs,
            n_jobs=self.njobs_spin.value(), chunk=self.chunk_spin.value(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._log)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)

        self.run_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.progress.setValue(0)
        self._log("── Starting batch ──")
        self.worker.start()

    def _abort(self):
        if self.worker is not None:
            self.worker.abort()
            self._log("Abort requested (finishing current chunks)…")

    def _on_progress(self, done, total):
        self.progress.setMaximum(total)
        self.progress.setValue(done)

    def _on_done(self, batch_result, elapsed):
        self.run_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        n = self.controller.dataset.n_patterns * len(batch_result["saved_rois"])
        self._log(f"Done in {elapsed/60:.1f} min ({elapsed/max(n,1)*1000:.1f} ms/fit).")
        for ri in range(len(batch_result["saved_rois"])):
            n2 = int(np.sum(np.isfinite(batch_result["chi2_maps"][ri])))
            n1 = int(np.sum(np.isfinite(batch_result["chi1_maps"][ri]))) - n2
            self._log(f"ROI {ri+1}: 1-fibre pixels = {n1}, 2-fibre (crossing) = {n2}")
        # hand straight to visualization
        self.controller.set_batch_result(batch_result)
        self._log("Result loaded into the Visualization tab.")

    def _on_failed(self, msg):
        self.run_btn.setEnabled(True)
        self.abort_btn.setEnabled(False)
        self._log("ERROR: " + msg)
        QMessageBox.critical(self, "Batch", msg)

    def _save(self):
        br = self.controller.batch_result
        if br is None:
            return
        default = ""
        ds = self.controller.dataset
        if ds is not None:
            base = os.path.splitext(os.path.basename(ds.scan_path))[0]
            default = os.path.join(os.path.dirname(ds.scan_path), base + "_fiber_maps.npz")
        fn, _ = QFileDialog.getSaveFileName(self, "Save fibre maps", default, "NumPy (*.npz)")
        if not fn:
            return
        np.savez_compressed(
            fn,
            chi1_maps=br["chi1_maps"], dchi1_maps=br["dchi1_maps"],
            chi2_maps=br["chi2_maps"], dchi2_maps=br["dchi2_maps"],
            npeaks_maps=br["npeaks_maps"],
            saved_rois=np.array(br["saved_rois"]),
            n_rows=br["n_rows"], n_cols=br["n_cols"],
        )
        self._log(f"Saved: {fn}")
