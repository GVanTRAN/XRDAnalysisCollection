"""
Main window.

Holds the shared state (the loaded Dataset, the list of submitted ROIs, the
current batch_result) and wires the four tabs together. The tabs never talk to
each other directly — they read/write this controller, so submitting a ROI in
Inspection makes it available to Batch, and finishing a Batch feeds Visualization.
"""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QMainWindow, QTabWidget, QApplication

from .tab_dataset import DatasetTab
from .tab_inspection import InspectionTab
from .tab_batch import BatchTab
from .tab_visualization import VisualizationTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Azimuthal Fit")
        self.resize(1200, 820)

        # ---- shared state ----
        self.dataset = None
        self.n_rows = 440
        self.n_cols = 160
        self.serpentine = False
        self.saved_rois = []          # list of (2theta_min, 2theta_max)
        self.batch_result = None

        # ---- tabs ----
        self.tabs = QTabWidget()
        self.dataset_tab = DatasetTab(self)
        self.inspection_tab = InspectionTab(self)
        self.batch_tab = BatchTab(self)
        self.viz_tab = VisualizationTab(self)

        self.tabs.addTab(self.dataset_tab, "1 · Dataset")
        self.tabs.addTab(self.inspection_tab, "2 · Inspection")
        self.tabs.addTab(self.batch_tab, "3 · Batch fit")
        self.tabs.addTab(self.viz_tab, "4 · Visualization")
        self.setCentralWidget(self.tabs)

        self.statusBar().showMessage("Load an HDF5 scan in the Dataset tab to begin.")

    # ---- state setters called by tabs -----------------------------------
    def set_dataset(self, dataset, n_rows, n_cols, serpentine):
        if self.dataset is not None:
            self.dataset.close()
        self.dataset = dataset
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.serpentine = serpentine
        self.inspection_tab.on_dataset_loaded()
        self.batch_tab.on_dataset_loaded()
        self.statusBar().showMessage(
            f"Dataset loaded — {dataset.n_patterns} patterns. "
            "Define ROIs in the Inspection tab.")
        self.tabs.setCurrentWidget(self.inspection_tab)

    def set_batch_result(self, batch_result):
        self.batch_result = batch_result
        self.viz_tab.on_batch_result()

    def closeEvent(self, event):
        if self.dataset is not None:
            self.dataset.close()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
