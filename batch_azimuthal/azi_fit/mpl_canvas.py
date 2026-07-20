"""Embedded matplotlib canvas (Qt Agg backend) with the standard toolbar."""

from __future__ import annotations

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

from PyQt6.QtWidgets import QWidget, QVBoxLayout


class MplCanvas(QWidget):
    def __init__(self, figsize=(8, 6), toolbar=True):
        super().__init__()
        self.fig = Figure(figsize=figsize, constrained_layout=False)
        self.canvas = FigureCanvas(self.fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if toolbar:
            self.toolbar = NavigationToolbar(self.canvas, self)
            layout.addWidget(self.toolbar)
        else:
            self.toolbar = None
        layout.addWidget(self.canvas)

    def draw(self):
        self.canvas.draw_idle()
