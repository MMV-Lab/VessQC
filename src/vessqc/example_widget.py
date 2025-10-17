"""
example_widget.py
=================

Minimalbeispiel für ein Napari-Plugin-Widget, das drei orthogonale
Ansichten eines 3D-Images im *gleichen* Viewer zeigt.
"""

from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton
import numpy as np


class ExampleQWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        layout = QVBoxLayout(self)

        # Ein einfacher Button zum Laden eines Test-3D-Arrays
        self.btn = QPushButton("Testbild laden (3D)")
        self.btn.clicked.connect(self.load_test_image)
        layout.addWidget(self.btn)

        # drei Buttons für Ansichten
        self.btn_xy = QPushButton("XY-Ansicht")
        self.btn_xz = QPushButton("XZ-Ansicht")
        self.btn_yz = QPushButton("YZ-Ansicht")

        self.btn_xy.clicked.connect(lambda: self.set_view("xy"))
        self.btn_xz.clicked.connect(lambda: self.set_view("xz"))
        self.btn_yz.clicked.connect(lambda: self.set_view("yz"))

        layout.addWidget(self.btn_xy)
        layout.addWidget(self.btn_xz)
        layout.addWidget(self.btn_yz)

        self.setLayout(layout)

    def load_test_image(self):
        """Lädt ein kleines 3D-Testarray, um sicher zu gehen, dass alles funktioniert."""
        data = np.random.random((64, 64, 64))
        # Falls schon etwas da ist, entferne alte Layer
        for layer in list(self.viewer.layers):
            self.viewer.layers.remove(layer)
        self.layer = self.viewer.add_image(data, name="Test3D")
        self.viewer.dims.ndisplay = 2  # 2D-Schnittansicht

    def set_view(self, mode: str):
        """Setze die Kameraperspektive auf XY, XZ oder YZ."""
        if not hasattr(self, "layer"):
            print("Bitte zuerst ein 3D-Bild laden.")
            return

        if mode == "xy":
            self.viewer.dims.order = (0, 1, 2)
            self.viewer.camera.angles = (0, 0, 0)
        elif mode == "xz":
            self.viewer.dims.order = (1, 0, 2)
            self.viewer.camera.angles = (90, 0, 0)
        elif mode == "yz":
            self.viewer.dims.order = (2, 1, 0)
            self.viewer.camera.angles = (0, 90, 0)

        print(f"Ansicht gesetzt auf {mode.upper()}")
