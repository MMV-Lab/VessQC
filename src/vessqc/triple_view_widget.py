"""
TripleViewWidget
================

Dieses Modul stellt ein QWidget bereit, das drei orthogonale Napari-Viewer
zur gleichzeitigen Ansicht eines 3D-Images enthält.

Jeder Viewer zeigt eine Schnittebene (axial, sagittal, koronar).
Die Schnitte sind synchronisiert, sodass eine Änderung in einer Ansicht
die Position der anderen automatisch aktualisiert.

Verwendet Napari und PyQt5.
"""

from qtpy.QtWidgets import QWidget, QGridLayout
from qtpy.QtCore import Qt
import napari
import numpy as np


class TripleViewWidget(QWidget):
    """Ein QWidget mit drei synchronisierten Napari-Viewern für 3D-Daten."""

    def __init__(self, image: np.ndarray, parent=None):
        super().__init__(parent)
        self.image = image
        self._init_ui()

    def _init_ui(self):
        """Erzeuge drei Viewer und richte Layout und Synchronisation ein."""
        layout = QGridLayout(self)
        layout.setSpacing(5)

        # Drei Viewer für drei Richtungen
        self.viewer_xy = napari.Viewer(ndisplay=2, title="XY-Schnitt")
        self.viewer_xz = napari.Viewer(ndisplay=2, title="XZ-Schnitt")
        self.viewer_yz = napari.Viewer(ndisplay=2, title="YZ-Schnitt")

        # Bilder laden
        self.layer_xy = self.viewer_xy.add_image(self.image, name="Image", blending="additive")
        self.layer_xz = self.viewer_xz.add_image(self.image, name="Image", blending="additive")
        self.layer_yz = self.viewer_yz.add_image(self.image, name="Image", blending="additive")

        # Viewer in Grid einfügen
        layout.addWidget(self.viewer_xy.window._qt_window, 0, 0)
        layout.addWidget(self.viewer_xz.window._qt_window, 0, 1)
        layout.addWidget(self.viewer_yz.window._qt_window, 1, 0, 1, 2)

        self.setLayout(layout)

        # Anfangsorientierung festlegen
        self._set_orientations()

        # Synchronisation aktivieren
        self._connect_sync_events()

    def _set_orientations(self):
        """Setze die Ansichten so, dass jede orthogonal zur anderen ist."""
        # XY-Schnitt = Z konstant
        self.viewer_xy.dims.order = (0, 1, 2)
        self.viewer_xy.camera.angles = (0, 0, 0)

        # XZ-Schnitt = Y konstant
        self.viewer_xz.dims.order = (1, 0, 2)
        self.viewer_xz.camera.angles = (90, 0, 0)

        # YZ-Schnitt = X konstant
        self.viewer_yz.dims.order = (2, 1, 0)
        self.viewer_yz.camera.angles = (0, 90, 0)

    def _connect_sync_events(self):
        """Synchronisiere die aktuelle Schnittposition zwischen allen Viewern."""

        def sync_from(source_viewer):
            zpos = list(source_viewer.dims.current_step)
            # zpos ist (z, y, x)
            for viewer in [self.viewer_xy, self.viewer_xz, self.viewer_yz]:
                if viewer is not source_viewer:
                    viewer.dims.set_current_step(0, zpos[0])
                    viewer.dims.set_current_step(1, zpos[1])
                    viewer.dims.set_current_step(2, zpos[2])

        # Verbinde die Ereignisse
        for viewer in [self.viewer_xy, self.viewer_xz, self.viewer_yz]:
            viewer.dims.events.current_step.connect(lambda e, v=viewer: sync_from(v))
