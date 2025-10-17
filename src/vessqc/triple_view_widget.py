"""
TripleViewWidget (robuste Einbettung)
=====================================

Erzeugt drei Napari-Viewer als eingebettete Qt-Widgets ohne separate Fenster.
Wichtig: Viewer werden mit show=False erstellt und das Qt-Widget der Viewer
wird als Child in das Layout eingebunden.
"""

from qtpy.QtWidgets import QWidget, QGridLayout
from qtpy.QtCore import Qt
import napari
import numpy as np


class TripleViewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self._init_ui()

    def _init_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(5)

        # Viewer mit show=False anlegen — verhindert sofortiges Aufpoppen von Fenstern
        self.viewer_xy = napari.Viewer(ndisplay=2, title="XY-Schnitt", show=False)
        self.viewer_xz = napari.Viewer(ndisplay=2, title="XZ-Schnitt", show=False)
        self.viewer_yz = napari.Viewer(ndisplay=2, title="YZ-Schnitt", show=False)

        # Zugriff auf die Qt-Widgets der Viewer
        qt_xy = self.viewer_xy.window._qt_window
        qt_xz = self.viewer_xz.window._qt_window
        qt_yz = self.viewer_yz.window._qt_window

        # Sicherstellen, dass die Widgets als eingebettete Kinder arbeiten
        for w in (qt_xy, qt_xz, qt_yz):
            w.setParent(self)
            # Entferne WindowFlags, damit es keine eigenständigen Fenster sind
            w.setWindowFlags(Qt.Widget)
            # Optional: damit das Schließen nicht den Viewer zerstört
            w.setAttribute(Qt.WA_DeleteOnClose, False)

        # Layout: zwei oben, eines groß unten
        layout.addWidget(qt_xy, 0, 0)
        layout.addWidget(qt_xz, 0, 1)
        layout.addWidget(qt_yz, 1, 0, 1, 2)

        self.setLayout(layout)

        # Orientierung & Synchronisation vorbereiten
        self._set_orientations()
        #self._connect_sync_events()

    def load_image(self, image: np.ndarray):
        """Bild nachträglich laden. Bestehende Image-Layer werden ersetzt."""
        self.image = image

        # Entferne nur Image-Layer (keine vollständige clear(), um Nebeneffekte zu reduzieren)
        for v in (self.viewer_xy, self.viewer_xz, self.viewer_yz):
            # nur Image-Layer entfernen, falls vorhanden
            image_layers = [ly for ly in v.layers if getattr(ly, "is_image", False) or ly.ndim == 3]
            for ly in image_layers:
                v.layers.remove(ly)

        # Bild in alle Viewer einfügen
        self.viewer_xy.add_image(self.image, name="Image", blending="additive")
        self.viewer_xz.add_image(self.image, name="Image", blending="additive")
        self.viewer_yz.add_image(self.image, name="Image", blending="additive")

        # Setze die gleiche aktuelle Position in allen Viewern (sinnvoll als Start)
        steps = list(self.viewer_xy.dims.current_step)
        for v in (self.viewer_xy, self.viewer_xz, self.viewer_yz):
            for dim_idx, step in enumerate(steps):
                try:
                    v.dims.set_current_step(dim_idx, int(step))
                except Exception:
                    # einige viewer haben andere dims.order — sichere Ignorierung
                    pass

    def _set_orientations(self):
        # konservative, klassische Orientierungseinstellung
        try:
            self.viewer_xy.dims.order = (0, 1, 2)
            self.viewer_xy.camera.angles = (0, 0, 0)

            self.viewer_xz.dims.order = (1, 0, 2)
            self.viewer_xz.camera.angles = (90, 0, 0)

            self.viewer_yz.dims.order = (2, 1, 0)
            self.viewer_yz.camera.angles = (0, 90, 0)
        except Exception:
            # Falls dims.order momentan nicht gesetzt werden kann — ignorieren, läuft trotzdem weiter
            pass

    def _connect_sync_events(self):
        """Synchronisiere Schnittpositionen (robust gegen Events beim Laden)."""

        def sync_from(source_viewer, _event=None):
            # defensiv: lese aktuelle steps, ignoriere Ausnahmen
            try:
                steps = list(source_viewer.dims.current_step)
            except Exception:
                return
            for viewer in (self.viewer_xy, self.viewer_xz, self.viewer_yz):
                if viewer is source_viewer:
                    continue
                # Versuche, passende dims zu setzen — falls IndexError, ignoriere
                for i, s in enumerate(steps):
                    try:
                        viewer.dims.set_current_step(i, int(s))
                    except Exception:
                        # unterschiedliche dims.order möglich -> sichere Ignorierung
                        pass

        # Verbinde Ereignisse. Verwende eine benannte Funktion statt Lambdas in einer Schleife.
        self.viewer_xy.dims.events.current_step.connect(lambda e: sync_from(self.viewer_xy, e))
        self.viewer_xz.dims.events.current_step.connect(lambda e: sync_from(self.viewer_xz, e))
        self.viewer_yz.dims.events.current_step.connect(lambda e: sync_from(self.viewer_yz, e))
