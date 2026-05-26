"""
Multiple viewer widget
======================

This is an example of how to have more than one viewer in the same napari window.
Additional viewers state will be synchronized with the main viewer.
Switching to 3D display will only impact the main viewer.

This example also contains the option to enable cross that will be moved to the
current dims point (`viewer.dims.point`).

.. tags:: gui
"""

# Added defensive checks for layer existence during viewer teardown.
# Required for stable operation with newer napari / Python versions.

from copy import deepcopy

import numpy as np
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QSplitter,
)
from superqt.utils import qthrottled
from typing import Optional         # ChatGPT

import napari
from napari.components.viewer_model import ViewerModel
from napari.layers import Labels, Layer, Vectors
from napari.qt import QtViewer
from napari.utils.action_manager import action_manager
from napari.utils.events.event import WarningEmitter
from napari.utils.notifications import show_info

def copy_layer(layer: Layer, name: str = ''):
    res_layer = Layer.create(*layer.as_layer_data_tuple())
    res_layer.metadata['viewer_name'] = name
    return res_layer


def get_property_names(layer: Layer):
    klass = layer.__class__
    res = []
    for event_name, event_emitter in layer.events.emitters.items():
        if isinstance(event_emitter, WarningEmitter):
            continue
        if event_name in ('thumbnail', 'name'):
            continue
        if (
            isinstance(getattr(klass, event_name, None), property)
            and getattr(klass, event_name).fset is not None
        ):
            res.append(event_name)
    return res


def center_cross_on_mouse(
    viewer_model: napari.components.viewer_model.ViewerModel,
):
    """move the cross to the mouse position"""

    if not getattr(viewer_model, 'mouse_over_canvas', True):
        # There is no way for napari 0.4.15 to check if mouse is over sending canvas.
        show_info(
            'Mouse is not over the canvas. You may need to click on the canvas.'
        )
        return

    viewer_model.dims.current_step = tuple(
        np.round(
            [
                max(min_, min(p, max_)) / step
                for p, (min_, max_, step) in zip(
                    viewer_model.cursor.position, viewer_model.dims.range, strict=False
                )
            ]
        ).astype(int)
    )


action_manager.register_action(
    name='napari:move_point',
    command=center_cross_on_mouse,
    description='Move dims point to mouse position',
    keymapprovider=ViewerModel,
)

action_manager.bind_shortcut('napari:move_point', 'C')


class own_partial:
    """
    Workaround for deepcopy not copying partial functions
    (Qt widgets are not serializable)
    """

    def __init__(self, func, *args, **kwargs) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return self.func(*(self.args + args), **{**self.kwargs, **kwargs})

    def __deepcopy__(self, memodict=None):
        if memodict is None:
            memodict = {}
        return own_partial(
            self.func,
            *deepcopy(self.args, memodict),
            **deepcopy(self.kwargs, memodict),
        )


class QtViewerWrap(QtViewer):
    def __init__(self, main_viewer, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.main_viewer = main_viewer

    def _qt_open(
        self,
        filenames: list,
        stack: bool,
        plugin: Optional[str] = None,       # ChatGPT
        layer_type: Optional[str] = None,
        **kwargs,
    ):
        """for drag and drop open files"""
        self.main_viewer.window._qt_viewer._qt_open(
            filenames, stack, plugin, layer_type, **kwargs
        )


class CrossWidget(QCheckBox):
    """
    Widget to control the layer representing cross.
    Because of the performance reasons,
    the update of cross is throttled
    """

    # def __init__(self, viewer: napari.Viewer) -> None:
        # super().__init__('Add cross layer')
    def __init__(self, viewer: napari.Viewer, parent=None) -> None:
        super().__init__('Add cross layer', parent=parent)
        self.viewer = viewer
        self.setChecked(False)
        self.stateChanged.connect(self._update_cross_visibility)
        self.layer = None
        self._layer_updates_blocked = False
        self._dataset_load_in_progress = False
        self.viewer.dims.events.order.connect(self.update_cross)
        self.viewer.dims.events.ndim.connect(self._update_ndim)
        self.viewer.dims.events.current_step.connect(self.update_cross)
        self._extent = None

        self._update_extent()
        self.viewer.dims.events.connect(self._update_extent)

    def set_dataset_load_in_progress(self, active: bool) -> None:
        """Suppress throttled extent/cross updates while a dataset is loading."""
        self._dataset_load_in_progress = active

    def _compute_extent(self) -> None:
        """Calculate data extent (excluding the cross layer)."""
        layers = [
            layer
            for layer in self.viewer.layers
            if layer is not self.layer and layer.name != '.cross'
        ]
        if not layers:
            self._extent = None
            return
        self._extent = self.viewer.layers.get_extent(layers)

    def _refresh_extent_now(self) -> None:
        """Synchronous extent refresh (throttled handler may not have run yet)."""
        self._compute_extent()

    @qthrottled(leading=False)
    def _update_extent(self):
        if self._dataset_load_in_progress:
            return
        self._compute_extent()
        self.update_cross()

    def _make_cross_layer(self, ndim: int) -> Vectors:
        layer = Vectors(name='.cross', ndim=ndim)
        layer.edge_width = 1.5
        layer.vector_style = 'line'
        return layer

    def _append_cross_layer(self) -> None:
        """Add cross to the main viewer once (no auto-select)."""
        if self.layer is None or self.layer in self.viewer.layers:
            return
        layers = self.viewer.layers
        restore_activate = layers._activate_on_insert
        layers._activate_on_insert = False
        try:
            layers.append(self.layer)
        finally:
            layers._activate_on_insert = restore_activate

    def detach_from_viewer(self, *, block_updates: bool = False) -> None:
        """Remove cross from the layer list (e.g. before loading a new dataset)."""
        if block_updates:
            self._layer_updates_blocked = True
        if self.layer is not None and self.layer in self.viewer.layers:
            self.viewer.layers.remove(self.layer)

    def _has_data_layers(self) -> bool:
        """True if the viewer has at least one non-cross layer."""
        return any(layer.name != '.cross' for layer in self.viewer.layers)

    def attach_after_dataset_load(self, visible: bool, *, on_complete=None) -> None:
        """
        Insert cross once after data layers exist (hidden or visible).

        Deferred to the next event-loop tick so extent is current and layer
        controls are not built while the load path is still unwinding.
        """
        self._layer_updates_blocked = False
        self.layer = self._make_cross_layer(self.viewer.dims.ndim)

        def _finish_attach() -> None:
            if self.layer is None:
                if on_complete is not None:
                    on_complete()
                return
            if self.layer not in self.viewer.layers:
                self._append_cross_layer()
            self._refresh_extent_now()
            self.layer.visible = visible
            self.update_cross()
            if on_complete is not None:
                on_complete()

        QTimer.singleShot(0, _finish_attach)

    def _update_ndim(self, event):
        if self._layer_updates_blocked:
            self.layer = self._make_cross_layer(event.value)
            return

        was_in_list = self.layer is not None and self.layer in self.viewer.layers
        was_visible = was_in_list and self.layer.visible
        if was_in_list:
            self.viewer.layers.remove(self.layer)
        self.layer = self._make_cross_layer(event.value)
        if was_in_list or self.isChecked():
            self._append_cross_layer()
            self.layer.visible = was_visible if was_in_list else self.isChecked()
        self.update_cross()

    def _ensure_cross_layer(self) -> None:
        """Create the vectors layer once dims are known (if ndim event has not)."""
        if self.layer is not None:
            return
        self.layer = self._make_cross_layer(self.viewer.dims.ndim)

    def _update_cross_visibility(self, state):
        # Toggle visibility only when cross is already in the layer list.
        checked = state == Qt.Checked
        self._ensure_cross_layer()
        if checked and self.layer not in self.viewer.layers:
            if self._has_data_layers():
                self._append_cross_layer()
            else:
                return
        if self.layer in self.viewer.layers:
            self.layer.visible = checked
            self.update_cross()

    def update_cross(self):
        if self.layer is None or self.layer not in self.viewer.layers:
            return
        if self._extent is None:
            return

        step = np.asarray(self._extent.step, dtype=float)
        if step.size != self.layer.ndim:
            self._refresh_extent_now()
            if self._extent is None:
                return
            step = np.asarray(self._extent.step, dtype=float)
            if step.size != self.layer.ndim:
                return

        point = self.viewer.dims.current_step
        vec = []
        for i, (lower, upper) in enumerate(self._extent.world.T):
            if (upper - lower) / self._extent.step[i] == 1:
                continue
            point1 = list(point)
            point1[i] = (lower + self._extent.step[i] / 2) / self._extent.step[
                i
            ]
            point2 = [0 for _ in point]
            point2[i] = (upper - lower) / self._extent.step[i]
            vec.append((point1, point2))
        scale = np.asarray(self.layer.scale, dtype=float)
        if scale.shape == step.shape and np.any(scale != step):
            self.layer.scale = step
        self.layer.data = vec


class MultipleViewerWidget(QSplitter):
    """The main widget of the example."""

    # def __init__(self, viewer: napari.Viewer) -> None:
        # super().__init__()
    def __init__(self, viewer: napari.Viewer, parent=None) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self.viewer_model1 = ViewerModel(title='model1')
        self.viewer_model2 = ViewerModel(title='model2')
        self._block = False
        self._defer_aux_sync = False
        viewer_splitter = QSplitter(self)
        viewer_splitter.setOrientation(Qt.Orientation.Vertical)
        self.qt_viewer1 = QtViewerWrap(viewer, self.viewer_model1)
        self.qt_viewer2 = QtViewerWrap(viewer, self.viewer_model2)
        self.qt_viewer1.setParent(viewer_splitter)
        self.qt_viewer2.setParent(viewer_splitter)
        viewer_splitter.addWidget(self.qt_viewer1)
        viewer_splitter.addWidget(self.qt_viewer2)
        viewer_splitter.setContentsMargins(0, 0, 0, 0)

        self.addWidget(viewer_splitter)

        self.viewer.layers.events.inserted.connect(self._layer_added)
        self.viewer.layers.events.removed.connect(self._layer_removed)
        self.viewer.layers.events.moved.connect(self._layer_moved)
        self.viewer.layers.selection.events.active.connect(
            self._layer_selection_changed
        )
        self.viewer.dims.events.current_step.connect(self._point_update)
        self.viewer_model1.dims.events.current_step.connect(self._point_update)
        self.viewer_model2.dims.events.current_step.connect(self._point_update)
        self.viewer.dims.events.order.connect(self._order_update)
        self.viewer.events.reset_view.connect(self._reset_view)
        self.viewer_model1.events.status.connect(self._status_update)
        self.viewer_model2.events.status.connect(self._status_update)

        """ Lennarts Methode
        def temp():
            QtViewer._instances.clear()
            #raise NotImplementedError(f"Instances: {QtViewer._instances}")

        self.viewer.window._qt_window.destroyed.connect(temp)   # new line
        """

    def _status_update(self, event):
        self.viewer.status = event.value

    def _reset_view(self):
        self.viewer_model1.reset_view()
        self.viewer_model2.reset_view()

    def _layer_selection_changed(self, event):
        """
        update of current active layer
        """
        if self._block:
            return

        if event.value is None:
            self.viewer_model1.layers.selection.active = None
            self.viewer_model2.layers.selection.active = None
            return

        name = event.value.name                 # ChatGPT
        if name not in self.viewer_model1.layers:
            return
        if name not in self.viewer_model2.layers:
            return

        self.viewer_model1.layers.selection.active = self.viewer_model1.layers[
            event.value.name
        ]
        self.viewer_model2.layers.selection.active = self.viewer_model2.layers[
            event.value.name
        ]

    def _point_update(self, event):
        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:
            if model.dims is event.source:
                continue
            if len(self.viewer.layers) != len(model.layers):
                continue
            model.dims.current_step = event.value

    def _order_update(self):
        order = list(self.viewer.dims.order)
        if len(order) <= 2:

            if self.viewer_model1 is None:      # ChatGPT
                return
            if self.viewer_model2 is None:
                return

            self.viewer_model1.dims.order = order
            self.viewer_model2.dims.order = order
            return

        order[-3:] = order[-2], order[-3], order[-1]
        self.viewer_model1.dims.order = tuple(order)
        order = list(self.viewer.dims.order)
        order[-3:] = order[-1], order[-2], order[-3]
        self.viewer_model2.dims.order = tuple(order)

    def begin_dataset_load(self) -> None:
        """Defer layer mirroring until load completes (no hide/show)."""
        self._defer_aux_sync = True

    def end_dataset_load(self) -> None:
        """Mirror all data layers to aux viewers once after load."""
        self._defer_aux_sync = False
        self._sync_aux_layers_from_main()

    def set_aux_sync_deferred(self, deferred: bool) -> None:
        """Defer mirroring layers to auxiliary viewers (e.g. during dataset load)."""
        self._defer_aux_sync = deferred
        if not deferred:
            self._sync_aux_layers_from_main()

    def _sync_aux_layers_from_main(self) -> None:
        """Rebuild auxiliary viewer layers from the main viewer (excluding .cross)."""
        for model in (self.viewer_model1, self.viewer_model2):
            while len(model.layers) > 0:
                model.layers.pop()
        aux_index = 0
        for layer in self.viewer.layers:
            if layer.name == '.cross':
                continue
            self._add_layer_to_aux_viewers(aux_index, layer)
            aux_index += 1
        self._order_update()

    def _add_layer_to_aux_viewers(self, index: int, layer: Layer) -> None:
        """Mirror one main-viewer layer into the auxiliary viewers."""
        self.viewer_model1.layers.insert(index, copy_layer(layer, 'model1'))
        self.viewer_model2.layers.insert(index, copy_layer(layer, 'model2'))
        for name in get_property_names(layer):
            getattr(layer.events, name).connect(
                own_partial(self._property_sync, name)
            )

        if isinstance(layer, Labels):
            layer.events.set_data.connect(self._set_data_refresh)
            layer.events.labels_update.connect(self._set_data_refresh)
            self.viewer_model1.layers[layer.name].events.set_data.connect(
                self._set_data_refresh
            )
            self.viewer_model2.layers[layer.name].events.set_data.connect(
                self._set_data_refresh
            )
            layer.events.labels_update.connect(self._set_data_refresh)
            self.viewer_model1.layers[layer.name].events.labels_update.connect(
                self._set_data_refresh
            )
            self.viewer_model2.layers[layer.name].events.labels_update.connect(
                self._set_data_refresh
            )
        self.viewer_model1.layers[layer.name].events.data.connect(
            self._sync_data
        )
        self.viewer_model2.layers[layer.name].events.data.connect(
            self._sync_data
        )

        layer.events.name.connect(self._sync_name)

    def _layer_added(self, event):
        """add layer to additional viewers and connect all required events"""
        # Cross lives only on the main viewer; mirroring it spins up extra
        # QtViewer/VisPy work and can flash brief empty top-level windows.
        if event.value.name == '.cross':
            return
        if self._defer_aux_sync:
            return

        self._add_layer_to_aux_viewers(event.index, event.value)
        self._order_update()

    def _sync_name(self, event):
        """sync name of layers"""
        index = self.viewer.layers.index(event.source)
        self.viewer_model1.layers[index].name = event.source.name
        self.viewer_model2.layers[index].name = event.source.name

    def _sync_data(self, event):
        """sync data modification from additional viewers"""
        if self._block:
            return
        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:

            # layer = model.layers[event.source.name]
            name = event.source.name        # ChatGPT
            if name not in model.layers:
                continue
            layer = model.layers[name]

            if layer is event.source:
                continue
            try:
                self._block = True
                layer.data = event.source.data
            finally:
                self._block = False

    def _set_data_refresh(self, event):
        """
        synchronize data refresh between layers
        """
        if self._block:
            return
        for model in [self.viewer, self.viewer_model1, self.viewer_model2]:

            # layer = model.layers[event.source.name]
            name = event.source.name            # ChatGPT
            if name not in model.layers:
                continue
            layer = model.layers[name]

            if layer is event.source:
                continue
            try:
                self._block = True
                layer.refresh()
            finally:
                self._block = False

    def _layer_removed(self, event):
        """remove layer in all viewers"""
        if event.value.name == '.cross':
            return

        if event.index >= len(self.viewer_model1.layers):       # ChatGPT
            return
        if event.index >= len(self.viewer_model2.layers):
            return

        self.viewer_model1.layers.pop(event.index)
        self.viewer_model2.layers.pop(event.index)

    def _layer_moved(self, event):
        """update order of layers"""
        dest_index = (
            event.new_index
            if event.new_index < event.index
            else event.new_index + 1
        )
        self.viewer_model1.layers.move(event.index, dest_index)
        self.viewer_model2.layers.move(event.index, dest_index)

    def _property_sync(self, name, event):
        """Sync layers properties (except the name)"""
        if event.source not in self.viewer.layers:
            return
        try:
            self._block = True

            name2 = event.source.name        # ChatGPT
            if name2 not in self.viewer_model1.layers:
                return
            if name2 not in self.viewer_model2.layers:
                return

            setattr(
                self.viewer_model1.layers[event.source.name],
                name,
                getattr(event.source, name),
            )
            setattr(
                self.viewer_model2.layers[event.source.name],
                name,
                getattr(event.source, name),
            )
        finally:
            self._block = False


if __name__ == '__main__':
    from qtpy import QtWidgets
    QtWidgets.QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    # above two lines are needed to allow undocking the widget with
    # additional viewers
    view = napari.Viewer()
    dock_widget = MultipleViewerWidget(view)
    cross = CrossWidget(view)

    view.window.add_dock_widget(dock_widget, name='Sample')
    view.window.add_dock_widget(cross, name='Cross', area='left')

    view.open_sample('napari', 'cells3d')

    napari.run()
