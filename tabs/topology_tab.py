import copy
import hashlib
import os

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSizePolicy,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QListWidget,
    QListWidgetItem, QCheckBox, QRadioButton, QButtonGroup, QGroupBox,
    QAbstractItemView, QApplication, QMessageBox, QSpinBox,
)
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter, QIcon
from PyQt5.QtCore import Qt, pyqtSignal, QThread

from components.ui_help import HELP_RSML, show_help
from core.model import mask_is_disconnected
from core.root_topology import (
    Topology,
    TopologyError,
    automatic_topology,
    build_graph,
    import_plant,
    read_rsml_plants,
    restore_saved,
)

# Same order colors as the Phenomics Inspector.
_ORDER_COLORS = {
    0: (255, 0, 0),
    1: (0, 255, 0),
    2: (0, 255, 255),
    3: (255, 0, 255),
}
_DEEPER_COLOR = (255, 255, 0)
# Tab20 without the two reds, so a lateral is never mistaken for the main root.
_TAB20 = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34),
    (23, 190, 207), (174, 199, 232), (255, 187, 120), (152, 223, 138),
    (197, 176, 213), (196, 156, 148), (247, 182, 210), (199, 199, 199),
    (219, 219, 141), (158, 218, 229),
]
_HIT_PX = 10


class _RsmlView(QGraphicsView):
    """Zoomable plant view. Wheel zooms, Ctrl or the middle button pans."""

    clicked = pyqtSignal(int, int, int)
    double_clicked = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self._scene = QGraphicsScene(self)
        self._item = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self.setBackgroundBrush(QColor("#1e1e1e"))
        self.setDragMode(QGraphicsView.NoDrag)
        self.setFrameShape(QGraphicsView.NoFrame)
        self._panning = False
        self._press = None
        self._press_mods = Qt.NoModifier
        self._saw_double = False

    def set_image(self, pixmap, fit=False):
        self._item.setPixmap(pixmap)
        self._scene.setSceneRect(0, 0, max(1, pixmap.width()), max(1, pixmap.height()))
        if fit:
            self.fit_image()

    def fit_image(self):
        if self._item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self._item, Qt.KeepAspectRatio)

    def zoom_by(self, factor):
        self.scale(factor, factor)

    def wheelEvent(self, event):
        self.zoom_by(1.15 if event.angleDelta().y() > 0 else 1 / 1.15)

    def mousePressEvent(self, event):
        self._press = event.pos()
        self._press_mods = event.modifiers()
        if event.button() == Qt.MiddleButton or (
            event.button() == Qt.LeftButton and (event.modifiers() & Qt.ControlModifier)
        ):
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._panning = True
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        self._saw_double = True
        if event.button() == Qt.LeftButton:
            scene = self.mapToScene(event.pos())
            self.double_clicked.emit(int(round(scene.x())), int(round(scene.y())))
        event.accept()

    def mouseReleaseEvent(self, event):
        press = self._press
        mods = self._press_mods
        self._press = None
        saw_double = self._saw_double
        self._saw_double = False
        if self._panning:
            self.setDragMode(QGraphicsView.NoDrag)
            self._panning = False
        super().mouseReleaseEvent(event)
        if saw_double or event.button() != Qt.LeftButton or press is None:
            return
        if (event.pos() - press).manhattanLength() > 4:
            return
        scene = self.mapToScene(event.pos())
        self.clicked.emit(int(round(scene.x())), int(round(scene.y())), int(mods))


class _TraceWorker(QThread):
    """Build one plant's RSML off the interface thread."""

    ready = pyqtSignal(int, object)
    failed = pyqtSignal(int, str, bool)

    def __init__(self, serial, func, args):
        super().__init__()
        self.serial = serial
        self.func = func
        self.args = args

    def run(self):
        try:
            self.ready.emit(self.serial, self.func(*self.args))
        except TopologyError as exc:
            self.failed.emit(self.serial, str(exc), True)
        except Exception as exc:
            self.failed.emit(self.serial, str(exc), False)


def _trace_plant(
    patch, x_off, y_off, bin_crop, img_shape, prune, image_name, genotype, uid,
    record, decided, rsml_path,
):
    built = build_graph(patch, x_off, y_off, bin_crop, img_shape, prune)
    if built is None:
        raise TopologyError("This plant has no skeleton to trace.")
    graph, colored, full_skel = built
    note = ""
    if record:
        plant_xml = read_rsml_plants(rsml_path).get(int(uid)) if rsml_path else None
        restored = restore_saved(
            graph, colored, full_skel, image_name, uid, genotype, prune, record, plant_xml,
        )
        if restored is not None:
            return {"uid": uid, "topology": restored, "note": ""}
        note = "The applied topology no longer matches this skeleton. Showing the automatic tracing."
    if not decided:
        plant_xml = read_rsml_plants(rsml_path).get(int(uid)) if rsml_path else None
        if plant_xml is not None:
            imported = import_plant(graph, colored, plant_xml, prune)
            if imported is not None:
                # The file is already on disk. Keeping this record only for the view;
                # copying it into the overrides would make the plate look unsaved.
                return {"uid": uid, "topology": imported, "note": note}
            note = "The exported RSML does not match this skeleton. Showing the automatic tracing."
    topology = automatic_topology(graph, colored, full_skel, image_name, uid, genotype, prune)
    return {"uid": uid, "topology": topology, "note": note}


def _swatch(color):
    pixmap = QPixmap(14, 14)
    pixmap.fill(QColor(*color))
    return QIcon(pixmap)


class RsmlCanvasTab(QWidget):
    state_changed = pyqtSignal()
    topology_applied = pyqtSignal(int)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.path_provider = None
        self.prune_provider = None
        self.scale_provider = None
        self.topology = None
        self.crop_x = 0
        self.crop_y = 0
        self._image = None
        self.selected_ids = []
        self.selected_edge = None
        self._click_point = None
        self.line_width = 2
        self._tool = None
        self._load_serial = 0
        self._trace_workers = []
        self._topology_cache = {}
        self.split_mode = False
        self.isolate = False
        self.view_mode = "lateral"
        self._colors = {}
        self._filling_list = False
        self._undo = []
        self._suspend = False
        self._note = ""
        self._loaded_uid = None
        self._fit_next = False
        self._seen_token = {}
        self._content_epoch = -1
        self._baseline = None
        self._baseline_topology = None
        self._reverting = False
        self.init_ui()
        self.model.register_data_callback(self.on_data_changed)
        self.model.register_selection_callback(self.on_selection_changed)

    def init_ui(self):
        layout = QVBoxLayout(self)
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setStyleSheet(
            "font-size: 14px; font-weight: bold; margin: 5px; background-color: #eee; padding: 6px;"
        )
        layout.addWidget(self.info_label)

        self.view = _RsmlView()
        self.view.clicked.connect(self.on_clicked)
        self.view.double_clicked.connect(self.on_double_clicked)

        body = QHBoxLayout()
        side = QVBoxLayout()
        view_group = QGroupBox("View")
        view_layout = QVBoxLayout()
        self.rad_view_raw = QRadioButton("Raw image")
        self.rad_view_dense = QRadioButton("Dense segmentation")
        self.rad_view_rsml = QRadioButton("RSML")
        self.rad_view_lateral = QRadioButton("Each lateral root")
        self.rad_view_raw.setToolTip("Scanner image without overlays.")
        self.rad_view_dense.setToolTip("Main root and lateral root labels painted in Annotation.")
        self.rad_view_rsml.setToolTip("Main red, lateral green, tertiary cyan, quaternary magenta.")
        self.rad_view_lateral.setToolTip("Main root in red. Every other root has its own color.")
        self.rad_view_lateral.setChecked(True)
        self.view_group = QButtonGroup(self)
        for rad in (self.rad_view_raw, self.rad_view_dense, self.rad_view_rsml, self.rad_view_lateral):
            self.view_group.addButton(rad)
            rad.toggled.connect(self._on_view_mode)
            view_layout.addWidget(rad)
        view_group.setLayout(view_layout)
        side.addWidget(view_group)
        self.chk_only = QCheckBox("Show only the selected root")
        self.chk_only.toggled.connect(self._set_isolate)
        side.addWidget(self.chk_only)
        self.chk_direction = QCheckBox("Show direction")
        self.chk_direction.setToolTip("Draw an arrow from the beginning of each root to its end.")
        self.chk_direction.toggled.connect(lambda _checked: self._redraw())
        side.addWidget(self.chk_direction)
        self.chk_nodes = QCheckBox("Show nodes")
        self.chk_nodes.setToolTip("Draw the joins between segments.")
        self.chk_nodes.toggled.connect(lambda _checked: self._redraw())
        side.addWidget(self.chk_nodes)
        self.root_list = QListWidget()
        self.root_list.setFixedWidth(250)
        self.root_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.root_list.itemClicked.connect(self._on_item_clicked)
        side.addWidget(self.root_list, stretch=1)
        body.addLayout(side)
        body.addWidget(self.view, stretch=1)
        layout.addLayout(body, stretch=1)
        self.lbl_navigation = QLabel(
            "Control click and drag to pan, use wheel to zoom | "
            "Click a segment | Shift or Ctrl-click adds a second | "
            "Double-click selects the whole root"
        )
        self.lbl_navigation.setStyleSheet("color: #666; font-style: italic; padding: 2px;")
        self.lbl_navigation.setAlignment(Qt.AlignCenter)
        self.lbl_navigation.setWordWrap(True)
        layout.addWidget(self.lbl_navigation)
        self._clear()

    def _clear(self):
        pixmap = QPixmap(100, 100)
        pixmap.fill(QColor("black"))
        self.view.set_image(pixmap, fit=True)
        self._image = None
        self.root_list.clear()
        self._colors = {}

    def remap_tracking(self, mapping):
        """Move cached tracings with the plants when ids are reassigned."""
        if not mapping:
            return

        def carry(store):
            return {mapping.get(uid, uid): value for uid, value in store.items()}

        self._seen_token = carry(self._seen_token)
        self._topology_cache = carry(self._topology_cache)
        if self._loaded_uid is not None:
            self._loaded_uid = mapping.get(self._loaded_uid, self._loaded_uid)

    def on_data_changed(self):
        if self.model.callbacks_muted or self._suspend:
            return
        # A newly opened plate replaces every mask. That is not an edit, so the
        # exported RSML can still be read. A later change of a plant's mask or
        # class labels throws away that plant's fixes and blocks the old file.
        epoch = self.model.content_epoch
        if epoch != self._content_epoch:
            self._content_epoch = epoch
            self._seen_token = {
                uid: self._content_token(uid) for uid in self.model.plant_uids()
            }
            self._topology_cache.clear()
            self.topology = None
            self._baseline = None
            self._baseline_topology = None
            self._undo = []
            self._tool = None
            self._loaded_uid = None
            if self.isVisible():
                self.load_plant()
            return
        affected = False
        live = set(self.model.plant_uids())
        for uid in list(self._seen_token):
            if uid not in live:
                self._discard_rsml(uid)
        changed = self.model._content_changed
        targets = live if changed is None else (live & changed)
        for uid in targets:
            token = self._content_token(uid)
            previous = self._seen_token.get(uid)
            if previous is not None and previous != token:
                self._discard_rsml(uid)
                affected = affected or uid == self._loaded_uid
            self._seen_token[uid] = token
        if self.isVisible() and (affected or self.topology is None):
            self.load_plant()

    def _discard_rsml(self, uid):
        """Segmentation edits invalidate every RSML fix for this plant."""
        self.model.topology_overrides.pop(uid, None)
        self.model.topology_decided.add(uid)
        self._seen_token.pop(uid, None)
        self._topology_cache.pop(uid, None)
        if uid != self._loaded_uid:
            return
        self._undo = []
        self._baseline = None
        self._baseline_topology = None
        self._tool = None
        self.topology = None
        self.selected_ids = []
        self.selected_edge = None

    def on_selection_changed(self):
        if self._reverting or not self.isVisible():
            return
        new_uid = self.model.active_uid
        if new_uid != self._loaded_uid and self.has_unsaved_edits():
            self._reverting = True
            if self._loaded_uid is not None:
                self.model.set_selection([self._loaded_uid])
            self._reverting = False
            if not self.confirm_leave(reset_on_discard=False):
                return
            self._reverting = True
            self.model.set_selection([] if new_uid is None else [new_uid])
            self.load_plant()
            self._reverting = False
            return
        self.load_plant()

    def has_unsaved_edits(self):
        if self.topology is None or self._baseline is None:
            return False
        return self.topology.to_record() != self._baseline

    def confirm_leave(self, reset_on_discard=True):
        """Ask before leaving edits that were not applied. False means stay."""
        if not self.has_unsaved_edits():
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Unsaved root fixes")
        box.setText("Keep the root fixes for this plant before leaving?")
        apply_button = box.addButton("Keep fixes", QMessageBox.AcceptRole)
        discard_button = box.addButton("Discard", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.exec_()
        chosen = box.clickedButton()
        if chosen == apply_button:
            return bool(self.apply_edit())
        if chosen == discard_button:
            if reset_on_discard and self._baseline_topology is not None:
                self.topology = self._baseline_topology.clone()
                self._undo = []
                self._tool = None
                self.selected_ids = []
                self.selected_edge = None
                self._click_point = None
                self._refresh_status()
                self._redraw()
            return True
        return False

    def showEvent(self, event):
        super().showEvent(event)
        uid = self.model.active_uid
        if uid != self._loaded_uid and self.has_unsaved_edits():
            self._reverting = True
            if self._loaded_uid is not None:
                self.model.set_selection([self._loaded_uid])
            self._reverting = False
            if not self.confirm_leave(reset_on_discard=False):
                return
            self._reverting = True
            self.model.set_selection([] if uid is None else [uid])
            self._reverting = False
        token = self._content_token(uid)
        if self.topology is None or uid != self._loaded_uid or token != self._seen_token.get(uid):
            self.load_plant()
        else:
            self._redraw()

    def _set_isolate(self, enabled):
        self.isolate = bool(enabled)
        self._redraw()

    def _on_view_mode(self):
        if self.rad_view_raw.isChecked():
            self.view_mode = "raw"
        elif self.rad_view_dense.isChecked():
            self.view_mode = "dense"
        elif self.rad_view_rsml.isChecked():
            self.view_mode = "rsml"
        else:
            self.view_mode = "lateral"
        if self.topology is not None:
            self._colors = self._build_colors()
            self._fill_list()
            self._redraw()

    def load_plant(self):
        if self.model.content_epoch != self._content_epoch:
            # A new plate can select a plant before this tab has dropped the
            # previous plate. That is not a segmentation edit.
            return
        uid = self.model.active_uid
        self._load_serial += 1
        serial = self._load_serial
        self.selected_ids = []
        self.selected_edge = None
        self._click_point = None
        self.split_mode = False
        self._tool = None
        self._undo = []
        self._note = ""
        self.topology = None
        self._fit_next = True
        self._loaded_uid = uid
        if uid is None or uid not in self.model.masks or self.model.raw_image is None:
            self._clear()
            self._set_info("")
            self.state_changed.emit()
            return
        if mask_is_disconnected(self.model.masks[uid]):
            self._clear()
            self._popup(
                "This mask is disconnected. Merge the pieces in Annotation, "
                "or split them into separate plants."
            )
            self.state_changed.emit()
            return
        patch_data = self.model._get_class_patch(uid)
        if not patch_data:
            self._clear()
            self._set_info("")
            self._popup("This plant has no root-part labels.")
            self.state_changed.emit()
            return
        token = self._content_token(uid)
        skeleton_changed = uid in self._seen_token and self._seen_token[uid] != token
        if skeleton_changed:
            self._topology_cache.pop(uid, None)
        cached = self._topology_cache.get(uid)
        if cached is not None and cached["token"] == token and not skeleton_changed:
            self._show_cached(uid, cached)
            return
        if skeleton_changed:
            self.model.topology_overrides.pop(uid, None)
            self.model.topology_decided.add(uid)
        patch, x_off, y_off = patch_data
        patch = np.array(patch, copy=True)
        y2 = y_off + patch.shape[0]
        x2 = x_off + patch.shape[1]
        bin_crop = np.array(self.model.masks[uid][y_off:y2, x_off:x2], copy=True)
        meta = self.model.plants_meta.get(uid) or {}
        record = None if skeleton_changed else copy.deepcopy(self.model.topology_overrides.get(uid))
        self._clear()
        self._set_info("Building the RSML…")
        self.state_changed.emit()
        worker = _TraceWorker(serial, _trace_plant, (
            patch, x_off, y_off, bin_crop, self.model.raw_image.shape[:2], self._prune(),
            _basename(self.model.image_path), meta.get("genotype") or "Unknown", uid,
            record, uid in self.model.topology_decided,
            self.path_provider() if self.path_provider else None,
        ))
        worker.ready.connect(self._on_trace_ready)
        worker.failed.connect(self._on_trace_failed)
        worker.finished.connect(self._drop_trace_worker)
        self._trace_workers.append(worker)
        worker.start()

    def _on_trace_ready(self, serial, payload):
        if serial != self._load_serial or payload.get("uid") != self._loaded_uid:
            return
        self._seen_token[payload["uid"]] = self._content_token(payload["uid"])
        self._note = payload.get("note") or ""
        self.topology = payload["topology"]
        self._prepare_view(payload["uid"])
        self._colors = self._build_colors()
        self._baseline = self.topology.to_record()
        self._baseline_topology = self.topology.clone()
        self._remember_topology(payload["uid"])
        self._refresh_status()
        self._redraw()
        self.state_changed.emit()

    def _show_cached(self, uid, cached):
        self._note = cached.get("note") or ""
        self.topology = cached["topology"].clone()
        self._prepare_view(uid)
        self._colors = self._build_colors()
        self._baseline = self.topology.to_record()
        self._baseline_topology = self.topology.clone()
        self._seen_token[uid] = cached["token"]
        self._refresh_status()
        self._redraw()
        self.state_changed.emit()

    def _remember_topology(self, uid):
        """Keep the tracing that should reopen: a kept edit, otherwise the generated one."""
        if self.topology is None or uid is None:
            return
        self._topology_cache[uid] = {
            "token": self._content_token(uid),
            "note": self._note,
            "topology": self.topology.clone(),
        }

    def _on_trace_failed(self, serial, message, is_topology_error):
        if serial != self._load_serial:
            return
        self._clear()
        self._set_info("")
        self._popup(message if is_topology_error else f"Could not build the RSML: {message}")
        self.state_changed.emit()

    def _drop_trace_worker(self):
        worker = self.sender()
        if worker in self._trace_workers:
            self._trace_workers.remove(worker)
        worker.deleteLater()

    def _content_token(self, uid):
        """Changes when the mask, the root-part labels, or the prune setting change."""
        if uid is None or uid not in self.model.masks:
            return None
        mask = self.model.masks[uid]
        box = self.model.bboxes.get(uid) or (0, 0, mask.shape[1], mask.shape[0])
        x, y, width, height = box
        x2 = min(mask.shape[1], x + width)
        y2 = min(mask.shape[0], y + height)
        x = max(0, x)
        y = max(0, y)
        crop = np.ascontiguousarray(mask[y:y2, x:x2])
        parts = [str(self._prune()).encode(), f"{x},{y}".encode(), crop.tobytes()]
        patch_data = self.model._get_class_patch(uid)
        if patch_data is not None:
            parts.append(np.ascontiguousarray(patch_data[0]).tobytes())
        return hashlib.sha1(b"\0".join(parts)).hexdigest()

    def _prune(self):
        return self.prune_provider() if self.prune_provider else 5

    def _prepare_view(self, uid):
        x, y, w, h = self.model.bboxes.get(uid, (0, 0, 0, 0))
        if w == 0:
            self.crop_x = self.crop_y = 0
            self._image = self.model.raw_image.copy()
            return
        img_h, img_w = self.model.raw_image.shape[:2]
        pad = 30
        self.crop_x = max(0, x - pad)
        self.crop_y = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)
        self._image = self.model.raw_image[self.crop_y:y2, self.crop_x:x2].copy()

    def arm_tool(self, mode):
        """None or 'point'. The next click cuts the segment there."""
        self._tool = mode
        self._refresh_status()
        self.state_changed.emit()

    def on_double_clicked(self, x, y):
        if self.topology is None or self._tool:
            return
        hit = self._hit_segment((x + self.crop_x, y + self.crop_y))
        if hit is None or hit[0][0] != "root":
            return
        self.selected_ids = [hit[0]]
        self.selected_edge = None
        self._click_point = None
        self._redraw()
        self.state_changed.emit()

    def on_clicked(self, x, y, modifiers=0):
        if self.topology is None:
            return
        point = (x + self.crop_x, y + self.crop_y)
        self._note = ""
        hit = self._hit_segment(point)
        if self._tool == "point":
            self._split_at_point(hit, point)
            return
        extend = Qt.KeyboardModifiers(int(modifiers)) & (Qt.ShiftModifier | Qt.ControlModifier)
        if hit is None:
            if not extend:
                self.selected_ids = []
                self.selected_edge = None
        else:
            key, edge_index = hit
            if extend and self.selected_ids and key != self.selected_ids[0]:
                if key in self.selected_ids:
                    self.selected_ids = [chosen for chosen in self.selected_ids if chosen != key]
                elif len(self.selected_ids) >= 2:
                    self.selected_ids = [self.selected_ids[0], key]
                else:
                    self.selected_ids.append(key)
                self.selected_edge = None
            else:
                self.selected_ids = [key]
                self.selected_edge = edge_index
                self._click_point = point
        self._redraw()
        self.state_changed.emit()

    def connect_selection(self):
        if self.topology is None:
            return
        if len(self.selected_ids) != 2:
            self._popup("Shift-click or Ctrl-click a second segment. Connect joins only pieces that touch.")
            return
        target, source = (self._chain_ref(key) for key in self.selected_ids)
        self._snapshot()
        try:
            self._note = self.topology.connect(target, source)
        except TopologyError as exc:
            self._undo.pop()
            self._popup(str(exc))
            return
        self.selected_ids = []
        self.selected_edge = None
        self._after_edit()

    def reverse_selection(self):
        if self.topology is None or len(self.selected_ids) != 1:
            self._popup("Select one root, then Reverse.")
            return
        self._snapshot()
        try:
            self.topology.reverse_chain(self._chain_ref(self.selected_ids[0]))
        except TopologyError as exc:
            self._undo.pop()
            self._popup(str(exc))
            return
        self._after_edit()

    def reuse_selection(self):
        if self.topology is None or len(self.selected_ids) != 1:
            self._popup("Select the root to copy, or one of its segments.")
            return
        key = self.selected_ids[0]
        if key[0] != "root":
            self._popup("Copy a root or a segment that already belongs to a root.")
            return
        self._snapshot()
        try:
            if self.selected_edge is None:
                created = self.topology.copy_root(key[1])
            else:
                created = self.topology.copy_segment(key[1], self.selected_edge)
        except TopologyError as exc:
            self._undo.pop()
            self._popup(str(exc))
            return
        if created is not None:
            self.selected_ids = [("root", created)]
            self.selected_edge = 0
        else:
            self.selected_ids = [("avail", len(self.topology.available) - 1)]
            self.selected_edge = None
        self._after_edit()

    def remove_selection(self):
        if self.topology is None or len(self.selected_ids) != 1:
            return
        key = self.selected_ids[0]
        self._snapshot()
        try:
            if key[0] == "avail":
                self.topology.discard_available(key[1])
            elif self.selected_edge is None:
                raise TopologyError("Click the extra copy of the segment first.")
            else:
                self.topology.remove_copy(key[1], self.selected_edge)
        except TopologyError as exc:
            self._undo.pop()
            self._popup(str(exc))
            return
        self.selected_ids = []
        self.selected_edge = None
        self._after_edit()

    def undo_edit(self):
        if not self._undo or self.topology is None:
            return
        self.topology = self._undo.pop()
        self.selected_ids = []
        self.selected_edge = None
        self.split_mode = False
        self._tool = None
        self._after_edit()

    def reset_edit(self):
        uid = self.model.active_uid
        if uid is None:
            return
        self.model.topology_overrides.pop(uid, None)
        self.model.topology_decided.add(uid)
        self._topology_cache.pop(uid, None)
        self.load_plant()
        self.topology_applied.emit(int(uid))

    def apply_edit(self):
        uid = self._loaded_uid
        if uid is None or self.topology is None:
            return False
        self.topology.reconnect_main()
        if not self.topology.can_apply():
            self._popup(self.topology.status_text())
            self._refresh_status()
            self._redraw()
            self.state_changed.emit()
            return False
        record = self.topology.to_record()
        self.model.topology_overrides[uid] = record
        self.model.topology_decided.add(uid)
        self._baseline = record
        self._baseline_topology = self.topology.clone()
        self._remember_topology(uid)
        self._refresh_status()
        self.state_changed.emit()
        self.topology_applied.emit(int(uid))
        return True

    def split_intersection_selection(self):
        """Separate the selected segment from the rest of its lateral. A lone segment stays."""
        self._tool = None
        if self.topology is None or len(self.selected_ids) != 1:
            return
        key = self.selected_ids[0]
        if key[0] != "root" or self.selected_edge is None:
            return
        if key[1] == self.topology.main_id:
            self._popup("You cannot split the main root.")
            return
        edges = self._edges_of(key)
        if len(edges) <= 1:
            return
        boundary = self.selected_edge if self.selected_edge > 0 else 1
        self._snapshot()
        try:
            before = set(self.topology.roots)
            self.topology.split(key[1], boundary)
        except TopologyError as exc:
            self._undo.pop()
            self._popup(str(exc))
            return
        created = [rid for rid in self.topology.roots if rid not in before]
        self.selected_ids = [("root", created[-1])] if created else [key]
        self.selected_edge = 0 if created else None
        self._after_edit()

    def _split_at_point(self, hit, point):
        if hit is None or hit[0][0] != "root":
            self._popup("Click a point on a root segment.")
            return
        key, edge_index = hit
        self._snapshot()
        try:
            self.selected_edge = self.topology.cut_polyline(key[1], edge_index, point)
        except TopologyError as exc:
            self._undo.pop()
            self._popup(str(exc))
            return
        self.selected_ids = [key]
        self._click_point = point
        self._tool = None
        self._after_edit()

    def set_line_width(self, width):
        self.line_width = max(1, int(width))
        self._redraw()

    def _boundaries(self):
        if self.topology is None or len(self.selected_ids) != 1:
            return []
        key = self.selected_ids[0]
        if key[0] == "root" and key[1] in self.topology.roots:
            return self.topology.boundaries(key[1])
        if key[0] == "avail" and key[1] < len(self.topology.available):
            return self.topology.chain_boundaries(self.topology.available[key[1]])
        return []

    def _edge_at(self, key, point):
        edges = self._edges_of(key)
        best = None
        best_dist = _HIT_PX + 1
        for index, signed in enumerate(edges):
            dist = _polyline_distance(point, self.topology.edges[abs(signed)].points)
            if dist < best_dist:
                best, best_dist = index, dist
        return best

    def _edges_of(self, key):
        if self.topology is None:
            return []
        if key[0] == "root" and key[1] in self.topology.roots:
            return self.topology.roots[key[1]]["edges"]
        if key[0] == "avail" and key[1] < len(self.topology.available):
            return self.topology.available[key[1]]
        return []

    def _chain_ref(self, key):
        return key[1] if key[0] == "root" else f"a{key[1]}"

    def _snapshot(self):
        self._undo.append(self.topology.clone())
        if len(self._undo) > 40:
            self._undo.pop(0)

    def _after_edit(self):
        self._colors = self._build_colors()
        self._fill_list()
        self._refresh_status()
        self._redraw()
        self.state_changed.emit()

    def _set_info(self, text):
        self.info_label.setText(text)

    def _popup(self, text):
        QMessageBox.warning(self, "RSML connections", text)

    def _refresh_status(self):
        if self._tool == "point":
            self._set_info("Click where the new split point should be. This also works on the main root.")
            return
        if self.model.active_uid is None or self.topology is None:
            self._set_info("")
            return
        self._set_info(self.topology.status_text())

    def _hit_segment(self, point):
        best = None
        best_dist = _HIT_PX + 1
        for key in self._draw_order():
            for index, signed in enumerate(self._edges_of(key)):
                dist = _polyline_distance(point, self.topology.edges[abs(signed)].points)
                if dist <= best_dist:
                    best = (key, index)
                    best_dist = dist
        return best

    def _draw_order(self):
        """Unselected roots first, so the selection is painted on top."""
        keys = self._visible_keys()
        return [key for key in keys if key not in self.selected_ids] + [
            key for key in keys if key in self.selected_ids
        ]

    def _visible_keys(self):
        keys = [("root", root_id) for root_id in self._root_order()]
        keys.extend(("avail", index) for index in range(len(self.topology.available)))
        if self.isolate and self.selected_ids:
            keys = [key for key in keys if key in self.selected_ids]
        return keys

    def _redraw(self):
        if self._image is None or self.topology is None:
            return
        canvas = self._image.copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        if self.view_mode == "dense":
            self._overlay_dense(canvas)
        self._fill_list()
        show_color = self.view_mode in ("rsml", "lateral")
        for key in self._draw_order():
            color = self._colors.get(key, _TAB20[0])
            edges = self._edges_of(key)
            for index, signed in enumerate(edges):
                points = self.topology.edges[abs(signed)].points
                if signed < 0:
                    points = list(reversed(points))
                chosen = self.selected_edge if len(self.selected_ids) == 1 and self.selected_ids[0] == key else None
                selected = key in self.selected_ids and (chosen is None or chosen == index)
                width = self.line_width
                if key[0] == "avail":
                    self._polyline(canvas, _gapped(points), (255, 220, 0), width + 2)
                if selected:
                    self._polyline(canvas, _gapped(points), (255, 255, 255), width + 2)
                if show_color:
                    self._polyline(canvas, _gapped(points), color, width)
            if show_color and self.chk_nodes.isChecked():
                for _index, joint in self.topology.chain_boundaries(edges):
                    center = self._to_crop(joint)
                    cv2.circle(canvas, center, 3, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(canvas, center, 3, color, 1, cv2.LINE_AA)
        if show_color:
            self._draw_direction(canvas)
        height, width = canvas.shape[:2]
        image = QImage(canvas.data, width, height, 3 * width, QImage.Format_RGB888)
        self.view.set_image(QPixmap.fromImage(image.copy()), fit=self._fit_next)
        self._fit_next = False

    def _overlay_dense(self, canvas):
        uid = self.model.active_uid
        patch_data = self.model._get_class_patch(uid) if uid is not None else None
        if not patch_data:
            return
        patch, x_off, y_off = patch_data
        dx, dy = int(x_off - self.crop_x), int(y_off - self.crop_y)
        height, width = canvas.shape[:2]
        patch_h, patch_w = patch.shape[:2]
        start_x, start_y = max(0, dx), max(0, dy)
        patch_x, patch_y = max(0, -dx), max(0, -dy)
        end_x, end_y = min(width, dx + patch_w), min(height, dy + patch_h)
        if end_x <= start_x or end_y <= start_y:
            return
        patch_slice = patch[patch_y:patch_y + (end_y - start_y), patch_x:patch_x + (end_x - start_x)]
        crop_slice = canvas[start_y:end_y, start_x:end_x]
        main_mask = patch_slice == 1
        lat_mask = patch_slice == 2
        if np.any(main_mask):
            crop_slice[main_mask] = (
                crop_slice[main_mask].astype(np.float32) * 0.4
                + np.array([255, 100, 100], dtype=np.float32) * 0.6
            ).astype(np.uint8)
        if np.any(lat_mask):
            crop_slice[lat_mask] = (
                crop_slice[lat_mask].astype(np.float32) * 0.4
                + np.array([100, 255, 100], dtype=np.float32) * 0.6
            ).astype(np.uint8)

    def _points_of(self, key):
        return self.topology.chain_points(self._edges_of(key))

    def _polyline(self, canvas, points, color, width):
        shifted = [self._to_crop(point) for point in points]
        for start, end in zip(shifted, shifted[1:]):
            cv2.line(canvas, start, end, color, width, cv2.LINE_AA)

    def _draw_direction(self, canvas):
        if not self.chk_direction.isChecked():
            return
        for key in self._draw_order():
            points = [self._to_crop(point) for point in self._points_of(key)]
            for tip, direction in _direction_marks(points, spacing=70):
                _draw_chevron(canvas, tip, direction)

    def _to_crop(self, point):
        return (int(point[0] - self.crop_x), int(point[1] - self.crop_y))

    def _build_colors(self):
        colors = {}
        if self.topology is None:
            return colors
        if self.view_mode == "rsml":
            for root_id in self.topology.roots:
                order = self.topology.order_of(root_id)
                colors[("root", root_id)] = _ORDER_COLORS.get(order, _DEEPER_COLOR)
        else:
            colors[("root", self.topology.main_id)] = _ORDER_COLORS[0]
            others = [root_id for root_id in self._root_order() if root_id != self.topology.main_id]
            for index, root_id in enumerate(others):
                colors[("root", root_id)] = _TAB20[index % len(_TAB20)]
        for index in range(len(self.topology.available)):
            colors[("avail", index)] = _TAB20[(index + 3) % len(_TAB20)]
        return colors

    def _root_order(self):
        if self.topology is None or self.topology.main_id not in self.topology.roots:
            return []
        ordered = []

        def visit(root_id):
            if root_id in ordered:
                return
            ordered.append(root_id)
            children = [
                other_id for other_id, root in self.topology.roots.items()
                if root["parent"] == root_id
            ]
            children.sort(key=lambda other_id: self._attach_key(root_id, other_id))
            for child_id in children:
                visit(child_id)

        visit(self.topology.main_id)
        for root_id in sorted(self.topology.roots):
            if root_id not in ordered:
                visit(root_id)
        return ordered

    def _attach_key(self, parent_id, child_id):
        nodes = self.topology.chain_nodes(self.topology.roots[parent_id]["edges"])
        start = self.topology.edge_ends(self.topology.roots[child_id]["edges"][0])[0]
        return nodes.index(start) if start in nodes else 10 ** 9

    def _rows(self):
        rows = []
        if self.topology is None:
            return rows
        sibling = {}
        names = {}

        def add(root_id):
            if root_id in names:
                return
            order = self.topology.order_of(root_id)
            parent = self.topology.roots[root_id]["parent"]
            parent_name = names.get(parent, "Main root")
            key = (parent, order)
            sibling[key] = sibling.get(key, 0) + 1
            number = sibling[key]
            if order == 0:
                name = "Main root"
            elif order == 1:
                name = f"Lateral {number}"
            elif order == 2:
                name = f"{parent_name} · tertiary {number}"
            elif order == 3:
                name = f"{parent_name} · quaternary {number}"
            elif order is None:
                name = f"Unassigned {number}"
            else:
                name = f"{parent_name} · branch {order}.{number}"
            names[root_id] = name
            rows.append((("root", root_id), name, self._colors.get(("root", root_id), _TAB20[0])))
            children = [
                other_id for other_id, root in self.topology.roots.items() if root["parent"] == root_id
            ]
            children.sort(key=lambda other_id: self._attach_key(root_id, other_id))
            for child_id in children:
                add(child_id)

        if self.topology.available:
            rows.append((None, "Available segments", None))
            for index in range(len(self.topology.available)):
                rows.append((
                    ("avail", index), f"Segment {index + 1}",
                    self._colors.get(("avail", index), _TAB20[index % len(_TAB20)]),
                ))
        if self.topology.main_id in self.topology.roots:
            add(self.topology.main_id)
        for root_id in sorted(self.topology.roots):
            if root_id not in names:
                add(root_id)
        return rows

    def _fill_list(self):
        self._filling_list = True
        self.root_list.blockSignals(True)
        self.root_list.clear()
        for key, name, color in self._rows():
            label = name
            if key is not None and key in self.selected_ids and len(self.selected_ids) == 2:
                role = "1 Target" if self.selected_ids[0] == key else "2 Source"
                label = f"{role}  {name}"
            elif key is not None and key in self.selected_ids:
                label = f"▸ {name}"
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, key)
            if color is not None:
                item.setIcon(_swatch(color))
            else:
                item.setFlags(Qt.NoItemFlags)
            if key is not None and key in self.selected_ids:
                item.setBackground(QColor("#d9e8fb"))
            self.root_list.addItem(item)
        self.root_list.blockSignals(False)
        self._filling_list = False

    def _on_item_clicked(self, item):
        if self._filling_list or self.topology is None:
            return
        key = item.data(Qt.UserRole)
        if key is None:
            return
        modifiers = QApplication.keyboardModifiers()
        if modifiers & (Qt.ControlModifier | Qt.ShiftModifier):
            if key in self.selected_ids:
                self.selected_ids = [chosen for chosen in self.selected_ids if chosen != key]
            elif len(self.selected_ids) >= 2:
                self.selected_ids = [self.selected_ids[0], key]
            else:
                self.selected_ids.append(key)
        else:
            self.selected_ids = [key]
        self.selected_edge = None
        self._fill_list()
        self._refresh_status()
        self._redraw()
        self.state_changed.emit()


def _direction_marks(points, spacing):
    """Tip position and a unit direction, every `spacing` pixels. Size is fixed later."""
    if len(points) < 2:
        return []
    marks = []
    traveled = 0.0
    next_at = float(spacing)
    for start, stop in zip(points, points[1:]):
        step = _distance(start, stop)
        if step <= 0:
            continue
        direction = ((stop[0] - start[0]) / step, (stop[1] - start[1]) / step)
        while traveled + step >= next_at:
            scale = (next_at - traveled) / step
            tip = (
                int(round(start[0] + (stop[0] - start[0]) * scale)),
                int(round(start[1] + (stop[1] - start[1]) * scale)),
            )
            marks.append((tip, direction))
            next_at += spacing
        traveled += step
    if not marks:
        step = _distance(points[0], points[-1])
        if step <= 0:
            return []
        direction = ((points[-1][0] - points[0][0]) / step, (points[-1][1] - points[0][1]) / step)
        marks.append((points[-1], direction))
    return marks


def _draw_chevron(canvas, tip, direction):
    """A 6-pixel chevron. Its size does not depend on the root length."""
    dx, dy = direction
    px, py = -dy, dx
    back = 6
    wing = 3
    left = (int(round(tip[0] - dx * back + px * wing)), int(round(tip[1] - dy * back + py * wing)))
    right = (int(round(tip[0] - dx * back - px * wing)), int(round(tip[1] - dy * back - py * wing)))
    cv2.line(canvas, tip, left, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, tip, right, (255, 255, 255), 1, cv2.LINE_AA)


def _gapped(points, gap=4):
    """Pull the ends in so consecutive segments stay visually separate."""
    if len(points) < 2:
        return points
    opened = list(points)

    def pull(origin, toward):
        dx = toward[0] - origin[0]
        dy = toward[1] - origin[1]
        length = max(1.0, float(np.hypot(dx, dy)))
        step = min(gap, length / 3.0)
        return (int(round(origin[0] + dx / length * step)), int(round(origin[1] + dy / length * step)))

    opened[0] = pull(opened[0], opened[1])
    opened[-1] = pull(opened[-1], opened[-2])
    return opened


def _distance(left, right):
    return float(np.hypot(left[0] - right[0], left[1] - right[1]))


def _polyline_distance(point, points):
    if not points:
        return 1e9
    array = np.asarray(points, dtype=float)
    return float(np.min(np.hypot(array[:, 0] - point[0], array[:, 1] - point[1])))


def _basename(path):
    return os.path.basename(path) if path else "plate.png"


class RsmlToolPanel(QWidget):
    def __init__(self, shared_model, canvas_tab: RsmlCanvasTab):
        super().__init__()
        self.model = shared_model
        self.canvas_tab = canvas_tab
        self.init_ui()
        self.canvas_tab.state_changed.connect(self._sync_buttons)
        self.model.register_selection_callback(lambda: self._sync_buttons())

    def init_ui(self):
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.addWidget(QLabel("<b>Lateral root fixes</b>"))
        header.addStretch()
        help_button = QPushButton("Help")
        help_button.clicked.connect(lambda: show_help(self, "RSML Lateral Root Fixes", HELP_RSML))
        header.addWidget(help_button)
        layout.addLayout(header)

        width_row = QHBoxLayout()
        width_row.addWidget(QLabel("Line width"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(1, 8)
        self.width_spin.setValue(2)
        self.width_spin.setMaximumWidth(65)
        self.width_spin.setToolTip("Thickness of the root lines, in pixels. Use a thinner line when zoomed out.")
        self.width_spin.valueChanged.connect(self.canvas_tab.set_line_width)
        width_row.addWidget(self.width_spin)
        width_row.addStretch(1)
        layout.addLayout(width_row)

        self.btn_connect = QPushButton("Connect")
        self.btn_connect.setToolTip(
            "First click keeps its direction. The second piece must touch its beginning or its end. "
            "Pieces that do not touch are left apart."
        )
        self.btn_connect.clicked.connect(self.canvas_tab.connect_selection)
        layout.addWidget(self.btn_connect)

        self.btn_reverse = QPushButton("Reverse direction")
        self.btn_reverse.setToolTip(
            "Swap the beginning and the end. Only when both ends are branching nodes, not a tip."
        )
        self.btn_reverse.clicked.connect(self.canvas_tab.reverse_selection)
        layout.addWidget(self.btn_reverse)

        self.btn_split = QPushButton("Split at intersection")
        self.btn_split.setToolTip(
            "Select a lateral segment that is joined to others, then split it off. "
            "A segment that is already on its own is left as it is. The main root cannot be split this way."
        )
        self.btn_split.clicked.connect(self.canvas_tab.split_intersection_selection)
        layout.addWidget(self.btn_split)

        self.btn_split_point = QPushButton("Generate a split point")
        self.btn_split_point.setCheckable(True)
        self.btn_split_point.setToolTip(
            "Press this, then click a point on a segment, including the main root. "
            "The segment is cut there so a small piece can be copied."
        )
        self.btn_split_point.clicked.connect(self._arm_split_point)
        layout.addWidget(self.btn_split_point)

        self.btn_reuse = QPushButton("Copy segment")
        self.btn_reuse.setToolTip(
            "Copy the selected root, or one of its segments, so another lateral can use it."
        )
        self.btn_reuse.clicked.connect(self.canvas_tab.reuse_selection)
        layout.addWidget(self.btn_reuse)

        self.btn_remove = QPushButton("Remove copy")
        self.btn_remove.setToolTip(
            "Remove the selected extra copy. The last copy of a segment stays."
        )
        self.btn_remove.clicked.connect(self.canvas_tab.remove_selection)
        layout.addWidget(self.btn_remove)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setToolTip("Undo the last connect, split, copy, or remove.")
        self.btn_undo.clicked.connect(self.canvas_tab.undo_edit)
        layout.addWidget(self.btn_undo)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setToolTip("Discard the topology kept in memory and trace this plant again.")
        self.btn_reset.clicked.connect(self.canvas_tab.reset_edit)
        layout.addWidget(self.btn_reset)

        self.btn_apply = QPushButton("Keep fixes for this plant")
        self.btn_apply.setToolTip(
            "Keep these roots for measurement and export. It does not write a file."
        )
        self.btn_apply.setStyleSheet(
            "background-color: #d4edda; font-weight: bold; color: #155724; font-size: 12px;"
        )
        self.btn_apply.clicked.connect(self.canvas_tab.apply_edit)
        layout.addWidget(self.btn_apply)
        layout.addStretch(1)
        self._sync_buttons()

    def _set_checked(self, button, checked):
        button.blockSignals(True)
        button.setChecked(checked)
        button.blockSignals(False)

    def _arm_split_point(self):
        if self.btn_split_point.isChecked():
            self.canvas_tab.arm_tool("point")
            self._set_checked(self.btn_split, False)
        elif self.canvas_tab._tool == "point":
            self.canvas_tab.arm_tool(None)

    def _sync_buttons(self):
        has_plant = self.model.active_uid is not None
        topo = self.canvas_tab.topology
        one = len(self.canvas_tab.selected_ids) == 1
        edge = self.canvas_tab.selected_edge is not None
        self.btn_connect.setEnabled(has_plant and topo is not None)
        reversible = False
        if has_plant and one and topo is not None:
            reversible = topo.can_reverse(self.canvas_tab._chain_ref(self.canvas_tab.selected_ids[0]))
        self.btn_reverse.setEnabled(reversible)
        self.btn_split.setEnabled(has_plant and topo is not None)
        self.btn_split_point.setEnabled(has_plant and topo is not None)
        self._set_checked(self.btn_split_point, self.canvas_tab._tool == "point")
        self.btn_reuse.setEnabled(
            has_plant and one and self.canvas_tab.selected_ids[0][0] == "root"
        )
        self.btn_remove.setEnabled(has_plant and one and (edge or self.canvas_tab.selected_ids[0][0] == "avail"))
        self.btn_undo.setEnabled(bool(self.canvas_tab._undo))
        self.btn_reset.setEnabled(has_plant)
        self.btn_apply.setEnabled(has_plant and topo is not None)
