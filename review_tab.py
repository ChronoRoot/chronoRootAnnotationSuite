from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, 
                             QListWidgetItem, QPushButton, QLabel, QGraphicsView, 
                             QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, 
                             QMessageBox, QSlider, QTreeWidget, QTreeWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QPainterPath, QPen, QColor, 
                         QBrush, QIcon)

class SortableTreeItem(QTreeWidgetItem):
    def __lt__(self, other):
        col = self.treeWidget().sortColumn()
        # Sort numerically using the hidden UserRole data (either ID or Area)
        return self.data(col, Qt.UserRole) < other.data(col, Qt.UserRole)
    
# ==========================================
# HELPER: DATA TO GUI TRANSLATION
# ==========================================
def _bytes_to_pixmap(data_tuple, is_rgba=True):
    """
    Safely converts the raw bytes from the isolated model into a QPixmap.
    Creates a deep copy so Qt owns the memory and avoids segfaults.
    """
    if not data_tuple:
        return QPixmap()
        
    b_data, w, h, bpl = data_tuple
    fmt = QImage.Format_RGBA8888 if is_rgba else QImage.Format_RGB888
    
    # .copy() is critical here to prevent memory corruption
    qimg = QImage(b_data, w, h, bpl, fmt).copy()
    return QPixmap.fromImage(qimg)

# ==========================================
# UNIFIED CANVAS
# ==========================================
class BaseCanvas(QGraphicsView):
    zoom_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        self.base_pixmap = QGraphicsPixmapItem()
        self.base_pixmap.setZValue(0)
        self.overlay_pixmap = QGraphicsPixmapItem()
        self.overlay_pixmap.setZValue(1)
        
        self.scene.addItem(self.base_pixmap)
        self.scene.addItem(self.overlay_pixmap)
        
        self.is_panning = False
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def update_view(self, base_pmap, overlay_pmap):
        """Now strictly accepts pre-rendered QPixmaps."""
        self.base_pixmap.setPixmap(base_pmap)
        self.overlay_pixmap.setPixmap(overlay_pmap)
        if not base_pmap.isNull():
            self.scene.setSceneRect(0, 0, base_pmap.width(), base_pmap.height())

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0: self.scale(1.15, 1.15)
        else: self.scale(1 / 1.15, 1 / 1.15)
        self.zoom_changed.emit(self.transform().m11())

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton or (event.modifiers() & Qt.ControlModifier):
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self.is_panning = True
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self.is_panning:
            self.setDragMode(QGraphicsView.NoDrag)
            self.is_panning = False
        super().mouseReleaseEvent(event)
        
class PaintCanvas(BaseCanvas):
    on_stroke_finished = pyqtSignal(list, bool)
    on_click = pyqtSignal(int, int, bool)
    on_split_finish = pyqtSignal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.brush_cursor = None
        self.brush_size = 5 
        self.mode = "SELECT"
        
        self.is_painting = False
        self.is_erasing = False
        self.temp_item = None
        self.points_buffer = []
        
        self.poly_points = []
        self.poly_lines = []
        self.rubber_band = None
        
        self.bbox_group = self.scene.createItemGroup([])
        self.bbox_group.setZValue(2) 
        self.update_cursor_visual()

    def set_mode(self, mode):
        self.mode = mode
        if mode != "SPLIT": self.clear_poly_visuals()
            
        if mode == "SELECT":
            self.setCursor(Qt.ArrowCursor)
            if self.brush_cursor: self.brush_cursor.setVisible(False)
        elif mode in ["PAINT", "SEMANTIC"]:
            self.setCursor(Qt.CrossCursor)
            if self.brush_cursor: self.brush_cursor.setVisible(True)
        elif mode == "SPLIT":
            self.setCursor(Qt.PointingHandCursor)
            if self.brush_cursor: self.brush_cursor.setVisible(False)
            
    def update_cursor_visual(self):
        if self.brush_cursor: self.scene.removeItem(self.brush_cursor)
        s = self.brush_size
        self.brush_cursor = self.scene.addEllipse(0, 0, s, s, 
                                                QPen(Qt.white, 1), QBrush(QColor(255, 255, 255, 30)))
        self.brush_cursor.setZValue(100)
        self.brush_cursor.setVisible(self.mode in ["PAINT", "SEMANTIC"])

    def clear_poly_visuals(self):
        for line in self.poly_lines: 
            self.scene.removeItem(line)
        self.poly_lines.clear()
        if self.rubber_band:
            self.scene.removeItem(self.rubber_band)
            self.rubber_band = None
        self.poly_points.clear()
        
    def set_brush_size(self, s): 
        self.brush_size = s
        self.update_cursor_visual()      
    
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.is_panning: return
        sp = self.mapToScene(event.pos())
        
        if self.mode == "SELECT":
            if event.button() == Qt.LeftButton:
                shift = (event.modifiers() & Qt.ShiftModifier)
                self.on_click.emit(int(sp.x()), int(sp.y()), shift)
        
        elif self.mode in ["PAINT", "SEMANTIC"]:
            if self.is_painting or self.is_erasing:
                return
                
            if event.button() == Qt.LeftButton:
                self.is_painting = True
                self.start_stroke(event.pos(), is_erase=False)
            elif event.button() == Qt.RightButton:
                self.is_erasing = True
                self.start_stroke(event.pos(), is_erase=True)

        elif self.mode == "SPLIT":
            if event.button() == Qt.LeftButton:
                self.poly_points.append(sp)
                if len(self.poly_points) > 1:
                    last_p = self.poly_points[-2]
                    line = self.scene.addLine(last_p.x(), last_p.y(), sp.x(), sp.y(), QPen(Qt.red, 4))
                    self.poly_lines.append(line)

    def mouseReleaseEvent(self, event):
        if (self.is_painting and event.button() == Qt.LeftButton) or \
           (self.is_erasing and event.button() == Qt.RightButton):
            
            if self.temp_item:
                self.scene.removeItem(self.temp_item)
                self.temp_item = None
                
            self.on_stroke_finished.emit(self.points_buffer, self.is_erasing)
            
            # Reset state
            self.is_painting = False
            self.is_erasing = False
            self.points_buffer = []
            
        super().mouseReleaseEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        if self.mode == "SPLIT" and len(self.poly_points) > 1:
            self.on_split_finish.emit(self.poly_points)
            self.clear_poly_visuals()
        super().mouseDoubleClickEvent(event)
        
    def start_stroke(self, pos, is_erase):
        sp = self.mapToScene(pos)
        self.points_buffer = [sp]
        self.temp_path = QPainterPath(sp)
        color = QColor(0, 255, 0, 150) if not is_erase else QColor(255, 0, 0, 150)
        pen = QPen(color)
        pen.setWidth(self.brush_size)
        pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin)
        self.temp_item = self.scene.addPath(self.temp_path, pen)
        self.temp_item.setZValue(50)

    def mouseMoveEvent(self, event):
        sp = self.mapToScene(event.pos())
        
        if self.mode in ["PAINT", "SEMANTIC"] and self.brush_cursor:
            offset = self.brush_size / 2.0
            self.brush_cursor.setRect(sp.x() - offset, sp.y() - offset, 
                                    self.brush_size, self.brush_size)

        if (self.is_painting or self.is_erasing) and self.temp_item:
            self.points_buffer.append(sp)
            self.temp_path.lineTo(sp)
            self.temp_item.setPath(self.temp_path)

        if self.mode == "SPLIT" and self.poly_points:
            last_p = self.poly_points[-1]
            if self.rubber_band: self.scene.removeItem(self.rubber_band)
            pen = QPen(Qt.red, 4, Qt.DashLine)
            pen.setCosmetic(True)
            self.rubber_band = self.scene.addLine(last_p.x(), last_p.y(), sp.x(), sp.y(), pen)
            
        super().mouseMoveEvent(event)

    def update_bboxes(self, bboxes, color_map, visible):
        for item in self.bbox_group.childItems(): 
            self.scene.removeItem(item)
        if visible:
            for uid, (x, y, w, h) in bboxes.items():
                if uid in color_map:
                    c = color_map[uid]
                    pen = QPen(QColor(c[0], c[1], c[2]), 2)
                    pen.setCosmetic(True) 
                    rect = QGraphicsRectItem(x, y, w, h)
                    rect.setPen(pen)
                    self.bbox_group.addToGroup(rect)

# ==========================================
# UNIFIED REVIEW TAB
# ==========================================
class ReviewTab(QWidget):
    def __init__(self, shared_model):
        super().__init__()
        self.model = shared_model
        
        # State
        self.selected_ids = set()
        self.active_uid = None
        self.current_mode = "SELECT" 
        self.previous_view_mode = "SELECT" 
        self.show_bboxes = True
        self.current_image_path = None

        # Connect to the pure python backend using the callback API
        self.model.register_callback(self.populate_list)
        
        # Layouts
        main_layout = QHBoxLayout(self)
        side_panel = QWidget()
        side_panel.setFixedWidth(320)
        sl = QVBoxLayout(side_panel)
        
        # UI Definitions
        self.lbl_instance_count = QLabel("<b>PLANT INSTANCES (0 Total):</b>")
        sl.addWidget(self.lbl_instance_count)
        
        self.list_instances = QTreeWidget()
        self.list_instances.setHeaderLabels(["Plant ID", "Area (px)"])
        self.list_instances.setSortingEnabled(True)
        self.list_instances.setRootIsDecorated(False) # Hides the expansion arrows
        self.list_instances.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.list_instances.setFixedHeight(250)
        self.list_instances.itemSelectionChanged.connect(self.on_list_selection_changed)
        
        # Make the ID column fit snugly, and the Area column stretch
        self.list_instances.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.list_instances.header().setSectionResizeMode(1, QHeaderView.Stretch)
        sl.addWidget(self.list_instances)
        
        self.btn_new = QPushButton("+ Create New Plant")
        self.btn_new.clicked.connect(self.action_new_instance)
        sl.addWidget(self.btn_new)
        
        row_actions = QHBoxLayout()
        self.btn_unselect = QPushButton("Clear Selection")
        self.btn_unselect.clicked.connect(self.unselect_instance)
        
        self.btn_bbox = QPushButton("Show Bounding Boxes")
        self.btn_bbox.setCheckable(True)
        self.btn_bbox.setChecked(True)
        self.btn_bbox.clicked.connect(self.toggle_bbox)
        
        row_actions.addWidget(self.btn_unselect)
        row_actions.addWidget(self.btn_bbox)
        sl.addLayout(row_actions)
        
        self.btn_global = QPushButton("Global Multi-Class View")
        self.btn_global.setCheckable(True)
        self.btn_global.clicked.connect(lambda: self.toggle_mode("GLOBAL"))
        sl.addWidget(self.btn_global)
        
        sl.addSpacing(15)
        sl.addWidget(QLabel("<b>TOOLS:</b>"))

        self.btn_paint = QPushButton("Shape Paint")
        self.btn_paint.setCheckable(True)
        self.btn_paint.clicked.connect(lambda: self.toggle_mode("PAINT"))
        sl.addWidget(self.btn_paint)

        self.btn_split = QPushButton("Split (Knife)")
        self.btn_split.setCheckable(True)
        self.btn_split.clicked.connect(lambda: self.toggle_mode("SPLIT"))
        sl.addWidget(self.btn_split)
        
        self.btn_merge = QPushButton("Merge Selected")
        self.btn_merge.clicked.connect(self.action_merge)
        sl.addWidget(self.btn_merge)
        
        self.btn_del = QPushButton("Delete Selected")
        self.btn_del.clicked.connect(self.action_delete)
        sl.addWidget(self.btn_del)
        
        self.btn_undo = QPushButton("Undo Last Action")
        self.btn_undo.clicked.connect(self.model.undo)
        sl.addWidget(self.btn_undo)
        
        self.btn_semantic = QPushButton("Multi-Class Paint") 
        self.btn_semantic.setCheckable(True)
        self.btn_semantic.clicked.connect(lambda: self.toggle_mode("SEMANTIC"))
        sl.addWidget(self.btn_semantic)

        self.list_classes = QListWidget()
        self.list_classes.setFixedHeight(120)
        self.populate_class_palette()
        self.list_classes.setVisible(False)
        sl.addWidget(self.list_classes)
        
        sl.addWidget(QLabel("Brush Size:"))
        size_layout = QHBoxLayout()
        self.slider_size = QSlider(Qt.Horizontal)
        self.slider_size.setRange(1, 30); self.slider_size.setValue(5)
        self.lbl_size_val = QLabel("5px") 
        self.lbl_size_val.setFixedWidth(40)
        self.slider_size.valueChanged.connect(self.update_brush_size) 
        size_layout.addWidget(self.slider_size)
        size_layout.addWidget(self.lbl_size_val)
        sl.addLayout(size_layout)
        
        sl.addWidget(QLabel("Overlay Opacity:"))
        op_layout = QHBoxLayout()
        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(0, 100); self.slider_opacity.setValue(40)
        self.lbl_op_val = QLabel("40%")
        self.lbl_op_val.setFixedWidth(40)
        self.slider_opacity.valueChanged.connect(lambda v: self.lbl_op_val.setText(f"{v}%"))
        self.slider_opacity.sliderReleased.connect(self.refresh_canvas)
        op_layout.addWidget(self.slider_opacity)
        op_layout.addWidget(self.lbl_op_val)
        sl.addLayout(op_layout)

        sl.addStretch()
        main_layout.addWidget(side_panel)
        
        # --- RIGHT CANVAS ---
        right_panel = QWidget()
        rl = QVBoxLayout(right_panel)
        
        # 1. Top Info Bar
        self.lbl_info = QLabel()
        self.lbl_info.setStyleSheet("background-color: #eee; padding: 5px; font-weight: bold;")
        rl.addWidget(self.lbl_info)
        
        # 2. Main Canvas
        self.canvas = PaintCanvas()
        self.canvas.on_stroke_finished.connect(self.handle_stroke)
        self.canvas.on_click.connect(self.handle_canvas_click)
        self.canvas.on_split_finish.connect(self.handle_split)
        rl.addWidget(self.canvas)
        
        # 3. NEW: Bottom Navigation Instructions
        self.lbl_navigation = QLabel("Control click and drag for pan, use wheel to zoom")
        self.lbl_navigation.setStyleSheet("color: #666; font-style: italic; padding: 2px;")
        self.lbl_navigation.setAlignment(Qt.AlignCenter)
        rl.addWidget(self.lbl_navigation)
        
        main_layout.addWidget(right_panel)
        main_layout.setStretch(1, 1)
        
        # Initialize the dynamic text
        self.update_info_label()
        self.apply_tooltips()

    def apply_tooltips(self):
        """Centralized location for all Review Tab tooltips."""
        # General Actions
        self.btn_new.setToolTip("<b>Create New Plant</b><br>Generates a new plant ID and enters paint mode.")
        self.btn_unselect.setToolTip("<b>Clear Selection</b><br>Deselect current plants to view the whole image.")
        self.btn_bbox.setToolTip("<b>Toggle Bounding Boxes</b><br>Show or hide the rectangular bounds around plants.")
        self.btn_global.setToolTip("<b>Global View</b><br>View all plant organs across all plants simultaneously.")
        
        # Tools
        self.btn_paint.setToolTip("<b>Shape Paint Mode</b><br>Modify the general mask for the selected plant.<br>• <i>Left-Click:</i> Paint<br>• <i>Right-Click:</i> Erase")
        self.btn_split.setToolTip("<b>Split Instance</b><br>Draw a line across a plant to cut it into two separate IDs.<br>• <i>Click:</i> Add points<br>• <i>Double-Click:</i> Finish cut")
        self.btn_merge.setToolTip("<b>Merge Selected</b><br>Combine two or more selected plants from the list into a single ID.")
        self.btn_del.setToolTip("<b>Delete Selected</b><br>Permanently remove the selected plants and their masks.")
        self.btn_undo.setToolTip("<b>Undo</b><br>Revert the last paint, split, merge, or delete action.")
        self.btn_semantic.setToolTip("<b>Multi-Class Paint</b><br>Paint specific biological parts (e.g., Lateral Root, Hypocotyl) inside the selected plant's mask.")
        
        # Controls
        self.slider_size.setToolTip("Adjust the thickness of the painting brush.")
        self.slider_opacity.setToolTip("Adjust how transparent the colored mask overlays are.")
        self.list_instances.setToolTip("<b>Plant Instances</b><br>• <i>Click:</i> Select plant<br>• <i>Ctrl/Shift+Click:</i> Select multiple for merging")
        
    # ==========================================
    # MODE MANAGEMENT
    # ==========================================
    def toggle_bbox(self):
        self.show_bboxes = self.btn_bbox.isChecked()
        self.refresh_canvas()

    def update_brush_size(self, val):
        self.lbl_size_val.setText(f"{val}px")
        self.canvas.set_brush_size(val)
        
    def toggle_mode(self, target_mode):
        if self.current_mode == target_mode:
            self.set_mode(self.previous_view_mode) # SMART FALLBACK
            return

        if target_mode == "GLOBAL":
            self.unselect_instance() 
            self.set_mode("GLOBAL")
            return

        if not self.active_uid and target_mode != "SELECT":
            QMessageBox.warning(self, "Selection Required", "Please select a plant first.")
            self.set_mode(self.previous_view_mode) # SMART FALLBACK
            return
            
        self.set_mode(target_mode)

    def set_mode(self, mode):
        if self.current_mode in ["SELECT", "GLOBAL"] and mode not in ["SELECT", "GLOBAL"]:
            self.previous_view_mode = self.current_mode
            
        self.current_mode = mode
        
        self.btn_paint.setChecked(mode == "PAINT")
        self.btn_semantic.setChecked(mode == "SEMANTIC")
        self.btn_split.setChecked(mode == "SPLIT")
        self.btn_global.setChecked(mode == "GLOBAL") 
        
        self.list_classes.setVisible(mode == "SEMANTIC")
        self.btn_semantic.setEnabled(self.active_uid is not None)
        
        if mode == "SELECT":
            self.canvas.set_mode("SELECT")
        elif mode == "GLOBAL": 
            self.selected_ids.clear()
            self.active_uid = None
            self.list_instances.blockSignals(True)
            self.list_instances.clearSelection()
            self.list_instances.blockSignals(False)
            self.canvas.set_mode("SELECT") 
        elif mode == "PAINT":
            self.canvas.set_mode("PAINT")
        elif mode == "SEMANTIC":
            self.canvas.set_mode("SEMANTIC") 
        elif mode == "SPLIT":
            self.canvas.set_mode("SPLIT")
            
        self.update_info_label() 
        self.refresh_canvas()

    def refresh_canvas(self):
        raw_data = self.model.get_raw_image_data()
        if not raw_data: return
        
        base_pixmap = _bytes_to_pixmap(raw_data, is_rgba=False)
        op = self.slider_opacity.value() / 100.0
        
        # 1. GLOBAL MULTI-CLASS VIEW
        if self.current_mode == "GLOBAL":
            opacity = self.slider_opacity.value() / 100.0
            overlay_data = self.model.get_full_class_overlay_data(selected_ids=list(self.selected_ids), opacity=opacity)
            
            # Convert the raw bytes to a memory-safe QPixmap, then use the correct canvas method
            overlay_pixmap = _bytes_to_pixmap(overlay_data, is_rgba=True)
            self.canvas.update_view(base_pixmap, overlay_pixmap)
            self.canvas.update_bboxes(self.model.bboxes, self.model.color_map, self.show_bboxes)

        # 2. SELECT MODE
        elif self.current_mode == "SELECT":
            valid_selection = [uid for uid in self.selected_ids if uid in self.model.masks]
            overlay_data = self.model.get_overlay_data(selected_ids=valid_selection, opacity=op)
            overlay_pixmap = _bytes_to_pixmap(overlay_data, is_rgba=True)
            
            self.canvas.update_view(base_pixmap, overlay_pixmap)
            self.canvas.update_bboxes(self.model.bboxes, self.model.color_map, self.show_bboxes)
            
        # 3. INDIVIDUAL MULTI-CLASS EDITING
        elif self.current_mode == "SEMANTIC" and self.active_uid in self.model.masks:
            # We draw the multi-class directly onto the base background
            painter = QPainter(base_pixmap)
            
            class_data, cx, cy = self.model.get_class_overlay_data(self.active_uid, opacity=op)
            if class_data:
                painter.drawPixmap(cx, cy, _bytes_to_pixmap(class_data, is_rgba=True))
            
            cont_data, bx, by = self.model.get_binary_contours_data(self.active_uid)
            if cont_data:
                painter.drawPixmap(bx, by, _bytes_to_pixmap(cont_data, is_rgba=True))
                
            painter.end()
            
            self.canvas.update_view(base_pixmap, QPixmap()) 
            self.canvas.update_bboxes({}, {}, False)
            
        # 4. ISOLATION MODE (Paint/Split/New Plant)
        else: 
            if self.active_uid and self.active_uid in self.model.masks:
                overlay_data = self.model.get_overlay_data(isolate_uids=[self.active_uid], opacity=op)
                overlay_pixmap = _bytes_to_pixmap(overlay_data, is_rgba=True)
                self.canvas.update_view(base_pixmap, overlay_pixmap)
            else:
                self.canvas.update_view(base_pixmap, QPixmap())
            self.canvas.update_bboxes({}, {}, False)

    # ==========================================
    # SELECTION HANDLING
    # ==========================================
    def handle_canvas_click(self, x, y, shift):
        uid = self.model.get_id_at(x, y)
        
        # 1. Handle deselecting (clicking the background)
        if uid == 0:
            if not shift: self.unselect_instance()
            return

        # 2. Handle selecting
        if shift:
            if uid in self.selected_ids: self.selected_ids.remove(uid)
            else: self.selected_ids.add(uid)
        else:
            self.selected_ids = {uid}

        self.active_uid = uid if len(self.selected_ids) == 1 else None
        self.btn_semantic.setEnabled(self.active_uid is not None)
        
        # 3. SMART MODE FALLBACKS
        if self.active_uid is None:
            if self.current_mode in ["PAINT", "SPLIT", "SEMANTIC"]:
                self.set_mode(self.previous_view_mode)

        self.update_info_label()
        self.sync_list_selection()

    def on_list_selection_changed(self):
        self.selected_ids = {item.data(0, Qt.UserRole) for item in self.list_instances.selectedItems()}
        
        if self.selected_ids:
            if len(self.selected_ids) == 1:
                self.active_uid = list(self.selected_ids)[0]
                self.zoom_to_plant(self.active_uid)
            else:
                self.active_uid = None
                if self.current_mode in ["PAINT", "SPLIT", "SEMANTIC"]:
                    self.set_mode(self.previous_view_mode)
        else:
            self.active_uid = None
            if self.current_mode not in ["GLOBAL", "SELECT"]:
                self.set_mode(self.previous_view_mode)
                
        self.btn_semantic.setEnabled(self.active_uid is not None)
        
        self.update_info_label()
        self.refresh_canvas()

    def sync_list_selection(self):
        self.list_instances.blockSignals(True)
        for i in range(self.list_instances.topLevelItemCount()):
            item = self.list_instances.topLevelItem(i)
            uid = item.data(0, Qt.UserRole)
            item.setSelected(uid in self.selected_ids)
            if uid == self.active_uid:
                self.list_instances.scrollToItem(item)
        self.list_instances.blockSignals(False)
        self.refresh_canvas()

    def unselect_instance(self):
        self.selected_ids.clear()
        self.active_uid = None
        
        self.list_instances.blockSignals(True)
        self.list_instances.clearSelection()
        self.list_instances.blockSignals(False)
        
        # SMART FALLBACK: Return to your remembered overview
        if self.current_mode not in ["GLOBAL", "SELECT"]:
            self.set_mode(self.previous_view_mode) 
            
        self.btn_semantic.setEnabled(False)
        self.update_info_label()
        self.refresh_canvas()

    # ==========================================
    # ACTIONS (Converting QPointF to Python tuples)
    # ==========================================
    def handle_stroke(self, points, is_erase):
        if not self.active_uid: return
        
        # Translate QPointF objects to basic tuples (x, y) for the pure python model
        raw_points = [(p.x(), p.y()) for p in points]
        
        if self.current_mode == "SEMANTIC":
            item = self.list_classes.currentItem()
            cid = item.data(Qt.UserRole) if item else 1
            if not is_erase: 
                self.model.apply_class_stroke(self.active_uid, raw_points, self.slider_size.value(), cid)
        
        elif self.current_mode == "PAINT":
            # Check if this is the very first stroke of a newly created plant
            if self.active_uid not in self.model.masks:
                if not is_erase: 
                    self.selected_ids = {self.active_uid}
                    self.model.commit_new_instance(self.active_uid, raw_points, self.slider_size.value())
                    self.btn_semantic.setEnabled(True)
                    self.update_info_label()
                    self.sync_list_selection() 
            else:
                # Standard painting on an existing plant
                self.model.apply_stroke(self.active_uid, raw_points, self.slider_size.value(), is_erase)
                
    def handle_split(self, pts):
        if self.active_uid and self.current_mode == "SPLIT":
            # Translate to basic tuples
            raw_pts = [(p.x(), p.y()) for p in pts]
            success = self.model.split_instance(self.active_uid, raw_pts)
            
            if success:
                self.lbl_info.setText("Split successful.")
                self.unselect_instance() 
            else:
                self.lbl_info.setText("Split failed (lines didn't cross plant fully).")

    def action_new_instance(self):
        self.unselect_instance()
        self.active_uid = self.model.prepare_new_uid()
        self.current_mode = "PAINT" 
        self.btn_paint.setChecked(True)
        self.canvas.set_mode("PAINT")
        self.lbl_info.setText(f"CREATION MODE: Paint new Plant ID {self.active_uid}")
        self.refresh_canvas()

    def action_merge(self):
        if len(self.selected_ids) < 2: return
        new_id = self.model.merge_instances(list(self.selected_ids))
        if new_id is not None:
            self.selected_ids = {new_id}
            self.active_uid = new_id
        self.sync_list_selection()
        self.update_info_label()
        
    def action_delete(self):
        if self.selected_ids:
            self.model.delete_instances(list(self.selected_ids))
            self.unselect_instance()

    # ==========================================
    # UTILS
    # ==========================================
    def update_selection_from_mapping(self, mapping):
        """Translates the active selection to the new UIDs after a save reindexes them."""
        if not mapping: return
        
        new_selected = set()
        for old_id in self.selected_ids:
            if old_id in mapping:
                new_selected.add(mapping[old_id])
                
        self.selected_ids = new_selected
        
        if self.active_uid in mapping:
            self.active_uid = mapping[self.active_uid]
        else:
            self.active_uid = None
            
    def populate_class_palette(self):
        self.list_classes.clear()
        classes = [(1, "Main Root", "red"), (2, "Lateral Root", "green"), (3, "Seed", "blue"),
                   (4, "Hypocotyl", "yellow"), (5, "Leaves/Aerial", "cyan"), (6, "Petiole", "magenta")]
        for cid, name, color in classes:
            item = QListWidgetItem(f"{cid} - {name}")
            item.setData(Qt.UserRole, cid)
            pix = QPixmap(16,16); pix.fill(QColor(color))
            item.setIcon(QIcon(pix))
            self.list_classes.addItem(item)
        self.list_classes.setCurrentRow(0)

    def populate_list(self):
        """Called automatically when the Model's data changes via the callback pattern."""
        current_uids = set(self.model.masks.keys())
        existing_uids = {self.list_instances.topLevelItem(i).data(0, Qt.UserRole) for i in range(self.list_instances.topLevelItemCount())}
        
        # 1. Detect if we loaded a completely new image
        image_changed = (getattr(self, 'current_image_path', None) != self.model.image_path)
        if image_changed:
            self.current_image_path = self.model.image_path
            # Reset sorting back to Plant ID (Column 0) so the new image loads in logical order
            self.list_instances.sortItems(0, Qt.AscendingOrder)
        
        # 2. PERFORMANCE BOOST: If plants didn't change (just a paint stroke), update areas and abort!
        if current_uids == existing_uids and not image_changed:
            
            # BUGFIX: We MUST disable sorting while updating, otherwise items jump rows mid-loop!
            self.list_instances.setSortingEnabled(False)
            
            for i in range(self.list_instances.topLevelItemCount()):
                item = self.list_instances.topLevelItem(i)
                uid = item.data(0, Qt.UserRole)
                area = self.model.areas.get(uid, 0)
                item.setText(1, f"{area:,}")
                item.setData(1, Qt.UserRole, area)
                
            self.list_instances.setSortingEnabled(True) # Re-enable sorting safely
            self.refresh_canvas()
            return

        # 3. FULL REBUILD (New Image, Split, Merge, or Create)
        self.lbl_instance_count.setText(f"<b>PLANT INSTANCES ({len(self.model.masks)} Total):</b>")
        
        self.selected_ids = {uid for uid in self.selected_ids if uid in self.model.masks}
        if self.active_uid not in self.model.masks:
            self.active_uid = None
            if self.current_mode not in ["SELECT", "GLOBAL"]:
                self.set_mode("SELECT")

        old_sel = set(self.selected_ids)
        self.list_instances.blockSignals(True)
        
        self.list_instances.clearSelection()
        self.list_instances.setSortingEnabled(False)
        self.list_instances.clear()
        
        for uid in sorted(self.model.masks.keys()):
            area = self.model.areas.get(uid, 0)
            item = SortableTreeItem([f"Plant {uid}", f"{area:,}"])
            item.setData(0, Qt.UserRole, uid)   # ID for math sorting
            item.setData(1, Qt.UserRole, area)  # Area for math sorting
            
            if uid in self.model.color_map:
                c = self.model.color_map[uid]
                pix = QPixmap(16, 16); pix.fill(QColor(c[0], c[1], c[2]))
                item.setIcon(0, QIcon(pix))
                
            self.list_instances.addTopLevelItem(item)
            if uid in old_sel: item.setSelected(True)
            
        self.list_instances.setSortingEnabled(True) # Turn sorting back on
        self.list_instances.blockSignals(False)
        self.refresh_canvas()

    def zoom_to_plant(self, uid):
        if uid in self.model.bboxes:
            x, y, w, h = self.model.bboxes[uid]
            margin = 80
            self.canvas.fitInView(QRectF(x-margin, y-margin, w+margin*2, h+margin*2), Qt.KeepAspectRatio)
            
    def update_info_label(self):
        """Dynamically updates the top info bar based on current mode and selection."""
        mode = self.current_mode
        
        if mode == "SELECT":
            if self.active_uid:
                self.lbl_info.setText(f"SELECTION MODE: Plant ID {self.active_uid} Selected. (Shift+Click to add more)")
            elif len(self.selected_ids) > 1:
                ids_str = ", ".join(str(i) for i in sorted(self.selected_ids))
                self.lbl_info.setText(f"SELECTION MODE: {len(self.selected_ids)} Plants Selected (IDs: {ids_str}).")
            else:
                self.lbl_info.setText("SELECTION MODE: No plant selected. Click plants on the canvas or in the list to select.")
                
        elif mode == "GLOBAL": 
            self.lbl_info.setText("GLOBAL VIEW: Showing all Semantic Classes across the image.")
            
        elif mode == "PAINT":
            self.lbl_info.setText(f"SHAPE PAINT [Plant ID {self.active_uid}]: Left-Click to Paint | Right-Click to Erase.")
            
        elif mode == "SEMANTIC":
            self.lbl_info.setText(f"MULTI-CLASS [Plant ID {self.active_uid}]: Paint specific biology classes.")
             
        elif mode == "SPLIT":
            self.lbl_info.setText(f"SPLIT MODE [Plant ID {self.active_uid}]: Draw a red line across the plant to cut it.")