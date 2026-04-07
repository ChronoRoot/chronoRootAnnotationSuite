from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, 
                             QListWidgetItem, QPushButton, QLabel, QGraphicsView, 
                             QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, 
                             QMessageBox, QSlider, QTreeWidgetItem)

from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QPainterPath, QPen, QColor, 
                         QBrush, QIcon)

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


class ReviewCanvasTab(QWidget):
    """
    Handles ONLY the visual canvas, overlay rendering, and stroke translation.
    Listens to the shared PlantImageModel for state changes.
    """
    def __init__(self, shared_model):
        super().__init__()
        self.model = shared_model
        
        # Canvas State
        self.current_mode = "SELECT" 
        self.previous_view_mode = "SELECT" 
        self.show_bboxes = True
        self.opacity = 0.40
        self.brush_size = 5
        self.active_class_id = 1
        
        self.init_ui()
        
        # Connect to the pure python backend callbacks
        self.model.register_data_callback(self.on_data_changed)
        self.model.register_selection_callback(self.on_selection_changed)

    def init_ui(self):
        rl = QVBoxLayout(self)
        
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
        
        # 3. Bottom Navigation Instructions
        self.lbl_navigation = QLabel("Control click and drag for pan, use wheel to zoom")
        self.lbl_navigation.setStyleSheet("color: #666; font-style: italic; padding: 2px;")
        self.lbl_navigation.setAlignment(Qt.AlignCenter)
        rl.addWidget(self.lbl_navigation)
        
        self.update_info_label()

    # ==========================================
    # CANVAS STATE & RENDERING
    # ==========================================
    def set_mode(self, mode):
        if self.current_mode in ["SELECT", "GLOBAL"] and mode not in ["SELECT", "GLOBAL"]:
            self.previous_view_mode = self.current_mode
            
        self.current_mode = mode
        self.canvas.set_mode(mode)
        self.update_info_label()
        self.refresh_canvas()

    def set_opacity(self, val):
        self.opacity = val / 100.0
        self.refresh_canvas()

    def set_brush_size(self, val):
        self.brush_size = val
        self.canvas.set_brush_size(val)

    def set_show_bboxes(self, show):
        self.show_bboxes = show
        self.refresh_canvas()

    def set_active_class(self, class_id):
        self.active_class_id = class_id

    def refresh_canvas(self):
        raw_data = self.model.get_raw_image_data()
        if not raw_data: return
        
        base_pixmap = _bytes_to_pixmap(raw_data, is_rgba=False)
        
        # 1. GLOBAL MULTI-CLASS VIEW
        if self.current_mode == "GLOBAL":
            overlay_data = self.model.get_full_class_overlay_data(selected_ids=list(self.model.selected_uids), opacity=self.opacity)
            overlay_pixmap = _bytes_to_pixmap(overlay_data, is_rgba=True)
            self.canvas.update_view(base_pixmap, overlay_pixmap)
            self.canvas.update_bboxes(self.model.bboxes, self.model.color_map, self.show_bboxes)

        # 2. SELECT MODE
        elif self.current_mode == "SELECT":
            valid_selection = [uid for uid in self.model.selected_uids if uid in self.model.masks]
            overlay_data = self.model.get_overlay_data(selected_ids=valid_selection, opacity=self.opacity)
            overlay_pixmap = _bytes_to_pixmap(overlay_data, is_rgba=True)
            
            self.canvas.update_view(base_pixmap, overlay_pixmap)
            self.canvas.update_bboxes(self.model.bboxes, self.model.color_map, self.show_bboxes)
            
        # 3. INDIVIDUAL MULTI-CLASS EDITING
        elif self.current_mode == "SEMANTIC" and self.model.active_uid in self.model.masks:
            painter = QPainter(base_pixmap)
            class_data, cx, cy = self.model.get_class_overlay_data(self.model.active_uid, opacity=self.opacity)
            if class_data:
                painter.drawPixmap(cx, cy, _bytes_to_pixmap(class_data, is_rgba=True))
            
            cont_data, bx, by = self.model.get_binary_contours_data(self.model.active_uid)
            if cont_data:
                painter.drawPixmap(bx, by, _bytes_to_pixmap(cont_data, is_rgba=True))
                
            painter.end()
            self.canvas.update_view(base_pixmap, QPixmap()) 
            self.canvas.update_bboxes({}, {}, False)
            
        # 4. ISOLATION MODE (Paint/Split/New Plant)
        else: 
            if self.model.active_uid and self.model.active_uid in self.model.masks:
                overlay_data = self.model.get_overlay_data(isolate_uids=[self.model.active_uid], opacity=self.opacity)
                overlay_pixmap = _bytes_to_pixmap(overlay_data, is_rgba=True)
                self.canvas.update_view(base_pixmap, overlay_pixmap)
            else:
                self.canvas.update_view(base_pixmap, QPixmap())
            self.canvas.update_bboxes({}, {}, False)

    def update_info_label(self):
        """Dynamically updates the top info bar based on current mode and selection."""
        mode = self.current_mode
        active = self.model.active_uid
        selected = self.model.selected_uids
        
        if mode == "SELECT":
            if active:
                self.lbl_info.setText(f"SELECTION MODE: Plant ID {active} Selected. (Shift+Click to add more)")
            elif len(selected) > 1:
                ids_str = ", ".join(str(i) for i in sorted(selected))
                self.lbl_info.setText(f"SELECTION MODE: {len(selected)} Plants Selected (IDs: {ids_str}).")
            else:
                self.lbl_info.setText("SELECTION MODE: No plant selected. Click plants on the canvas or in the list to select.")
        elif mode == "GLOBAL": 
            self.lbl_info.setText("GLOBAL VIEW: Showing all Semantic Classes across the image.")
        elif mode == "PAINT":
            self.lbl_info.setText(f"SHAPE PAINT [Plant ID {active}]: Left-Click to Paint | Right-Click to Erase.")
        elif mode == "SEMANTIC":
            self.lbl_info.setText(f"MULTI-CLASS [Plant ID {active}]: Paint specific biology classes.")
        elif mode == "SPLIT":
            self.lbl_info.setText(f"SPLIT MODE [Plant ID {active}]: Draw a red line across the plant to cut it.")

    def zoom_to_plant(self, uid):
        if uid in self.model.bboxes:
            x, y, w, h = self.model.bboxes[uid]
            margin = 80
            self.canvas.fitInView(QRectF(x-margin, y-margin, w+margin*2, h+margin*2), Qt.KeepAspectRatio)

    # ==========================================
    # USER INTERACTIONS
    # ==========================================
    def handle_canvas_click(self, x, y, shift):
        uid = self.model.get_id_at(x, y)
        
        if uid == 0:
            if not shift: self.model.set_selection([])
            return

        current_sel = set(self.model.selected_uids)
        if shift:
            if uid in current_sel: current_sel.remove(uid)
            else: current_sel.add(uid)
        else:
            current_sel = {uid}

        # Updating the model automatically triggers on_selection_changed
        self.model.set_selection(list(current_sel))

    def handle_stroke(self, points, is_erase):
        uid = self.model.active_uid
        if not uid: return
        
        raw_points = [(p.x(), p.y()) for p in points]
        
        if self.current_mode == "SEMANTIC":
            if not is_erase: 
                self.model.apply_class_stroke(uid, raw_points, self.brush_size, self.active_class_id)
        
        elif self.current_mode == "PAINT":
            if uid not in self.model.masks:
                if not is_erase: 
                    self.model.set_selection([uid])
                    self.model.commit_new_instance(uid, raw_points, self.brush_size)
            else:
                self.model.apply_stroke(uid, raw_points, self.brush_size, is_erase)

    def handle_split(self, pts):
        if self.model.active_uid and self.current_mode == "SPLIT":
            raw_pts = [(p.x(), p.y()) for p in pts]
            success = self.model.split_instance(self.model.active_uid, raw_pts)
            
            if success:
                self.lbl_info.setText("Split successful.")
                self.model.set_selection([]) 
            else:
                self.lbl_info.setText("Split failed (lines didn't cross plant fully).")

    # ==========================================
    # MODEL CALLBACKS
    # ==========================================
    def on_data_changed(self):
        """Fires when masks are edited."""
        self.refresh_canvas()
        
    def on_selection_changed(self):
        """Fires when the global selection state changes."""            

        if self.model.active_uid is None:
            if self.current_mode in ["PAINT", "SPLIT", "SEMANTIC"]:
                self.set_mode(self.previous_view_mode)
                
        self.update_info_label()
        self.refresh_canvas()


class ReviewToolPanel(QWidget):
    """
    Handles ONLY the UI controls for the Annotation canvas.
    Requires a reference to ReviewCanvasTab to push mode/slider changes directly.
    """
    def __init__(self, shared_model, canvas_tab: ReviewCanvasTab):
        super().__init__()
        self.model = shared_model
        self.canvas_tab = canvas_tab
        
        self.init_ui()
        self.model.register_selection_callback(self.on_selection_changed)

    def init_ui(self):
        sl = QVBoxLayout(self)
        sl.setContentsMargins(5, 5, 5, 5)
        
        row_actions = QHBoxLayout()
        self.btn_unselect = QPushButton("Clear Selection")
        self.btn_unselect.clicked.connect(lambda: self.model.set_selection([]))
        
        self.btn_bbox = QPushButton("Show Bounding Boxes")
        self.btn_bbox.setCheckable(True)
        self.btn_bbox.setChecked(True)
        self.btn_bbox.clicked.connect(lambda: self.canvas_tab.set_show_bboxes(self.btn_bbox.isChecked()))
        
        row_actions.addWidget(self.btn_unselect)
        row_actions.addWidget(self.btn_bbox)
        sl.addLayout(row_actions)
        
        self.btn_global = QPushButton("Global Multi-Class View")
        self.btn_global.setCheckable(True)
        self.btn_global.clicked.connect(lambda: self.toggle_mode("GLOBAL"))
        sl.addWidget(self.btn_global)
        
        sl.addSpacing(15)
        
        self.btn_paint = QPushButton("Shape Paint")
        self.btn_paint.setCheckable(True)
        self.btn_paint.clicked.connect(lambda: self.toggle_mode("PAINT"))
        sl.addWidget(self.btn_paint)

        self.btn_split = QPushButton("Split (Knife)")
        self.btn_split.setCheckable(True)
        self.btn_split.clicked.connect(lambda: self.toggle_mode("SPLIT"))
        sl.addWidget(self.btn_split)
        
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
        self.list_classes.itemSelectionChanged.connect(self.on_class_selected)
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
        self.slider_opacity.sliderReleased.connect(lambda: self.canvas_tab.set_opacity(self.slider_opacity.value()))
        op_layout.addWidget(self.slider_opacity)
        op_layout.addWidget(self.lbl_op_val)
        sl.addLayout(op_layout)

        sl.addStretch()

    # ==========================================
    # TOOL LOGIC
    # ==========================================
    def toggle_mode(self, target_mode):
        current = self.canvas_tab.current_mode
        if current == target_mode:
            self.force_mode(self.canvas_tab.previous_view_mode) 
            return

        if target_mode == "GLOBAL":
            self.model.set_selection([]) 
            self.force_mode("GLOBAL")
            return

        if not self.model.active_uid and target_mode != "SELECT":
            QMessageBox.warning(self, "Selection Required", "Please select a plant first.")
            self.force_mode(self.canvas_tab.previous_view_mode) 
            return
            
        self.force_mode(target_mode)

    def force_mode(self, mode):
        """Forces the UI buttons and canvas into a specific mode. Used by MainWindow when creating a new plant."""
        self.btn_paint.setChecked(mode == "PAINT")
        self.btn_semantic.setChecked(mode == "SEMANTIC")
        self.btn_split.setChecked(mode == "SPLIT")
        self.btn_global.setChecked(mode == "GLOBAL") 
        
        self.list_classes.setVisible(mode == "SEMANTIC")
        self.canvas_tab.set_mode(mode)

    def update_brush_size(self, val):
        self.lbl_size_val.setText(f"{val}px")
        self.canvas_tab.set_brush_size(val)

    def on_class_selected(self):
        item = self.list_classes.currentItem()
        if item:
            self.canvas_tab.set_active_class(item.data(Qt.UserRole))

    def on_selection_changed(self):
        """Updates the enable/disable state of tools based on global selection."""
        has_active = self.model.active_uid is not None
        self.btn_semantic.setEnabled(has_active)
        
        # If selection was cleared and we were in a targeted mode, fallback
        if not has_active and self.canvas_tab.current_mode in ["PAINT", "SPLIT", "SEMANTIC"]:
            self.force_mode(self.canvas_tab.previous_view_mode)

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