import math
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, 
                             QListWidgetItem, QPushButton, QLabel, QGraphicsView, 
                             QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, 
                             QMessageBox, QSlider, QGraphicsSimpleTextItem, QSizePolicy)

from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import (QImage, QPixmap, QPainter, QPainterPath, QPen, QColor, 
                         QBrush, QIcon, QFont, QFontMetrics)

from components.ui_help import HELP_ANNOTATION, show_help

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
        self.ruler_pad_w = 0
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def update_view(self, base_pmap, overlay_pmap):
        self.base_pixmap.setPixmap(base_pmap)
        self.overlay_pixmap.setPixmap(overlay_pmap)
        if not base_pmap.isNull():
            pad = self.ruler_pad_w or 0
            self.scene.setSceneRect(-pad, 0, base_pmap.width() + pad, base_pmap.height())

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
    on_marquee_select = pyqtSignal(QRectF, bool) # --- NEW: Drag & Select Signal ---
    on_split_finish = pyqtSignal(list)
    distance_measured = pyqtSignal(float)  
    point_picked = pyqtSignal(int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.brush_cursor = None
        self.brush_size = 5 
        self.mode = "SELECT"
        
        self.is_painting = False
        self.is_erasing = False
        self.temp_item = None
        self.points_buffer = []
        
        # Marquee State
        self.marquee_start = None
        self.marquee_rect_item = None
        
        self.poly_points = []
        self.poly_lines = []
        self.rubber_band = None
        
        self.bbox_group = self.scene.createItemGroup([])
        self.bbox_group.setZValue(2) 
        
        self.cm_per_px = 1.0 
        self.ruler_line = None
        self.ruler_text = None
        self.is_calibrating = False 
        
        self.update_cursor_visual()

    def set_mode(self, mode):
        self.mode = mode
        if mode not in ["SPLIT", "RULER"]: self.clear_poly_visuals()
            
        if mode in ["SELECT", "GLOBAL"]:
            self.setCursor(Qt.ArrowCursor)
            if self.brush_cursor: self.brush_cursor.setVisible(False)
        elif mode in ["PAINT", "SEMANTIC"]:
            self.setCursor(Qt.CrossCursor)
            if self.brush_cursor: self.brush_cursor.setVisible(True)
        elif mode == "SPLIT":
            self.setCursor(Qt.PointingHandCursor)
            if self.brush_cursor: self.brush_cursor.setVisible(False)
        elif mode in ["RULER", "SEED"]:
            self.setCursor(Qt.CrossCursor)
            if self.brush_cursor: self.brush_cursor.setVisible(False)
            
    def update_cursor_visual(self):
        if self.brush_cursor: self.scene.removeItem(self.brush_cursor)
        s = self.brush_size
        self.brush_cursor = self.scene.addEllipse(0, 0, s, s, QPen(Qt.white, 1), QBrush(QColor(255, 255, 255, 30)))
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
        
        # --- NEW: Marquee Origin Setup ---
        if self.mode in ["SELECT", "GLOBAL"]:
            if event.button() == Qt.LeftButton:
                self.marquee_start = sp
                pen = QPen(Qt.white, 1, Qt.DashLine)
                pen.setCosmetic(True)
                self.marquee_rect_item = self.scene.addRect(QRectF(sp, sp), pen, QBrush(QColor(255, 255, 255, 50)))
                self.marquee_rect_item.setZValue(100)
        
        elif self.mode in ["PAINT", "SEMANTIC"]:
            if self.is_painting or self.is_erasing: return
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
                    
        elif self.mode == "SEED":
            if event.button() == Qt.LeftButton:
                self.point_picked.emit(int(sp.x()), int(sp.y()))

        elif self.mode == "RULER":
            if event.button() == Qt.LeftButton:
                self.clear_poly_visuals() 
                self.poly_points = [sp]
                
                pen = QPen(Qt.yellow, 3, Qt.DashLine)
                pen.setCosmetic(True)
                self.ruler_line = self.scene.addLine(sp.x(), sp.y(), sp.x(), sp.y(), pen)
                self.ruler_line.setZValue(100)
                
                self.ruler_text = self.scene.addText("0.00 cm")
                self.ruler_text.setDefaultTextColor(Qt.yellow)
                self.ruler_text.setPos(sp.x(), sp.y() - 20)
                self.ruler_text.setZValue(100)
                
                self.poly_lines.append(self.ruler_line)
                self.poly_lines.append(self.ruler_text)

    def mouseReleaseEvent(self, event):
        sp = self.mapToScene(event.pos())
        
        # --- NEW: Marquee Release Calculation ---
        if self.mode in ["SELECT", "GLOBAL"] and self.marquee_start:
            rect = QRectF(self.marquee_start, sp).normalized()
            shift = (event.modifiers() & Qt.ShiftModifier)
            
            # If the user just clicked without dragging, treat it as a standard click
            if rect.width() < 5 and rect.height() < 5:
                self.on_click.emit(int(sp.x()), int(sp.y()), shift)
            else:
                self.on_marquee_select.emit(rect, shift)
                
            if self.marquee_rect_item:
                self.scene.removeItem(self.marquee_rect_item)
                self.marquee_rect_item = None
            self.marquee_start = None
            
        elif (self.is_painting and event.button() == Qt.LeftButton) or \
             (self.is_erasing and event.button() == Qt.RightButton):
            
            if self.temp_item:
                self.scene.removeItem(self.temp_item)
                self.temp_item = None
                
            self.on_stroke_finished.emit(self.points_buffer, self.is_erasing)
            self.is_painting = False
            self.is_erasing = False
            self.points_buffer = []
            
        elif self.mode == "RULER" and event.button() == Qt.LeftButton:
            if self.poly_points:
                start_p = self.poly_points[0]
                dist_px = math.hypot(sp.x() - start_p.x(), sp.y() - start_p.y())
                self.distance_measured.emit(dist_px)
            self.poly_points.clear() 
            
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
        
        # --- NEW: Marquee Drag Animation ---
        if self.mode in ["SELECT", "GLOBAL"] and self.marquee_start and self.marquee_rect_item:
            rect = QRectF(self.marquee_start, sp).normalized()
            self.marquee_rect_item.setRect(rect)
        
        elif self.mode in ["PAINT", "SEMANTIC"] and self.brush_cursor:
            offset = self.brush_size / 2.0
            self.brush_cursor.setRect(sp.x() - offset, sp.y() - offset, self.brush_size, self.brush_size)

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
            
        if self.mode == "RULER" and self.poly_points and self.ruler_line and self.ruler_text:
            start_p = self.poly_points[0]
            self.ruler_line.setLine(start_p.x(), start_p.y(), sp.x(), sp.y())
            
            dist_px = math.hypot(sp.x() - start_p.x(), sp.y() - start_p.y())
            if not self.is_calibrating and self.cm_per_px is not None and self.cm_per_px > 0:
                dist_cm = dist_px * self.cm_per_px
                self.ruler_text.setPlainText(f"{dist_cm:.2f} cm")
            else:
                self.ruler_text.setPlainText(f"{dist_px:.1f} px")
                
            self.ruler_text.setPos(sp.x() + 10, sp.y() - 10)
            
        super().mouseMoveEvent(event)

    def update_overlays(self, bboxes, color_map, labels_dict, show_bboxes, show_labels, show_num=False, show_geno=False, geno_fmt="TEXT"):
        """Draws boundaries and dynamically constructs labels based on toggled preferences."""
        for item in self.bbox_group.childItems(): 
            self.scene.removeItem(item)
            
        if not show_bboxes and not show_labels:
            return
            
        if self.cm_per_px and self.cm_per_px > 0:
            font_px = max(12, int(0.2 / self.cm_per_px))
        else:
            font_px = 12
            
        font = QFont("Sans Serif")
        font.setPixelSize(font_px)
        font.setBold(True)

        for uid, (x, y, w, h) in bboxes.items():
            c = color_map.get(uid, (255, 255, 255))
            
            if show_bboxes:
                pen = QPen(QColor(c[0], c[1], c[2]), 2)
                pen.setCosmetic(True) 
                rect = QGraphicsRectItem(x, y, w, h)
                rect.setPen(pen)
                self.bbox_group.addToGroup(rect)
                
            if show_labels and uid in labels_dict:
                data = labels_dict[uid]
                text_str = ""
                
                # --- BACKWARD COMPATIBILITY: Annotation App sends a raw string ---
                if isinstance(data, str):
                    text_str = data
                    
                # --- NEW BEHAVIOR: Analyzer App sends a structured dictionary ---
                elif isinstance(data, dict):
                    lines = []
                    if show_num and data.get('plant_num'):
                        lines.append(f"Nº {data['plant_num']}")
                        
                    if show_geno:
                        if geno_fmt == "TEXT":
                            lines.append(data.get('geno_text', ''))
                        else:
                            lines.append(f"G: {data.get('geno_num', '?')}")
                            
                    text_str = "\n".join(lines).strip()
                
                # Only render if there is actual text to display
                if text_str:
                    text_item = QGraphicsSimpleTextItem(text_str)
                    text_item.setFont(font)
                    text_item.setBrush(QBrush(Qt.white))
                    
                    bg_rect = QGraphicsRectItem(text_item.boundingRect())
                    bg_rect.setBrush(QBrush(QColor(0, 0, 0, 160)))
                    bg_rect.setPen(QPen(Qt.NoPen))
                    
                    x_pos = x + w / 2 - bg_rect.boundingRect().height() / 2
                    y_pos = y - bg_rect.boundingRect().height() - 2
                    bg_rect.setPos(x_pos, y_pos)
                    text_item.setPos(x_pos, y_pos)
                    
                    self.bbox_group.addToGroup(bg_rect)
                    self.bbox_group.addToGroup(text_item)


class ReviewCanvasTab(QWidget):
    distance_measured = pyqtSignal(float)  
    point_picked = pyqtSignal(int, int)
    
    def __init__(self, shared_model):
        super().__init__()
        self.model = shared_model
        
        self.current_mode = "SELECT" 
        self.previous_view_mode = "SELECT" 
        
        self.show_bboxes = True
        
        # --- DEFAULT OFF FOR ANNOTATION APP COMPATIBILITY ---
        self.show_labels = False 
        self.label_show_num = False
        self.label_show_geno = False
        self.label_geno_format = "TEXT"
        
        self.overlay_labels = {}
        
        self.opacity = 0.40
        self.brush_size = 5
        self.active_class_id = 1
        self._batch_refresh = False
        self._batch_refresh_needed = False
        
        self.init_ui()
        
        self.model.register_data_callback(self.on_data_changed)
        self.model.register_selection_callback(self.on_selection_changed)

    def init_ui(self):
        rl = QVBoxLayout(self)
        
        self.lbl_info = QLabel()
        self.lbl_info.setStyleSheet("background-color: #eee; padding: 5px; font-weight: bold;")
        rl.addWidget(self.lbl_info)
        
        self.canvas = PaintCanvas()
        self.canvas.on_stroke_finished.connect(self.handle_stroke)
        self.canvas.on_click.connect(self.handle_canvas_click)
        self.canvas.on_split_finish.connect(self.handle_split)
        self.canvas.on_marquee_select.connect(self.handle_marquee_select) # Hooked up Drag
        
        self.canvas.distance_measured.connect(self.distance_measured.emit)
        self.canvas.point_picked.connect(self.point_picked.emit)
        
        rl.addWidget(self.canvas)
        
        self.lbl_navigation = QLabel("Control click and drag for pan, use wheel to zoom | Drag to marquee select in Select mode")
        self.lbl_navigation.setStyleSheet("color: #666; font-style: italic; padding: 2px;")
        self.lbl_navigation.setAlignment(Qt.AlignCenter)
        rl.addWidget(self.lbl_navigation)
        
        self.update_info_label()

    # ==========================================
    # CANVAS STATE & RENDERING
    # ==========================================
    def set_overlay_labels(self, label_dict):
        """Ingests the IDs and Genotypes from the Main Window."""
        self.overlay_labels = label_dict
        self.refresh_canvas()

    def set_show_labels(self, show):
        self.show_labels = show
        self.refresh_canvas()

    def set_calibrating(self, state):
        self.canvas.is_calibrating = state

    def set_mode(self, mode):
        if self.current_mode == "SELECT" and mode not in ["SELECT", "GLOBAL"]:
            self.previous_view_mode = "SELECT"
            
        self.current_mode = mode
        self.canvas.set_mode(mode)
        self.update_info_label()
        self.refresh_canvas()

    def set_label_preferences(self, show_num, show_geno, geno_format):
        """Receives display preferences from the Analyzer UI and updates the canvas."""
        # Auto-enable the master label toggle if either specific option is checked
        self.show_labels = show_num or show_geno 
        self.label_show_num = show_num
        self.label_show_geno = show_geno
        self.label_geno_format = geno_format
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

    def update_scene_ruler(self, cm_per_px):
        self.canvas.cm_per_px = cm_per_px
        
        if hasattr(self, 'scene_ruler_items'):
            for item in self.scene_ruler_items:
                self.canvas.scene.removeItem(item)
        self.scene_ruler_items = []
        
        if cm_per_px is None or cm_per_px <= 0 or not hasattr(self, '_cached_base_pixmap'):
            self.canvas.ruler_pad_w = 0
            return
            
        img_w = self._cached_base_pixmap.width()
        img_h = self._cached_base_pixmap.height()
        pixels_per_cm = int(1.0 / cm_per_px)
        
        if pixels_per_cm < 10:
            self.canvas.ruler_pad_w = 0
            return

        font_px = max(12, int(0.2 / cm_per_px))
        font = QFont("Sans Serif")
        font.setPixelSize(font_px)
        font.setBold(True)
        metrics = QFontMetrics(font)
        label_w = metrics.boundingRect("000cm").width()
        gutter_pad = max(4, font_px // 5)
        tick_len = max(10, int(font_px * 0.8))
        stroke = max(2, font_px // 12)
        pad_w = gutter_pad + label_w + gutter_pad + tick_len + gutter_pad
        line_x = -gutter_pad

        self.canvas.ruler_pad_w = pad_w
        self.canvas.scene.setSceneRect(-pad_w, 0, img_w + pad_w, img_h)

        bg = self.canvas.scene.addRect(-pad_w, 0, pad_w, img_h, QPen(Qt.NoPen), QBrush(Qt.black))
        bg.setZValue(2)
        self.scene_ruler_items.append(bg)

        pen = QPen(Qt.white, stroke)
        line = self.canvas.scene.addLine(line_x, 0, line_x, img_h, pen)
        line.setZValue(3)
        self.scene_ruler_items.append(line)

        for cm_val in range(0, int(img_h / pixels_per_cm)):
            y = int(cm_val * pixels_per_cm)
            tick = self.canvas.scene.addLine(line_x - tick_len, y, line_x, y, pen)
            tick.setZValue(3)
            self.scene_ruler_items.append(tick)

            text = QGraphicsSimpleTextItem(f"{cm_val}cm")
            text.setFont(font)
            text.setBrush(QBrush(Qt.white))
            text.setPos(-pad_w + gutter_pad, y - metrics.height() / 2)
            text.setZValue(3)
            self.canvas.scene.addItem(text)
            self.scene_ruler_items.append(text)

    def begin_batch_refresh(self):
        self._batch_refresh = True
        self._batch_refresh_needed = False

    def end_batch_refresh(self):
        self._batch_refresh = False
        if self._batch_refresh_needed:
            self._batch_refresh_needed = False
            self.refresh_canvas()

    def refresh_canvas(self):
        if self._batch_refresh:
            self._batch_refresh_needed = True
            return
        if not hasattr(self, '_cached_base_pixmap') or getattr(self, '_last_image_path', None) != self.model.image_path:
            raw_data = self.model.get_raw_image_data()
            if not raw_data:
                return
            self._cached_base_pixmap = _bytes_to_pixmap(raw_data, is_rgba=False)
            self._last_image_path = self.model.image_path
            
        base_pixmap = self._cached_base_pixmap
        
        # Safely fetch the label preferences (Fallbacks ensure the Annotation App doesn't crash)
        s_num = getattr(self, 'label_show_num', False)
        s_geno = getattr(self, 'label_show_geno', False)
        g_fmt = getattr(self, 'label_geno_format', "TEXT")
        
        # 1. GLOBAL MULTI-CLASS VIEW
        if self.current_mode == "GLOBAL":
            overlay_data = self.model.get_full_class_overlay_data(selected_ids=list(self.model.selected_uids), opacity=self.opacity)
            overlay_pixmap = _bytes_to_pixmap(overlay_data, is_rgba=True)
            self.canvas.update_view(base_pixmap, overlay_pixmap)
            self.canvas.update_overlays(self.model.bboxes, self.model.color_map, self.overlay_labels, 
                                        self.show_bboxes, self.show_labels, s_num, s_geno, g_fmt)

        # 2. SELECT MODE (SEED keeps the same view so seeds can be placed between plants)
        elif self.current_mode in ["SELECT", "SEED"]:
            valid_selection = [uid for uid in self.model.selected_uids if uid in self.model.masks]
            overlay_data = self.model.get_overlay_data(selected_ids=valid_selection, opacity=self.opacity)
            overlay_pixmap = _bytes_to_pixmap(overlay_data, is_rgba=True)
            self.canvas.update_view(base_pixmap, overlay_pixmap)
            self.canvas.update_overlays(self.model.bboxes, self.model.color_map, self.overlay_labels, 
                                        self.show_bboxes, self.show_labels, s_num, s_geno, g_fmt)
            
        # 3. INDIVIDUAL MULTI-CLASS EDITING
        elif self.current_mode == "SEMANTIC" and self.model.active_uid in self.model.masks:
            overlay_pixmap = QPixmap(base_pixmap.size())
            overlay_pixmap.fill(Qt.transparent)
            painter = QPainter(overlay_pixmap)
            
            class_data, cx, cy = self.model.get_class_overlay_data(self.model.active_uid, opacity=self.opacity)
            if class_data:
                painter.drawPixmap(cx, cy, _bytes_to_pixmap(class_data, is_rgba=True))
            
            cont_data, bx, by = self.model.get_binary_contours_data(self.model.active_uid)
            if cont_data:
                painter.drawPixmap(bx, by, _bytes_to_pixmap(cont_data, is_rgba=True))
                
            painter.end()
            self.canvas.update_view(base_pixmap, overlay_pixmap) 
            self.canvas.update_overlays({}, {}, {}, False, False, True, True, "TEXT")
            
        # 4. ISOLATION MODE
        else: 
            if self.model.active_uid and self.model.active_uid in self.model.masks:
                overlay_data = self.model.get_overlay_data(isolate_uids=[self.model.active_uid], opacity=self.opacity)
                overlay_pixmap = _bytes_to_pixmap(overlay_data, is_rgba=True)
                self.canvas.update_view(base_pixmap, overlay_pixmap)
            else:
                self.canvas.update_view(base_pixmap, QPixmap())
            self.canvas.update_overlays({}, {}, {}, False, False, True, True, "TEXT")

    def update_info_label(self):
        mode = self.current_mode
        active = self.model.active_uid
        selected = self.model.selected_uids
        
        if mode == "SELECT":
            if active:
                self.lbl_info.setText(
                    f"Plant {active} selected. Shift+click to add more plants."
                )
            elif len(selected) > 1:
                ids_str = ", ".join(str(i) for i in sorted(selected))
                self.lbl_info.setText(f"{len(selected)} plants selected ({ids_str}).")
            else:
                self.lbl_info.setText(
                    "No plant selected. Click a plant on the canvas or in the list."
                )
        elif mode == "GLOBAL":
            self.lbl_info.setText("Review all labels — every plant and root part on the plate.")
        elif mode == "PAINT":
            self.lbl_info.setText(
                f"Plant {active} — Outline: left = add, right = erase."
            )
        elif mode == "SEMANTIC":
            self.lbl_info.setText(
                f"Plant {active} — Paint root part labels (Main Root, Lateral Root, etc.)."
            )
        elif mode == "SPLIT":
            self.lbl_info.setText(
                f"Plant {active} — Draw a red line across the plant to cut; double-click to finish."
            )
        elif mode == "RULER":
            self.lbl_info.setText(
                "Ruler: click and drag to measure. Set calibration in Plant Metadata for cm."
            )
        elif mode == "SEED":
            self.lbl_info.setText(
                "Non-germinated seed: click its position on the plate. One click places one seed."
            )

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

        self.model.set_selection(list(current_sel))

    def handle_marquee_select(self, rect, shift):
        """Select plants whose actual mask pixels touch the marquee."""
        selected_uids = []
        for uid, bbox in self.model.bboxes.items():
            bx, by, bw, bh = bbox
            if not rect.intersects(QRectF(bx, by, bw, bh)):
                continue

            mask = self.model.masks.get(uid)
            if mask is None:
                continue

            x1 = max(0, bx, math.floor(rect.left()))
            y1 = max(0, by, math.floor(rect.top()))
            x2 = min(mask.shape[1], bx + bw, math.floor(rect.right()) + 1)
            y2 = min(mask.shape[0], by + bh, math.floor(rect.bottom()) + 1)
            if x1 < x2 and y1 < y2 and mask[y1:y2, x1:x2].any():
                selected_uids.append(uid)
                
        if not selected_uids:
            if not shift: self.model.set_selection([])
            return
            
        current_sel = set(self.model.selected_uids)
        if shift:
            for u in selected_uids:
                if u in current_sel: current_sel.remove(u)
                else: current_sel.add(u)
        else:
            current_sel = set(selected_uids)
            
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
        self.refresh_canvas()
        
    def on_selection_changed(self):
        self.update_info_label()
        self.refresh_canvas()


class ReviewToolPanel(QWidget):
    def __init__(self, shared_model, canvas_tab: ReviewCanvasTab):
        super().__init__()
        self.model = shared_model
        self.canvas_tab = canvas_tab
        
        self.init_ui()
        self.model.register_selection_callback(self.on_selection_changed)
        self.model.register_history_callback(self.update_undo_state)
        self.update_undo_state()

    def init_ui(self):
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        sl = QVBoxLayout(self)
        sl.setContentsMargins(5, 5, 5, 5)
        sl.setSpacing(4)

        tools_header = QHBoxLayout()
        self.lbl_workflow_hint = QLabel(
            "<span style='color:#555; font-size:11px;'>"
            "Select a plant in the list, then use the tools below.</span>"
        )
        tools_header.addWidget(self.lbl_workflow_hint)
        tools_header.addStretch()
        self.btn_ann_help = QPushButton("Help")
        self.btn_ann_help.setToolTip("Open annotation workflow help")
        self.btn_ann_help.clicked.connect(
            lambda: show_help(self, "Annotation Help", HELP_ANNOTATION)
        )
        tools_header.addWidget(self.btn_ann_help)
        sl.addLayout(tools_header)
        
        row_actions = QHBoxLayout()
        self.btn_unselect = QPushButton("Clear Selection")
        self.btn_unselect.setToolTip("Clear the current plant selection.")
        self.btn_unselect.clicked.connect(lambda: self.model.set_selection([]))
        
        self.btn_bbox = QPushButton("Bounding Boxes")
        self.btn_bbox.setCheckable(True)
        self.btn_bbox.setChecked(True)
        self.btn_bbox.setToolTip("Show or hide bounding boxes around each plant.")
        self.btn_bbox.clicked.connect(lambda: self.canvas_tab.set_show_bboxes(self.btn_bbox.isChecked()))
        
        row_actions.addWidget(self.btn_unselect)
        row_actions.addWidget(self.btn_bbox)
        sl.addLayout(row_actions)
        
        self.btn_global = QPushButton("Review all labels")
        self.btn_global.setCheckable(True)
        self.btn_global.setToolTip("Show every plant and root part label on the plate at once.")
        self.btn_global.clicked.connect(lambda: self.toggle_mode("GLOBAL"))
        sl.addWidget(self.btn_global)
        
        sl.addSpacing(15)
        
        self.btn_paint = QPushButton("Shape Paint")
        self.btn_paint.setCheckable(True)
        self.btn_paint.setToolTip("Paint or erase the plant outline (left = add, right = erase).")
        self.btn_paint.clicked.connect(lambda: self.toggle_mode("PAINT"))
        sl.addWidget(self.btn_paint)

        self.btn_split = QPushButton("Split (Knife)")
        self.btn_split.setCheckable(True)
        self.btn_split.setToolTip("Draw a line across fused plants; double-click to cut.")
        self.btn_split.clicked.connect(lambda: self.toggle_mode("SPLIT"))
        sl.addWidget(self.btn_split)

        self.btn_split_parts = QPushButton("Split disconnected fragments")
        self.btn_split_parts.setToolTip(
            "Split one plant ID into separate IDs for each disconnected mask piece."
        )
        self.btn_split_parts.clicked.connect(self.action_split_parts)
        sl.addWidget(self.btn_split_parts)

        self.btn_undo = QPushButton("Undo Last Action")
        self.btn_undo.setToolTip(
            "Undo the last annotation action, including paint, seeds, split, merge, "
            "delete, Graph, or Frangi. History resets after save or load."
        )
        self.btn_undo.clicked.connect(self.model.undo)
        sl.addWidget(self.btn_undo)
        
        self.btn_semantic = QPushButton("Multi-Class Paint")
        self.btn_semantic.setCheckable(True)
        self.btn_semantic.setToolTip("Assign Main Root, Lateral Root, Hypocotyl, and other root part labels.")
        self.btn_semantic.clicked.connect(lambda: self.toggle_mode("SEMANTIC"))
        sl.addWidget(self.btn_semantic)

        self.list_classes = QListWidget()
        self.list_classes.setFixedHeight(120)
        self.list_classes.setToolTip("Select which root part label to paint.")
        self.list_classes.currentItemChanged.connect(self.on_class_selected)
        self.populate_class_palette()
        self.list_classes.setVisible(False)
        sl.addWidget(self.list_classes)
        
        sl.addWidget(QLabel("Brush Size:"))
        size_layout = QHBoxLayout()
        self.slider_size = QSlider(Qt.Horizontal)
        self.slider_size.setRange(1, 30); self.slider_size.setValue(5)
        self.slider_size.setToolTip("Width of the paint and erase brush.")
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
        self.slider_opacity.setToolTip("Transparency of painted masks over the raw image.")
        self.lbl_op_val = QLabel("40%")
        self.lbl_op_val.setFixedWidth(40)
        self.slider_opacity.valueChanged.connect(lambda v: self.lbl_op_val.setText(f"{v}%"))
        self.slider_opacity.sliderReleased.connect(lambda: self.canvas_tab.set_opacity(self.slider_opacity.value()))
        op_layout.addWidget(self.slider_opacity)
        op_layout.addWidget(self.lbl_op_val)
        sl.addLayout(op_layout)

        self.setFixedHeight(self.sizeHint().height())

    def toggle_mode(self, target_mode):
        current = self.canvas_tab.current_mode
        if target_mode == "GLOBAL":
            if current == "GLOBAL":
                self.force_mode("SELECT")
            else:
                self.model.set_selection([])
                self.force_mode("GLOBAL")
            return

        if current == target_mode:
            restore = self.canvas_tab.previous_view_mode
            if restore == target_mode:
                restore = "SELECT"
            self.force_mode(restore)
            return

        if not self.model.active_uid and target_mode != "SELECT":
            QMessageBox.warning(self, "Selection Required", "Please select a plant first.")
            restore = self.canvas_tab.previous_view_mode
            if restore == target_mode:
                restore = "SELECT"
            self.force_mode(restore)
            return
            
        self.force_mode(target_mode)

    def force_mode(self, mode):
        self.btn_paint.setChecked(mode == "PAINT")
        self.btn_semantic.setChecked(mode == "SEMANTIC")
        self.btn_split.setChecked(mode == "SPLIT")
        self.btn_global.setChecked(mode == "GLOBAL") 
        
        self.list_classes.setVisible(mode == "SEMANTIC")
        if mode == "SEMANTIC":
            self.on_class_selected(self.list_classes.currentItem())
        self.canvas_tab.set_mode(mode)
        self.setFixedHeight(self.sizeHint().height())

    def update_brush_size(self, val):
        self.lbl_size_val.setText(f"{val}px")
        self.canvas_tab.set_brush_size(val)

    def update_undo_state(self):
        self.btn_undo.setEnabled(
            bool(self.model.history) and not self.model.callbacks_muted
        )

    def on_class_selected(self, item, previous=None):
        if item is not None:
            self.canvas_tab.set_active_class(int(item.data(Qt.UserRole)))

    def on_selection_changed(self):
        has_active = self.model.active_uid is not None
        self.btn_semantic.setEnabled(has_active)
        self.btn_split_parts.setEnabled(has_active) 
        
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
        
    def action_split_parts(self):
        uid = self.model.active_uid
        if not uid:
            QMessageBox.warning(self, "Selection Required", "Please select a plant first.")
            return
            
        success = self.model.split_disconnected_components(uid)
        if success:
            self.model.set_selection([])
            self.force_mode("SELECT")
            self.canvas_tab.lbl_info.setText("Separated parts split successfully.")
        else:
            QMessageBox.information(self, "Split disconnected fragments", "No disconnected parts found in this plant.")