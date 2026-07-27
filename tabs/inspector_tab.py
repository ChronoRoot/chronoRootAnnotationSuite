import cv2
import numpy as np
import xml.etree.ElementTree as ET
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QCheckBox,
                             QLabel, QTextEdit, QSizePolicy, QPushButton,
                             QGroupBox, QRadioButton, QButtonGroup, QFileDialog, QMessageBox,
                             QSpinBox, QFormLayout, QStackedWidget)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QSize, QRect, pyqtSignal
from PyQt5.QtSvg import QSvgGenerator

from components.ui_help import HELP_INSPECTOR, show_help

class AspectRatioLabel(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(100, 100) 
        self._pixmap = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update()

    def clear(self):
        self._pixmap = None
        self.update()

    def paintEvent(self, event):
        if self._pixmap and not self._pixmap.isNull():
            p = QPainter(self)
            rect = self.contentsRect()
            scaled_pixmap = self._pixmap.scaled(rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = int((rect.width() - scaled_pixmap.width()) / 2)
            y = int((rect.height() - scaled_pixmap.height()) / 2)
            p.drawPixmap(x, y, scaled_pixmap)

class PhenomicsInspectorTab(QWidget):
    run_analysis_requested = pyqtSignal()
    go_to_metadata_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.current_uid = None
        self.current_data = None
        self.current_viz = None
        self.current_plant_meta = {}
        self.current_raw = None
        self.current_bbox = None
        self.current_cm_px = None
        self.loaded_rsml_roots = []
        self.init_ui()

    def init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()

        self.empty_page = QWidget()
        empty_layout = QVBoxLayout(self.empty_page)
        empty_layout.addStretch()
        self.lbl_empty = QLabel(
            "<b>No measurements yet.</b><br><br>"
            "Open <b>Plant Metadata</b>, set scale and genotypes, "
            "then click <b>Run Measurements</b>."
        )
        self.lbl_empty.setWordWrap(True)
        self.lbl_empty.setAlignment(Qt.AlignCenter)
        self.lbl_empty.setStyleSheet("color: #555; font-size: 14px; padding: 24px;")
        empty_layout.addWidget(self.lbl_empty)

        self.btn_run_analysis = QPushButton("Go to Plant Metadata")
        self.btn_run_analysis.setToolTip("Open Plant Metadata to set scale, genotypes, and run measurements.")
        self.btn_run_analysis.setStyleSheet(
            "background-color: #007bff; color: white; font-weight: bold; padding: 12px;"
        )
        self.btn_run_analysis.clicked.connect(self.go_to_metadata_requested.emit)
        empty_layout.addWidget(self.btn_run_analysis, alignment=Qt.AlignCenter)
        empty_layout.addStretch()

        self.content_page = QWidget()
        insp_layout = QHBoxLayout(self.content_page)
        
        # ==========================================
        # LEFT PANEL: Canvas
        # ==========================================
        self.lbl_inspector_img = AspectRatioLabel()
        self.lbl_inspector_img.setStyleSheet("background-color: #1e1e1e; border: 1px solid #333;")
        insp_layout.addWidget(self.lbl_inspector_img, stretch=2)
        
        # ==========================================
        # RIGHT PANEL: Controls
        # ==========================================
        right_layout = QVBoxLayout()

        insp_header = QHBoxLayout()
        insp_header.addWidget(QLabel("<b>Phenomics review</b>"))
        insp_header.addStretch()
        btn_help = QPushButton("Help")
        btn_help.setToolTip("Open inspector help")
        btn_help.clicked.connect(lambda: show_help(self, "Inspector Help", HELP_INSPECTOR))
        insp_header.addWidget(btn_help)
        right_layout.addLayout(insp_header)
        
        # --- Group 1: Core Visualization ---
        group_view = QGroupBox("Root Representation")
        vbox_view = QVBoxLayout()
        
        self.rad_view_model = QRadioButton("Colored root parts (annotation)")
        self.rad_view_model.setToolTip("Semantic labels painted in Annotation.")
        self.rad_view_graph_mask = QRadioButton("Colored skeleton")
        self.rad_view_graph_mask.setToolTip("Mask pixels validated by the root tracing graph.")
        self.rad_view_rsml = QRadioButton("RSML architecture preview")
        self.rad_view_rsml.setToolTip("RSML architecture overlay.")
        self.rad_view_none = QRadioButton("Raw image only")
        self.rad_view_none.setToolTip("Scanner image without overlays.")
        self.rad_view_model.setChecked(True)

        self.view_group = QButtonGroup()
        for rad in [self.rad_view_model, self.rad_view_graph_mask, self.rad_view_rsml, self.rad_view_none]:
            self.view_group.addButton(rad)
            rad.toggled.connect(self._trigger_redraw)
            vbox_view.addWidget(rad)
            
        group_view.setLayout(vbox_view)
        right_layout.addWidget(group_view)

        # --- Group 2: Features & Angles ---
        group_angles = QGroupBox("Biological Features")
        vbox_ang = QVBoxLayout()
        
        self.rad_ang_none = QRadioButton("Hide Angles")
        self.rad_ang_emergence = QRadioButton("Lateral emergence angle (2 mm)")
        self.rad_ang_emergence.setToolTip("Angle where lateral roots emerge from the main root.")
        self.rad_ang_tip = QRadioButton("Overall Tip Angles")
        self.rad_ang_none.setChecked(True)

        self.angle_group = QButtonGroup()
        for rad in [self.rad_ang_none, self.rad_ang_emergence, self.rad_ang_tip]:
            self.angle_group.addButton(rad)
            rad.toggled.connect(self._trigger_redraw)
            vbox_ang.addWidget(rad)
            
        self.chk_hull = QCheckBox("Show Convex Hull")
        self.chk_hull.setChecked(True)
        self.chk_hull.stateChanged.connect(self._trigger_redraw)
        vbox_ang.addWidget(self.chk_hull)
            
        group_angles.setLayout(vbox_ang)
        right_layout.addWidget(group_angles)

        # --- Group 3: Rendering Settings ---
        group_settings = QGroupBox("Render Settings")
        form_set = QFormLayout()
        self.spin_width = QSpinBox()
        self.spin_width.setRange(1, 10)
        self.spin_width.setValue(2)
        self.spin_width.valueChanged.connect(self._trigger_redraw)
        form_set.addRow("Line/Mask Width:", self.spin_width)
        group_settings.setLayout(form_set)
        right_layout.addWidget(group_settings)
        
        # --- Group 4: Metrics Readout ---
        self.txt_metrics = QTextEdit()
        self.txt_metrics.setReadOnly(True)
        self.txt_metrics.setStyleSheet("font-family: monospace; font-size: 13px; background-color: #f8f9fa; color: #333;")
        right_layout.addWidget(self.txt_metrics, stretch=1)
        
        # --- Export Button ---
        self.btn_export_svg = QPushButton("Export Editable Vector (.SVG)")
        self.btn_export_svg.setToolTip("Export the current inspector view as an SVG figure.")
        self.btn_export_svg.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 10px;")
        self.btn_export_svg.clicked.connect(self.export_to_svg)
        right_layout.addWidget(self.btn_export_svg)

        insp_layout.addLayout(right_layout, stretch=1)

        self.stack.addWidget(self.empty_page)
        self.stack.addWidget(self.content_page)
        root_layout.addWidget(self.stack)
        self.stack.setCurrentIndex(0)

    def show_no_measurements_state(self):
        self.stack.setCurrentIndex(0)
        self.lbl_inspector_img.clear()
        self.txt_metrics.clear()
        self.loaded_rsml_roots = []
        self.current_uid = None
        self.current_data = None
        self.current_viz = None
        self.current_plant_meta = {}

    def show_content_state(self):
        self.stack.setCurrentIndex(1)

    def _trigger_redraw(self):
        # Prevent unnecessary parsing, just trigger the paint event
        self._render_current_state()

    def _get_shifted_coord(self, pt, x1, y1, pad_ruler):
        return (int(pt[0]) - x1 + pad_ruler, int(pt[1]) - y1)
    
    def refresh_plant_metadata(self, plant_meta, data, uid):
        """Update genotype/plant number readout without recomputing graph visuals."""
        self.current_plant_meta = plant_meta or {}
        if not data:
            return
        display_genotype = self.current_plant_meta.get("genotype") or data.get("genotype", "N/A")
        display_plant_num = self.current_plant_meta.get("plant_num") or data.get("plant_num", "N/A")

        report = f"--- REPORT ID: {uid} ---\n"
        report += f"Genotype: {display_genotype}\n"
        report += f"Plant #:  {display_plant_num}\n\n"

        report += "=== BASIC ARCHITECTURE ===\n"
        report += f"Main Root Length (MR):       {data.get('mr_length_mm', 0.0):.2f} mm\n"
        report += f"Lateral Root Length (LR):    {data.get('lr_length_mm', 0.0):.2f} mm\n"
        report += f"Total Root Length (TR):      {data.get('tr_length_mm', 0.0):.2f} mm\n"
        report += f"Number of Lateral Roots:     {data.get('lr_count', 0)} count\n"
        report += f"Discrete LR Density:         {data.get('lr_density_cm', 0.0):.2f} LRs/cm\n"
        report += f"Main Over Total Root (MR/TR): {data.get('mr_over_tr_ratio', 0.0):.2f}\n\n"

        report += "=== SPATIAL DISTRIBUTION ===\n"
        report += f"Convex Hull Area:            {data.get('hull_area_mm2', 0.0):.2f} mm²\n"
        report += f"Convex Hull Width:           {data.get('hull_width_mm', 0.0):.2f} mm\n"
        report += f"Convex Hull Height:          {data.get('hull_height_mm', 0.0):.2f} mm\n"
        report += f"Root Density:                {data.get('root_density_mm_mm2', 0.0):.4f} mm/mm²\n"
        report += f"Aspect Ratio (Height/Width): {data.get('aspect_ratio', 0.0):.2f}\n\n"

        report += "=== ROOT ANGLES ===\n"
        report += f"Mean Tip Angle:              {data.get('tip_angle_deg', 0.0):.1f}°\n"
        report += f"Mean Emerg. Angle (2mm):     {data.get('emergence_angle_deg', 0.0):.1f}°\n"

        self.txt_metrics.setText(report)

    def update_view(self, uid, data, raw_image, bbox, cm_per_px, plant_meta=None, viz_data=None):
        self.current_uid, self.current_data, self.current_raw = uid, data, raw_image
        self.current_bbox, self.current_cm_px = bbox, cm_per_px
        self.current_plant_meta = plant_meta or {}
        self.current_viz = viz_data
        
        if not uid or not data:
            self.show_content_state()
            self.lbl_inspector_img.clear()
            self.txt_metrics.clear()
            self.loaded_rsml_roots = []
            return

        self.show_content_state()
            
        # ==========================================
        # 1. LIVE RSML PARSING (In-Memory)
        # ==========================================
        self.loaded_rsml_roots = []
        rsml_source = viz_data or {}
        if rsml_source.get("rsml_xml") is not None:
            plant_xml = rsml_source["rsml_xml"]
            
            def parse_root_node(elem, current_order):
                pts = []
                polyline = elem.find("./geometry/polyline")
                if polyline is not None:
                    for pt in polyline.findall("point"):
                        pts.append((float(pt.get('x')), float(pt.get('y'))))
                if pts:
                    self.loaded_rsml_roots.append({'order': current_order, 'points': pts})
                
                # Recurse for all children roots
                for child in elem.findall("./root"):
                    parse_root_node(child, current_order + 1)

            for main_root in plant_xml.findall("./root"):
                parse_root_node(main_root, current_order=0)

        # ==========================================
        # 2. MANUSCRIPT METRICS REPORT READOUT
        # ==========================================
        self.refresh_plant_metadata(self.current_plant_meta, data, uid)
        
        # Trigger the visual update
        self._render_current_state()

    def _render_current_state(self):
        """Dedicated rendering block so UI toggles don't trigger re-parsing."""
        if not self.current_uid or not self.current_data:
            return

        x, y, w, h = self.current_bbox
        if w == 0: return
        
        pad, pad_ruler = 40, 80
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(self.current_raw.shape[1], x+w+pad), min(self.current_raw.shape[0], y+h+pad)
        
        padded_rgb = self._generate_raster_canvas(
            self.current_data, self.current_viz, self.current_raw, x1, y1, x2, y2, pad_ruler
        )
        
        h_img, w_img, _ = padded_rgb.shape
        qimg = QImage(padded_rgb.data, w_img, h_img, 3 * w_img, QImage.Format_RGB888)
        self.lbl_inspector_img.setPixmap(QPixmap.fromImage(qimg))

    def _generate_raster_canvas(self, data, viz_data, raw_image, x1, y1, x2, y2, pad_ruler):
        crop_rgb = raw_image[y1:y2, x1:x2].copy()
        padded_rgb = cv2.copyMakeBorder(crop_rgb, 0, 0, pad_ruler, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        h_img, w_img, _ = padded_rgb.shape
        viz = viz_data or {}
        cx_off, cy_off = viz.get("crop_offset", (0, 0))
        line_w = self.spin_width.value()
        
        # --- RULER ---
        cm_in_px = int(1.0 / self.current_cm_px) if self.current_cm_px else 100
        if cm_in_px > 10:
            cv2.line(padded_rgb, (pad_ruler - 15, 10), (pad_ruler - 15, h_img - 10), (200, 200, 200), 2)
            for tick_y in range(10, h_img - 10, cm_in_px):
                cv2.line(padded_rgb, (pad_ruler - 25, tick_y), (pad_ruler - 15, tick_y), (200, 200, 200), 2)
                cv2.putText(padded_rgb, f"{(tick_y-10)//cm_in_px}cm", (5, tick_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # --- MODE 1: MANUAL/MODEL SEMANTIC MASK ---
        if self.rad_view_model.isChecked() and "semantic_patch" in viz:
            orig_patch = viz["semantic_patch"]
            dx, dy = cx_off - x1, cy_off - y1
            h_patch, w_patch = orig_patch.shape
            h_crop, w_crop = crop_rgb.shape[:2]
            
            start_x, start_y = max(0, dx), max(0, dy)
            patch_start_x, patch_start_y = max(0, -dx), max(0, -dy)
            end_x, end_y = min(w_crop, dx + w_patch), min(h_crop, dy + h_patch)
            patch_end_x = patch_start_x + (end_x - start_x)
            patch_end_y = patch_start_y + (end_y - start_y)
            
            if end_x > start_x and end_y > start_y:
                patch_slice = orig_patch[patch_start_y:patch_end_y, patch_start_x:patch_end_x]
                crop_slice = crop_rgb[start_y:end_y, start_x:end_x]
                
                main_mask = (patch_slice == 1)
                lat_mask = (patch_slice == 2)
                
                if np.any(main_mask):
                    crop_slice[main_mask] = (crop_slice[main_mask].astype(np.float32) * 0.4 + np.array([255, 100, 100], dtype=np.float32) * 0.6).astype(np.uint8)
                if np.any(lat_mask):
                    crop_slice[lat_mask] = (crop_slice[lat_mask].astype(np.float32) * 0.4 + np.array([100, 255, 100], dtype=np.float32) * 0.6).astype(np.uint8)
            
            padded_rgb[0:y2-y1, pad_ruler:pad_ruler+x2-x1] = crop_rgb

        # --- MODE 2: GRAPH VALIDATED MASK ---
        elif self.rad_view_graph_mask.isChecked() and "colored_skel_crop" in viz:
            skel_crop = viz["colored_skel_crop"]
            kernel = np.ones((line_w, line_w), np.uint8)
            main_mask = cv2.dilate((np.isin(skel_crop, viz["main_root_colors"])).astype(np.uint8), kernel)
            lat_mask = cv2.dilate((~np.isin(skel_crop, viz["main_root_colors"]) & (skel_crop > 0)).astype(np.uint8), kernel)
            
            for mask, color in [(main_mask, [255, 0, 0]), (lat_mask, [0, 255, 0])]:
                s_ys, s_xs = np.where(mask > 0)
                for sy, sx in zip(s_ys, s_xs):
                    gx, gy = sx + cx_off, sy + cy_off
                    px, py = gx - x1 + pad_ruler, gy - y1
                    if 0 <= px < w_img and 0 <= py < h_img:
                        padded_rgb[py, px] = color

        # --- MODE 3: LIVE RSML OVERLAY ---
        elif self.rad_view_rsml.isChecked() and self.loaded_rsml_roots:
            # Array is pushed to QImage.Format_RGB888, so we use pure RGB tuples!
            order_colors = {
                0: (255, 0, 0),    # Main Root = Red
                1: (0, 255, 0),    # 1st Order (Lateral) = Green
                2: (0, 255, 255),  # 2nd Order (Tertiary) = Cyan
                3: (255, 0, 255)   # 3rd Order (Quaternary) = Magenta
            }
            default_color = (255, 255, 0) # Anything deeper is Yellow
            
            for root_segment in self.loaded_rsml_roots:
                pts = root_segment['points']
                order = root_segment['order']
                col = order_colors.get(order, default_color)
                
                shifted_pts = [self._get_shifted_coord(p, x1, y1, pad_ruler) for p in pts]
                
                for i in range(len(shifted_pts) - 1):
                    p1, p2 = shifted_pts[i], shifted_pts[i+1]
                    cv2.line(padded_rgb, p1, p2, col, line_w, cv2.LINE_AA)

        # --- CONVEX HULL ---
        if self.chk_hull.isChecked() and data.get("hull_pts"):
            pts_arr = np.array([self._get_shifted_coord(p, x1, y1, pad_ruler) for p in data["hull_pts"]], np.int32).reshape((-1, 1, 2))
            cv2.polylines(padded_rgb, [pts_arr], True, (255, 0, 255), 2)

        # --- ANGLES (With Text Geometry) ---
        if not self.rad_ang_none.isChecked() and "lateral_pts_list" in viz:
            dist_px_2mm = max(1, int(0.2 / self.current_cm_px)) if self.current_cm_px else 5
            
            for lat_pts in viz["lateral_pts_list"]:
                if len(lat_pts) < 2: continue
                start = self._get_shifted_coord(lat_pts[0], x1, y1, pad_ruler)
                
                cv2.circle(padded_rgb, start, max(3, line_w), (0, 255, 255), -1)
                
                target_idx = min(dist_px_2mm, len(lat_pts) - 1)
                target_pt = lat_pts[target_idx] if self.rad_ang_emergence.isChecked() else lat_pts[-1]
                end_pt = self._get_shifted_coord(target_pt, x1, y1, pad_ruler)
                
                color = (255, 255, 255) if self.rad_ang_emergence.isChecked() else (0, 255, 255)
                cv2.line(padded_rgb, start, end_pt, color, line_w)

                dx = target_pt[0] - lat_pts[0][0]
                dy = target_pt[1] - lat_pts[0][1]
                hyp = np.hypot(dx, dy)
                ang_deg = np.degrees(np.arccos(np.clip(dy / hyp, -1.0, 1.0))) if hyp > 0 else 0
                
                txt_x = start[0] - 45 if dx < 0 else start[0] + 15
                txt_y = start[1] + 15
                cv2.putText(padded_rgb, f"{ang_deg:.1f}", (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        return padded_rgb

    def export_to_svg(self):
        if not self.current_data:
            QMessageBox.warning(self, "Export Error", "No data available to export.")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save Vector Graphic", f"Plant_{self.current_uid}_Analysis.svg", "SVG Files (*.svg)")
        if not path: return
        
        data = self.current_data
        viz = self.current_viz or {}
        x, y, w, h = self.current_bbox
        pad, pad_ruler = 40, 80
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(self.current_raw.shape[1], x+w+pad), min(self.current_raw.shape[0], y+h+pad)
        
        crop_rgb = self.current_raw[y1:y2, x1:x2].copy()
        padded_rgb = cv2.copyMakeBorder(crop_rgb, 0, 0, pad_ruler, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        h_img, w_img, _ = padded_rgb.shape
        qimg_bg = QImage(padded_rgb.data, w_img, h_img, 3 * w_img, QImage.Format_RGB888)
        
        generator = QSvgGenerator()
        generator.setFileName(path)
        generator.setSize(QSize(w_img, h_img))
        generator.setViewBox(QRect(0, 0, w_img, h_img))
        
        painter = QPainter()
        painter.begin(generator)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawImage(0, 0, qimg_bg)
        
        line_w = self.spin_width.value()

        # LIVE RSML SVG Export
        if self.rad_view_rsml.isChecked() and self.loaded_rsml_roots:
            q_colors = {
                0: QColor(255, 0, 0),
                1: QColor(0, 255, 0),
                2: QColor(0, 255, 255),
                3: QColor(255, 0, 255)
            }
            default_color = QColor(255, 255, 0)
            
            for root_segment in self.loaded_rsml_roots:
                pts = root_segment['points']
                order = root_segment['order']
                qcol = q_colors.get(order, default_color)
                
                painter.setPen(QPen(qcol, line_w))
                shifted_pts = [self._get_shifted_coord(p, x1, y1, pad_ruler) for p in pts]
                
                for i in range(len(shifted_pts) - 1):
                    p1, p2 = shifted_pts[i], shifted_pts[i+1]
                    painter.drawLine(p1[0], p1[1], p2[0], p2[1])

        if self.chk_hull.isChecked() and data.get("hull_pts"):
            painter.setPen(QPen(QColor(255, 0, 255), 2, Qt.DashLine))
            pts = [self._get_shifted_coord(p, x1, y1, pad_ruler) for p in data["hull_pts"]]
            for i in range(len(pts)):
                p1, p2 = pts[i], pts[(i+1) % len(pts)]
                painter.drawLine(p1[0], p1[1], p2[0], p2[1])

        if not self.rad_ang_none.isChecked() and "lateral_pts_list" in viz:
            dist_px_2mm = max(1, int(0.2 / self.current_cm_px)) if self.current_cm_px else 5
            
            for lat_pts in viz["lateral_pts_list"]:
                if len(lat_pts) < 2: continue
                start = self._get_shifted_coord(lat_pts[0], x1, y1, pad_ruler)
                
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, 255, 255))
                painter.drawEllipse(start[0] - max(3, line_w), start[1] - max(3, line_w), max(3, line_w)*2, max(3, line_w)*2)
                
                target_idx = min(dist_px_2mm, len(lat_pts) - 1)
                target_pt = lat_pts[target_idx] if self.rad_ang_emergence.isChecked() else lat_pts[-1]
                end_pt = self._get_shifted_coord(target_pt, x1, y1, pad_ruler)
                
                color = QColor(255, 255, 255) if self.rad_ang_emergence.isChecked() else QColor(0, 255, 255)
                painter.setPen(QPen(color, line_w))
                painter.drawLine(start[0], start[1], end_pt[0], end_pt[1])
                    
        painter.end()
        QMessageBox.information(self, "Export Complete", f"Successfully exported vector graphic to:\n{path}")