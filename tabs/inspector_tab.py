import cv2
import numpy as np
import os
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, 
                             QLabel, QTextEdit, QSizePolicy, QPushButton, 
                             QGroupBox, QRadioButton, QButtonGroup, QFileDialog, QMessageBox, QSpinBox, QFormLayout)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from PyQt5.QtCore import Qt, QSize, QRect
from PyQt5.QtSvg import QSvgGenerator

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
    def __init__(self):
        super().__init__()
        self.current_uid = None
        self.current_data = None
        self.current_raw = None
        self.current_bbox = None
        self.current_cm_px = 1.0
        self.init_ui()

    def init_ui(self):
        insp_layout = QHBoxLayout(self)
        
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
        
        # --- Group 1: Core Visualization (Mutually Exclusive) ---
        group_view = QGroupBox("1. Root Representation")
        vbox_view = QVBoxLayout()
        
        self.rad_view_model = QRadioButton("Manual/Model Semantic Mask")
        self.rad_view_graph_mask = QRadioButton("Graph-Validated Mask (Pixels)")
        self.rad_view_hard_graph = QRadioButton("Hard Topological Graph (Lines)")
        self.rad_view_none = QRadioButton("Raw Image Only")
        self.rad_view_model.setChecked(True)

        self.view_group = QButtonGroup()
        for rad in [self.rad_view_model, self.rad_view_graph_mask, self.rad_view_hard_graph, self.rad_view_none]:
            self.view_group.addButton(rad)
            rad.toggled.connect(self._trigger_redraw)
            vbox_view.addWidget(rad)
            
        group_view.setLayout(vbox_view)
        right_layout.addWidget(group_view)

        # --- Group 2: Features & Angles (Mutually Exclusive) ---
        group_angles = QGroupBox("2. Biological Features")
        vbox_ang = QVBoxLayout()
        
        self.rad_ang_none = QRadioButton("Hide Angles")
        self.rad_ang_emergence = QRadioButton("Emergence Angles (2mm)")
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
        group_settings = QGroupBox("3. Render Settings")
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
        self.btn_export_svg.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 10px;")
        self.btn_export_svg.clicked.connect(self.export_to_svg)
        right_layout.addWidget(self.btn_export_svg)

        insp_layout.addLayout(right_layout, stretch=1)

    def _trigger_redraw(self):
        self.update_view(self.current_uid, self.current_data, self.current_raw, self.current_bbox, self.current_cm_px)

    def _get_shifted_coord(self, pt, x1, y1, pad_ruler):
        return (int(pt[0]) - x1 + pad_ruler, int(pt[1]) - y1)
    
    def update_view(self, uid, data, raw_image, bbox, cm_per_px):
        self.current_uid, self.current_data, self.current_raw = uid, data, raw_image
        self.current_bbox, self.current_cm_px = bbox, cm_per_px
        
        if not uid or not data:
            self.lbl_inspector_img.clear()
            self.txt_metrics.clear()
            return
            
        # ==========================================
        # MANUSCRIPT METRICS REPORT READOUT
        # ==========================================
        report = f"--- CHRONOROOT ID: {uid} ---\n"
        report += f"Genotype: {data.get('genotype', 'N/A')}\n"
        report += f"Plant #:  {data.get('plant_num', 'N/A')}\n\n"
        
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
        
        # ==========================================
        # RENDER IMAGE VIEWER
        # ==========================================
        x, y, w, h = bbox
        if w == 0: return
        
        pad, pad_ruler = 40, 80
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(raw_image.shape[1], x+w+pad), min(raw_image.shape[0], y+h+pad)
        
        padded_rgb = self._generate_raster_canvas(data, raw_image, x1, y1, x2, y2, pad_ruler)
        
        h_img, w_img, _ = padded_rgb.shape
        qimg = QImage(padded_rgb.data, w_img, h_img, 3 * w_img, QImage.Format_RGB888)
        self.lbl_inspector_img.setPixmap(QPixmap.fromImage(qimg))

    def _generate_raster_canvas(self, data, raw_image, x1, y1, x2, y2, pad_ruler):
        crop_rgb = raw_image[y1:y2, x1:x2].copy()
        padded_rgb = cv2.copyMakeBorder(crop_rgb, 0, 0, pad_ruler, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        h_img, w_img, _ = padded_rgb.shape
        cx_off, cy_off = data["crop_offset"]
        line_w = self.spin_width.value()
        
        # --- RULER ---
        cm_in_px = int(1.0 / self.current_cm_px)
        if cm_in_px > 10:
            cv2.line(padded_rgb, (pad_ruler - 15, 10), (pad_ruler - 15, h_img - 10), (200, 200, 200), 2)
            for tick_y in range(10, h_img - 10, cm_in_px):
                cv2.line(padded_rgb, (pad_ruler - 25, tick_y), (pad_ruler - 15, tick_y), (200, 200, 200), 2)
                cv2.putText(padded_rgb, f"{(tick_y-10)//cm_in_px}cm", (5, tick_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # --- MODE 1: MANUAL/MODEL SEMANTIC MASK ---
        if self.rad_view_model.isChecked() and "semantic_patch" in data:
            orig_patch = data["semantic_patch"]
            
            # 1. Calculate the relative offset of the patch INSIDE the UI crop
            # This accounts for the 40px padding you see in the viewer
            dx = cx_off - x1
            dy = cy_off - y1
            
            h_patch, w_patch = orig_patch.shape
            h_crop, w_crop = crop_rgb.shape[:2]
            
            # 2. Determine safe intersection boundaries to prevent NumPy crashes
            start_x = max(0, dx)
            start_y = max(0, dy)
            patch_start_x = max(0, -dx)
            patch_start_y = max(0, -dy)
            
            end_x = min(w_crop, dx + w_patch)
            end_y = min(h_crop, dy + h_patch)
            patch_end_x = patch_start_x + (end_x - start_x)
            patch_end_y = patch_start_y + (end_y - start_y)
            
            # 3. Apply blending only to the mathematically aligned region
            if end_x > start_x and end_y > start_y:
                patch_slice = orig_patch[patch_start_y:patch_end_y, patch_start_x:patch_end_x]
                crop_slice = crop_rgb[start_y:end_y, start_x:end_x]
                
                main_mask = (patch_slice == 1)
                lat_mask = (patch_slice == 2)
                
                # Pure NumPy Alpha Blending (Original_Pixel * 0.4 + Overlay * 0.6)
                if np.any(main_mask):
                    crop_slice[main_mask] = (
                        crop_slice[main_mask].astype(np.float32) * 0.4 + 
                        np.array([255, 100, 100], dtype=np.float32) * 0.6
                    ).astype(np.uint8)
                    
                if np.any(lat_mask):
                    crop_slice[lat_mask] = (
                        crop_slice[lat_mask].astype(np.float32) * 0.4 + 
                        np.array([100, 255, 100], dtype=np.float32) * 0.6
                    ).astype(np.uint8)
            
            # 4. Put the fully modified crop back onto the padded ruler canvas
            padded_rgb[0:y2-y1, pad_ruler:pad_ruler+x2-x1] = crop_rgb

        # --- MODE 2: GRAPH VALIDATED MASK ---
        elif self.rad_view_graph_mask.isChecked():
            skel_crop = data["colored_skel_crop"]
            kernel = np.ones((line_w, line_w), np.uint8)
            main_mask = cv2.dilate((np.isin(skel_crop, data["main_root_colors"])).astype(np.uint8), kernel)
            lat_mask = cv2.dilate((~np.isin(skel_crop, data["main_root_colors"]) & (skel_crop > 0)).astype(np.uint8), kernel)
            
            for mask, color in [(main_mask, [255, 0, 0]), (lat_mask, [0, 255, 0])]:
                s_ys, s_xs = np.where(mask > 0)
                for sy, sx in zip(s_ys, s_xs):
                    gx, gy = sx + cx_off, sy + cy_off
                    px, py = gx - x1 + pad_ruler, gy - y1
                    if 0 <= px < w_img and 0 <= py < h_img:
                        padded_rgb[py, px] = color

        # --- MODE 3: HARD TOPOLOGICAL GRAPH ---
        elif self.rad_view_hard_graph.isChecked():
            for u, v, r_type in data["graph_edges"]:
                col = (255, 0, 0) if r_type == 1 else (0, 255, 0)
                cv2.line(padded_rgb, self._get_shifted_coord(u, x1, y1, pad_ruler), self._get_shifted_coord(v, x1, y1, pad_ruler), col, line_w)

        # --- CONVEX HULL ---
        if self.chk_hull.isChecked() and data["hull_pts"]:
            pts_arr = np.array([self._get_shifted_coord(p, x1, y1, pad_ruler) for p in data["hull_pts"]], np.int32).reshape((-1, 1, 2))
            cv2.polylines(padded_rgb, [pts_arr], True, (255, 0, 255), 2)

        # --- ANGLES (With Text Geometry) ---
        if not self.rad_ang_none.isChecked():
            dist_px_2mm = int(0.2 / self.current_cm_px)
            for lat_pts in data["lateral_pts_list"]:
                if len(lat_pts) < 2: continue
                start = self._get_shifted_coord(lat_pts[0], x1, y1, pad_ruler)
                
                cv2.circle(padded_rgb, start, max(3, line_w), (0, 255, 255), -1)
                
                target_pt = lat_pts[min(dist_px_2mm, len(lat_pts)) - 1] if self.rad_ang_emergence.isChecked() else lat_pts[-1]
                end_pt = self._get_shifted_coord(target_pt, x1, y1, pad_ruler)
                
                # Draw Line
                color = (255, 255, 255) if self.rad_ang_emergence.isChecked() else (0, 255, 255)
                cv2.line(padded_rgb, start, end_pt, color, line_w)

                # Calculate Angle for Label
                dx = target_pt[0] - lat_pts[0][0]
                dy = target_pt[1] - lat_pts[0][1]
                hyp = np.hypot(dx, dy)
                ang_deg = np.degrees(np.arccos(np.clip(dy / hyp, -1.0, 1.0))) if hyp > 0 else 0
                
                # Text Placement Logic
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

        if self.rad_view_hard_graph.isChecked():
            for u, v, r_type in data["graph_edges"]:
                color = QColor(255, 0, 0) if r_type == 1 else QColor(0, 255, 0)
                painter.setPen(QPen(color, line_w))
                p1, p2 = self._get_shifted_coord(u, x1, y1, pad_ruler), self._get_shifted_coord(v, x1, y1, pad_ruler)
                painter.drawLine(p1[0], p1[1], p2[0], p2[1])

        if self.chk_hull.isChecked() and data["hull_pts"]:
            painter.setPen(QPen(QColor(255, 0, 255), 2, Qt.DashLine))
            pts = [self._get_shifted_coord(p, x1, y1, pad_ruler) for p in data["hull_pts"]]
            for i in range(len(pts)):
                p1, p2 = pts[i], pts[(i+1) % len(pts)]
                painter.drawLine(p1[0], p1[1], p2[0], p2[1])

        if not self.rad_ang_none.isChecked():
            dist_px_2mm = int(0.2 / self.current_cm_px)
            for lat_pts in data["lateral_pts_list"]:
                if len(lat_pts) < 2: continue
                start = self._get_shifted_coord(lat_pts[0], x1, y1, pad_ruler)
                
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, 255, 255))
                painter.drawEllipse(start[0] - max(3, line_w), start[1] - max(3, line_w), max(3, line_w)*2, max(3, line_w)*2)
                
                target_pt = lat_pts[min(dist_px_2mm, len(lat_pts)) - 1] if self.rad_ang_emergence.isChecked() else lat_pts[-1]
                end_pt = self._get_shifted_coord(target_pt, x1, y1, pad_ruler)
                
                color = QColor(255, 255, 255) if self.rad_ang_emergence.isChecked() else QColor(0, 255, 255)
                painter.setPen(QPen(color, line_w))
                painter.drawLine(start[0], start[1], end_pt[0], end_pt[1])
                    
        painter.end()
        QMessageBox.information(self, "Export Complete", f"Successfully exported vector graphic to:\n{path}")