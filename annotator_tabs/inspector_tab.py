import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, 
                             QLabel, QTextEdit, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import Qt

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
        self.init_ui()

    def init_ui(self):
        insp_layout = QHBoxLayout(self)
        
        # Left side: Image and Controls
        controls_layout = QVBoxLayout()
        
        # EXACT TOGGLES REQUESTED
        self.chk_seg = QCheckBox("Graph-Colored Segmentation (Pixel-Perfect)"); self.chk_seg.setChecked(True)
        self.chk_graph = QCheckBox("Hard Graph (Straight Lines)"); self.chk_graph.setChecked(False)
        self.chk_hull = QCheckBox("Convex Hull (Magenta)"); self.chk_hull.setChecked(True)
        self.chk_angles = QCheckBox("Lateral Root Angles (Init, 2mm Emg, Tip)"); self.chk_angles.setChecked(True)
        
        for chk in [self.chk_seg, self.chk_graph, self.chk_hull, self.chk_angles]:
            chk.stateChanged.connect(lambda: self._trigger_redraw())
            controls_layout.addWidget(chk)
            
        self.lbl_inspector_img = AspectRatioLabel()
        self.lbl_inspector_img.setStyleSheet("background-color: #1e1e1e;")
        controls_layout.addWidget(self.lbl_inspector_img, stretch=1)
        insp_layout.addLayout(controls_layout, stretch=2)
        
        # Right side: Metrics Readout
        self.txt_metrics = QTextEdit()
        self.txt_metrics.setReadOnly(True)
        self.txt_metrics.setStyleSheet("font-family: monospace; font-size: 14px;")
        insp_layout.addWidget(self.txt_metrics, stretch=1)
        
        self.current_uid = None
        self.current_data = None
        self.current_raw = None
        self.current_bbox = None
        self.current_cm_px = 1.0

    def _trigger_redraw(self):
        self.update_view(self.current_uid, self.current_data, self.current_raw, self.current_bbox, self.current_cm_px)

    def update_view(self, uid, data, raw_image, bbox, cm_per_px):
        self.current_uid = uid
        self.current_data = data
        self.current_raw = raw_image
        self.current_bbox = bbox
        self.current_cm_px = cm_per_px
        
        if not uid or not data:
            self.lbl_inspector_img.clear()
            self.txt_metrics.clear()
            return
            
        # 1. Update Text Readout
        report = f"--- AUTOMATIC ID: {uid} ---\n\n"
        report += f"Genotype: {data['genotype']}\n"
        report += f"Plant #: {data['plant_num']}\n\n"
        report += f"Main Root: {data['mr_length_cm']:.2f} cm\n"
        report += f"Lateral Roots: {data['lr_length_cm']:.2f} cm\n"
        report += f"LR Count: {data['lr_count']}\n\n"
        report += f"Mean Tip Angle: {data['tip_angle_deg']:.1f}°\n"
        report += f"Mean Emerg. Angle (2mm): {data['emergence_angle_deg']:.1f}°\n\n"
        report += f"Hull Area: {data['hull_area_cm2']:.2f} cm²\n"
        report += f"Width: {data['width_cm']:.2f} cm\n"
        report += f"Height: {data['height_cm']:.2f} cm\n"
        self.txt_metrics.setText(report)
        
        # 2. Base Setup
        x, y, w, h = bbox
        if w == 0: return
        
        pad = 40
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(raw_image.shape[1], x+w+pad), min(raw_image.shape[0], y+h+pad)
        
        crop_rgb = raw_image[y1:y2, x1:x2].copy()
        
        # --- PADDED RULER ---
        pad_ruler = 80
        padded_rgb = cv2.copyMakeBorder(crop_rgb, 0, 0, pad_ruler, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        h_img, w_img, _ = padded_rgb.shape
        
        cm_in_px = int(1.0 / cm_per_px)
        if cm_in_px > 10:
            cv2.line(padded_rgb, (pad_ruler - 15, 10), (pad_ruler - 15, h_img - 10), (255, 255, 255), 2)
            for tick_y in range(10, h_img - 10, cm_in_px):
                cv2.line(padded_rgb, (pad_ruler - 25, tick_y), (pad_ruler - 15, tick_y), (255, 255, 255), 2)
                cv2.putText(padded_rgb, f"{(tick_y-10)//cm_in_px}cm", (5, tick_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        def shift(pt): return (int(pt[0]) - x1 + pad_ruler, int(pt[1]) - y1)

        # --- A. GRAPH COLORED SEGMENTATION ---
        if self.chk_seg.isChecked():
            skel_crop = data["colored_skel_crop"]
            cx_off, cy_off = data["crop_offset"]
            s_ys, s_xs = np.where(skel_crop > 0)
            
            for sy, sx in zip(s_ys, s_xs):
                val = skel_crop[sy, sx]
                # Map back to global coordinates
                gx, gy = sx + cx_off, sy + cy_off
                # Map to the padded view
                px, py = gx - x1 + pad_ruler, gy - y1
                
                if 0 <= px < padded_rgb.shape[1] and 0 <= py < padded_rgb.shape[0]:
                    if val in data["main_root_colors"]: padded_rgb[py, px] = [255, 0, 0] # Main
                    else: padded_rgb[py, px] = [0, 255, 0] # Lateral

        # --- B. HARD GRAPH (Straight Lines) ---
        if self.chk_graph.isChecked():
            for u, v, r_type in data["graph_edges"]:
                col = (255, 0, 0) if r_type == 1 else (0, 255, 0)
                cv2.line(padded_rgb, shift(u), shift(v), col, 2)

        # --- C. CONVEX HULL ---
        if self.chk_hull.isChecked():
            hull_pts = data["hull_pts"]
            if hull_pts:
                pts_arr = np.array([shift(p) for p in hull_pts], np.int32).reshape((-1, 1, 2))
                cv2.polylines(padded_rgb, [pts_arr], True, (255, 0, 255), 2)

        # --- D. ANGLES (Init, 2mm, Tip) ---
        if self.chk_angles.isChecked():
            dist_px_2mm = int(0.2 / cm_per_px)
            
            for lat_pts in data["lateral_pts_list"]:
                if len(lat_pts) < 2: continue
                start = shift(lat_pts[0])
                
                # Initiation Point (Yellow)
                cv2.circle(padded_rgb, start, 4, (0, 255, 255), -1)
                
                # Vertical Ref
                cv2.line(padded_rgb, start, (start[0], start[1] + 30), (200, 200, 200), 1)
                
                # Emergence Vector @ 2mm (White)
                em_end = lat_pts[min(dist_px_2mm, len(lat_pts)) - 1]
                cv2.line(padded_rgb, start, shift(em_end), (255, 255, 255), 2)
                
                # Tip Vector (Cyan)
                cv2.line(padded_rgb, start, shift(lat_pts[-1]), (0, 255, 255), 2)

        qimg = QImage(padded_rgb.data, w_img, h_img, 3 * w_img, QImage.Format_RGB888)
        self.lbl_inspector_img.setPixmap(QPixmap.fromImage(qimg))