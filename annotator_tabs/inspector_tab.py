import cv2
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, 
                             QLabel, QTextEdit, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap, QPainter
from PyQt5.QtCore import Qt
from core.analyzer_engine import emergenceAngle

class AspectRatioLabel(QWidget):
    """Prevents PyQt from locking the layout to massive image sizes."""
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
        self.chk_show_graph = QCheckBox("Show Roots (MR=Red, LR=Green)"); self.chk_show_graph.setChecked(True)
        self.chk_show_initiation = QCheckBox("Highlight LR Initiation Points"); self.chk_show_initiation.setChecked(True)
        self.chk_show_em_angles = QCheckBox("Draw Emergence Angles"); self.chk_show_em_angles.setChecked(True)
        self.chk_show_tip_angles = QCheckBox("Draw Tip Angles")
        
        for chk in [self.chk_show_graph, self.chk_show_initiation, self.chk_show_em_angles, self.chk_show_tip_angles]:
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

    def update_view(self, uid, data, raw_image, bbox, cm_per_px):
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
        report += f"Mean Emerg. Angle: {data['emergence_angle_deg']:.1f}°\n\n"
        report += f"Hull Area: {data['hull_area_cm2']:.2f} cm²\n"
        report += f"Width: {data['width_cm']:.2f} cm\n"
        report += f"Height: {data['height_cm']:.2f} cm\n"
        self.txt_metrics.setText(report)
        
        # 2. Draw Visuals
        x, y, w, h = bbox
        if w == 0: return
        
        pad = 40
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(raw_image.shape[1], x+w+pad), min(raw_image.shape[0], y+h+pad)
        
        crop_rgb = raw_image[y1:y2, x1:x2].copy()
        
        # --- ADD PADDED RULER TO THE LEFT ---
        pad_ruler = 80
        # Add black space to the left of the image
        padded_rgb = cv2.copyMakeBorder(crop_rgb, 0, 0, pad_ruler, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        h_img, w_img, _ = padded_rgb.shape
        
        cm_in_px = int(1.0 / cm_per_px)
        if cm_in_px > 10:
            cv2.line(padded_rgb, (pad_ruler - 15, 10), (pad_ruler - 15, h_img - 10), (255, 255, 255), 2)
            for tick_y in range(10, h_img - 10, cm_in_px):
                cv2.line(padded_rgb, (pad_ruler - 25, tick_y), (pad_ruler - 15, tick_y), (255, 255, 255), 2)
                cv2.putText(padded_rgb, f"{(tick_y-10)//cm_in_px}cm", (5, tick_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Helper to shift X coordinates by both the crop offset AND the new ruler padding
        def shift(pt): return (int(pt[0]) - x1 + pad_ruler, int(pt[1]) - y1)

        # Draw Graph Overlay
        if self.chk_show_graph.isChecked():
            pts_main = data["main_pts"]
            if len(pts_main) > 1:
                pts_arr = np.array([shift(p) for p in pts_main], np.int32).reshape((-1, 1, 2))
                cv2.polylines(padded_rgb, [pts_arr], False, (255, 0, 0), 2)
            
            for lat_pts in data["lateral_pts_list"]:
                if len(lat_pts) < 2: continue
                pts_arr = np.array([shift(p) for p in lat_pts], np.int32).reshape((-1, 1, 2))
                cv2.polylines(padded_rgb, [pts_arr], False, (0, 255, 0), 2)

        for lat_pts in data["lateral_pts_list"]:
            if len(lat_pts) < 2: continue
            start = shift(lat_pts[0])
            
            if self.chk_show_initiation.isChecked():
                cv2.circle(padded_rgb, start, 4, (0, 255, 255), -1)
                
            if self.chk_show_em_angles.isChecked():
                cv2.line(padded_rgb, start, (start[0], start[1] + 30), (200, 200, 200), 1)
                em_end = lat_pts[min(cm_in_px, len(lat_pts)) - 1]
                cv2.line(padded_rgb, start, shift(em_end), (255, 255, 255), 2)
                
            if self.chk_show_tip_angles.isChecked():
                if not self.chk_show_em_angles.isChecked():
                    cv2.line(padded_rgb, start, (start[0], start[1] + 30), (200, 200, 200), 1)
                cv2.line(padded_rgb, start, shift(lat_pts[-1]), (0, 255, 255), 2)

        qimg = QImage(padded_rgb.data, w_img, h_img, 3 * w_img, QImage.Format_RGB888)
        self.lbl_inspector_img.setPixmap(QPixmap.fromImage(qimg))