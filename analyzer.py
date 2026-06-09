import sys
import os
import json
import cv2
import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QSplitter, QLabel, QLineEdit, 
                             QPushButton, QFormLayout, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QMessageBox, 
                             QFileDialog, QTabWidget, QListWidget, QListWidgetItem,
                             QStyle, QProgressDialog, QTextEdit, QCheckBox, QSizePolicy, QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPixmap, QPainter

from core.model import PlantImageModel
from core.analyzer_engine import extract_plate_metrics, export_rsml_and_json, tipAngle, emergenceAngle
from annotator_tabs.review_tab import ReviewCanvasTab

APP_NAME = "chronorootAnalyzer"
GLOBAL_CONFIG_DIR = os.path.expanduser(f"~/.config/{APP_NAME}")
GLOBAL_CONFIG_FILE = os.path.join(GLOBAL_CONFIG_DIR, "analyzerConfig.json")
os.makedirs(GLOBAL_CONFIG_DIR, exist_ok=True)

if not os.path.exists(GLOBAL_CONFIG_FILE):
    default_config = {"input_root": ".", "output_root": ".", "calib_mode": "Scanner DPI", "calib_val": "600"}
    with open(GLOBAL_CONFIG_FILE, 'w') as f: json.dump(default_config, f, indent=4)
    GLOBAL_CONFIG = default_config
else:
    with open(GLOBAL_CONFIG_FILE, 'r') as f: GLOBAL_CONFIG = json.load(f)


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

class AnalyzerFileStatsWidget(QWidget):
    def __init__(self, name, annotator_status, analyzer_status, plant_count):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5); layout.setSpacing(2)
        top_row = QHBoxLayout()
        icon_label = QLabel()
        
        if analyzer_status == "Analyzed": icon = QApplication.style().standardIcon(QStyle.SP_DialogApplyButton)
        elif analyzer_status == "Ready": icon = QApplication.style().standardIcon(QStyle.SP_FileIcon)
        else: icon = QApplication.style().standardIcon(QStyle.SP_MessageBoxWarning)
            
        icon_label.setPixmap(icon.pixmap(16, 16))
        top_row.addWidget(icon_label); top_row.addWidget(QLabel(name)); top_row.addStretch()
        layout.addLayout(top_row)
        
        lbl_stats = QLabel(f"Status: {analyzer_status} | Nº Plants: {plant_count}")
        if analyzer_status == "Analyzed": lbl_stats.setStyleSheet("color: #28a745; font-size: 10px; font-weight: bold;")
        elif analyzer_status == "Ready": lbl_stats.setStyleSheet("color: #007bff; font-size: 10px; font-weight: bold;")
        else: lbl_stats.setStyleSheet("color: #dc3545; font-size: 10px;")
        layout.addWidget(lbl_stats)

class AnalyzerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChronoRoot Analyzer - Phenomics Extractor")
        self.resize(1600, 900)
        
        self.model = PlantImageModel()
        self.active_workers = set()
        self.measurements_cache = {} 
        self.current_cm_per_px = 1.0 # Stored globally for UI redrawing
        
        self.in_dir = GLOBAL_CONFIG.get("input_root", ".")
        self.out_dir = GLOBAL_CONFIG.get("output_root", ".")
        self.current_dir = self.in_dir
        self.current_base_name = ""
        self.current_task_path = ""
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        
        # 1. LEFT PANE
        lp = QWidget()
        lp_layout = QVBoxLayout(lp)
        lp_layout.addWidget(QLabel("<b>Input Directory:</b>"))
        path_layout = QHBoxLayout()
        self.lbl_path = QLineEdit(self.in_dir)
        self.btn_browse_in = QPushButton("..."); self.btn_browse_in.setFixedWidth(30)
        self.btn_browse_in.clicked.connect(self.change_input_dir)
        path_layout.addWidget(self.lbl_path); path_layout.addWidget(self.btn_browse_in)
        lp_layout.addLayout(path_layout)
        
        lp_layout.addWidget(QLabel("<b>Output Directory:</b>"))
        out_layout = QHBoxLayout()
        self.lbl_out_path = QLineEdit(self.out_dir)
        self.btn_browse_out = QPushButton("..."); self.btn_browse_out.setFixedWidth(30)
        self.btn_browse_out.clicked.connect(self.change_output_dir)
        out_layout.addWidget(self.lbl_out_path); out_layout.addWidget(self.btn_browse_out)
        lp_layout.addLayout(out_layout)
        
        nav_layout = QHBoxLayout()
        self.btn_up = QPushButton("⬆ Up Level"); self.btn_up.clicked.connect(self.navigate_up)
        self.btn_refresh = QPushButton("Refresh"); self.btn_refresh.clicked.connect(self.populate_browser)
        nav_layout.addWidget(self.btn_up); nav_layout.addWidget(self.btn_refresh)
        lp_layout.addLayout(nav_layout)
        
        self.task_list = QListWidget()
        self.task_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.task_list.itemClicked.connect(self.on_item_clicked)
        lp_layout.addWidget(self.task_list)
        splitter.addWidget(lp)
        
        # 2. MIDDLE PANE
        mp = QWidget()
        mp_layout = QVBoxLayout(mp)
        
        mp_layout.addWidget(QLabel("<b>Plate Metadata & Calibration:</b>"))
        form = QFormLayout()
        self.in_plate_id = QLineEdit("Plate_01")
        self.in_condition = QLineEdit("Control")
        self.in_timepoint = QLineEdit("Day_07")
        
        # --- NEW: Robust Calibration Inputs ---
        self.cb_calibration = QComboBox()
        self.cb_calibration.addItems(["Scanner DPI", "Known Image Height (cm)", "Known Image Width (cm)"])
        self.cb_calibration.setCurrentText(GLOBAL_CONFIG.get("calib_mode", "Scanner DPI"))
        
        self.in_calib_val = QLineEdit(GLOBAL_CONFIG.get("calib_val", "600"))
        
        form.addRow("Plate ID:", self.in_plate_id)
        form.addRow("Condition:", self.in_condition)
        form.addRow("Timepoint (Stage):", self.in_timepoint)
        form.addRow("Calibration Method:", self.cb_calibration)
        form.addRow("Calibration Value:", self.in_calib_val)
        mp_layout.addLayout(form)
        
        mp_layout.addSpacing(10)
        mp_layout.addWidget(QLabel("<b>Plant Identification:</b>"))
        
        self.table_plants = QTableWidget(0, 3)
        self.table_plants.setHorizontalHeaderLabels(["Automatic ID", "Genotype", "Plant #"])
        self.table_plants.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_plants.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_plants.itemSelectionChanged.connect(self.on_table_selection)
        mp_layout.addWidget(self.table_plants)
        
        btn_layout = QHBoxLayout()
        self.btn_measure = QPushButton("1. Measure & Inspect")
        self.btn_measure.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 10px;")
        self.btn_measure.clicked.connect(self.run_measurements)
        
        self.btn_export = QPushButton("2. Export JSON & RSML")
        self.btn_export.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 10px;")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.run_export)
        
        btn_layout.addWidget(self.btn_measure); btn_layout.addWidget(self.btn_export)
        mp_layout.addLayout(btn_layout)
        splitter.addWidget(mp)
        
        # 3. RIGHT PANE
        rp = QWidget()
        rp_layout = QVBoxLayout(rp)
        
        self.tabs = QTabWidget()
        self.canvas_review = ReviewCanvasTab(self.model)
        self.canvas_review.set_mode("SELECT")
        self.tabs.addTab(self.canvas_review, "Plate Overview")
        
        self.inspector_widget = QWidget()
        insp_layout = QHBoxLayout(self.inspector_widget)
        
        # --- NEW: Checkbox controls for the Inspector ---
        controls_layout = QVBoxLayout()
        self.chk_show_graph = QCheckBox("Show Roots (MR=Red, LR=Green)"); self.chk_show_graph.setChecked(True)
        self.chk_show_initiation = QCheckBox("Highlight LR Initiation Points"); self.chk_show_initiation.setChecked(True)
        self.chk_show_em_angles = QCheckBox("Draw Emergence Angles"); self.chk_show_em_angles.setChecked(True)
        self.chk_show_tip_angles = QCheckBox("Draw Tip Angles")
        
        for chk in [self.chk_show_graph, self.chk_show_initiation, self.chk_show_em_angles, self.chk_show_tip_angles]:
            chk.stateChanged.connect(self.update_inspector_view)
            controls_layout.addWidget(chk)
            
        self.lbl_inspector_img = AspectRatioLabel()
        controls_layout.addWidget(self.lbl_inspector_img, stretch=1)
        insp_layout.addLayout(controls_layout, stretch=2)
        
        self.txt_metrics = QTextEdit()
        self.txt_metrics.setReadOnly(True)
        self.txt_metrics.setStyleSheet("font-family: monospace; font-size: 14px;")
        insp_layout.addWidget(self.txt_metrics, stretch=1)
        
        self.tabs.addTab(self.inspector_widget, "Phenomics Inspector")
        rp_layout.addWidget(self.tabs)
        
        splitter.addWidget(rp)
        splitter.setSizes([300, 400, 900])

        self.model.register_data_callback(self.populate_plant_table)
        self.model.register_selection_callback(self.sync_canvas_to_table)
        self.populate_browser()

    # --- FILE BROWSER LOGIC ---
    def change_input_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Input Folder", self.in_dir)
        if d: self.in_dir = self.current_dir = d; self.lbl_path.setText(d); self.populate_browser()

    def change_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.out_dir)
        if d: self.out_dir = d; self.lbl_out_path.setText(d); self.populate_browser()

    def navigate_up(self):
        parent = os.path.dirname(os.path.abspath(self.current_dir))
        self.current_dir = parent
        self.populate_browser()

    def analyzer_scan_routine(self, folder_path, out_directory):
        contents = self.model.scan_directory(folder_path)
        for item in contents:
            if item["type"] == "file":
                if item["status"] != "Completed": item["analyzer_status"] = "Not Annotated"
                else:
                    metrics_path = os.path.join(out_directory, f"{item['name']}_Metrics.json")
                    item["analyzer_status"] = "Analyzed" if os.path.exists(metrics_path) else "Ready"
        return contents

    def populate_browser(self):
        self.lbl_path.setText(self.current_dir)
        self.task_list.clearSelection(); self.task_list.clear()
        self.show_loading(f"Scanning directory for analysis...\n{self.current_dir}")
        worker = ModelWorker(self.analyzer_scan_routine, self.current_dir, self.out_dir)
        self.active_workers.add(worker)
        worker.finished.connect(self._on_scan_finished)
        worker.error.connect(self._on_thread_error)
        worker.start()

    def _on_scan_finished(self, contents):
        self.hide_loading()
        worker = self.sender()
        if worker in self.active_workers:
            self.active_workers.remove(worker)
            worker.deleteLater()
            
        for item in contents:
            if item["type"] == "dir":
                list_item = QListWidgetItem(f"📁 {item['name']}")
                list_item.setData(Qt.UserRole, {"type": "dir", "path": item["path"]})
                self.task_list.addItem(list_item)
            elif item["type"] == "file":
                list_item = QListWidgetItem()
                list_item.setData(Qt.UserRole, {"type": "file", "path": item["path"], "status": item["analyzer_status"]})
                widget = AnalyzerFileStatsWidget(item["name"], item["status"], item["analyzer_status"], item.get("plant_count", 0))
                list_item.setSizeHint(widget.sizeHint())
                self.task_list.addItem(list_item)
                self.task_list.setItemWidget(list_item, widget)

    def on_item_double_clicked(self, item):
        data = item.data(Qt.UserRole)
        if data["type"] == "dir":
            self.current_dir = data["path"]
            self.populate_browser()

    def on_item_clicked(self, item):
        data = item.data(Qt.UserRole)
        if data["type"] != "file": return
        
        if data["status"] == "Not Annotated":
            QMessageBox.warning(self, "Skip", "This file has not been completed in the Annotation Suite yet.")
            return

        file_path = data["path"]
        self.current_task_path = os.path.dirname(file_path)
        self.current_base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        self.model.callbacks_muted = True 
        self.model.set_selection([])
        self.show_loading(f"Loading {self.current_base_name}...")
        
        worker = ModelWorker(self.model.load_task, self.current_task_path, self.current_base_name)
        self.active_workers.add(worker)
        worker.finished.connect(self._on_load_finished)
        worker.error.connect(self._on_thread_error)
        worker.start()
        
    def _on_load_finished(self, _):
        self.hide_loading()
        worker = self.sender()
        if worker in self.active_workers:
            self.active_workers.remove(worker)
            worker.deleteLater()
            
        self.model.callbacks_muted = False
        self.model._notify_data_changed()
        self.canvas_review.refresh_canvas()

    def populate_plant_table(self):
        self.table_plants.blockSignals(True)
        self.table_plants.setRowCount(0)
        self.measurements_cache.clear()
        self.btn_export.setEnabled(False)
        self.txt_metrics.clear()
        self.lbl_inspector_img.clear()
        
        for uid in sorted(self.model.masks.keys()):
            row = self.table_plants.rowCount()
            self.table_plants.insertRow(row)
            
            item_uid = QTableWidgetItem(f"{uid}")
            item_uid.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            item_uid.setData(Qt.UserRole, uid)
            
            self.table_plants.setItem(row, 0, item_uid)
            self.table_plants.setItem(row, 1, QTableWidgetItem("Col-0"))
            self.table_plants.setItem(row, 2, QTableWidgetItem(str(row + 1)))
            
        self.table_plants.blockSignals(False)

    def on_table_selection(self):
        selected_items = self.table_plants.selectedItems()
        if not selected_items: return
        uid = selected_items[0].data(Qt.UserRole)
        
        self.model.callbacks_muted = True 
        self.model.set_selection([uid])
        self.model.callbacks_muted = False
        
        self.canvas_review.refresh_canvas()
        self.canvas_review.zoom_to_plant(uid)
        self.update_inspector_view()

    def sync_canvas_to_table(self):
        uid = self.model.active_uid
        if not uid: return
        self.table_plants.blockSignals(True)
        for row in range(self.table_plants.rowCount()):
            if self.table_plants.item(row, 0).data(Qt.UserRole) == uid:
                self.table_plants.selectRow(row)
                break
        self.table_plants.blockSignals(False)
        self.update_inspector_view()

    # --- EXTRACTION ENGINE ---
    def get_cm_per_px(self):
        calib_mode = self.cb_calibration.currentText()
        try:
            val = float(self.in_calib_val.text())
            if val <= 0: raise ValueError
        except ValueError:
            return None
            
        if "DPI" in calib_mode:
            return 2.54 / val
        else:
            if self.model.raw_image is None: return None
            h_px, w_px = self.model.raw_image.shape[:2]
            if "Height" in calib_mode: return val / h_px
            elif "Width" in calib_mode: return val / w_px
        return None

    def run_measurements(self):
        if not self.model.masks: return
        
        cm_per_px = self.get_cm_per_px()
        if cm_per_px is None:
            QMessageBox.warning(self, "Calibration Error", "Invalid Calibration Value.")
            return
            
        self.current_cm_per_px = cm_per_px

        plants_meta = []
        for row in range(self.table_plants.rowCount()):
            plants_meta.append({
                "uid": self.table_plants.item(row, 0).data(Qt.UserRole),
                "genotype": self.table_plants.item(row, 1).text(),
                "plant_num": self.table_plants.item(row, 2).text()
            })

        self.show_loading("Measuring Morphometrics & Tracing RSML...")
        worker = ModelWorker(extract_plate_metrics, self.model, plants_meta, cm_per_px)
        self.active_workers.add(worker)
        worker.finished.connect(self._on_measure_finished)
        worker.error.connect(self._on_thread_error)
        worker.start()

    def _on_measure_finished(self, results_dict):
        self.hide_loading()
        worker = self.sender()
        if worker in self.active_workers:
            self.active_workers.remove(worker)
            worker.deleteLater()
            
        self.measurements_cache = results_dict
        self.btn_export.setEnabled(True)
        
        self.tabs.setCurrentIndex(1)
        self.update_inspector_view()

    def update_inspector_view(self):
        uid = self.model.active_uid
        if not uid or uid not in self.measurements_cache:
            if uid: self.lbl_inspector_img.setText("Press 'Measure & Inspect' to view data.")
            return
            
        data = self.measurements_cache[uid]
        cm_per_px = self.current_cm_per_px
        
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
        x, y, w, h = self.model.bboxes.get(uid, (0,0,0,0))
        if w > 0:
            pad = 40
            x1, y1 = max(0, x-pad), max(0, y-pad)
            x2, y2 = min(self.model.raw_image.shape[1], x+w+pad), min(self.model.raw_image.shape[0], y+h+pad)
            
            crop_rgb = self.model.raw_image[y1:y2, x1:x2].copy()
            h_img, w_img, _ = crop_rgb.shape
            
            # Draw Dynamic Ruler (1cm ticks)
            cm_in_px = int(1.0 / cm_per_px)
            if cm_in_px > 10: # Safety to prevent drawing thousands of lines if scaling is broken
                cv2.line(crop_rgb, (10, 10), (10, h_img - 10), (255, 255, 255), 2)
                for tick_y in range(10, h_img - 10, cm_in_px):
                    cv2.line(crop_rgb, (5, tick_y), (15, tick_y), (255, 255, 255), 2)
                    cv2.putText(crop_rgb, f"{(tick_y-10)//cm_in_px}cm", (20, tick_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Draw Graph Overlay
            if self.chk_show_graph.isChecked():
                # Main Root (RED)
                pts_main = data["main_pts"]
                if len(pts_main) > 1:
                    pts_arr = np.array([[int(p[0]) - x1, int(p[1]) - y1] for p in pts_main], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(crop_rgb, [pts_arr], False, (255, 0, 0), 2)
                
                # Lateral Roots (GREEN)
                for lat_pts in data["lateral_pts_list"]:
                    if len(lat_pts) < 2: continue
                    pts_arr = np.array([[int(p[0]) - x1, int(p[1]) - y1] for p in lat_pts], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(crop_rgb, [pts_arr], False, (0, 255, 0), 2)

            for lat_pts in data["lateral_pts_list"]:
                if len(lat_pts) < 2: continue
                start = (int(lat_pts[0][0]) - x1, int(lat_pts[0][1]) - y1)
                
                # Initiation Points (YELLOW)
                if self.chk_show_initiation.isChecked():
                    cv2.circle(crop_rgb, start, 4, (0, 255, 255), -1)
                    
                # Emergence Angle Vector (WHITE)
                if self.chk_show_em_angles.isChecked():
                    cv2.line(crop_rgb, start, (start[0], start[1] + 30), (200, 200, 200), 1) # Vertical Ref
                    em_end = lat_pts[min(cm_in_px, len(lat_pts)) - 1]
                    em_pt = (int(em_end[0]) - x1, int(em_end[1]) - y1)
                    cv2.line(crop_rgb, start, em_pt, (255, 255, 255), 2)
                    
                # Tip Angle Vector (CYAN)
                if self.chk_show_tip_angles.isChecked():
                    if not self.chk_show_em_angles.isChecked():
                        cv2.line(crop_rgb, start, (start[0], start[1] + 30), (200, 200, 200), 1) # Vertical Ref
                    tip_pt = (int(lat_pts[-1][0]) - x1, int(lat_pts[-1][1]) - y1)
                    cv2.line(crop_rgb, start, tip_pt, (0, 255, 255), 2)

            qimg = QImage(crop_rgb.data, w_img, h_img, 3 * w_img, QImage.Format_RGB888)
            self.lbl_inspector_img.setPixmap(QPixmap.fromImage(qimg))

    def run_export(self):
        if not self.measurements_cache: return
        os.makedirs(self.out_dir, exist_ok=True)
        
        cm_per_px = self.get_cm_per_px()
        if cm_per_px is None: return
            
        plate_meta = {
            "plate_id": self.in_plate_id.text(),
            "condition": self.in_condition.text(),
            "timepoint": self.in_timepoint.text(),
            "calibration_mode": self.cb_calibration.currentText(),
            "calibration_val": self.in_calib_val.text(),
            "scale_cm_px": cm_per_px,
        }
        
        self.show_loading("Exporting JSON and RSML...")
        worker = ModelWorker(export_rsml_and_json, self.out_dir, self.current_base_name, plate_meta, self.measurements_cache)
        self.active_workers.add(worker)
        worker.finished.connect(self._on_export_finished)
        worker.error.connect(self._on_thread_error)
        worker.start()
        
    def _on_export_finished(self, _):
        self.hide_loading()
        worker = self.sender()
        if worker in self.active_workers:
            self.active_workers.remove(worker)
            worker.deleteLater()
            
        self.populate_browser()
        QMessageBox.information(self, "Export Complete", f"Successfully exported to:\n{self.out_dir}")

    def show_loading(self, message):
        self.progress = QProgressDialog(message, None, 0, 0, self)
        self.progress.setWindowTitle("Please Wait")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None) 
        self.progress.show()

    def hide_loading(self):
        if hasattr(self, 'progress') and self.progress:
            self.progress.close()
            self.progress.deleteLater()
            self.progress = None
            
    def _on_thread_error(self, err_msg):
        self.hide_loading()
        worker = self.sender()
        if worker in getattr(self, 'active_workers', set()):
            self.active_workers.remove(worker)
            worker.deleteLater()
        QMessageBox.critical(self, "Processing Error", f"A background task failed:\n{err_msg}")

    def closeEvent(self, event):
        GLOBAL_CONFIG["input_root"] = self.in_dir
        GLOBAL_CONFIG["output_root"] = self.out_dir
        GLOBAL_CONFIG["calib_mode"] = self.cb_calibration.currentText()
        GLOBAL_CONFIG["calib_val"] = self.in_calib_val.text()
        with open(GLOBAL_CONFIG_FILE, 'w') as f: json.dump(GLOBAL_CONFIG, f, indent=4)
        event.accept()

class ModelWorker(QThread):
    finished = pyqtSignal(object); error = pyqtSignal(str)
    def __init__(self, func, *args, **kwargs):
        super().__init__(); self.func = func; self.args = args; self.kwargs = kwargs
    def run(self):
        try: result = self.func(*self.args, **self.kwargs); self.finished.emit(result)
        except Exception as e: self.error.emit(str(e))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AnalyzerWindow()
    window.show()
    sys.exit(app.exec_())