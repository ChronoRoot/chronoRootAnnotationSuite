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
                             QStyle, QProgressDialog, QComboBox, QInputDialog)
from PyQt5.QtGui import QRegularExpressionValidator
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRegularExpression

# --- MODULAR CORE IMPORTS ---
from core.model import PlantImageModel
from core.analyzer_engine import extract_plate_metrics, export_rsml_and_json
from tabs.review_tab import ReviewCanvasTab
from tabs.inspector_tab import PhenomicsInspectorTab 
from tabs.report_tab import ComparisonReportTab

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

# ==========================================
# MAIN APPLICATION WINDOW
# ==========================================
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
        
        lbl_stats = QLabel(f"Status: {analyzer_status} | Num Plants: {plant_count}")
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
        self.current_cm_per_px = 1.0 
        
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
        self.btn_up = QPushButton("Up Level"); self.btn_up.clicked.connect(self.navigate_up)
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
        
        # Calibration Dropdown
        self.cb_calibration = QComboBox()
        self.cb_calibration.addItems(["Scanner DPI", "Known Image Height (cm)", "Known Image Width (cm)", "Custom Ratio (px/cm)"])
        self.cb_calibration.setCurrentText(GLOBAL_CONFIG.get("calib_mode", "Scanner DPI"))
        self.in_calib_val = QLineEdit(GLOBAL_CONFIG.get("calib_val", "600"))
        
        self.cb_calibration.currentIndexChanged.connect(self.update_canvas_ruler)
        
        # --- MINIMAL CHANGE: Dedicated Calibration Trigger Button ---
        calib_layout = QHBoxLayout()
        calib_layout.addWidget(self.in_calib_val)
        
        self.btn_set_calib = QPushButton("Set via Measurement")
        self.btn_set_calib.clicked.connect(self.start_calibration_flow)
        calib_layout.addWidget(self.btn_set_calib)
        
        form.addRow("Plate ID:", self.in_plate_id)
        form.addRow("Condition:", self.in_condition)
        form.addRow("Timepoint (Stage):", self.in_timepoint)
        form.addRow("Calibration Method:", self.cb_calibration)
        form.addRow("Calibration Value:", calib_layout) # Replaced QLineEdit with Layout
        mp_layout.addLayout(form)
        
        self.btn_measure_tool = QPushButton("Test Distance Tool")
        self.btn_measure_tool.setCheckable(True)
        self.btn_measure_tool.setStyleSheet("background-color: #ffc107; color: black; font-weight: bold;")
        self.btn_measure_tool.clicked.connect(self.toggle_ruler_mode)
        mp_layout.addWidget(self.btn_measure_tool)
        
        # Flag to track if the ruler is currently hijacking the calibration
        self.is_calibrating = False
        
        # Explicitly allow digits with either a dot OR a comma at the hardware level
        reg_ex = QRegularExpression(r"^[0-9]+[.,]?[0-9]*$")
        num_validator = QRegularExpressionValidator(reg_ex, self)
        self.in_calib_val.setValidator(num_validator)
        
        # Keep your metadata connections the same
        self.in_plate_id.editingFinished.connect(lambda: self.sanitize_line_edit(self.in_plate_id, "Plate ID"))
        self.in_condition.editingFinished.connect(lambda: self.sanitize_line_edit(self.in_condition, "Condition"))
        self.in_timepoint.editingFinished.connect(lambda: self.sanitize_line_edit(self.in_timepoint, "Timepoint"))
        self.in_calib_val.editingFinished.connect(self.validate_and_update_ruler)
        
        mp_layout.addSpacing(10)
        mp_layout.addWidget(QLabel("<b>Plant Identification:</b>"))
        
        self.table_plants = QTableWidget(0, 3)
        self.table_plants.setHorizontalHeaderLabels(["Automatic ID", "Genotype", "Plant #"])
        self.table_plants.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_plants.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_plants.itemSelectionChanged.connect(self.on_table_selection)
        self.table_plants.itemChanged.connect(self.on_table_item_changed)
        mp_layout.addWidget(self.table_plants)
        
        
        btn_layout = QHBoxLayout()
        self.btn_measure = QPushButton("1. Measure")
        self.btn_measure.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 10px;")
        self.btn_measure.clicked.connect(self.run_measurements)
        
        self.btn_export = QPushButton("2. Export")
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
        
        # --- NEW: Connect the distance tool signal from canvas ---
        # Assuming your ReviewCanvasTab emits a `distance_measured(float)` signal containing the pixel length
        if hasattr(self.canvas_review, 'distance_measured'):
            self.canvas_review.distance_measured.connect(self.on_distance_measured)
            
        self.tabs.addTab(self.canvas_review, "Plate Overview")
        
        self.inspector_tab = PhenomicsInspectorTab()
        self.tabs.addTab(self.inspector_tab, "Phenomics Inspector")
        
        # Add the new Report tab
        self.report_tab = ComparisonReportTab(self)
        self.tabs.addTab(self.report_tab, "Report & Plots")
        
        # Trigger refresh of reports when tab is clicked
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        rp_layout.addWidget(self.tabs)
        
        splitter.addWidget(rp)
        splitter.setSizes([300, 400, 900])

        self.model.register_data_callback(self.populate_plant_table)
        self.model.register_selection_callback(self.sync_canvas_to_table)
        self.populate_browser()

    def on_tab_changed(self, index):
        # Refresh the report list if the user navigates to the Report tab (Index 2)
        if index == 2:
            self.report_tab.refresh_file_list()

    # --- CALIBRATION LOGIC ---
    def get_cm_per_px(self):
        calib_mode = self.cb_calibration.currentText()
        try:
            val = float(self.in_calib_val.text())
            if val <= 0: raise ValueError
        except ValueError:
            return None
            
        if "DPI" in calib_mode:
            return 2.54 / val
        elif "Custom Ratio" in calib_mode:
            return 1.0 / val
        else:
            if self.model.raw_image is None: return None
            h_px, w_px = self.model.raw_image.shape[:2]
            if "Height" in calib_mode: return val / h_px
            elif "Width" in calib_mode: return val / w_px
        return None

    def start_calibration_flow(self):
        """Forces the canvas into Ruler mode specifically for calibration."""
        self.is_calibrating = True
        self.btn_measure_tool.setChecked(True)
        self.canvas_review.set_mode("RULER")
        self.btn_measure_tool.setText("Stop Measuring")
        self.tabs.setCurrentIndex(0)
        QMessageBox.information(self, "Calibration Mode", "Click and drag across a known distance (e.g., a ruler in the photo).")

    def toggle_ruler_mode(self):
        """Standard toggle for visual inspection only."""
        if self.btn_measure_tool.isChecked():
            self.is_calibrating = False # Normal visual inspection, no prompt
            self.canvas_review.set_mode("RULER")
            self.btn_measure_tool.setText("Stop Measuring")
            self.tabs.setCurrentIndex(0)
        else:
            self.is_calibrating = False
            self.canvas_review.set_mode("SELECT")
            self.btn_measure_tool.setText("Test Distance Tool")

    def validate_and_update_ruler(self):
        """Validates the main calibration entry field on focus out / enter press."""
        text_val = self.in_calib_val.text().strip()
        
        if not text_val:
            self.in_calib_val.setText("1.0")
            text_val = "1.0"
        
        # 1. Convert commas to dots first
        if "," in text_val:
            text_val = text_val.replace(",", ".")
            
        # 2. Strict Check: If there's more than 1 dot, keep only the first one
        if text_val.count('.') > 1:
            parts = text_val.split('.')
            # Stitch back together: first_part . everything_else_joined
            text_val = parts[0] + '.' + ''.join(parts[1:])
            
            QMessageBox.warning(
                self, "Invalid Format", 
                "Multiple decimal points detected. The value has been truncated to a valid number."
            )
            
        # Update the UI field with the final sanitized string safely
        self.in_calib_val.blockSignals(True)
        self.in_calib_val.setText(text_val)
        self.in_calib_val.blockSignals(False)
            
        self.update_canvas_ruler()
        
    def on_distance_measured(self, pixel_distance):
        """Validates numerical entries generated through the test distance routine."""
        if not self.is_calibrating:
            return 
            
        self.btn_measure_tool.setChecked(False)
        self.canvas_review.set_mode("SELECT")
        self.btn_measure_tool.setText("Test Distance Tool")
        
        text_val, ok = QInputDialog.getText(
            self, "Calibration Setup", 
            f"Line measured as {pixel_distance:.2f} pixels.\nWhat is this distance in real life (cm)?",
            text="1.0"
        )
        
        if ok and text_val.strip():
            raw_text = text_val.strip().replace(" ", "_").replace(",", ".")
            
            # Strict Check: Fix multiple dots in the popup window
            if raw_text.count('.') > 1:
                parts = raw_text.split('.')
                raw_text = parts[0] + '.' + ''.join(parts[1:])
                QMessageBox.warning(
                    self, "Format Correction",
                    "Multiple decimal points detected. The value has been automatically corrected."
                )
            
            try:
                real_cm = float(raw_text)
                if real_cm > 0:
                    px_per_cm = pixel_distance / real_cm
                    self.cb_calibration.setCurrentText("Custom Ratio (px/cm)")
                    self.in_calib_val.setText(f"{px_per_cm:.2f}")
                    self.update_canvas_ruler()
                else:
                    raise ValueError
            except ValueError:
                QMessageBox.critical(self, "Processing Failure", "Failed to resolve value. Enter a positive number.")
            
        self.is_calibrating = False

    def update_canvas_ruler(self):
        text_val = self.in_calib_val.text()
        
        # --- MINIMAL CHANGE: Validate and auto-correct comma to dot ---
        if "," in text_val:
            # Block signals temporarily to prevent infinite event loops while fixing text
            self.in_calib_val.blockSignals(True)
            corrected_text = text_val.replace(",", ".")
            self.in_calib_val.setText(corrected_text)
            self.in_calib_val.blockSignals(False)
            
            QMessageBox.warning(
                self, "Invalid Format", 
                "ChronoRoot requires dots (.) instead of commas (,) for decimal numbers.\n\n"
                "The value has been automatically corrected for you."
            )
            
        cm_per_px = self.get_cm_per_px()
        self.canvas_review.update_scene_ruler(cm_per_px)
            
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
                    # Dynamically look for any Metrics.json related to this base image
                    # Since renaming, we check if ANY file starts with PlateID and contains the base name
                    metrics_found = False
                    if os.path.exists(out_directory):
                        for f in os.listdir(out_directory):
                            if item['name'] in f and f.endswith("_Metrics.json"):
                                metrics_found = True
                                break
                    item["analyzer_status"] = "Analyzed" if metrics_found else "Ready"
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
                # Fetch the native standard directory icon from the OS theme
                icon = QApplication.style().standardIcon(QStyle.SP_DirIcon)
                list_item = QListWidgetItem(icon, item['name'])
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
        self.update_canvas_ruler()

    def populate_plant_table(self):
        self.table_plants.blockSignals(True)
        self.table_plants.setRowCount(0)
        self.measurements_cache.clear()
        self.btn_export.setEnabled(False)
        self.inspector_tab.update_view(None, None, None, None, None)
        
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
        
    def sanitize_line_edit(self, line_edit, field_name):
        """Replaces illegal spacing with underscores in plate metadata text entries."""
        raw_text = line_edit.text()
        if " " in raw_text:
            line_edit.blockSignals(True)
            sanitized = raw_text.replace(" ", "_")
            line_edit.setText(sanitized)
            line_edit.blockSignals(False)
            
            QMessageBox.warning(
                self, "Naming Convention Warning", 
                f"Spaces are not allowed in the '{field_name}' field to prevent "
                "downstream file parsing and directory path breaks.\n\n"
                "Spaces have been automatically replaced with underscores (_)."
            )

    def on_table_item_changed(self, item):
        """Sanitizes manual user string edits inside Genotype or Plant Number table cells."""
        if item.column() == 0: 
            return # Ignore read-only automatic UID column
            
        raw_text = item.text()
        if " " in raw_text:
            self.table_plants.blockSignals(True)
            sanitized = raw_text.replace(" ", "_")
            item.setText(sanitized)
            self.table_plants.blockSignals(False)
            
            QMessageBox.warning(
                self, "Naming Convention Warning", 
                "Spaces are not permitted in Genotype or Plant Number identifiers to maintain clean metrics maps.\n\n"
                "Spaces have been automatically converted to underscores (_)."
            )

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
    def run_measurements(self):
        if not self.model.masks: return
        
        if not self.table_plants.selectedItems() and self.table_plants.rowCount() > 0:
            self.table_plants.selectRow(0)
            
        cm_per_px = self.get_cm_per_px()
        if cm_per_px is None:
            QMessageBox.warning(self, "Error", "Invalid Calibration Value.")
            return

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
            self.inspector_tab.update_view(None, None, None, None, None)
            return
            
        data = self.measurements_cache[uid]
        bbox = self.model.bboxes.get(uid, (0,0,0,0))
        self.inspector_tab.update_view(uid, data, self.model.raw_image, bbox, self.current_cm_per_px)

    def run_export(self):
        if not self.measurements_cache: return
        os.makedirs(self.out_dir, exist_ok=True)
        cm_per_px = self.get_cm_per_px()
        if cm_per_px is None: return
        
        # 1. Grab UI inputs
        plate_id = self.in_plate_id.text().strip()
        condition = self.in_condition.text().strip()
        timepoint = self.in_timepoint.text().strip()
        
        # 2. Build the export filename standard
        export_base_name = f"{plate_id}_{condition}_{timepoint}"
        
        # 3. Store reference to the actual image
        original_img_name = os.path.basename(self.model.image_path) if self.model.image_path else "unknown"
            
        plate_meta = {
            "original_image": original_img_name,
            "plate_id": plate_id,
            "condition": condition,
            "timepoint": timepoint,
            "calibration_mode": self.cb_calibration.currentText(),
            "calibration_val": self.in_calib_val.text(),
            "scale_cm_px": cm_per_px,
        }
        
        self.show_loading("Exporting JSON and RSML...")
        
        # Pass the dynamic export_base_name instead of self.current_base_name
        worker = ModelWorker(export_rsml_and_json, self.out_dir, export_base_name, plate_meta, self.measurements_cache)
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
        self.report_tab.refresh_file_list()
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