import sys
import os

os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.warning=false"

import json

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QSplitter, QStackedWidget, QProgressDialog, QMessageBox,
                             QLabel, QFrame, QInputDialog, QTabWidget, QGroupBox,
                             QCheckBox, QRadioButton)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from core.model import PlantImageModel
from core.analyzer_engine import extract_plate_metrics, export_rsml_and_json

from components.file_browser import UnifiedFileBrowser
from components.analyzer_panel import PhenomicsControlPanel
from components.genotype_manager import GenotypeManagerDialog
from components.workspace_manager import WorkspaceManager
from components.instance_list import InstanceListPanel

from tabs.review_tab import ReviewToolPanel
from tabs.frangi_tab import FrangiToolPanel
from tabs.graph_tab import GraphToolPanel

APP_NAME = "chronoroot"
CONFIG_DIR = os.path.expanduser(f"~/.config/{APP_NAME}")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")


class ModelWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class ChronoRootSuite(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChronoRoot - Unified Phenomics Suite")
        self.resize(1600, 900)

        self.global_model = PlantImageModel()
        self.active_workers = set()
        self.measurements_cache = {}
        self.is_calibrating = False

        self.current_task_path = ""
        self.current_base_name = ""
        self.current_file_path = ""
        self._pending_load_data = None
        self._pending_mark_finished = False
        self._is_closing = False
        self._selection_sync_guard = False

        self.config_file = CONFIG_FILE
        self.load_config()

        self.init_ui()
        self.wire_signals()
        self.scan_directory(self.browser.current_dir)

    def load_config(self):
        os.makedirs(CONFIG_DIR, exist_ok=True)
        self.config = {
            "database_root": ".",
            "input_root": ".",
            "output_root": ".",
            "output_mode": "task_folder",
            "saved_genotypes": ["Col-0", "Ler", "Cvi-0"],
            "calib_mode": "Scanner DPI",
            "calib_val": "600",
        }
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    loaded = json.load(f)
                    self.config.update(loaded)
            except (json.JSONDecodeError, ValueError):
                pass

        if "database_root" in self.config and self.config.get("input_root") == ".":
            self.config["input_root"] = self.config["database_root"]
        if "output_mode" not in self.config:
            self.config["output_mode"] = "task_folder"
        if "saved_genotypes" not in self.config:
            self.config["saved_genotypes"] = ["Col-0", "Ler", "Cvi-0"]

    def save_interface_config(self):
        try:
            try:
                with open(self.config_file, "r") as f:
                    cfg = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                cfg = {}

            cfg.update(self.config)
            cfg["database_root"] = self.browser.in_dir
            cfg["input_root"] = self.browser.in_dir
            cfg["output_root"] = self.browser.out_dir
            cfg["output_mode"] = self.config.get("output_mode", "task_folder")
            cfg["calib_mode"] = self.panel_phenomics.cb_calibration.currentText()
            cfg["calib_val"] = self.panel_phenomics.in_calib_val.text()

            cfg["review"] = {
                "brush_size": self.panel_review.slider_size.value(),
                "opacity": self.panel_review.slider_opacity.value(),
                "show_bboxes": self.panel_review.btn_bbox.isChecked(),
            }

            f_targets = [
                self.panel_frangi.list_target_classes.item(i).data(Qt.UserRole)
                for i in range(self.panel_frangi.list_target_classes.count())
                if self.panel_frangi.list_target_classes.item(i).checkState() == Qt.Checked
            ]
            cfg["frangi"] = {
                "search_range": self.panel_frangi.sp_search.value(),
                "bridge_gaps": self.panel_frangi.sp_bridge.value(),
                "min_part": self.panel_frangi.sp_min_part.value(),
                "faint_sens": self.panel_frangi.sp_f_low.value(),
                "strong_conf": self.panel_frangi.sp_f_high.value(),
                "final_thick": self.panel_frangi.sp_thick.value(),
                "centerline_correction": self.panel_frangi.chk_correction.isChecked(),
                "roots_dark": self.panel_frangi.chk_dark.isChecked(),
                "allow_disconnected": self.panel_frangi.chk_disconnected.isChecked(),
                "channel_mode": self.panel_frangi.cb_channel.currentText(),
                "clahe_mode": self.panel_frangi.cb_clahe.currentText(),
                "smooth_mode": self.panel_frangi.cb_smooth.currentText(),
                "target_classes": f_targets,
            }

            g_targets = [
                self.panel_graph.list_target_classes.item(i).data(Qt.UserRole)
                for i in range(self.panel_graph.list_target_classes.count())
                if self.panel_graph.list_target_classes.item(i).checkState() == Qt.Checked
            ]
            cfg["graph"] = {
                "prune": self.panel_graph.sp_prune.value(),
                "thick": self.panel_graph.sp_thick.value(),
                "target_classes": g_targets,
            }

            self.config = cfg
            with open(self.config_file, "w") as f:
                json.dump(cfg, f, indent=4)
                f.flush()
        except Exception as e:
            print(f"CRITICAL ERROR SAVING CONFIG: {e}")

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        self.splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(self.splitter)

        self.browser = UnifiedFileBrowser(
            self.config.get("input_root", "."),
            self.config.get("output_root", "."),
            config_file=self.config_file,
            show_save_actions=True,
            unified_file_stats=True,
            output_mode=self.config.get("output_mode", "task_folder"),
        )
        self.browser.setMinimumWidth(250)
        self.browser.setMaximumWidth(400)
        self.splitter.addWidget(self.browser)

        self.workspaces = WorkspaceManager(self, self.global_model)
        self._inject_canvas_configs()

        self.middle_stack = QStackedWidget()
        self.middle_stack.setMinimumWidth(300)
        self.middle_stack.setMaximumWidth(450)

        annotator_container = QWidget()
        al = QVBoxLayout(annotator_container)
        al.setContentsMargins(5, 5, 5, 5)
        al.setAlignment(Qt.AlignTop)

        self.tool_stack = QStackedWidget()
        self.panel_review = ReviewToolPanel(self.global_model, self.workspaces.canvas_review)
        self.panel_frangi = FrangiToolPanel(self.global_model, self.workspaces.canvas_frangi)
        self.panel_graph = GraphToolPanel(self.global_model, self.workspaces.canvas_graph)

        self.instance_list = InstanceListPanel(
            self.global_model, self.workspaces, review_tool_panel=self.panel_review, annotation_tab_index=0
        )
        al.addWidget(self.instance_list)

        al.addSpacing(10)
        al.addWidget(QFrame(frameShape=QFrame.HLine, frameShadow=QFrame.Sunken))
        al.addSpacing(10)
        al.addWidget(QLabel("<b>Control Panel:</b>"))

        r_cfg = self.config.get("review", {})
        if "brush_size" in r_cfg:
            self.panel_review.slider_size.setValue(r_cfg["brush_size"])
        if "opacity" in r_cfg:
            self.panel_review.slider_opacity.setValue(r_cfg["opacity"])
        if "show_bboxes" in r_cfg:
            self.panel_review.btn_bbox.setChecked(r_cfg["show_bboxes"])
            self.workspaces.canvas_review.set_show_bboxes(r_cfg["show_bboxes"])

        self.tool_stack.addWidget(self.panel_review)
        self.tool_stack.addWidget(self.panel_frangi)
        self.tool_stack.addWidget(self.panel_graph)
        al.addWidget(self.tool_stack)

        self.panel_phenomics = PhenomicsControlPanel(self.config, config_file=self.config_file)
        self.control_tabs = QTabWidget()
        self.control_tabs.addTab(annotator_container, "Annotation Controls")
        self.control_tabs.addTab(self.panel_phenomics, "Plant Metadata")

        self.global_label_group = QGroupBox("Canvas Label Settings")
        gl = QVBoxLayout(self.global_label_group)

        row_show = QHBoxLayout()
        self.chk_show_num_global = QCheckBox("Plant #")
        self.chk_show_geno_global = QCheckBox("Genotype")
        self.chk_show_num_global.setChecked(True)
        self.chk_show_geno_global.setChecked(True)
        row_show.addWidget(QLabel("Show:"))
        row_show.addWidget(self.chk_show_num_global)
        row_show.addWidget(self.chk_show_geno_global)
        row_show.addStretch()

        row_fmt = QHBoxLayout()
        self.rad_fmt_text_global = QRadioButton("Text")
        self.rad_fmt_num_global = QRadioButton("Index")
        self.rad_fmt_text_global.setChecked(True)
        row_fmt.addWidget(QLabel("Format:"))
        row_fmt.addWidget(self.rad_fmt_text_global)
        row_fmt.addWidget(self.rad_fmt_num_global)
        row_fmt.addStretch()

        gl.addLayout(row_show)
        gl.addLayout(row_fmt)

        controls_container = QWidget()
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(self.global_label_group)
        controls_layout.addWidget(self.control_tabs)
        self.middle_stack.addWidget(controls_container)
        self.middle_stack.addWidget(QWidget())

        self.splitter.addWidget(self.middle_stack)
        self.splitter.addWidget(self.workspaces)
        self.splitter.setSizes([300, 350, 950])

    def _inject_canvas_configs(self):
        f_cfg = self.config.get("frangi", {})
        frangi = self.workspaces.canvas_frangi
        frangi.p_search_range = f_cfg.get("search_range", 0)
        frangi.p_bridge_gaps = f_cfg.get("bridge_gaps", 1)
        frangi.p_min_part = f_cfg.get("min_part", 5)
        frangi.p_faint_sens = f_cfg.get("faint_sens", 0.05)
        frangi.p_strong_conf = f_cfg.get("strong_conf", 0.20)
        frangi.p_final_thick = f_cfg.get("final_thick", 1)
        frangi.p_centerline_correction = f_cfg.get("centerline_correction", True)
        frangi.p_roots_dark = f_cfg.get("roots_dark", False)
        frangi.p_allow_disconnected = f_cfg.get("allow_disconnected", False)
        frangi.p_channel_mode = f_cfg.get("channel_mode", "Red-Blue Avg (RB)")
        frangi.p_clahe_mode = f_cfg.get("clahe_mode", "None")
        frangi.p_smooth_mode = f_cfg.get("smooth_mode", "Medium")
        frangi.p_target_classes = f_cfg.get("target_classes", [1, 2])

        g_cfg = self.config.get("graph", {})
        graph = self.workspaces.canvas_graph
        graph.p_prune = g_cfg.get("prune", 0)
        graph.p_thick = g_cfg.get("thick", 1)
        graph.p_target_classes = g_cfg.get("target_classes", [1, 2])

    def wire_signals(self):
        self.workspaces.workspace_changed.connect(self._on_workspace_changed)

        self.browser.refresh_requested.connect(self.scan_directory)
        self.browser.file_selected.connect(self.load_file)
        self.browser.save_progress_requested.connect(lambda: self.save_task(mark_finished=False))
        self.browser.finish_annotation_requested.connect(lambda: self.save_task(mark_finished=True))
        self.browser.input_dir_changed.connect(self._on_input_dir_changed)
        self.browser.output_dir_changed.connect(self._on_output_dir_changed)
        self.browser.output_mode_changed.connect(self._on_output_mode_changed)

        self.panel_phenomics.manage_genotypes_requested.connect(self.open_genotype_manager)
        self.panel_phenomics.measure_requested.connect(self.run_measurements)
        self.panel_phenomics.export_requested.connect(self.run_export)
        self.panel_phenomics.selection_changed.connect(self._on_phenomics_table_selection)
        self.panel_phenomics.set_calib_ruler_requested.connect(self.start_calibration_flow)
        self.panel_phenomics.test_ruler_toggled.connect(self.toggle_ruler_mode)
        self.panel_phenomics.calibration_changed.connect(self.validate_and_update_ruler)
        self.panel_phenomics.overlay_labels_changed.connect(self.update_overlay_labels)

        if hasattr(self.workspaces.canvas_review, "distance_measured"):
            self.workspaces.canvas_review.distance_measured.connect(self.on_distance_measured)

        self.workspaces.inspector_tab.run_analysis_requested.connect(
            self.panel_phenomics._on_measure_clicked
        )

        self.global_model.register_selection_callback(self._on_model_selection_changed)
        self.global_model.register_data_callback(self._on_model_data_changed)

        self.chk_show_num_global.toggled.connect(self._apply_global_label_preferences)
        self.chk_show_geno_global.toggled.connect(self._apply_global_label_preferences)
        self.rad_fmt_text_global.toggled.connect(self._apply_global_label_preferences)
        self.rad_fmt_num_global.toggled.connect(self._apply_global_label_preferences)
        self._apply_global_label_preferences()

    def _on_input_dir_changed(self, path):
        self.config["input_root"] = path
        self.config["database_root"] = path

    def _on_output_dir_changed(self, path):
        self.config["output_root"] = path
        self.config["output_mode"] = "fixed"

    def _on_output_mode_changed(self, mode):
        self.config["output_mode"] = mode
        if mode == "task_folder" and self.current_task_path:
            self.browser.out_dir = self.current_task_path
            self.config["output_root"] = self.current_task_path
        elif mode == "fixed":
            self.config["output_root"] = self.browser.out_dir

    def _on_workspace_changed(self, index):
        if index in [0, 1, 2, 3]:
            self.ensure_active_plant()

        if index in [0, 1, 2]:
            self.control_tabs.setEnabled(True)
            self.middle_stack.setCurrentIndex(0)
            self.tool_stack.setCurrentIndex(index)
            if index in [1, 2]:
                self.control_tabs.setCurrentIndex(0)
            if self.splitter.sizes()[1] == 0:
                self.splitter.setSizes([300, 350, 950])

            if index == 0:
                self.panel_review.force_mode("SELECT")
                if self.global_model.active_uid:
                    self.workspaces.canvas_review.zoom_to_plant(self.global_model.active_uid)

        elif index == 3:
            self.control_tabs.setEnabled(True)
            self.middle_stack.setCurrentIndex(0)
            self.control_tabs.setCurrentIndex(1)
            if self.splitter.sizes()[1] == 0:
                self.splitter.setSizes([300, 350, 950])
            self.sync_inspector()

        elif index == 4:
            self.control_tabs.setEnabled(False)
            self.middle_stack.setCurrentIndex(1)
            self.splitter.setSizes([300, 0, 1300])
            self.workspaces.report_tab.refresh_file_list()

        elif index in [5, 6]:
            self.control_tabs.setEnabled(False)
            self.middle_stack.setCurrentIndex(1)
            self.splitter.setSizes([300, 0, 1300])

    def ensure_active_plant(self):
        """Guarantee a valid active plant when masks exist but selection is empty or stale."""
        model = self.global_model
        if not model.masks:
            if model.selected_uids:
                self._selection_sync_guard = True
                model.set_selection([])
                self.panel_phenomics.set_table_selection(set())
                self._selection_sync_guard = False
            return

        valid_uids = set(model.masks.keys())
        selected = [uid for uid in model.selected_uids if uid in valid_uids]

        if not selected:
            first_uid = sorted(valid_uids)[0]
            self._apply_plant_selection([first_uid], source="auto", zoom=False)
        elif len(selected) != len(model.selected_uids):
            self._apply_plant_selection(selected, source="auto", zoom=False)

    def _apply_plant_selection(self, uids, source="external", zoom=False):
        """Single source-of-truth selection update for model-driven UI sync."""
        if self._selection_sync_guard:
            return

        valid_uids = [uid for uid in uids if uid in self.global_model.masks]
        if uids and not valid_uids:
            valid_uids = []

        self._selection_sync_guard = True
        self.global_model.set_selection(valid_uids)
        self.panel_phenomics.set_table_selection(set(valid_uids))
        self.workspaces.canvas_review.refresh_canvas()

        if zoom and self.global_model.active_uid:
            self.workspaces.canvas_review.zoom_to_plant(self.global_model.active_uid)

        self.sync_inspector()
        self._selection_sync_guard = False

    def _sync_metadata_table_with_model(self, uid_mapping=None):
        """Rebuild plant metadata table when mask topology changes (new/merge/delete/split)."""
        model_uids = set(self.global_model.masks.keys())
        table_uids = self.panel_phenomics.get_table_uids()
        if model_uids == table_uids and not uid_mapping:
            return

        preserved = self.panel_phenomics.capture_metadata()
        if uid_mapping:
            preserved = self.panel_phenomics.remap_metadata(preserved, uid_mapping)

        if model_uids:
            self.panel_phenomics.populate_table(
                list(model_uids),
                self.config.get("saved_genotypes", []),
                preserved_metadata=preserved,
            )
            self.panel_phenomics.enable_tools(True)
            self._selection_sync_guard = True
            self.panel_phenomics.set_table_selection(self.global_model.selected_uids)
            self._selection_sync_guard = False
        else:
            self.panel_phenomics.clear_table()

        if model_uids != table_uids or uid_mapping:
            self.measurements_cache.clear()
            self.panel_phenomics.reset_measurements()
            self.workspaces.inspector_tab.show_no_measurements_state()

    def _on_model_data_changed(self):
        self._sync_metadata_table_with_model()
        self.ensure_active_plant()

    def _on_model_selection_changed(self):
        if self._selection_sync_guard:
            return

        self.panel_phenomics.set_table_selection(self.global_model.selected_uids)
        self.sync_inspector()

        ws_index = self.workspaces.currentIndex()
        if ws_index == 0 and self.global_model.active_uid:
            self.workspaces.canvas_review.refresh_canvas()

    def _on_phenomics_table_selection(self, uids):
        if self._selection_sync_guard:
            return
        if not uids:
            return
        zoom = len(uids) == 1
        self._apply_plant_selection(uids, source="table", zoom=zoom)

    def update_overlay_labels(self):
        if hasattr(self.workspaces.canvas_review, "set_overlay_labels"):
            self.workspaces.canvas_review.set_overlay_labels(
                self.panel_phenomics.build_overlay_labels()
            )

    def _apply_global_label_preferences(self):
        show_num = self.chk_show_num_global.isChecked()
        show_geno = self.chk_show_geno_global.isChecked()
        self.rad_fmt_text_global.setEnabled(show_geno)
        self.rad_fmt_num_global.setEnabled(show_geno)
        fmt = "TEXT" if self.rad_fmt_text_global.isChecked() else "NUMBER"
        self.workspaces.canvas_review.set_label_preferences(show_num, show_geno, fmt)

    # --- Calibration (from Analyzer) ---
    def start_calibration_flow(self):
        self.is_calibrating = True
        self.panel_phenomics.btn_measure_tool.setChecked(True)
        self.workspaces.canvas_review.set_calibrating(True)
        self.workspaces.canvas_review.set_mode("RULER")
        self.panel_phenomics.btn_measure_tool.setText("Stop Measuring")
        self.workspaces.setCurrentIndex(0)
        QMessageBox.information(
            self, "Calibration Mode",
            "Click and drag across a known distance (e.g., a ruler in the photo)."
        )

    def toggle_ruler_mode(self, checked):
        if checked:
            self.is_calibrating = False
            self.workspaces.canvas_review.set_calibrating(False)
            self.workspaces.canvas_review.set_mode("RULER")
            self.panel_phenomics.btn_measure_tool.setText("Stop Measuring")
            self.workspaces.setCurrentIndex(0)
        else:
            self.is_calibrating = False
            self.workspaces.canvas_review.set_calibrating(False)
            self.workspaces.canvas_review.set_mode("SELECT")
            self.panel_phenomics.btn_measure_tool.setText("Test Distance Tool")

    def validate_and_update_ruler(self):
        self.panel_phenomics.validate_calib_field()
        self.update_canvas_ruler()

    def on_distance_measured(self, pixel_distance):
        if not self.is_calibrating:
            return

        self.panel_phenomics.btn_measure_tool.setChecked(False)
        self.workspaces.canvas_review.set_calibrating(False)
        self.workspaces.canvas_review.set_mode("SELECT")
        self.panel_phenomics.btn_measure_tool.setText("Test Distance Tool")

        text_val, ok = QInputDialog.getText(
            self, "Calibration Setup",
            f"Line measured as {pixel_distance:.2f} pixels.\nWhat is this distance in real life (cm)?",
            text="1.0",
        )

        if ok and text_val.strip():
            raw_text = text_val.strip().replace(" ", "_").replace(",", ".")
            if raw_text.count(".") > 1:
                parts = raw_text.split(".")
                raw_text = parts[0] + "." + "".join(parts[1:])
                QMessageBox.warning(
                    self, "Format Correction",
                    "Multiple decimal points detected. The value has been automatically corrected."
                )
            try:
                real_cm = float(raw_text)
                if real_cm > 0:
                    px_per_cm = pixel_distance / real_cm
                    self.panel_phenomics.cb_calibration.setCurrentText("Custom Ratio (px/cm)")
                    self.panel_phenomics.in_calib_val.setText(f"{px_per_cm:.2f}")
                    self.update_canvas_ruler()
                else:
                    raise ValueError
            except ValueError:
                QMessageBox.critical(
                    self, "Processing Failure",
                    "Failed to resolve value. Enter a positive number."
                )

        self.is_calibrating = False

    def update_canvas_ruler(self):
        text_val = self.panel_phenomics.in_calib_val.text()
        if "," in text_val:
            self.panel_phenomics.in_calib_val.blockSignals(True)
            self.panel_phenomics.in_calib_val.setText(text_val.replace(",", "."))
            self.panel_phenomics.in_calib_val.blockSignals(False)
            QMessageBox.warning(
                self, "Invalid Format",
                "ChronoRoot requires dots (.) instead of commas (,) for decimal numbers.\n\n"
                "The value has been automatically corrected for you."
            )

        shape = self.global_model.raw_image.shape if self.global_model.raw_image is not None else None
        cm_per_px = self.panel_phenomics.get_cm_per_px(shape)
        self.panel_phenomics.current_cm_per_px = cm_per_px
        self.workspaces.canvas_review.update_scene_ruler(cm_per_px)

    # --- Loading & Threading ---
    def show_loading(self, message):
        self.progress = QProgressDialog(message, None, 0, 0, self)
        self.progress.setWindowTitle("Please Wait")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None)
        self.progress.show()

    def hide_loading(self):
        if hasattr(self, "progress") and self.progress:
            self.progress.close()
            self.progress.deleteLater()
            self.progress = None

    def _enrich_with_analyzer_status(self, contents):
        for item in contents:
            if item["type"] != "file":
                continue
            if item.get("status") != "Completed":
                item["analyzer_status"] = "Not Annotated"
            else:
                metrics_found = False
                task_dir = os.path.dirname(item["path"])
                out_dir = self.browser.get_effective_output_dir(task_dir)
                if os.path.exists(out_dir):
                    for f in os.listdir(out_dir):
                        if item["name"] in f and f.endswith("_Metrics.json"):
                            metrics_found = True
                            break
                item["analyzer_status"] = "Analyzed" if metrics_found else "Ready"

    def scan_directory(self, folder_path, force_refresh=False):
        self.browser.current_dir = folder_path

        if force_refresh:
            keys_to_delete = [k for k in self.browser.folder_cache if k.startswith(folder_path)]
            for k in keys_to_delete:
                del self.browser.folder_cache[k]

        cached = self.browser.get_cached_contents(folder_path)
        if cached is not None and not force_refresh:
            self._enrich_with_analyzer_status(cached)
            self.browser.render_contents(cached, use_cache=False)
            return

        self.show_loading(f"Scanning directory stats...\n{folder_path}")
        worker = ModelWorker(self.global_model.scan_directory, folder_path)
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

        self._enrich_with_analyzer_status(contents)
        self.browser.render_contents(contents)

    def load_file(self, data):
        if self.global_model.dirty:
            reply = QMessageBox.question(
                self, "Save?", "Save changes?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            if reply == QMessageBox.Yes:
                self._pending_load_data = data
                self.save_task(mark_finished=False)
                return
            if reply == QMessageBox.Cancel:
                return

        self._execute_load(data)

    def _execute_load(self, data):
        file_path = data["path"]
        self.current_file_path = file_path
        self.current_task_path = os.path.dirname(file_path)
        self.current_base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Default behavior: export files beside the loaded image+annotation pair.
        # Users can still switch to fixed output by selecting an output folder.
        if self.config.get("output_mode", "task_folder") == "task_folder":
            self.browser.out_dir = self.current_task_path
            self.config["output_root"] = self.current_task_path

        self.global_model.callbacks_muted = True
        self.global_model.set_selection([])
        self.panel_review.force_mode("SELECT")

        self.show_loading(f"Loading {self.current_base_name}...\n(Parsing masks and NIfTI data)")
        worker = ModelWorker(
            self.global_model.load_task, self.current_task_path, self.current_base_name
        )
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

        self.global_model.callbacks_muted = False
        self.global_model._notify_data_changed()
        self.setWindowTitle(
            f"ChronoRoot | {self.current_base_name} [{self.global_model.status.upper()}]"
        )

        self.measurements_cache.clear()
        self.panel_phenomics.reset_measurements()
        self.workspaces.inspector_tab.show_no_measurements_state()

        if self.global_model.masks:
            uids = list(self.global_model.masks.keys())
            self.panel_phenomics.populate_table(uids, self.config.get("saved_genotypes", []))
            self.panel_phenomics.enable_tools(True)
            self.ensure_active_plant()
        else:
            self.panel_phenomics.clear_table()

        self.workspaces.canvas_review.refresh_canvas()
        self.update_canvas_ruler()
        self.workspaces.setCurrentIndex(0)

    def save_task(self, mark_finished=False):
        if not self.global_model.masks:
            return

        self.global_model.callbacks_muted = True
        self.show_loading("Saving progress...\n(Flattening multi-class masks and generating NIfTI)")
        self._pending_mark_finished = mark_finished

        worker = ModelWorker(
            self.global_model.save_current_task,
            self.current_task_path,
            self.current_base_name,
            mark_finished=mark_finished,
        )
        self.active_workers.add(worker)
        worker.finished.connect(self._on_save_finished)
        worker.error.connect(self._on_thread_error)
        worker.start()

    def _on_save_finished(self, mapping):
        self.hide_loading()
        self.global_model.callbacks_muted = False
        worker = self.sender()
        if worker in self.active_workers:
            self.active_workers.remove(worker)
            worker.deleteLater()

        if mapping is None:
            return

        new_selection = [mapping[uid] for uid in self.global_model.selected_uids if uid in mapping]
        self.global_model.set_selection(new_selection)
        self._sync_metadata_table_with_model(uid_mapping=mapping)
        self.global_model._notify_data_changed()
        self.ensure_active_plant()
        self.setWindowTitle(
            f"ChronoRoot | {self.current_base_name} [{self.global_model.status.upper()}]"
        )

        if self._is_closing:
            self.close()
        elif self._pending_load_data:
            data_to_load = self._pending_load_data
            self._pending_load_data = None
            self._execute_load(data_to_load)
        elif self.current_file_path:
            new_status = "Completed" if self._pending_mark_finished else "In Progress"
            self.browser.update_file_status_in_cache(
                self.current_file_path,
                new_status,
                new_plant_count=len(self.global_model.masks),
            )
        else:
            self.scan_directory(self.browser.current_dir, force_refresh=True)

    # --- Analysis Engine ---
    def run_measurements(self, plants_meta, cm_per_px):
        self.show_loading("Measuring Morphometrics & Tracing RSML...")
        worker = ModelWorker(extract_plate_metrics, self.global_model, plants_meta, cm_per_px)
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
        self.panel_phenomics.btn_export.setEnabled(True)
        self.workspaces.setCurrentIndex(3)
        self.sync_inspector()

    def sync_inspector(self):
        inspector = self.workspaces.inspector_tab
        if not self.measurements_cache:
            inspector.show_no_measurements_state()
            return

        uid = self.global_model.active_uid
        if not uid or uid not in self.measurements_cache:
            inspector.show_content_state()
            inspector.update_view(None, None, None, None, None)
            return

        data = self.measurements_cache[uid]
        bbox = self.global_model.bboxes.get(uid, (0, 0, 0, 0))
        shape = self.global_model.raw_image.shape if self.global_model.raw_image is not None else None
        live_cm_per_px = self.panel_phenomics.get_cm_per_px(shape)
        if live_cm_per_px is None:
            live_cm_per_px = 1.0
        else:
            self.panel_phenomics.current_cm_per_px = live_cm_per_px

        self.workspaces.inspector_tab.update_view(
            uid, data, self.global_model.raw_image, bbox, live_cm_per_px
        )

    def run_export(self, plate_meta):
        if not self.measurements_cache:
            return

        os.makedirs(self.browser.get_effective_output_dir(self.current_task_path), exist_ok=True)
        cm_per_px = plate_meta.get("scale_cm_px")
        if cm_per_px is None:
            QMessageBox.warning(self, "Error", "Invalid Calibration Value.")
            return

        out_dir = self.browser.get_effective_output_dir(self.current_task_path)
        export_base_name = (
            f"{plate_meta['plate_id']}_{plate_meta['condition']}_{plate_meta['timepoint']}"
        )
        json_check_path = os.path.join(out_dir, f"{export_base_name}_Metrics.json")
        rsml_check_path = os.path.join(out_dir, f"{export_base_name}_Topology.rsml")

        existing_json_payload = None
        if os.path.exists(json_check_path):
            try:
                with open(json_check_path, "r") as f:
                    existing_json_payload = json.load(f)
            except (json.JSONDecodeError, ValueError):
                existing_json_payload = None

        duplicate_msg = None
        if existing_json_payload and self.config.get("output_mode", "task_folder") == "task_folder":
            same_image = (
                existing_json_payload.get("original_image")
                == (os.path.basename(self.global_model.image_path) if self.global_model.image_path else "unknown")
            )
            same_plate = existing_json_payload.get("plate_id") == plate_meta.get("plate_id")
            same_condition = existing_json_payload.get("condition") == plate_meta.get("condition")
            same_timepoint = existing_json_payload.get("timepoint") == plate_meta.get("timepoint")
            if same_image or (same_plate and same_condition and same_timepoint):
                duplicate_msg = (
                    "An existing analysis JSON in this image folder appears to correspond to "
                    "the same image or plate metadata.\n\n"
                )

        if os.path.exists(json_check_path) or os.path.exists(rsml_check_path):
            reply = QMessageBox.question(
                self, "Confirm Export Overwrite",
                (duplicate_msg or "")
                + f"Warning: Files for '{export_base_name}' already exist in the output folder.\n\n"
                "Did you forget to update the Plate ID, Condition, or Timepoint parameters?\n\n"
                "Do you want to permanently overwrite the existing files?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        original_img_name = (
            os.path.basename(self.global_model.image_path)
            if self.global_model.image_path else "unknown"
        )
        plate_meta["original_image"] = original_img_name

        self.show_loading("Exporting JSON and RSML...")
        worker = ModelWorker(
            export_rsml_and_json,
            out_dir,
            export_base_name,
            plate_meta,
            self.measurements_cache,
        )
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

        self.scan_directory(self.browser.current_dir, force_refresh=True)
        self.workspaces.report_tab.refresh_file_list()
        out_dir = self.browser.get_effective_output_dir(self.current_task_path)
        QMessageBox.information(
            self, "Export Complete",
            f"Successfully exported to:\n{out_dir}"
        )

    def _on_thread_error(self, err_msg):
        self.hide_loading()
        self.global_model.callbacks_muted = False

        worker = self.sender()
        if worker in getattr(self, "active_workers", set()):
            self.active_workers.remove(worker)
            worker.deleteLater()

        QMessageBox.critical(self, "Processing Error", err_msg)
        self._is_closing = False
        self._pending_load_data = None

    # --- Genotype Management ---
    def open_genotype_manager(self):
        dialog = GenotypeManagerDialog(
            self.config.get("saved_genotypes", []),
            self,
            config_file=self.config_file,
        )
        dialog.genotypes_updated.connect(self.update_genotypes)
        dialog.exec_()

    def update_genotypes(self, new_genotypes):
        self.config["saved_genotypes"] = new_genotypes
        self.panel_phenomics.refresh_genotype_combos(new_genotypes)
        self.update_overlay_labels()

    def closeEvent(self, event):
        if self._is_closing:
            self.save_interface_config()
            event.accept()
            return

        if self.global_model.dirty:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes.\nDo you want to save your progress before exiting?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if reply == QMessageBox.Save:
                self._is_closing = True
                self.save_task(mark_finished=False)
                event.ignore()
            elif reply == QMessageBox.Discard:
                self.save_interface_config()
                event.accept()
            else:
                event.ignore()
        else:
            self.save_interface_config()
            event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ChronoRootSuite()
    window.show()
    sys.exit(app.exec_())
