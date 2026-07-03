import sys
import os

os.environ["QT_LOGGING_RULES"] = "*.debug=false;*.warning=false"

import json

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                             QSplitter, QStackedWidget, QMessageBox,
                             QLabel, QFrame, QInputDialog, QTabWidget, QGroupBox,
                             QCheckBox, QRadioButton, QPushButton, QSizePolicy, QToolButton,
                             QStyle)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer

from core.model import (
    PlantImageModel,
    annotation_status_display,
    metrics_path,
    topology_path,
    normalize_annotation_status,
)
from core.analyzer_engine import (
    analyze_single_plant,
    extract_plate_metrics,
    export_rsml_and_json,
    load_measurements_from_metrics_json,
    scalar_metrics_dict,
)

from components.file_browser import UnifiedFileBrowser
from components.analyzer_panel import PhenomicsControlPanel
from components.genotype_manager import GenotypeManagerDialog
from components.workspace_manager import WorkspaceManager
from components.instance_list import InstanceListPanel
from components.loading_overlay import LoadingOverlay

from tabs.review_tab import ReviewToolPanel
from tabs.frangi_tab import FrangiToolPanel
from tabs.graph_tab import GraphToolPanel

APP_NAME = "chronoRootAnnotationSuite"
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


class ChronoRootAnnotationSuite(QMainWindow):
    COLLAPSED_PANEL_WIDTH = 28
    LOADING_SHOW_DELAY_MS = 300

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChronoRoot Annotation Suite")
        self.resize(1600, 900)

        self.global_model = PlantImageModel()
        self.active_workers = set()
        self.measurements_cache = {}
        self.is_calibrating = False
        self.metadata_dirty = False

        self.current_task_path = ""
        self.current_base_name = ""
        self.current_file_path = ""
        self._pending_load_data = None
        self._pending_open_annotation_tab = False
        self._pending_mark_finished = False
        self._pending_export_plate_meta = None
        self._pending_measure = None
        self._is_closing = False
        self._selection_sync_guard = False
        self._focus_mode_active = False
        self._saved_splitter_sizes = None
        self._saved_browser_width = 250
        self._saved_middle_width = 320

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
            "report_search_roots": [],
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
        if "report_search_roots" not in self.config:
            self.config["report_search_roots"] = []
        if "panel_sizes_report" not in self.config:
            self.config["panel_sizes_report"] = [250, 320, 930]

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
            cfg["report_search_roots"] = self.config.get("report_search_roots", [])
            sizes = list(self.splitter.sizes())
            if not self._focus_mode_active:
                ws = self.workspaces.currentIndex()
                if sizes[1] == 0 and ws in (0, 1, 2, 3, 5):
                    sizes[1] = self._saved_middle_width or 320
                elif sizes[1] > 0:
                    self._saved_middle_width = sizes[1]
            cfg["panel_sizes_report"] = sizes
            cfg["calib_mode"] = self.panel_phenomics.cb_calibration.currentText()
            cfg["calib_val"] = self.panel_phenomics.in_calib_val.text()

            cfg["review"] = {
                "brush_size": self.panel_review.slider_size.value(),
                "opacity": self.panel_review.slider_opacity.value(),
                "show_bboxes": self.panel_review.btn_bbox.isChecked(),
            }

            f_targets = [
                cid for cid, chk in self.panel_frangi.class_checkboxes.items() if chk.isChecked()
            ]
            cfg["frangi"] = {
                "min_sigma": self.panel_frangi.sp_min_sigma.value(),
                "max_sigma": self.panel_frangi.sp_max_sigma.value(),
                "sigma_step": self.panel_frangi.sp_sigma_step.value(),
                "search_range": self.panel_frangi.sp_search.value(),
                "bridge_gaps": self.panel_frangi.sp_bridge.value(),
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
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(4, 4, 4, 4)

        self.splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.splitter, stretch=1)

        self.browser_panel = QWidget()
        browser_panel_layout = QVBoxLayout(self.browser_panel)
        browser_panel_layout.setContentsMargins(0, 0, 0, 0)
        browser_panel_layout.setSpacing(0)

        browser_header = QHBoxLayout()
        browser_header.setContentsMargins(4, 4, 4, 0)
        browser_header.addStretch()
        self.btn_toggle_files = self._create_panel_toggle_button(
            "Hide file browser",
            "Show file browser",
            self.toggle_browser_panel,
        )
        browser_header.addWidget(self.btn_toggle_files)
        browser_panel_layout.addLayout(browser_header)

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
        browser_panel_layout.addWidget(self.browser, stretch=1)
        self.splitter.addWidget(self.browser_panel)

        self.workspaces = WorkspaceManager(self, self.global_model)
        self._inject_canvas_configs()

        self.middle_stack = QStackedWidget()

        annotator_container = QWidget()
        al = QVBoxLayout(annotator_container)
        al.setContentsMargins(5, 5, 5, 5)

        self.tool_stack = QStackedWidget()
        self.panel_review = ReviewToolPanel(self.global_model, self.workspaces.canvas_review)
        self.panel_frangi = FrangiToolPanel(self.global_model, self.workspaces.canvas_frangi)
        self.panel_graph = GraphToolPanel(self.global_model, self.workspaces.canvas_graph)

        self.instance_list = InstanceListPanel(
            self.global_model, self.workspaces, review_tool_panel=self.panel_review, annotation_tab_index=0
        )
        al.addWidget(self.instance_list, stretch=1)

        al.addSpacing(6)
        al.addWidget(QFrame(frameShape=QFrame.HLine, frameShadow=QFrame.Sunken))
        al.addSpacing(4)
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
        self.tool_stack.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        al.addWidget(self.tool_stack, stretch=0)

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
        self.middle_stack.addWidget(self.workspaces.report_file_panel)

        self.middle_panel = QWidget()
        middle_panel_layout = QVBoxLayout(self.middle_panel)
        middle_panel_layout.setContentsMargins(0, 0, 0, 0)
        middle_panel_layout.setSpacing(0)

        middle_header = QHBoxLayout()
        middle_header.setContentsMargins(4, 4, 4, 0)
        middle_header.addStretch()
        self.btn_toggle_plates = self._create_panel_toggle_button(
            "Hide plate checklist",
            "Show plate checklist",
            self.toggle_middle_panel,
        )
        self.btn_toggle_plates.setVisible(False)
        middle_header.addWidget(self.btn_toggle_plates)
        middle_panel_layout.addLayout(middle_header)
        middle_panel_layout.addWidget(self.middle_stack, stretch=1)
        self.middle_panel.setMinimumWidth(300)
        self.middle_panel.setMaximumWidth(450)

        self.splitter.addWidget(self.middle_panel)
        self.splitter.addWidget(self.workspaces)
        default_sizes = self.config.get("panel_sizes_report", [250, 320, 930])
        if len(default_sizes) == 3:
            self.splitter.setSizes(default_sizes)
        else:
            self.splitter.setSizes([250, 320, 930])

        self.browser.set_browser_controls_visible(
            self._panel_expanded(self.splitter.sizes()[0])
        )

        self._loading_depth = 0
        self._loading_show_token = 0
        self._pending_loading_message = ""
        self._loading_overlay = LoadingOverlay(main_widget)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_loading_overlay") and self.centralWidget():
            self._loading_overlay.setGeometry(self.centralWidget().rect())

    def _panel_expanded(self, size):
        return size > self.COLLAPSED_PANEL_WIDTH

    def _visibility_icon(self, visible):
        name = "view-visible" if visible else "view-hidden"
        icon = QIcon.fromTheme(name)
        if not icon.isNull():
            return icon
        style = self.style()
        fallback = QStyle.SP_FileDialogDetailedView if visible else QStyle.SP_FileDialogListView
        return style.standardIcon(fallback)

    def _create_panel_toggle_button(self, hide_tooltip, show_tooltip, slot):
        btn = QToolButton()
        btn.setAutoRaise(True)
        btn.setToolTip(hide_tooltip)
        btn.setProperty("hide_tooltip", hide_tooltip)
        btn.setProperty("show_tooltip", show_tooltip)
        btn.clicked.connect(slot)
        self._set_panel_toggle_appearance(btn, expanded=True)
        return btn

    def _set_panel_toggle_appearance(self, btn, expanded):
        btn.setIcon(self._visibility_icon(expanded))
        btn.setToolTip(btn.property("hide_tooltip" if expanded else "show_tooltip"))
        if expanded:
            btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            btn.setText("Hide")
            btn.setFixedSize(QSize())
            btn.setMinimumSize(0, 26)
            btn.setMaximumSize(16777215, 26)
        else:
            btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
            btn.setText("")
            btn.setFixedSize(self.COLLAPSED_PANEL_WIDTH - 4, 26)

    def _splitter_sizes(self):
        return list(self.splitter.sizes())

    def _apply_splitter_sizes(self, left, middle, right):
        self.splitter.setSizes([max(0, int(left)), max(0, int(middle)), max(0, int(right))])

    def toggle_browser_panel(self):
        if self._focus_mode_active:
            return
        sizes = self._splitter_sizes()
        rail = self.COLLAPSED_PANEL_WIDTH
        if self._panel_expanded(sizes[0]):
            self._saved_browser_width = sizes[0]
            self.browser.hide()
            self.browser_panel.setMinimumWidth(rail)
            self.browser_panel.setMaximumWidth(rail)
            self._apply_splitter_sizes(rail, sizes[1], sizes[2] + sizes[0] - rail)
            self._set_panel_toggle_appearance(self.btn_toggle_files, expanded=False)
            self.browser.set_browser_controls_visible(False)
        else:
            self.browser.show()
            self.browser_panel.setMinimumWidth(0)
            self.browser_panel.setMaximumWidth(16777215)
            self.browser.setMinimumWidth(250)
            self.browser.setMaximumWidth(400)
            width = self._saved_browser_width or 250
            give_back = min(width, sizes[2])
            self._apply_splitter_sizes(width, sizes[1], sizes[2] - give_back)
            self._set_panel_toggle_appearance(self.btn_toggle_files, expanded=True)
            self.browser.set_browser_controls_visible(True)

    def toggle_middle_panel(self):
        if self._focus_mode_active or self.workspaces.currentIndex() != 4:
            return
        sizes = self._splitter_sizes()
        if self._panel_expanded(sizes[1]):
            self.collapse_middle_panel()
        else:
            self.expand_middle_panel()

    def collapse_middle_panel(self):
        if self._focus_mode_active:
            return
        sizes = self._splitter_sizes()
        if not self._panel_expanded(sizes[1]):
            return
        rail = self.COLLAPSED_PANEL_WIDTH
        self._saved_middle_width = sizes[1]
        self.middle_stack.hide()
        self.middle_panel.setMinimumWidth(rail)
        self.middle_panel.setMaximumWidth(rail)
        self._apply_splitter_sizes(sizes[0], rail, sizes[2] + sizes[1] - rail)
        self._set_panel_toggle_appearance(self.btn_toggle_plates, expanded=False)

    def expand_middle_panel(self):
        if self._focus_mode_active:
            return
        sizes = self._splitter_sizes()
        if self._panel_expanded(sizes[1]):
            return
        self.middle_stack.show()
        self.middle_panel.setMinimumWidth(300)
        self.middle_panel.setMaximumWidth(450)
        width = self._saved_middle_width or 320
        give_back = min(width, sizes[2])
        self._apply_splitter_sizes(sizes[0], width, sizes[2] - give_back)
        self._set_panel_toggle_appearance(self.btn_toggle_plates, expanded=True)

    def toggle_focus_mode(self):
        if not self._focus_mode_active:
            self._saved_splitter_sizes = self._splitter_sizes()
            total = sum(self._saved_splitter_sizes)
            self._apply_splitter_sizes(0, 0, total)
            self._focus_mode_active = True
            self.btn_toggle_files.setEnabled(False)
            self.btn_toggle_plates.setEnabled(False)
        else:
            restore = self._saved_splitter_sizes or [250, 320, 930]
            self._apply_splitter_sizes(*restore)
            self._focus_mode_active = False
            self.btn_toggle_files.setEnabled(True)
            self._sync_workspace_ui(self.workspaces.currentIndex())
            sizes = self._splitter_sizes()
            self._set_panel_toggle_appearance(
                self.btn_toggle_files, expanded=self._panel_expanded(sizes[0])
            )

    def _inject_canvas_configs(self):
        f_cfg = self.config.get("frangi", {})
        frangi = self.workspaces.canvas_frangi
        frangi.p_min_sigma = f_cfg.get("min_sigma", 1)
        frangi.p_max_sigma = f_cfg.get("max_sigma", 5)
        frangi.p_sigma_step = f_cfg.get("sigma_step", 1)
        frangi.p_search_range = f_cfg.get("search_range", 0)
        frangi.p_bridge_gaps = f_cfg.get("bridge_gaps", 1)
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
        self.browser.finish_annotation_requested.connect(self._on_finish_annotation_requested)
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
        self.panel_phenomics.calibration_changed.connect(self._on_metadata_dirty)
        self.panel_phenomics.overlay_labels_changed.connect(self.update_overlay_labels)
        self.panel_phenomics.metadata_dirty_changed.connect(self._on_metadata_dirty)
        self.panel_phenomics.auto_renumber_requested.connect(self._on_auto_renumber_plants)

        if hasattr(self.workspaces.canvas_review, "distance_measured"):
            self.workspaces.canvas_review.distance_measured.connect(self.on_distance_measured)

        self.workspaces.inspector_tab.go_to_metadata_requested.connect(
            self._go_to_plant_metadata
        )

        self.workspaces.report_file_panel.open_requested.connect(
            self.open_task_from_metrics
        )

        self.global_model.register_selection_callback(self._on_model_selection_changed)
        self.global_model.register_data_callback(self._on_model_data_changed)

        self.chk_show_num_global.toggled.connect(self._apply_global_label_preferences)
        self.chk_show_geno_global.toggled.connect(self._apply_global_label_preferences)
        self.rad_fmt_text_global.toggled.connect(self._apply_global_label_preferences)
        self.rad_fmt_num_global.toggled.connect(self._apply_global_label_preferences)
        self._apply_global_label_preferences()

        self._ensure_middle_panel_visible_on_startup()
        self._sync_workspace_ui(self.workspaces.currentIndex())

    def _ensure_middle_panel_visible_on_startup(self):
        if self._focus_mode_active:
            return
        if self.workspaces.currentIndex() == 5:
            return
        if not self._panel_expanded(self.splitter.sizes()[1]):
            self.expand_middle_panel()
        sizes = self._splitter_sizes()
        if sizes[0] == 0:
            self.browser.show()
            width = self._saved_browser_width or 250
            give_back = min(width, sizes[2])
            self._apply_splitter_sizes(width, sizes[1], sizes[2] - give_back)
            self._set_panel_toggle_appearance(self.btn_toggle_files, expanded=True)

    def _on_workspace_changed(self, index):
        self._sync_workspace_ui(index)

    def _sync_workspace_ui(self, index):
        if index in [0, 1, 2, 3]:
            self.ensure_active_plant()

        is_batch = index == 4
        self.btn_toggle_plates.setVisible(is_batch)
        self.btn_toggle_plates.setEnabled(is_batch and not self._focus_mode_active)
        if is_batch:
            self._set_panel_toggle_appearance(
                self.btn_toggle_plates,
                expanded=self._panel_expanded(self.splitter.sizes()[1]),
            )

        self.instance_list.sync_toolbar_for_tab(index)

        if index in [0, 1, 2]:
            self.control_tabs.setEnabled(True)
            self.middle_stack.setCurrentIndex(0)
            self.tool_stack.setCurrentIndex(index)
            if index in [1, 2]:
                self.control_tabs.setCurrentIndex(0)
            if not self._panel_expanded(self.splitter.sizes()[1]) and not self._focus_mode_active:
                self.expand_middle_panel()

            if index == 0:
                self.panel_review.force_mode("SELECT")
                if self.global_model.active_uid:
                    self.workspaces.canvas_review.zoom_to_plant(self.global_model.active_uid)

        elif index == 3:
            self.control_tabs.setEnabled(True)
            self.middle_stack.setCurrentIndex(0)
            self.control_tabs.setCurrentIndex(1)
            if not self._panel_expanded(self.splitter.sizes()[1]) and not self._focus_mode_active:
                self.expand_middle_panel()
            self.sync_inspector()

        elif index == 4:
            self.control_tabs.setEnabled(False)
            self.middle_stack.setCurrentIndex(1)
            if not self._focus_mode_active and self.splitter.sizes()[1] == 0:
                self.expand_middle_panel()
            self.refresh_report_plates_if_active()

        elif index == 5:
            self.control_tabs.setEnabled(False)
            self.middle_stack.setCurrentIndex(0)
            if not self._focus_mode_active:
                sizes = self._splitter_sizes()
                if self._panel_expanded(sizes[1]):
                    self._saved_middle_width = sizes[1]
                self.middle_stack.hide()
                self._apply_splitter_sizes(sizes[0], 0, sizes[2] + sizes[1])

    def _on_metadata_dirty(self):
        self.metadata_dirty = True
        if self.measurements_cache and self.global_model.active_uid:
            uid = self.global_model.active_uid
            if uid in self.measurements_cache:
                captured = self.panel_phenomics.capture_metadata()
                plant_meta = captured.get(
                    uid, self.global_model.get_plants_metadata().get(uid, {})
                )
                self.workspaces.inspector_tab.refresh_plant_metadata(
                    plant_meta,
                    self.measurements_cache[uid],
                    uid,
                )

    def _on_auto_renumber_plants(self):
        if not self.global_model.masks:
            return
        self.global_model.set_task_metadata(
            plants_meta=self.panel_phenomics.capture_metadata(),
        )
        self.global_model.sync_plant_numbers_to_uids(force=True)
        self.panel_phenomics.apply_plant_numbers(self.global_model.get_plants_metadata())
        self.metadata_dirty = True
        self.update_overlay_labels()

    def _is_task_dirty(self):
        return self.global_model.dirty or self.metadata_dirty

    def _resolve_metrics_path(self, task_dir=None, base_name=None):
        task_dir = task_dir or self.current_task_path
        base_name = base_name or self.current_base_name
        if not task_dir or not base_name:
            return None

        out_dir = self.browser.get_effective_output_dir(task_dir)
        path = metrics_path(out_dir, base_name)
        if not path and out_dir != task_dir:
            path = metrics_path(task_dir, base_name)
        return path

    def _resolve_topology_path(self, task_dir=None, base_name=None):
        task_dir = task_dir or self.current_task_path
        base_name = base_name or self.current_base_name
        if not task_dir or not base_name:
            return None

        out_dir = self.browser.get_effective_output_dir(task_dir)
        path = topology_path(out_dir, base_name)
        if not path and out_dir != task_dir:
            path = topology_path(task_dir, base_name)
        return path

    @staticmethod
    def _is_identity_uid_mapping(mapping):
        return not mapping or all(old == new for old, new in mapping.items())

    def _remap_measurements_cache(self, uid_mapping):
        if not self.measurements_cache or not uid_mapping:
            return
        remapped = {}
        for old_uid, new_uid in uid_mapping.items():
            if old_uid in self.measurements_cache:
                remapped[new_uid] = self.measurements_cache[old_uid]
        self.measurements_cache = remapped
        if remapped:
            self.panel_phenomics.btn_export.setEnabled(True)

    def _current_analysis_status_for_cache(self):
        metrics_file = self._resolve_metrics_path()
        if self.measurements_cache or (metrics_file and os.path.exists(metrics_file)):
            return "analyzed"
        if normalize_annotation_status(self.global_model.status) == "completed":
            return "not_analyzed"
        return "not_applicable"

    def refresh_report_plates_if_active(self):
        if self.workspaces.currentIndex() == 4:
            self.workspaces.report_file_panel.refresh_file_list()

    def _on_input_dir_changed(self, path):
        self.config["input_root"] = path
        self.config["database_root"] = path
        self.refresh_report_plates_if_active()

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

    def _go_to_plant_metadata(self):
        self.control_tabs.setCurrentIndex(1)
        if self.splitter.sizes()[1] == 0 and not self._focus_mode_active:
            self.expand_middle_panel()

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
        if model_uids == table_uids and self._is_identity_uid_mapping(uid_mapping):
            return

        mapping_changed = uid_mapping and not self._is_identity_uid_mapping(uid_mapping)
        topology_changed = model_uids != table_uids

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

        if topology_changed:
            self.measurements_cache.clear()
            self.panel_phenomics.reset_measurements()
            self.workspaces.inspector_tab.show_no_measurements_state()
        elif mapping_changed:
            self._remap_measurements_cache(uid_mapping)
            self.sync_inspector()

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
        self.panel_phenomics.btn_measure_tool.setText("Stop measuring")
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
            self.panel_phenomics.btn_measure_tool.setText("Stop measuring")
            self.workspaces.setCurrentIndex(0)
        else:
            self.is_calibrating = False
            self.workspaces.canvas_review.set_calibrating(False)
            self.workspaces.canvas_review.set_mode("SELECT")
            self.panel_phenomics.btn_measure_tool.setText("Check scale on image")

    def validate_and_update_ruler(self):
        self.panel_phenomics.validate_calib_field()
        self.update_canvas_ruler()

    def on_distance_measured(self, pixel_distance):
        if not self.is_calibrating:
            return

        self.panel_phenomics.btn_measure_tool.setChecked(False)
        self.workspaces.canvas_review.set_calibrating(False)
        self.workspaces.canvas_review.set_mode("SELECT")
        self.panel_phenomics.btn_measure_tool.setText("Check scale on image")

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
                "The app requires dots (.) instead of commas (,) for decimal numbers.\n\n"
                "The value has been automatically corrected for you."
            )

        shape = self.global_model.raw_image.shape if self.global_model.raw_image is not None else None
        cm_per_px = self.panel_phenomics.get_cm_per_px(shape)
        self.panel_phenomics.current_cm_per_px = cm_per_px
        self.workspaces.canvas_review.update_scene_ruler(cm_per_px)

    # --- Loading & Threading ---
    def show_loading(self, message):
        self._loading_depth += 1
        if self._loading_depth == 1:
            self._pending_loading_message = message
            self._loading_show_token += 1
            token = self._loading_show_token
            QTimer.singleShot(
                self.LOADING_SHOW_DELAY_MS,
                lambda: self._reveal_loading_if_needed(token),
            )

    def _reveal_loading_if_needed(self, token):
        if token != self._loading_show_token:
            return
        if self._loading_depth == 0:
            return
        self._loading_overlay.show_message(self._pending_loading_message)

    def hide_loading(self):
        if self._loading_depth > 0:
            self._loading_depth -= 1
        if self._loading_depth == 0:
            self._loading_show_token += 1
            self._loading_overlay.hide_overlay()

    def scan_directory(self, folder_path, force_refresh=False):
        self.browser.current_dir = folder_path

        if force_refresh:
            keys_to_delete = [k for k in self.browser.folder_cache if k.startswith(folder_path)]
            for k in keys_to_delete:
                del self.browser.folder_cache[k]

        cached = self.browser.get_cached_contents(folder_path)
        if cached is not None and not force_refresh:
            self.browser.render_contents(cached, use_cache=False)
            return

        out_dir = self.browser.get_effective_output_dir(folder_path)
        fixed_output = self.config.get("output_mode", "task_folder") == "fixed"
        self.show_loading(f"Scanning directory stats...\n{folder_path}")
        worker = ModelWorker(
            self.global_model.scan_directory, folder_path, out_dir, fixed_output
        )
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

        self.browser.render_contents(contents)
        self.refresh_report_plates_if_active()

    def _save_prompt_message(self):
        if self.metadata_dirty and not self.global_model.dirty:
            return "Save unsaved annotation metadata (genotypes, plate fields)?"
        if self.metadata_dirty and self.global_model.dirty:
            return "Save unsaved annotation changes and metadata?"
        return "Save annotation changes?"

    def load_file(self, data):
        if self._is_task_dirty():
            reply = QMessageBox.question(
                self, "Save?",
                self._save_prompt_message(),
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
            )
            if reply == QMessageBox.Yes:
                self._pending_load_data = data
                self.save_task(mark_finished=False)
                return
            if reply == QMessageBox.Cancel:
                self._pending_open_annotation_tab = False
                return

        self._execute_load(data)

    def open_task_from_metrics(self, metrics_path):
        from tabs.report_tab import resolve_task_image_path

        image_path = resolve_task_image_path(metrics_path)
        if not image_path:
            QMessageBox.warning(
                self,
                "Source Image Not Found",
                "Could not resolve the original image for this exported plate.\n"
                f"File: {metrics_path}",
            )
            return
        self._pending_open_annotation_tab = True
        self.load_file({"path": image_path})

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

        self.show_loading(f"Loading {self.current_base_name}...\n(Loading plate data…)")
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
        self.setWindowTitle(
            f"ChronoRoot Annotation Suite | {self.current_base_name} [{self.global_model.status.upper()}]"
        )

        self.measurements_cache.clear()
        self.panel_phenomics.reset_measurements()
        self.workspaces.inspector_tab.show_no_measurements_state()

        self.panel_phenomics.restore_plate_meta(self.global_model.get_plate_meta())

        if self.global_model.masks:
            uids = list(self.global_model.masks.keys())
            self.panel_phenomics.populate_table(
                uids,
                self.config.get("saved_genotypes", []),
                preserved_metadata=self.global_model.get_plants_metadata(),
            )
            self.panel_phenomics.enable_tools(True)
            self.ensure_active_plant()
        else:
            self.panel_phenomics.clear_table()

        # Notify views only after the metadata table matches the freshly loaded
        # model. Doing this earlier makes the table-vs-model sync see stale UIDs
        # and clear measurements_cache before the restore offer runs.
        self.global_model._notify_data_changed()

        self.metadata_dirty = False
        self._offer_analysis_restore()

        if self._pending_open_annotation_tab:
            self._pending_open_annotation_tab = False
            self.workspaces.setCurrentIndex(0)

        self.workspaces.canvas_review.refresh_canvas()
        self.update_canvas_ruler()
        self.workspaces.setCurrentIndex(0)

        self.workspaces.canvas_review.update_info_label()

    def _offer_analysis_restore(self):
        metrics_path = self._resolve_metrics_path()
        if not metrics_path:
            return

        reply = QMessageBox.question(
            self,
            "Existing Analysis",
            f"Found prior analysis export:\n{os.path.basename(metrics_path)}\n\n"
            "• Yes — restore measurements into the inspector\n"
            "• No — delete previous analysis files\n"
            "• Cancel — keep files on disk but do not load them now",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes,
        )

        if reply == QMessageBox.Cancel:
            return
        if reply == QMessageBox.No:
            self._delete_previous_analysis_files(metrics_path)
            return

        self._restore_analysis_from_metrics(metrics_path)

    def _delete_previous_analysis_files(self, metrics_path):
        paths_to_remove = [metrics_path]
        topo_path = self._resolve_topology_path()
        if topo_path:
            paths_to_remove.append(topo_path)

        for path in paths_to_remove:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
            except OSError as exc:
                QMessageBox.warning(self, "Delete Failed", f"Could not remove {path}:\n{exc}")
                return

        self.measurements_cache.clear()
        self.panel_phenomics.reset_measurements()
        self.workspaces.inspector_tab.show_no_measurements_state()
        if self.current_file_path:
            self.browser.update_file_status_in_cache(
                self.current_file_path,
                new_analysis_status="not_analyzed",
            )

    def _restore_analysis_from_metrics(self, metrics_path):
        try:
            restored, warnings = load_measurements_from_metrics_json(
                metrics_path,
                self.global_model.masks.keys(),
                self.global_model.get_plants_metadata(),
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            QMessageBox.warning(self, "Restore Failed", f"Could not load analysis data:\n{exc}")
            return

        if not restored:
            QMessageBox.warning(
                self, "Restore Failed",
                "No measurements could be matched to the current plants."
            )
            return

        self.measurements_cache = restored
        self.panel_phenomics.btn_export.setEnabled(True)
        self.sync_inspector()

        if warnings:
            QMessageBox.information(
                self, "Analysis Restored with Warnings",
                "Measurements were restored, but some plants could not be matched "
                "exactly and were remapped by plant number/genotype:\n\n"
                + "\n".join(f"• {w}" for w in warnings[:12])
                + ("\n• …" if len(warnings) > 12 else "")
            )

    def _on_finish_annotation_requested(self):
        self.save_task(mark_finished=True)

    def save_task(self, mark_finished=False):
        if not self.global_model.masks:
            return

        shape = self.global_model.raw_image.shape if self.global_model.raw_image is not None else None
        self.global_model.set_task_metadata(
            self.panel_phenomics.capture_metadata(),
            self.panel_phenomics.capture_plate_meta(shape),
        )

        self.global_model.callbacks_muted = True
        self.show_loading("Saving progress...\n(Saving masks…)")
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
        self._pending_mark_finished = False
        worker = self.sender()
        if worker in self.active_workers:
            self.active_workers.remove(worker)
            worker.deleteLater()

        if mapping is None:
            self._pending_export_plate_meta = None
            self._pending_measure = None
            return

        new_selection = [mapping[uid] for uid in self.global_model.selected_uids if uid in mapping]
        self.global_model.set_selection(new_selection)
        self._sync_metadata_table_with_model(uid_mapping=mapping)
        self.global_model._notify_data_changed()
        self.ensure_active_plant()
        self.setWindowTitle(
            f"ChronoRoot Annotation Suite | {self.current_base_name} [{self.global_model.status.upper()}]"
        )
        self.metadata_dirty = False

        if self._is_closing:
            self.close()
        elif self._pending_load_data:
            data_to_load = self._pending_load_data
            self._pending_load_data = None
            self._execute_load(data_to_load)
        elif self.current_file_path:
            ann_display = annotation_status_display(self.global_model.status)
            self.browser.update_file_status_in_cache(
                self.current_file_path,
                new_status=ann_display,
                new_plant_count=len(self.global_model.masks),
                new_analysis_status=self._current_analysis_status_for_cache(),
            )

        if self.measurements_cache:
            self.panel_phenomics.btn_export.setEnabled(True)
            self.sync_inspector()

        if self._pending_export_plate_meta is not None:
            plate_meta = self._pending_export_plate_meta
            self._pending_export_plate_meta = None
            self._run_export_worker(plate_meta)
        elif self._pending_measure is not None:
            _, cm_per_px = self._pending_measure
            self._pending_measure = None
            plants_meta = [
                {
                    "uid": uid,
                    "genotype": meta.get("genotype", "Unknown"),
                    "plant_num": meta.get("plant_num", str(uid)),
                }
                for uid, meta in self.panel_phenomics.capture_metadata().items()
            ]
            self._start_measure_worker(plants_meta, cm_per_px)

    # --- Analysis Engine ---
    def run_measurements(self, plants_meta, cm_per_px):
        if self._is_task_dirty():
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "We need to save before doing the measure.\n\n"
                + self._save_prompt_message(),
                QMessageBox.Save | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if reply == QMessageBox.Save:
                self._pending_measure = (plants_meta, cm_per_px)
                self.save_task(mark_finished=False)
                return
            return
        self._start_measure_worker(plants_meta, cm_per_px)

    def _start_measure_worker(self, plants_meta, cm_per_px):
        self.show_loading("Measuring traits…")
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

        self.measurements_cache = scalar_metrics_dict(results_dict)
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
        captured = self.panel_phenomics.capture_metadata()
        plant_meta = captured.get(
            uid, self.global_model.get_plants_metadata().get(uid, {})
        )
        genotype = plant_meta.get("genotype") or data.get("genotype", "Unknown")
        plant_num = plant_meta.get("plant_num") or data.get("plant_num", str(uid))

        bbox = self.global_model.bboxes.get(uid, (0, 0, 0, 0))
        shape = self.global_model.raw_image.shape if self.global_model.raw_image is not None else None
        live_cm_per_px = self.panel_phenomics.get_cm_per_px(shape)
        if live_cm_per_px is None:
            live_cm_per_px = 1.0
        else:
            self.panel_phenomics.current_cm_per_px = live_cm_per_px

        viz_data = analyze_single_plant(
            self.global_model, uid, genotype, plant_num, live_cm_per_px
        )

        self.workspaces.inspector_tab.update_view(
            uid,
            data,
            self.global_model.raw_image,
            bbox,
            live_cm_per_px,
            plant_meta=plant_meta,
            viz_data=viz_data,
        )

    def run_export(self, plate_meta):
        if not self.measurements_cache:
            return

        cm_per_px = plate_meta.get("scale_cm_px")
        if cm_per_px is None:
            QMessageBox.warning(self, "Error", "Invalid Calibration Value.")
            return

        is_completed = normalize_annotation_status(self.global_model.status) == "completed"
        mark_finished = False
        if not is_completed:
            finish_box = QMessageBox(
                QMessageBox.Question,
                "Finish Annotation",
                "We need to finish annotation before exporting.",
                QMessageBox.NoButton,
                self,
            )
            finish_btn = finish_box.addButton("Finish annotation", QMessageBox.AcceptRole)
            finish_box.addButton(QMessageBox.Cancel)
            finish_box.setDefaultButton(finish_btn)
            finish_box.exec_()
            if finish_box.clickedButton() != finish_btn:
                return
            mark_finished = True

        out_dir = self.browser.get_effective_output_dir(self.current_task_path)
        os.makedirs(out_dir, exist_ok=True)
        export_base_name = self.current_base_name
        json_check_path = os.path.join(out_dir, f"{export_base_name}_Metrics.json")
        rsml_check_path = os.path.join(out_dir, f"{export_base_name}_Topology.rsml")

        if os.path.exists(json_check_path) or os.path.exists(rsml_check_path):
            reply = QMessageBox.question(
                self, "Confirm Export Overwrite",
                f"Analysis files for '{export_base_name}' already exist in the output folder.\n\n"
                "Overwrite them with the current measurements?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        self._start_export_pipeline(plate_meta, mark_finished=mark_finished)

    def _start_export_pipeline(self, plate_meta, mark_finished=False):
        if mark_finished or self._is_task_dirty():
            self._pending_export_plate_meta = plate_meta
            self.save_task(mark_finished=mark_finished)
        else:
            self._run_export_worker(plate_meta)

    def _run_export_worker(self, plate_meta):
        out_dir = self.browser.get_effective_output_dir(self.current_task_path)
        os.makedirs(out_dir, exist_ok=True)
        export_base_name = self.current_base_name

        original_img_name = (
            os.path.basename(self.global_model.image_path)
            if self.global_model.image_path else "unknown"
        )
        plate_meta["original_image"] = original_img_name

        shape = self.global_model.raw_image.shape if self.global_model.raw_image is not None else None
        cm_per_px = plate_meta.get("scale_cm_px")
        if cm_per_px is None:
            cm_per_px = self.panel_phenomics.get_cm_per_px(shape)
        plants_meta = []
        for row_meta in self.panel_phenomics.capture_metadata().items():
            uid, meta = row_meta
            plants_meta.append({
                "uid": uid,
                "genotype": meta.get("genotype", "Unknown"),
                "plant_num": meta.get("plant_num", str(uid)),
            })

        self.show_loading("Exporting analysis…")
        worker = ModelWorker(self._export_analysis_bundle, out_dir, export_base_name, plate_meta, plants_meta, cm_per_px)
        self.active_workers.add(worker)
        worker.finished.connect(self._on_export_finished)
        worker.error.connect(self._on_thread_error)
        worker.start()

    def _export_analysis_bundle(self, out_dir, export_base_name, plate_meta, plants_meta, cm_per_px):
        full_results = extract_plate_metrics(self.global_model, plants_meta, cm_per_px)
        export_rsml_and_json(out_dir, export_base_name, plate_meta, full_results)
        return scalar_metrics_dict(full_results)

    def _on_export_finished(self, scalar_results):
        self.hide_loading()
        worker = self.sender()
        if worker in self.active_workers:
            self.active_workers.remove(worker)
            worker.deleteLater()

        if scalar_results:
            self.measurements_cache = scalar_results
            self.panel_phenomics.btn_export.setEnabled(True)

        out_dir = self.browser.get_effective_output_dir(self.current_task_path)

        if self.current_file_path:
            self.browser.update_file_status_in_cache(
                self.current_file_path,
                new_analysis_status="analyzed",
            )

        self.workspaces.report_tab.refresh_file_list()
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
        self._pending_export_plate_meta = None
        self._pending_measure = None

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

        if self._is_task_dirty():
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                self._save_prompt_message() + "\nDo you want to save before exiting?",
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
    window = ChronoRootAnnotationSuite()
    window.show()
    sys.exit(app.exec_())
