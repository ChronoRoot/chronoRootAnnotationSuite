import os
import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
    QPushButton, QComboBox, QFormLayout, QFileDialog, QMessageBox,
    QGroupBox, QProgressDialog, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QAbstractItemView, QRadioButton, QButtonGroup,
    QTextEdit, QDoubleSpinBox, QStackedWidget, QListWidget,
    QListWidgetItem, QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFontMetrics

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from core import convex_hull
from core import statistics as stats_module
from components.file_browser import qt_display_text
from components.ui_help import HELP_BATCH, show_help

METRIC_MAPPING = {
    "mr_length_mm": "Main Root Length (mm)",
    "lr_length_mm": "Lateral Root Length (mm)",
    "tr_length_mm": "Total Root Length (mm)",
    "lr_count": "Lateral Root Count",
    "lr_density_cm": "LR Density (LRs/cm)",
    "mr_over_tr_ratio": "MR/TR Ratio",
    "hull_area_mm2": "Convex Hull Area (mm²)",
    "hull_width_mm": "Convex Hull Width (mm)",
    "hull_height_mm": "Convex Hull Height (mm)",
    "root_density_mm_mm2": "Root Density (mm/mm²)",
    "aspect_ratio": "Aspect Ratio (H/W)",
    "tip_angle_deg": "Mean Tip Angle (°)",
    "emergence_angle_deg": "Mean Emergence Angle (°)",
}

PLOT_TYPES = [
    "Box Plot",
    "Swarm Plot",
    "Violin Plot",
    "Violin + Swarm Plot",
    "Line Plot (Means)",
    "Qualitative Atlas",
]

QUALITATIVE_PLOT_TYPE = "Qualitative Atlas"

AXIS_OPTIONS = ["timepoint", "condition", "genotype"]
HUE_OPTIONS = ["None"] + AXIS_OPTIONS
STAT_FACTOR_OPTIONS = ["genotype", "condition", "timepoint"]
STAT_STRATIFY_OPTIONS = ["(none)", "condition", "timepoint", "genotype"]
ERROR_BAR_OPTIONS = [
    "Standard Error (SE)",
    "Standard Deviation (SD)",
    "95% Confidence Interval (CI)",
]
ERROR_BAR_MAP = {
    "95% Confidence Interval (CI)": ("ci", 95),
    "Standard Error (SE)": "se",
    "Standard Deviation (SD)": "sd",
}

TEST_OPTIONS = ["auto", "mannwhitney", "kruskal", "anova"]

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def peek_metrics_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "plate_id": data.get("plate_id", ""),
        "condition": data.get("condition", ""),
        "timepoint": data.get("timepoint", ""),
        "plant_count": len(data.get("plants", [])),
        "original_image": data.get("original_image", ""),
        "json_dir": os.path.dirname(path),
    }


def resolve_task_image_path(metrics_path):
    peek = peek_metrics_metadata(metrics_path)
    json_dir = peek["json_dir"]
    original = peek.get("original_image", "")
    candidates = []
    if original:
        candidates.append(os.path.join(json_dir, original))
    base = os.path.basename(metrics_path).replace("_Metrics.json", "")
    for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        candidates.append(os.path.join(json_dir, base + ext))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _metrics_output_dir(main_window):
    if hasattr(main_window, "browser") and hasattr(main_window.browser, "out_dir"):
        browser = main_window.browser
        if getattr(browser, "output_mode", "task_folder") == "fixed":
            return browser.out_dir
        return getattr(browser, "root_dir", browser.out_dir)
    return getattr(main_window, "out_dir", ".")


def _metrics_search_roots(main_window):
    roots = []
    if hasattr(main_window, "browser"):
        browser = main_window.browser
        for candidate in (
            getattr(browser, "root_dir", None),
            getattr(browser, "in_dir", None),
            getattr(browser, "current_dir", None),
        ):
            if candidate and os.path.isdir(candidate):
                abs_path = os.path.abspath(candidate)
                if abs_path not in roots:
                    roots.append(abs_path)

    extra = []
    if hasattr(main_window, "config"):
        extra = main_window.config.get("report_search_roots", [])
    for candidate in extra:
        if candidate and os.path.isdir(candidate):
            abs_path = os.path.abspath(candidate)
            if abs_path not in roots:
                roots.append(abs_path)

    if not roots:
        out_dir = _metrics_output_dir(main_window)
        if os.path.isdir(out_dir):
            roots.append(os.path.abspath(out_dir))
    return roots


def discover_metrics_files(main_window):
    discovered = {}
    for root in _metrics_search_roots(main_window):
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if not filename.endswith("_Metrics.json"):
                    continue
                full_path = os.path.join(dirpath, filename)
                if full_path in discovered:
                    continue
                rel_folder = os.path.relpath(dirpath, root)
                if rel_folder == ".":
                    display = filename
                else:
                    display = f"{rel_folder}/{filename}"
                discovered[full_path] = {
                    "display": display,
                    "relative_folder": rel_folder if rel_folder != "." else "",
                    "search_root": root,
                }
    return discovered


METADATA_COLOR_PALETTES = {
    "genotype": "tab10",
    "timepoint": "husl",
    "condition": "Set2",
}


def _category_order(df, column):
    if column not in df.columns:
        return []
    series = df[column].dropna()
    if hasattr(series.dtype, "categories"):
        return list(series.cat.categories)
    values = series.unique().tolist()
    values.sort(key=natural_sort_key)
    return values


def build_metadata_color_map(df, column):
    if column not in df.columns:
        return {}
    values = _category_order(df, column)
    if not values:
        return {}
    palette_name = METADATA_COLOR_PALETTES.get(column, "tab10")
    colors = sns.color_palette(palette_name, n_colors=len(values))
    return {value: colors[i] for i, value in enumerate(values)}


def build_all_metadata_color_maps(df):
    return {
        column: build_metadata_color_map(df, column)
        for column in METADATA_COLOR_PALETTES
        if column in df.columns
    }


def _palette_for_column(df, column, color_maps=None):
    if color_maps and column in color_maps:
        return color_maps[column]
    return build_metadata_color_map(df, column)


def _list_palette_for_order(color_map, order):
    return [color_map[value] for value in order if value in color_map]


def _neutral_violin_palette(color_map, fill=0.88):
    return {key: (fill, fill, fill) for key in color_map}


def _base_categorical_kwargs(df, x_var, y_var, ax, hue_var=None):
    kwargs = dict(data=df, x=x_var, y=y_var, ax=ax)
    x_order = _category_order(df, x_var)
    if x_order:
        kwargs["order"] = x_order
    if hue_var:
        kwargs["hue"] = hue_var
        hue_order = _category_order(df, hue_var)
        if hue_order:
            kwargs["hue_order"] = hue_order
    return kwargs


def render_quantitative_plot(ax, df, plot_type, y_var, x_var, hue_var, error_bar=None,
                             color_maps=None):
    hue_var = None if not hue_var or hue_var == "None" else hue_var
    x_order = _category_order(df, x_var)
    hue_order = _category_order(df, hue_var) if hue_var else []
    hue_palette = _palette_for_column(df, hue_var, color_maps) if hue_var else {}
    x_palette = _palette_for_column(df, x_var, color_maps)

    common = _base_categorical_kwargs(df, x_var, y_var, ax, hue_var)

    if plot_type == "Box Plot":
        if hue_var:
            sns.boxplot(**common, palette=hue_palette, width=0.5, fliersize=3)
        else:
            sns.boxplot(
                **common, width=0.5, fliersize=3,
                palette=_list_palette_for_order(x_palette, x_order),
            )
    elif plot_type == "Swarm Plot":
        if hue_var:
            sns.swarmplot(**common, dodge=True, size=4, palette=hue_palette)
        else:
            sns.swarmplot(
                **common, dodge=False, size=4,
                palette=_list_palette_for_order(x_palette, x_order),
            )
    elif plot_type == "Violin Plot":
        if hue_var:
            sns.violinplot(
                **common, palette=hue_palette,
                inner="quartile", density_norm="width",
            )
        else:
            sns.violinplot(
                **common,
                palette=_list_palette_for_order(x_palette, x_order),
                inner="quartile", density_norm="width",
            )
    elif plot_type == "Violin + Swarm Plot":
        violin_kwargs = _base_categorical_kwargs(df, x_var, y_var, ax, hue_var)
        violin_kwargs.update(
            inner=None, density_norm="width", width=0.8, linewidth=1,
        )
        if hue_var:
            violin_kwargs["palette"] = _neutral_violin_palette(hue_palette)
        else:
            violin_kwargs["palette"] = _list_palette_for_order(x_palette, x_order)
        sns.violinplot(**violin_kwargs)

        swarm_kwargs = _base_categorical_kwargs(df, x_var, y_var, ax, hue_var)
        swarm_kwargs.update(
            dodge=bool(hue_var),
            size=4,
            alpha=0.85,
            legend=False,
        )
        if hue_var:
            swarm_kwargs["palette"] = hue_palette
        else:
            swarm_kwargs["palette"] = _list_palette_for_order(x_palette, x_order)
        sns.swarmplot(**swarm_kwargs)
    elif plot_type == "Line Plot (Means)":
        err_val = ERROR_BAR_MAP.get(error_bar or "Standard Error (SE)", "se")
        line_kwargs = dict(
            data=df, x=x_var, y=y_var, ax=ax,
            marker="o", err_style="bars", errorbar=err_val, sort=False,
        )
        if hue_var:
            line_kwargs["hue"] = hue_var
            if hue_order:
                line_kwargs["hue_order"] = hue_order
            sns.lineplot(**line_kwargs, palette=hue_palette)
        else:
            sns.lineplot(**line_kwargs, palette=x_palette)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

    ax.set_title(
        f"{y_var} by {x_var.capitalize()}"
        + (f" (Grouped by {hue_var.capitalize()})" if hue_var else ""),
        pad=15,
        fontweight="bold",
    )
    ax.set_xlabel(x_var.capitalize(), fontweight="bold")
    ax.set_ylabel(y_var, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    if hue_var:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(
                by_label.values(), by_label.keys(),
                title=hue_var.capitalize(), bbox_to_anchor=(1.05, 1), loc="upper left",
            )


class ReportFileListPanel(QWidget):
    """Metrics file picker with folder tree and metadata columns."""

    load_requested = pyqtSignal()
    open_requested = pyqtSignal(str)

    COL_INCLUDE = 0
    COL_NAME = 1
    COL_PLATE = 2
    COL_CONDITION = 3
    COL_TIMEPOINT = 4
    COL_PLANTS = 5
    COL_FILE = 6

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._block_checks = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        sel_btns = QHBoxLayout()
        sel_btns.addWidget(QLabel("<b>Processed Plates</b>"))
        self.btn_sel_all = QPushButton("All")
        self.btn_sel_all.setMaximumHeight(24)
        #self.btn_sel_all.setFlat(True)
        self.btn_sel_all.clicked.connect(lambda: self._set_all_checks(Qt.Checked))
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setMaximumHeight(24)
        #self.btn_clear.setFlat(True)
        self.btn_clear.clicked.connect(lambda: self._set_all_checks(Qt.Unchecked))
        sel_btns.addWidget(self.btn_sel_all)
        sel_btns.addWidget(self.btn_clear)
        layout.addLayout(sel_btns)

        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(
            ["Include", "Name", "Plate ID", "Condition", "Timepoint", "# Plants", "File"]
        )
        self.file_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_tree.setAlternatingRowColors(True)
        self.file_tree.itemChanged.connect(self._on_item_changed)
        self.file_tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        header = self.file_tree.header()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        layout.addWidget(self.file_tree)

        action_btns = QHBoxLayout()
        self.btn_open = QPushButton("Open Task")
        self.btn_open.clicked.connect(self._open_selected_task)
        self.btn_load_data = QPushButton("Load Selected Data")
        self.btn_load_data.setStyleSheet(
            "background-color: #007bff; color: white; font-weight: bold;"
        )
        self.btn_load_data.clicked.connect(self.load_requested.emit)
        action_btns.addWidget(self.btn_open)
        action_btns.addWidget(self.btn_load_data)
        layout.addLayout(action_btns)

    def _set_all_checks(self, state):
        self._block_checks = True
        for i in range(self.file_tree.topLevelItemCount()):
            self._set_subtree_checks(self.file_tree.topLevelItem(i), state)
        self._block_checks = False

    def _set_subtree_checks(self, item, state):
        item.setCheckState(self.COL_INCLUDE, state)
        for i in range(item.childCount()):
            self._set_subtree_checks(item.child(i), state)

    def _on_item_changed(self, item, column):
        if self._block_checks or column != self.COL_INCLUDE:
            return
        role = item.data(self.COL_INCLUDE, Qt.UserRole)
        if not role:
            return
        self._block_checks = True
        if role[0] == "folder":
            state = item.checkState(self.COL_INCLUDE)
            for i in range(item.childCount()):
                self._set_subtree_checks(item.child(i), state)
        else:
            parent = item.parent()
            while parent:
                self._update_folder_check_state(parent)
                parent = parent.parent()
        self._block_checks = False

    def _update_folder_check_state(self, folder_item):
        checked, total = self._count_file_checks(folder_item)
        if total == 0:
            return
        if checked == 0:
            folder_item.setCheckState(self.COL_INCLUDE, Qt.Unchecked)
        elif checked == total:
            folder_item.setCheckState(self.COL_INCLUDE, Qt.Checked)
        else:
            folder_item.setCheckState(self.COL_INCLUDE, Qt.PartiallyChecked)

    def _count_file_checks(self, item):
        role = item.data(self.COL_INCLUDE, Qt.UserRole)
        if role and role[0] == "file":
            return (
                (1, 1) if item.checkState(self.COL_INCLUDE) == Qt.Checked else (0, 1)
            )
        checked = 0
        total = 0
        for i in range(item.childCount()):
            c_checked, c_total = self._count_file_checks(item.child(i))
            checked += c_checked
            total += c_total
        return checked, total

    def _on_item_double_clicked(self, item, _column):
        role = item.data(self.COL_INCLUDE, Qt.UserRole)
        if role and role[0] == "file":
            self.open_requested.emit(role[1])

    def _open_selected_task(self):
        path = self._selected_file_path()
        if not path:
            QMessageBox.warning(self, "No File", "Select an exported plate to open.")
            return
        self.open_requested.emit(path)

    def _selected_file_path(self):
        items = self.file_tree.selectedItems()
        for item in items:
            role = item.data(self.COL_INCLUDE, Qt.UserRole)
            if role and role[0] == "file":
                return role[1]
        for path in self.get_checked_file_paths():
            return path
        return None

    def get_checked_file_paths(self):
        paths = []
        for i in range(self.file_tree.topLevelItemCount()):
            self._collect_checked_files(self.file_tree.topLevelItem(i), paths)
        return paths

    def _collect_checked_files(self, item, paths):
        role = item.data(self.COL_INCLUDE, Qt.UserRole)
        if role and role[0] == "file":
            if item.checkState(self.COL_INCLUDE) == Qt.Checked:
                paths.append(role[1])
            return
        for i in range(item.childCount()):
            self._collect_checked_files(item.child(i), paths)

    def refresh_file_list(self):
        self._block_checks = True
        self.file_tree.clear()
        metrics_files = discover_metrics_files(self.main_window)
        folder_nodes = {}

        for full_path in sorted(
            metrics_files.keys(),
            key=lambda p: natural_sort_key(metrics_files[p]["display"]),
        ):
            meta = metrics_files[full_path]
            rel = meta.get("relative_folder", "")
            parts = [p for p in rel.replace("\\", "/").split("/") if p] if rel else []

            parent = self.file_tree.invisibleRootItem()
            current_path = ""
            for part in parts:
                current_path = f"{current_path}/{part}" if current_path else part
                if current_path not in folder_nodes:
                    folder_item = QTreeWidgetItem(parent)
                    folder_item.setText(self.COL_NAME, qt_display_text(part))
                    folder_item.setFlags(
                        folder_item.flags()
                        | Qt.ItemIsUserCheckable
                        | Qt.ItemIsEnabled
                        | Qt.ItemIsSelectable
                    )
                    folder_item.setCheckState(self.COL_INCLUDE, Qt.Unchecked)
                    folder_item.setData(self.COL_INCLUDE, Qt.UserRole, ("folder", current_path))
                    folder_nodes[current_path] = folder_item
                    parent = folder_item
                else:
                    parent = folder_nodes[current_path]

            try:
                peek = peek_metrics_metadata(full_path)
            except (OSError, json.JSONDecodeError):
                peek = {
                    "plate_id": "?",
                    "condition": "?",
                    "timepoint": "?",
                    "plant_count": 0,
                }

            file_item = QTreeWidgetItem(parent)
            file_item.setFlags(
                file_item.flags()
                | Qt.ItemIsUserCheckable
                | Qt.ItemIsEnabled
                | Qt.ItemIsSelectable
            )
            file_item.setCheckState(self.COL_INCLUDE, Qt.Unchecked)
            file_item.setData(self.COL_INCLUDE, Qt.UserRole, ("file", full_path))
            file_item.setText(self.COL_NAME, qt_display_text(os.path.basename(full_path)))
            file_item.setText(self.COL_PLATE, str(peek.get("plate_id", "")))
            file_item.setText(self.COL_CONDITION, str(peek.get("condition", "")))
            file_item.setText(self.COL_TIMEPOINT, str(peek.get("timepoint", "")))
            file_item.setText(self.COL_PLANTS, str(peek.get("plant_count", 0)))
            file_item.setText(self.COL_FILE, meta.get("display", os.path.basename(full_path)))
            file_item.setToolTip(self.COL_NAME, full_path)

        self.file_tree.expandToDepth(1)
        self._block_checks = False


class ComparisonReportTab(QWidget):
    def __init__(self, main_window, file_panel):
        super().__init__()
        self.main_window = main_window
        self.file_panel = file_panel
        self.df = pd.DataFrame()
        self.color_maps = {}
        self.report_queue = []
        self.file_panel.load_requested.connect(self.load_selected_data)
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        self.explorer_panel = QWidget()
        self.explorer_panel.setMinimumWidth(280)
        controls_layout = QVBoxLayout(self.explorer_panel)
        controls_layout.setContentsMargins(6, 6, 6, 6)
        controls_layout.setSpacing(4)

        header_row = QHBoxLayout()
        header_row.addWidget(QLabel("<b>Report Explorer</b>"))
        
        self.btn_stats_help = QPushButton("Help")
        self.btn_stats_help.setToolTip("Open report and statistics help")
        self.btn_stats_help.clicked.connect(self._show_report_help)
        header_row.addWidget(self.btn_stats_help)
        
        self.btn_focus_plot = QPushButton("Focus on Plot")
        self.btn_focus_plot.setToolTip("Hide side panels to maximize the plot area")
        self.btn_focus_plot.clicked.connect(self._toggle_focus_mode)
        header_row.addWidget(self.btn_focus_plot)
        controls_layout.addLayout(header_row)

        self.lbl_batch_workflow = QLabel(
            "Check plates → Load Selected Data → tune plot → Add to Report → Generate"
        )
        self.lbl_batch_workflow.setWordWrap(True)
        self.lbl_batch_workflow.setStyleSheet("color: #555; font-size: 11px; padding: 2px 0;")
        controls_layout.addWidget(self.lbl_batch_workflow)

        plot_group = QGroupBox("Plot")
        plot_form = QFormLayout(plot_group)
        plot_form.setContentsMargins(6, 6, 6, 6)
        plot_form.setSpacing(4)

        self.cb_plot_type = QComboBox()
        self.cb_plot_type.addItems(PLOT_TYPES)
        self.cb_plot_type.currentIndexChanged.connect(self.on_plot_type_changed)

        self.cb_error_bar = QComboBox()
        self.cb_error_bar.addItems(ERROR_BAR_OPTIONS)
        self.cb_error_bar.setEnabled(False)
        self.cb_error_bar.currentIndexChanged.connect(self._on_plot_settings_changed)

        self.cb_y_metric = QComboBox()
        self.cb_y_metric.currentIndexChanged.connect(self._on_plot_settings_changed)

        self.cb_x_axis = QComboBox()
        self.cb_x_axis.addItems(AXIS_OPTIONS)
        self.cb_x_axis.currentIndexChanged.connect(self._on_plot_settings_changed)

        self.cb_hue = QComboBox()
        self.cb_hue.addItems(HUE_OPTIONS)
        self.cb_hue.setCurrentText("genotype")
        self.cb_hue.currentIndexChanged.connect(self._on_plot_settings_changed)

        plot_form.addRow("Plot Type:", self.cb_plot_type)
        plot_form.addRow("Line Error Bars:", self.cb_error_bar)
        plot_form.addRow("Y-Axis (Metric):", self.cb_y_metric)
        plot_form.addRow("X-Axis (Group):", self.cb_x_axis)
        plot_form.addRow("Hue (Color):", self.cb_hue)
        controls_layout.addWidget(plot_group)

        self.qual_group = QGroupBox("Atlas Columns")
        qual_form = QFormLayout(self.qual_group)
        qual_form.setContentsMargins(6, 6, 6, 6)
        qual_form.setSpacing(4)
        self.lbl_qual_rows = QLabel("Rows: timepoint (fixed)")
        self.col_group = QButtonGroup(self)
        self.rad_col_condition = QRadioButton("Condition")
        self.rad_col_genotype = QRadioButton("Genotype")
        self.rad_col_combined = QRadioButton("Condition | Genotype")
        self.rad_col_condition.setChecked(True)
        for rad in (self.rad_col_condition, self.rad_col_genotype, self.rad_col_combined):
            self.col_group.addButton(rad)
            rad.toggled.connect(self._on_plot_settings_changed)
        qual_form.addRow(self.lbl_qual_rows)
        qual_form.addRow(self.rad_col_condition)
        qual_form.addRow(self.rad_col_genotype)
        qual_form.addRow(self.rad_col_combined)
        self.qual_group.setVisible(False)
        controls_layout.addWidget(self.qual_group)

        self.stats_group = QGroupBox("Statistics")
        stats_form = QFormLayout(self.stats_group)
        stats_form.setContentsMargins(6, 6, 6, 6)
        stats_form.setSpacing(4)

        self.cb_stat_compare = QComboBox()
        self.cb_stat_compare.addItems(STAT_FACTOR_OPTIONS)

        self.cb_stat_within = QComboBox()
        self.cb_stat_within.addItems(STAT_STRATIFY_OPTIONS)

        self.cb_stat_and_within = QComboBox()
        self.cb_stat_and_within.addItems(STAT_STRATIFY_OPTIONS)
        self.cb_stat_and_within.setEnabled(False)

        self.cb_stat_test = QComboBox()
        for key in TEST_OPTIONS:
            self.cb_stat_test.addItem(stats_module.test_display_name(key), key)

        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.001, 0.5)
        self.spin_alpha.setSingleStep(0.01)
        self.spin_alpha.setDecimals(3)
        self.spin_alpha.setValue(0.05)

        stats_form.addRow("Compare groups by:", self.cb_stat_compare)
        stats_form.addRow("Within each:", self.cb_stat_within)
        stats_form.addRow("And within:", self.cb_stat_and_within)
        stats_form.addRow("Test:", self.cb_stat_test)
        stats_form.addRow("Alpha:", self.spin_alpha)
        controls_layout.addWidget(self.stats_group)

        self.lbl_test_hint = QLabel()
        self.lbl_test_hint.setWordWrap(True)
        self.lbl_test_hint.setStyleSheet("color: #666; font-size: 11px;")
        controls_layout.addWidget(self.lbl_test_hint)

        queue_group = QGroupBox("Report Queue")
        queue_layout = QVBoxLayout(queue_group)
        queue_layout.setContentsMargins(6, 6, 6, 6)

        add_row = QHBoxLayout()
        self.btn_add_to_report = QPushButton("Add to Report")
        self.btn_add_to_report.clicked.connect(self.add_current_to_report)
        self.btn_remove_from_report = QPushButton("Remove")
        self.btn_remove_from_report.clicked.connect(self.remove_selected_from_report)
        add_row.addWidget(self.btn_add_to_report)
        add_row.addWidget(self.btn_remove_from_report)
        queue_layout.addLayout(add_row)

        self.list_report_queue = QListWidget()
        queue_layout.addWidget(self.list_report_queue)
        controls_layout.addWidget(queue_group)

        self.btn_generate_report = QPushButton("Generate Report")
        self.btn_generate_report.setStyleSheet(
            "background-color: #28a745; color: white; font-weight: bold; padding: 10px;"
        )
        self.btn_generate_report.clicked.connect(self.generate_report)
        self.btn_generate_report.setEnabled(False)
        controls_layout.addWidget(self.btn_generate_report)

        self.btn_export_csv = QPushButton("Export Table (.CSV)")
        self.btn_export_csv.clicked.connect(self.export_data_csv)
        self.btn_export_csv.setEnabled(False)
        controls_layout.addWidget(self.btn_export_csv)

        canvas_column = QWidget()
        canvas_column_layout = QVBoxLayout(canvas_column)
        canvas_column_layout.setContentsMargins(0, 0, 0, 0)
        canvas_column_layout.setSpacing(0)

        canvas_splitter = QSplitter(Qt.Vertical)

        canvas_panel = QWidget()
        canvas_layout = QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas_stack = QStackedWidget()

        self.empty_canvas_page = QWidget()
        empty_layout = QVBoxLayout(self.empty_canvas_page)
        self.lbl_empty_plot = QLabel(
            "<b>No data loaded.</b><br><br>"
            "Check exported plates in the middle panel and click "
            "<i>Load Selected Data</i>."
        )
        self.lbl_empty_plot.setWordWrap(True)
        self.lbl_empty_plot.setAlignment(Qt.AlignCenter)
        self.lbl_empty_plot.setStyleSheet("color: #555; font-size: 14px; padding: 24px;")
        empty_layout.addStretch()
        empty_layout.addWidget(self.lbl_empty_plot)
        empty_layout.addStretch()

        self.plot_canvas_page = QWidget()
        plot_layout = QVBoxLayout(self.plot_canvas_page)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.figure.patch.set_facecolor("#f4f4f4")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, stretch=1)

        self.canvas_stack.addWidget(self.empty_canvas_page)
        self.canvas_stack.addWidget(self.plot_canvas_page)
        canvas_layout.addWidget(self.canvas_stack)

        self.stats_results_panel = QGroupBox("Statistical Results")
        stats_results_layout = QVBoxLayout(self.stats_results_panel)
        stats_results_layout.setContentsMargins(6, 6, 6, 6)
        self.txt_stat_results = QTextEdit()
        self.txt_stat_results.setReadOnly(True)
        self.txt_stat_results.setStyleSheet("font-family: monospace; font-size: 12px;")
        stats_results_layout.addWidget(self.txt_stat_results)
        self.stats_results_panel.setMinimumHeight(100)
        self.stats_results_panel.setMaximumHeight(220)
        self.stats_results_panel.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Preferred,
        )

        canvas_splitter.addWidget(canvas_panel)
        canvas_splitter.addWidget(self.stats_results_panel)
        canvas_splitter.setStretchFactor(0, 3)
        canvas_splitter.setStretchFactor(1, 1)
        canvas_column_layout.addWidget(canvas_splitter)

        splitter.addWidget(self.explorer_panel)
        splitter.addWidget(canvas_column)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])
        layout.addWidget(splitter)

        self.cb_stat_compare.currentIndexChanged.connect(self._refresh_stratify_options)
        self.cb_stat_compare.currentIndexChanged.connect(self._on_stat_settings_changed)
        self.cb_stat_within.currentIndexChanged.connect(self._on_within_changed)
        self.cb_stat_and_within.currentIndexChanged.connect(self._on_stat_settings_changed)
        self.cb_stat_test.currentIndexChanged.connect(self._on_stat_settings_changed)
        self.spin_alpha.valueChanged.connect(self._on_stat_settings_changed)

        self._setup_stat_test_tooltips()
        sns.set_theme(style="whitegrid")

    def _setup_stat_test_tooltips(self):
        for i in range(self.cb_stat_test.count()):
            key = self.cb_stat_test.itemData(i)
            tip = stats_module.TEST_TOOLTIPS.get(key, "")
            self.cb_stat_test.setItemData(i, tip, Qt.ToolTipRole)

    def _show_report_help(self):
        show_help(self, "Report Help", HELP_BATCH)

    def _toggle_focus_mode(self):
        if hasattr(self.main_window, "toggle_focus_mode"):
            self.main_window.toggle_focus_mode()
            active = getattr(self.main_window, "_focus_mode_active", False)
            self.btn_focus_plot.setText("Exit Focus" if active else "Focus on Plot")

    def _stat_test_key(self):
        key = self.cb_stat_test.currentData()
        return key if key else self.cb_stat_test.currentText()

    def _format_stat_config(self, entry=None):
        if entry:
            return stats_module.format_stat_config(
                entry.get("stat_compare", ""),
                entry.get("stat_within"),
                entry.get("stat_and_within"),
                entry.get("stat_test", "auto"),
                entry.get("stat_alpha", 0.05),
            )
        settings = self._current_stat_settings()
        return stats_module.format_stat_config(
            settings["stat_compare"],
            settings["stat_within"],
            settings["stat_and_within"],
            settings["stat_test"],
            settings["stat_alpha"],
        )

    def _count_compare_groups(self):
        if self.df.empty:
            return 0
        col = self.cb_stat_compare.currentText()
        if col not in self.df.columns:
            return 0
        return self.df[col].dropna().nunique()

    def _update_test_hint(self):
        if self._is_qualitative_plot():
            return
        n_groups = self._count_compare_groups()
        self.lbl_test_hint.setText(
            stats_module.describe_test_choice(self._stat_test_key(), n_groups)
        )

    def _get_col_group(self):
        if self.rad_col_genotype.isChecked():
            return "genotype"
        if self.rad_col_combined.isChecked():
            return "combined"
        return "condition"

    def _col_group_label(self):
        if self.rad_col_genotype.isChecked():
            return "genotype"
        if self.rad_col_combined.isChecked():
            return "condition | genotype"
        return "condition"

    def _is_qualitative_plot(self):
        return self.cb_plot_type.currentText() == QUALITATIVE_PLOT_TYPE

    def _get_output_dir(self):
        return _metrics_output_dir(self.main_window)

    def _discover_metrics_files(self):
        return discover_metrics_files(self.main_window)

    def refresh_file_list(self):
        self.file_panel.refresh_file_list()

    def _stratify_value(self, combo):
        text = combo.currentText()
        if text == "(none)":
            return None
        return text

    def _refresh_stratify_options(self):
        compare = self.cb_stat_compare.currentText()
        for combo in (self.cb_stat_within, self.cb_stat_and_within):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(none)")
            for col in STAT_FACTOR_OPTIONS:
                if col != compare:
                    combo.addItem(col)
            idx = combo.findText(current)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.blockSignals(False)
        self._on_within_changed()

    def _on_within_changed(self):
        has_within = self.cb_stat_within.currentText() != "(none)"
        self.cb_stat_and_within.setEnabled(has_within)
        compare = self.cb_stat_compare.currentText()
        within = self.cb_stat_within.currentText()
        current = self.cb_stat_and_within.currentText()
        self.cb_stat_and_within.blockSignals(True)
        self.cb_stat_and_within.clear()
        self.cb_stat_and_within.addItem("(none)")
        if has_within:
            for col in STAT_FACTOR_OPTIONS:
                if col not in (compare, within):
                    self.cb_stat_and_within.addItem(col)
        idx = self.cb_stat_and_within.findText(current)
        self.cb_stat_and_within.setCurrentIndex(idx if idx >= 0 else 0)
        self.cb_stat_and_within.blockSignals(False)
        self._on_stat_settings_changed()

    def _on_plot_settings_changed(self):
        self.update_plot()

    def _on_stat_settings_changed(self):
        self._update_test_hint()
        self.update_inline_stats()

    def on_plot_type_changed(self):
        plot_type = self.cb_plot_type.currentText()
        is_line_plot = "Line Plot" in plot_type
        is_atlas = plot_type == QUALITATIVE_PLOT_TYPE
        self.cb_error_bar.setEnabled(is_line_plot)
        self.cb_y_metric.setEnabled(not is_atlas)
        self.cb_x_axis.setEnabled(not is_atlas)
        self.cb_hue.setEnabled(not is_atlas)
        self.qual_group.setVisible(is_atlas)
        for widget in (
            self.stats_group, self.lbl_test_hint, self.stats_results_panel,
            self.btn_stats_help,
        ):
            widget.setVisible(not is_atlas)
        if not is_atlas:
            self.update_inline_stats()
            self._update_test_hint()
        self.update_plot()

    def _get_clean_dataframe(self):
        meta_cols = [
            "plate_id", "condition", "timepoint", "genotype", "plant_num",
            "relative_folder", "source_file",
        ]
        metric_cols = list(METRIC_MAPPING.values())
        desired_cols = [c for c in meta_cols + metric_cols if c in self.df.columns]
        return self.df[desired_cols].copy()

    def _set_data_loaded_state(self, enabled):
        self.btn_export_csv.setEnabled(enabled)
        self.btn_generate_report.setEnabled(enabled)
        self.btn_add_to_report.setEnabled(enabled)
        self.canvas_stack.setCurrentIndex(1 if enabled else 0)

    def load_selected_data(self):
        selected_paths = self.file_panel.get_checked_file_paths()
        if not selected_paths:
            QMessageBox.warning(
                self, "Selection Empty",
                "Please check at least one exported plate to include.",
            )
            return

        all_plants_data = []
        metrics_files = self._discover_metrics_files()

        for file_path in selected_paths:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                file_meta = metrics_files.get(file_path, {})
                relative_folder = file_meta.get("relative_folder", "")
                plate_meta = {
                    "plate_id": data.get("plate_id", "Unknown"),
                    "condition": data.get("condition", "Unknown"),
                    "timepoint": data.get("timepoint", "Unknown"),
                    "relative_folder": relative_folder or "root",
                    "source_file": os.path.basename(file_path),
                }
                for plant in data.get("plants", []):
                    all_plants_data.append({**plate_meta, **plant})
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

        if not all_plants_data:
            QMessageBox.warning(self, "No Data", "Could not extract plant data from selected files.")
            return

        self.df = pd.DataFrame(all_plants_data)
        self.df.rename(columns=METRIC_MAPPING, inplace=True)
        self.report_queue = []
        self._refresh_report_queue_list()

        numeric_metrics = []
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]) and col not in [
                "uid", "plant_num", "scale_cm_px",
            ]:
                numeric_metrics.append(col)

        for col in AXIS_OPTIONS:
            if col in self.df.columns:
                unique_vals = self.df[col].dropna().unique().tolist()
                unique_vals.sort(key=natural_sort_key)
                self.df[col] = pd.Categorical(
                    self.df[col], categories=unique_vals, ordered=True,
                )

        self.color_maps = build_all_metadata_color_maps(self.df)

        self.cb_y_metric.blockSignals(True)
        self.cb_y_metric.clear()
        self.cb_y_metric.addItems(sorted(numeric_metrics))
        self.cb_y_metric.blockSignals(False)

        self._refresh_stratify_options()
        self._update_test_hint()
        self._set_data_loaded_state(True)
        self.update_plot()

    def _current_stat_settings(self):
        return {
            "stat_compare": self.cb_stat_compare.currentText(),
            "stat_within": self._stratify_value(self.cb_stat_within),
            "stat_and_within": self._stratify_value(self.cb_stat_and_within),
            "stat_test": self._stat_test_key(),
            "stat_alpha": self.spin_alpha.value(),
        }

    def _run_stats_for_metric(self, metric):
        settings = self._current_stat_settings()
        return stats_module.run_comparisons_structured(
            self.df,
            metric,
            settings["stat_compare"],
            within_col=settings["stat_within"],
            and_within_col=settings["stat_and_within"],
            test=settings["stat_test"],
            alpha=settings["stat_alpha"],
        )

    def update_inline_stats(self):
        if self.df.empty or self._is_qualitative_plot():
            return
        metric = self.cb_y_metric.currentText()
        if not metric:
            self.txt_stat_results.clear()
            return
        results = self._run_stats_for_metric(metric)
        self.txt_stat_results.setPlainText(
            stats_module.results_to_text(results, alpha=self.spin_alpha.value())
        )

    def _queue_entry_label(self, entry, index):
        if entry["kind"] == "qualitative":
            return f"Fig {index}: Qualitative Atlas (columns: {entry['col_group_label']})"
        hue = entry.get("hue_var", "None")
        hue_part = f" × {hue}" if hue and hue != "None" else ""
        stat_line = self._format_stat_config(entry)
        return (
            f"Fig {index}: {entry['metric']} — {entry['plot_type']} "
            f"({entry['x_var']}{hue_part})\n"
            f"       Stats: {stat_line}"
        )

    def _queue_entry_tooltip(self, entry, index):
        if entry["kind"] == "qualitative":
            return (
                f"Figure {index}: Qualitative Atlas\n"
                f"Columns: {entry['col_group_label']}\n"
                "Exports PNG and SVG in the final report."
            )
        sig_count = sum(1 for r in entry.get("stat_results", []) if r.significant)
        return (
            f"Figure {index}: {entry['metric']}\n"
            f"Plot: {entry['plot_type']}\n"
            f"X: {entry['x_var']}, Hue: {entry.get('hue_var', 'None')}\n"
            f"Error bars: {entry.get('error_bar', '')}\n"
            f"{self._format_stat_config(entry)}\n"
            f"Significant comparisons: {sig_count}"
        )

    def _refresh_report_queue_list(self):
        self.list_report_queue.clear()
        for i, entry in enumerate(self.report_queue, start=1):
            item = QListWidgetItem(self._queue_entry_label(entry, i))
            item.setToolTip(self._queue_entry_tooltip(entry, i))
            self.list_report_queue.addItem(item)

    def _current_view_snapshot(self):
        if self._is_qualitative_plot():
            return {
                "kind": "qualitative",
                "plot_type": QUALITATIVE_PLOT_TYPE,
                "col_group": self._get_col_group(),
                "col_group_label": self._col_group_label(),
            }
        metric = self.cb_y_metric.currentText()
        if not metric:
            return None
        stat_results = self._run_stats_for_metric(metric)
        return {
            "kind": "quantitative",
            "metric": metric,
            "plot_type": self.cb_plot_type.currentText(),
            "x_var": self.cb_x_axis.currentText(),
            "hue_var": self.cb_hue.currentText(),
            "error_bar": self.cb_error_bar.currentText(),
            **self._current_stat_settings(),
            "stat_results": stat_results,
        }


    def add_current_to_report(self):
        if self.df.empty:
            return
        entry = self._current_view_snapshot()
        if not entry:
            QMessageBox.warning(self, "No Metric", "Select a metric before adding to the report.")
            return

        if entry["kind"] == "qualitative":
            self.report_queue = [
                e for e in self.report_queue if e.get("kind") != "qualitative"
            ]

        self.report_queue.append(entry)
        self._refresh_report_queue_list()

    def remove_selected_from_report(self):
        row = self.list_report_queue.currentRow()
        if row < 0:
            return
        if 0 <= row < len(self.report_queue):
            del self.report_queue[row]
            self._refresh_report_queue_list()

    def _render_to_figure(self, figure, plot_type, y_var=None, x_var=None, hue_var=None,
                          error_bar=None, col_group="condition"):
        figure.clf()
        if plot_type == QUALITATIVE_PLOT_TYPE:
            canvas_dims, dest_ini = convex_hull.calculate_optimal_canvas(self.df)
            convex_hull.draw_atlas_grid_on_figure(
                self.df, figure, canvas_dims, dest_ini, col_group=col_group,
            )
            return
        ax = figure.add_subplot(111)
        render_quantitative_plot(
            ax, self.df, plot_type, y_var, x_var, hue_var,
            error_bar=error_bar, color_maps=self.color_maps,
        )
        figure.tight_layout()

    def update_plot(self):
        if self.df.empty:
            return
        plot_type = self.cb_plot_type.currentText()
        try:
            if plot_type == QUALITATIVE_PLOT_TYPE:
                self._render_to_figure(
                    self.figure, plot_type, col_group=self._get_col_group(),
                )
            else:
                y_var = self.cb_y_metric.currentText()
                if not y_var:
                    return
                self._render_to_figure(
                    self.figure,
                    plot_type,
                    y_var=y_var,
                    x_var=self.cb_x_axis.currentText(),
                    hue_var=self.cb_hue.currentText(),
                    error_bar=self.cb_error_bar.currentText(),
                )
                self.update_inline_stats()
            self.ax = self.figure.axes[0] if self.figure.axes else None
            self.canvas.draw()
        except Exception as e:
            self.figure.clf()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Plotting Error:\n{str(e)}", ha="center", va="center", color="red")
            self.canvas.draw()

    def export_data_csv(self):
        if self.df.empty:
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Dataset",
            os.path.join(self._get_output_dir(), "Aggregated_Phenomics_Data.csv"),
            "CSV Files (*.csv)",
        )
        if out_path:
            self._get_clean_dataframe().to_csv(out_path, index=False)
            QMessageBox.information(self, "Export Complete", "Dataset exported successfully.")

    @staticmethod
    def _safe_figure_name(text):
        return (
            text.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace("°", "deg")
        )

    def generate_report(self):
        if self.df.empty:
            return
        if not self.report_queue:
            QMessageBox.warning(
                self, "Empty Report",
                "Add at least one figure to the report before exporting.",
            )
            return

        target_dir = QFileDialog.getExistingDirectory(
            self, "Select Folder for Report Generation", self._get_output_dir(),
        )
        if not target_dir:
            return

        csv_path = os.path.join(target_dir, "Master_Aggregated_Data.csv")
        self._get_clean_dataframe().to_csv(csv_path, index=False)

        fig_dir = os.path.join(target_dir, "Figures")
        os.makedirs(fig_dir, exist_ok=True)

        quant_entries = [e for e in self.report_queue if e["kind"] == "quantitative"]
        qual_entries = [e for e in self.report_queue if e["kind"] == "qualitative"]
        total_steps = len(quant_entries) + (1 if qual_entries else 0)
        progress = QProgressDialog("Generating report...", "Cancel", 0, max(total_steps, 1), self)
        progress.setWindowModality(Qt.WindowModal)
        step = 0

        fig_num = 0
        all_stat_results = []
        summary_sections = []

        for entry in self.report_queue:
            if entry["kind"] == "quantitative":
                if progress.wasCanceled():
                    break
                fig_num += 1
                self._render_to_figure(
                    self.figure,
                    entry["plot_type"],
                    y_var=entry["metric"],
                    x_var=entry["x_var"],
                    hue_var=entry["hue_var"],
                    error_bar=entry["error_bar"],
                )
                safe_name = self._safe_figure_name(entry["metric"])
                plot_slug = self._safe_figure_name(entry["plot_type"].split()[0])
                file_name = os.path.join(fig_dir, f"Fig_{fig_num:02d}_{safe_name}_{plot_slug}.svg")
                self.figure.savefig(file_name, format="svg", bbox_inches="tight")

                stat_results = entry.get("stat_results", [])
                all_stat_results.extend(stat_results)
                label = self._queue_entry_label(entry, fig_num)
                config_line = self._format_stat_config(entry)
                summary_sections.append(
                    f"\n{'=' * 50}\n{label}\n"
                    f"Statistical settings: {config_line}\n"
                    f"{'=' * 50}\n"
                    + stats_module.results_to_text(
                        stat_results, alpha=entry.get("stat_alpha", 0.05),
                    )
                )
                step += 1
                progress.setValue(step)

        if qual_entries and not progress.wasCanceled():
            qual = qual_entries[-1]
            convex_hull.generate_qualitative_grid(
                self.df,
                target_dir,
                col_group=qual["col_group"],
                export_svg=True,
            )
            step += 1
            progress.setValue(step)

        if all_stat_results:
            summary_path = os.path.join(target_dir, "Statistical_Summary.txt")
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("ChronoRoot Batch Report — Statistical Summary\n")
                f.write("\n".join(summary_sections))
            stats_module.results_to_dataframe(all_stat_results).to_csv(
                os.path.join(target_dir, "Statistical_Results.csv"),
                index=False,
            )

        self.canvas.draw()
        progress.setValue(max(total_steps, 1))
        QMessageBox.information(
            self, "Report Complete",
            f"Report saved to:\n{target_dir}\n\n"
            f"Figures: {len(quant_entries)} quantitative"
            + (", 1 qualitative atlas" if qual_entries else ""),
        )

    def generate_full_report(self):
        """Legacy entry point."""
        self.generate_report()
