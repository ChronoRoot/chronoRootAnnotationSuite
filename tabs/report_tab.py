import os
import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel,
                             QPushButton, QListWidget, QListWidgetItem, QComboBox,
                             QFormLayout, QFileDialog, QMessageBox, QAbstractItemView, QGroupBox, QProgressDialog)
from PyQt5.QtCore import Qt, pyqtSignal

# --- Matplotlib PyQt5 Integration ---
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from core import convex_hull
from components.file_browser import qt_display_text

# ==========================================
# METRIC NAME MAPPING (For Publication-Ready Plots)
# ==========================================
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
    "emergence_angle_deg": "Mean Emergence Angle (°)"
}

def natural_sort_key(s):
    """Splits strings into text/number chunks for intuitive human sorting (e.g. Day 2 before Day 10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


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

    if not roots:
        out_dir = _metrics_output_dir(main_window)
        if os.path.isdir(out_dir):
            roots.append(os.path.abspath(out_dir))
    return roots


def _metrics_output_dir(main_window):
    if hasattr(main_window, "browser") and hasattr(main_window.browser, "out_dir"):
        browser = main_window.browser
        if getattr(browser, "output_mode", "task_folder") == "fixed":
            return browser.out_dir
        return getattr(browser, "root_dir", browser.out_dir)
    return getattr(main_window, "out_dir", ".")


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


class ReportFileListPanel(QWidget):
    """Metrics file picker shown in the left/middle column during Batch Reports."""

    load_requested = pyqtSignal()

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.addWidget(QLabel("<b>Processed Plates</b>"))

        list_btns = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_file_list)
        self.btn_sel_all = QPushButton("Select All")
        self.btn_sel_all.clicked.connect(self.select_all_files)
        list_btns.addWidget(self.btn_refresh)
        list_btns.addWidget(self.btn_sel_all)
        layout.addLayout(list_btns)

        self.btn_load_data = QPushButton("Load Selected Data")
        self.btn_load_data.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")
        self.btn_load_data.clicked.connect(self.load_requested.emit)
        layout.addWidget(self.btn_load_data)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.file_list)

    def refresh_file_list(self):
        self.file_list.clear()
        metrics_files = discover_metrics_files(self.main_window)
        for full_path in sorted(
            metrics_files.keys(),
            key=lambda p: natural_sort_key(metrics_files[p]["display"]),
        ):
            meta = metrics_files[full_path]
            item = QListWidgetItem(qt_display_text(meta["display"]))
            item.setData(Qt.UserRole, full_path)
            item.setToolTip(full_path)
            self.file_list.addItem(item)

    def select_all_files(self):
        for i in range(self.file_list.count()):
            self.file_list.item(i).setSelected(True)


class ComparisonReportTab(QWidget):
    def __init__(self, main_window, file_panel):
        super().__init__()
        self.main_window = main_window
        self.file_panel = file_panel
        self.df = pd.DataFrame()
        self.file_panel.load_requested.connect(self.load_selected_data)
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)

        # Plot controls + matplotlib canvas (file list lives in the middle column)
        mid_panel = QWidget()
        mid_layout = QVBoxLayout(mid_panel)
        mid_layout.addWidget(QLabel("<b>Configure Plot</b>"))
        
        control_group = QGroupBox("Aesthetics")
        form = QFormLayout(control_group)
        
        # In init_ui, add the new plot type:
        self.cb_plot_type = QComboBox()
        self.cb_plot_type.addItems([
            "Box Plot", "Swarm Plot", "Violin Plot", "Violin + Swarm Plot", 
            "Line Plot (Means)", "Qualitative Plots"  
        ])
        self.cb_plot_type.currentIndexChanged.connect(self.on_plot_type_changed)
        
        self.cb_error_bar = QComboBox()
        self.cb_error_bar.addItems(["Standard Error (SE)", "Standard Deviation (SD)", "95% Confidence Interval (CI)"])
        self.cb_error_bar.setEnabled(False) # Only enable for line plot
        self.cb_error_bar.currentIndexChanged.connect(self.update_plot)

        self.cb_y_metric = QComboBox() 
        self.cb_y_metric.currentIndexChanged.connect(self.update_plot)
        
        self.cb_x_axis = QComboBox()
        self.cb_x_axis.addItems(["timepoint", "condition", "genotype", "relative_folder", "day_folder"])
        self.cb_x_axis.currentIndexChanged.connect(self.update_plot)
        
        self.cb_hue = QComboBox()
        self.cb_hue.addItems(["None", "genotype", "condition", "timepoint", "relative_folder", "day_folder"])
        self.cb_hue.setCurrentText("genotype")
        self.cb_hue.currentIndexChanged.connect(self.update_plot)
        
        form.addRow("Plot Type:", self.cb_plot_type)
        form.addRow("Line Error Bars:", self.cb_error_bar)
        form.addRow("Y-Axis (Metric):", self.cb_y_metric)
        form.addRow("X-Axis (Group):", self.cb_x_axis)
        form.addRow("Hue (Color):", self.cb_hue)
        
        mid_layout.addWidget(control_group)
        
        # Export Buttons
        self.btn_export_csv = QPushButton("Export Table (.CSV)")
        self.btn_export_csv.clicked.connect(self.export_data_csv)
        self.btn_export_csv.setEnabled(False)
        mid_layout.addWidget(self.btn_export_csv)

        self.btn_export_svg = QPushButton("Export Current Plot (.SVG)")
        self.btn_export_svg.clicked.connect(self.export_plot_to_svg)
        self.btn_export_svg.setEnabled(False)
        mid_layout.addWidget(self.btn_export_svg)
        
        self.btn_export_report = QPushButton("Generate Complete Report")
        self.btn_export_report.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 10px; margin-top: 15px;")
        self.btn_export_report.clicked.connect(self.generate_full_report)
        self.btn_export_report.setEnabled(False)
        mid_layout.addWidget(self.btn_export_report)
        
        mid_layout.addStretch()

        # ==========================================
        # RIGHT PANEL: The Matplotlib Canvas
        # ==========================================
        right_panel = QWidget()
        self.right_layout = QVBoxLayout(right_panel)
        
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.figure.patch.set_facecolor('#f4f4f4')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self) 
        
        self.right_layout.addWidget(self.toolbar)
        self.right_layout.addWidget(self.canvas, stretch=1)
        
        splitter.addWidget(mid_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([280, 920])
        layout.addWidget(splitter)
        
        sns.set_theme(style="whitegrid", palette="muted")

    def _get_output_dir(self):
        return _metrics_output_dir(self.main_window)

    def _get_search_roots(self):
        return _metrics_search_roots(self.main_window)

    def _discover_metrics_files(self):
        return discover_metrics_files(self.main_window)

    def refresh_file_list(self):
        self.file_panel.refresh_file_list()

    @staticmethod
    def _infer_day_folder(relative_folder):
        if not relative_folder:
            return ""
        for part in relative_folder.replace("\\", "/").split("/"):
            if re.match(r"(?i)^day[\s_-]?\d+", part) or re.match(r"(?i)^d\d+", part):
                return part
        return ""

    def on_plot_type_changed(self):
        plot_type = self.cb_plot_type.currentText()
        is_line_plot = "Line Plot" in plot_type
        is_atlas = "Qualitative" in plot_type
        
        # Only enable Error Bar dropdown if Line Plot is selected
        self.cb_error_bar.setEnabled(is_line_plot)
        
        # Disable X, Y, and Hue if Atlas is selected (Atlas uses strict grouping)
        self.cb_y_metric.setEnabled(not is_atlas)
        self.cb_x_axis.setEnabled(not is_atlas)
        self.cb_hue.setEnabled(not is_atlas)
        
        self.update_plot()

    def _get_clean_dataframe(self):
        """Strips system/JSON variables and returns a pure, publication-ready dataset."""
        meta_cols = [
            "plate_id", "condition", "timepoint", "genotype", "plant_num",
            "relative_folder", "day_folder", "source_file",
        ]
        metric_cols = list(METRIC_MAPPING.values())
        desired_cols = [c for c in meta_cols + metric_cols if c in self.df.columns]
        return self.df[desired_cols].copy()

    def load_selected_data(self):
        selected_items = self.file_panel.file_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Selection Empty", "Please select at least one JSON file.")
            return

        all_plants_data = []
        metrics_files = self._discover_metrics_files()

        for item in selected_items:
            file_path = item.data(Qt.UserRole)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                file_meta = metrics_files.get(file_path, {})
                relative_folder = file_meta.get("relative_folder", "")
                plate_meta = {
                    "plate_id": data.get("plate_id", "Unknown"),
                    "condition": data.get("condition", "Unknown"),
                    "timepoint": data.get("timepoint", "Unknown"),
                    "relative_folder": relative_folder or "root",
                    "day_folder": self._infer_day_folder(relative_folder) or "Unknown",
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
        
        # 1. RENAME METRICS TO HUMAN-READABLE
        self.df.rename(columns=METRIC_MAPPING, inplace=True)
        
        # Identify numeric columns for plotting (after renaming)
        numeric_metrics = []
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]) and col not in ["uid", "plant_num", "scale_cm_px"]:
                numeric_metrics.append(col)
                
        # 2. APPLY NATURAL SORTING TO CATEGORIES
        for col in ['timepoint', 'condition', 'genotype', 'relative_folder', 'day_folder']:
            if col in self.df.columns:
                unique_vals = self.df[col].dropna().unique().tolist()
                unique_vals.sort(key=natural_sort_key)
                # Convert to Pandas Categorical forcing the sorted order
                self.df[col] = pd.Categorical(self.df[col], categories=unique_vals, ordered=True)

        # Update UI Elements
        self.cb_y_metric.blockSignals(True)
        self.cb_y_metric.clear()
        self.cb_y_metric.addItems(sorted(numeric_metrics))
        self.cb_y_metric.blockSignals(False)
        
        self.btn_export_svg.setEnabled(True)
        self.btn_export_csv.setEnabled(True)
        self.btn_export_report.setEnabled(True)
        
        self.update_plot() 

    def update_plot(self):
        if self.df.empty: return
        
        plot_type = self.cb_plot_type.currentText()
        
        # We must clear the ENTIRE figure, not just the axis, because the 
        # Atlas generates a completely different multi-axis grid layout.
        self.figure.clf() 
        
        try:
            if plot_type == "Qualitative Plots": 
                canvas_dims, dest_ini = convex_hull.calculate_optimal_canvas(self.df)
                convex_hull.draw_atlas_grid_on_figure(self.df, self.figure, canvas_dims, dest_ini)
                
            else:
                # Standard Seaborn Plots: We need to recreate a single axis
                self.ax = self.figure.add_subplot(111)
                
                y_var = self.cb_y_metric.currentText()
                x_var = self.cb_x_axis.currentText()
                hue_var = self.cb_hue.currentText()
                
                if not y_var: return
                hue_var = None if hue_var == "None" else hue_var

                if plot_type == "Box Plot":
                    sns.boxplot(data=self.df, x=x_var, y=y_var, hue=hue_var, ax=self.ax, width=0.5, fliersize=3)
                elif plot_type == "Swarm Plot":
                    sns.swarmplot(data=self.df, x=x_var, y=y_var, hue=hue_var, ax=self.ax, dodge=True, size=4)
                elif plot_type == "Violin Plot":
                    sns.violinplot(data=self.df, x=x_var, y=y_var, hue=hue_var, ax=self.ax, inner="quartile", density_norm="width")
                elif plot_type == "Violin + Swarm Plot":
                    sns.violinplot(data=self.df, x=x_var, y=y_var, hue=hue_var, ax=self.ax, inner=None, color=".9", density_norm="width")
                    sns.swarmplot(data=self.df, x=x_var, y=y_var, hue=hue_var, ax=self.ax, dodge=True, size=4, palette="dark:black", alpha=0.6)
                elif plot_type == "Line Plot (Means)":
                    err_map = {"95% Confidence Interval (CI)": ("ci", 95), "Standard Error (SE)": "se", "Standard Deviation (SD)": "sd"}
                    err_val = err_map.get(self.cb_error_bar.currentText(), "se")
                    sns.lineplot(data=self.df, x=x_var, y=y_var, hue=hue_var, ax=self.ax, marker="o", err_style="bars", errorbar=err_val)

                # Formatting
                self.ax.set_title(f"{y_var} by {x_var.capitalize()}" + (f" (Grouped by {hue_var.capitalize()})" if hue_var else ""), pad=15, fontweight='bold')
                self.ax.set_xlabel(x_var.capitalize(), fontweight='bold')
                self.ax.set_ylabel(y_var, fontweight='bold')
                self.ax.tick_params(axis='x', rotation=45)
                
                if hue_var:
                    self.ax.legend(title=hue_var.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                self.figure.tight_layout() 
                
            self.canvas.draw()
            
        except Exception as e:
            self.figure.clf()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f"Plotting Error:\n{str(e)}", ha='center', va='center', color='red')
            self.canvas.draw()
    # --- EXPORTING LOGIC ---
    def export_plot_to_svg(self):
        if self.df.empty: return
        y_var = self.cb_y_metric.currentText().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        x_var = self.cb_x_axis.currentText()
        plot_type = self.cb_plot_type.currentText().split(" ")[0]
        
        default_name = f"{plot_type}_{y_var}_vs_{x_var}.svg"
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Vector Graphic", os.path.join(self._get_output_dir(), default_name), "SVG Files (*.svg)")
        if out_path:
            self.figure.savefig(out_path, format='svg', bbox_inches='tight')
            QMessageBox.information(self, "Export Complete", "Plot exported successfully.")

    def export_data_csv(self):
        if self.df.empty: return
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Dataset", os.path.join(self._get_output_dir(), "Aggregated_Phenomics_Data.csv"), "CSV Files (*.csv)")
        if out_path:
            clean_df = self._get_clean_dataframe() # <--- Use the clean version
            clean_df.to_csv(out_path, index=False)
            QMessageBox.information(self, "Export Complete", "Dataset exported successfully.")

    def generate_full_report(self):
        if self.df.empty: return
        
        target_dir = QFileDialog.getExistingDirectory(self, "Select Folder for Report Generation", self._get_output_dir())
        if not target_dir: return
        
        # 1. Save the Master CSV (Using the clean version)
        csv_path = os.path.join(target_dir, "Master_Aggregated_Data.csv")
        clean_df = self._get_clean_dataframe() # <--- Use the clean version
        clean_df.to_csv(csv_path, index=False)
        
        convex_hull.generate_qualitative_grid(self.df, target_dir)
        
        # 2. Iterate and generate plots
        progress = QProgressDialog("Generating Plots...", "Cancel", 0, self.cb_y_metric.count(), self)
        progress.setWindowModality(Qt.WindowModal)
        
        # Temporarily save current plot state to restore later
        original_metric_idx = self.cb_y_metric.currentIndex()
        original_plot_type = self.cb_plot_type.currentIndex()
        
        # Force standardized report settings
        self.cb_plot_type.setCurrentText("Line Plot (Means)")
        self.cb_x_axis.setCurrentText("timepoint")
        self.cb_hue.setCurrentText("genotype")
        self.cb_error_bar.setCurrentText("Standard Error (SE)")
        
        plot_dir = os.path.join(target_dir, "Report_Plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        for i in range(self.cb_y_metric.count()):
            if progress.wasCanceled(): break
            
            metric = self.cb_y_metric.itemText(i)
            self.cb_y_metric.setCurrentIndex(i)
            
            safe_name = metric.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("°", "deg")
            file_name = os.path.join(plot_dir, f"Fig_{i+1}_{safe_name}.svg")
            
            self.figure.savefig(file_name, format='svg', bbox_inches='tight')
            progress.setValue(i + 1)
            
        # Restore user settings
        self.cb_plot_type.setCurrentIndex(original_plot_type)
        self.cb_y_metric.setCurrentIndex(original_metric_idx)
        
        QMessageBox.information(self, "Report Complete", f"Data table and all SVG plots were successfully saved into:\n{target_dir}")