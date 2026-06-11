import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, 
                             QPushButton, QListWidget, QListWidgetItem, QComboBox, 
                             QFormLayout, QFileDialog, QMessageBox, QAbstractItemView, QGroupBox)
from PyQt5.QtCore import Qt

# --- Matplotlib PyQt5 Integration ---
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

class ComparisonReportTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.df = pd.DataFrame()  # Master DataFrame for all loaded plates
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        
        # ==========================================
        # LEFT PANEL: File Selection & Data Loading
        # ==========================================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("<b>1. Select Processed Plates</b>"))
        
        self.btn_refresh = QPushButton("Refresh Folder")
        self.btn_refresh.clicked.connect(self.refresh_file_list)
        left_layout.addWidget(self.btn_refresh)
        
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection) # Allow Multi-Select
        left_layout.addWidget(self.file_list)
        
        self.btn_load_data = QPushButton("Load Selected Data")
        self.btn_load_data.setStyleSheet("background-color: #007bff; color: white; font-weight: bold;")
        self.btn_load_data.clicked.connect(self.load_selected_data)
        left_layout.addWidget(self.btn_load_data)
        
        # ==========================================
        # MIDDLE PANEL: Plot Controls
        # ==========================================
        mid_panel = QWidget()
        mid_layout = QVBoxLayout(mid_panel)
        mid_layout.addWidget(QLabel("<b>2. Configure Plot</b>"))
        
        control_group = QGroupBox("Aesthetics")
        form = QFormLayout(control_group)
        
        self.cb_plot_type = QComboBox()
        self.cb_plot_type.addItems(["Box Plot", "Swarm Plot", "Violin Plot", "Line Plot (Means)"])
        self.cb_plot_type.currentIndexChanged.connect(self.update_plot)
        
        self.cb_y_metric = QComboBox() # Populated dynamically from JSON keys
        self.cb_y_metric.currentIndexChanged.connect(self.update_plot)
        
        self.cb_x_axis = QComboBox()
        self.cb_x_axis.addItems(["timepoint", "condition", "genotype"])
        self.cb_x_axis.currentIndexChanged.connect(self.update_plot)
        
        self.cb_hue = QComboBox()
        self.cb_hue.addItems(["None", "genotype", "condition", "timepoint"])
        self.cb_hue.currentIndexChanged.connect(self.update_plot)
        
        form.addRow("Plot Type:", self.cb_plot_type)
        form.addRow("Y-Axis (Metric):", self.cb_y_metric)
        form.addRow("X-Axis (Group):", self.cb_x_axis)
        form.addRow("Hue (Color):", self.cb_hue)
        
        mid_layout.addWidget(control_group)
        
        self.btn_export_svg = QPushButton("Export Current Plot (.SVG)")
        self.btn_export_svg.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        self.btn_export_svg.clicked.connect(self.export_plot_to_svg)
        self.btn_export_svg.setEnabled(False)
        mid_layout.addWidget(self.btn_export_svg)
        mid_layout.addStretch()

        # ==========================================
        # RIGHT PANEL: The Matplotlib Canvas
        # ==========================================
        right_panel = QWidget()
        self.right_layout = QVBoxLayout(right_panel)
        
        # Create Figure and Canvas
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.figure.patch.set_facecolor('#f4f4f4')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self) # Adds zoom/pan tools
        
        self.right_layout.addWidget(self.toolbar)
        self.right_layout.addWidget(self.canvas, stretch=1)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(mid_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([200, 250, 750])
        layout.addWidget(splitter)
        
        # Apply Seaborn Theme globally
        sns.set_theme(style="whitegrid", palette="muted")
        
    # --- 1. DATA INGESTION ---
    def refresh_file_list(self):
        self.file_list.clear()
        out_dir = self.main_window.out_dir
        if not os.path.exists(out_dir): return
        
        for f in sorted(os.listdir(out_dir)):
            if f.endswith("_Metrics.json"):
                item = QListWidgetItem(f)
                item.setData(Qt.UserRole, os.path.join(out_dir, f))
                self.file_list.addItem(item)
                
    def load_selected_data(self):
        """Reads multiple JSONs and flattens them into a Pandas DataFrame."""
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Selection Empty", "Please select at least one JSON file.")
            return
            
        all_plants_data = []
        numeric_metrics = set()
        
        for item in selected_items:
            file_path = item.data(Qt.UserRole)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # 1. Extract the top-level plate metadata
                plate_meta = {
                    "plate_id": data.get("plate_id", "Unknown"),
                    "condition": data.get("condition", "Unknown"),
                    "timepoint": data.get("timepoint", "Unknown"),
                    "scale_cm_px": data.get("scale_cm_px", 1.0)
                }
                
                # 2. Flatten the plant data with the plate metadata
                for plant in data.get("plants", []):
                    # Merge the two dictionaries
                    row = {**plate_meta, **plant}
                    all_plants_data.append(row)
                    
                    # 3. Identify numeric metrics for the visualization dropdowns
                    for k, v in plant.items():
                        # Exclude IDs, strings, or lists from the Y-axis metric options
                        if isinstance(v, (int, float)) and k not in ["uid", "plant_num"]:
                            numeric_metrics.add(k)
                            
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")

        if not all_plants_data:
            QMessageBox.warning(self, "No Data", "Could not extract plant data from selected files.")
            return

        # Build the Master DataFrame
        self.df = pd.DataFrame(all_plants_data)
        
        # Populate the UI dropdowns with the discovered metrics
        self.cb_y_metric.blockSignals(True)
        self.cb_y_metric.clear()
        
        # Sort alphabetically for a cleaner UI
        sorted_metrics = sorted(list(numeric_metrics))
        self.cb_y_metric.addItems(sorted_metrics)
        self.cb_y_metric.blockSignals(False)
        
        self.btn_export_svg.setEnabled(True)
        
        # Trigger the initial plot render (assuming update_plot is implemented)
        self.update_plot() 
        
        QMessageBox.information(self, "Data Loaded", f"Successfully loaded {len(self.df)} plant records.")

    # --- 2. PLOTTING ENGINE ---
    def update_plot(self):
        if self.df.empty: return
        
        y_var = self.cb_y_metric.currentText()
        x_var = self.cb_x_axis.currentText()
        hue_var = self.cb_hue.currentText()
        plot_type = self.cb_plot_type.currentText()
        
        if not y_var: return
        hue_var = None if hue_var == "None" else hue_var

        self.ax.clear()
        
        try:
            # Dynamically route to the correct Seaborn plotting function
            if plot_type == "Box Plot":
                sns.boxplot(data=self.df, x=x_var, y=y_var, hue=hue_var, ax=self.ax, width=0.5, fliersize=3)
            elif plot_type == "Swarm Plot":
                sns.swarmplot(data=self.df, x=x_var, y=y_var, hue=hue_var, ax=self.ax, dodge=True, size=4)
            elif plot_type == "Violin Plot":
                sns.violinplot(data=self.df, x=x_var, y=y_var, hue=hue_var, ax=self.ax, inner="quartile", density_norm="width")
            elif plot_type == "Line Plot (Means)":
                # Lineplot automatically calculates means and 95% confidence intervals
                sns.lineplot(data=self.df, x=x_var, y=y_var, hue=hue_var, ax=self.ax, marker="o", err_style="bars")

            # Formatting
            self.ax.set_title(f"{y_var} by {x_var}" + (f" (Grouped by {hue_var})" if hue_var else ""), pad=15, fontweight='bold')
            self.ax.set_xlabel(x_var.capitalize(), fontweight='bold')
            self.ax.set_ylabel(y_var.replace("_", " ").title(), fontweight='bold')
            
            # Fix overlapping X-axis labels
            self.ax.tick_params(axis='x', rotation=45)
            
            if hue_var:
                self.ax.legend(title=hue_var.capitalize(), bbox_to_anchor=(1.05, 1), loc='upper left')
                
            self.figure.tight_layout() # Ensures legend and labels aren't clipped
            self.canvas.draw()
            
        except Exception as e:
            self.ax.clear()
            self.ax.text(0.5, 0.5, f"Plotting Error:\n{str(e)}", ha='center', va='center', color='red')
            self.canvas.draw()

    # --- 3. EXPORTING ---
    def export_plot_to_svg(self):
        if self.df.empty: return
        
        # Build a smart default filename based on current selections
        y_var = self.cb_y_metric.currentText()
        x_var = self.cb_x_axis.currentText()
        hue_var = self.cb_hue.currentText()
        plot_type = self.cb_plot_type.currentText().split(" ")[0]
        
        hue_str = f"_by_{hue_var}" if hue_var != "None" else ""
        default_name = f"Report_{plot_type}_{y_var}_vs_{x_var}{hue_str}.svg"
        
        out_path, _ = QFileDialog.getSaveFileName(self, "Save Vector Graphic", os.path.join(self.main_window.out_dir, default_name), "SVG Files (*.svg)")
        if not out_path: return
        
        try:
            self.figure.savefig(out_path, format='svg', bbox_inches='tight')
            QMessageBox.information(self, "Export Complete", f"Successfully saved plot to:\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))