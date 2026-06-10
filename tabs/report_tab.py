import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QLabel, QPushButton, QListWidget, QListWidgetItem
from PyQt5.QtCore import Qt

class ComparisonReportTab(QWidget):
    def __init__(self, main_window):
        """
        Standalone UI Tab focused on handling batch run evaluations,
        file indexing, and analytics comparison views.
        """
        super().__init__()
        self.main_window = main_window
        self.init_ui()
        
    def init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        
        # --- LEFT PANEL: Batch Run File Picker ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("<b>Processed Plates (.json)</b>"))
        
        self.btn_refresh = QPushButton("Refresh List")
        self.btn_refresh.clicked.connect(self.refresh_file_list)
        left_layout.addWidget(self.btn_refresh)
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.on_file_selected)
        left_layout.addWidget(self.file_list)
        
        # --- RIGHT PANEL: Analysis Dashboard Placer ---
        right_panel = QWidget()
        self.right_layout = QVBoxLayout(right_panel)
        self.right_layout.addWidget(QLabel("<b>Data Visualization & Plots</b>"))
        
        self.lbl_plot_placeholder = QLabel("Select files from the left to generate comparison plots.")
        self.lbl_plot_placeholder.setAlignment(Qt.AlignCenter)
        self.lbl_plot_placeholder.setStyleSheet("background-color: #2b2b2b; color: #aaa; border: 1px dashed #444; font-family: sans-serif;")
        self.right_layout.addWidget(self.lbl_plot_placeholder, stretch=1)
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])
        layout.addWidget(splitter)
        
    def refresh_file_list(self):
        """Scans the output target directory for newly configured standardized metrics files."""
        self.file_list.clear()
        out_dir = self.main_window.out_dir
        if not os.path.exists(out_dir): return
        
        for f in sorted(os.listdir(out_dir)):
            if f.endswith("_Metrics.json"):
                item = QListWidgetItem(f)
                item.setData(Qt.UserRole, os.path.join(out_dir, f))
                self.file_list.addItem(item)
                
    def on_file_selected(self, item):
        """Callback hook ready to feed data matrices straight into visualization plotting engines."""
        file_path = item.data(Qt.UserRole)
        # Structural payload check ready for your Matplotlib/PyQtGraph scripts
        self.lbl_plot_placeholder.setText(f"Ready to plot structured batch data for:\n{os.path.basename(file_path)}")