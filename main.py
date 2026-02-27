import os
import json
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QApplication,
                            QHBoxLayout, QLineEdit, QFileDialog, QListWidget, QListWidgetItem, 
                            QSplitter, QTabWidget, QStyle, QMainWindow, QProgressDialog)
from PyQt5.QtGui import QPainter, QColor, QBrush
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# ==========================================
# IMPORT REFACTORED MODULES
# ==========================================
from model import PlantImageModel
from review_tab import ReviewTab
from guidelines_tab import GuidelinesTab
from about_tab import AboutTab

# ==========================================
# CONFIGURATION
# ==========================================

APP_NAME = "chronorootAnnotationSuite"
GLOBAL_CONFIG_DIR = os.path.expanduser(f"~/.config/{APP_NAME}")
GLOBAL_CONFIG_FILE = os.path.join(GLOBAL_CONFIG_DIR, "interfaceConfig.json")
os.makedirs(GLOBAL_CONFIG_DIR, exist_ok=True)

if not os.path.exists(GLOBAL_CONFIG_FILE):
    default_config = {
        "database_root": "."
    }
    with open(GLOBAL_CONFIG_FILE, 'w') as f:
        print(f"Created default config at {GLOBAL_CONFIG_FILE}")
        json.dump(default_config, f, indent=4)
    DATABASE_ROOT = "."
else:
    with open(GLOBAL_CONFIG_FILE, 'r') as f:
        config = json.load(f)
        DATABASE_ROOT = config.get("database_root", ".")
        
        
class FolderStatsWidget(QWidget):
    def __init__(self, text, total, in_progress, completed, is_folder=True):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        
        # Top Row: Icon + Name
        top_row = QHBoxLayout()
        icon_label = QLabel()
        if is_folder:
            icon = QApplication.style().standardIcon(QStyle.SP_DirIcon)
        else:
            icon = QApplication.style().standardIcon(QStyle.SP_FileIcon)
        icon_label.setPixmap(icon.pixmap(16, 16))
        
        name_label = QLabel(text)
        name_label.setStyleSheet("font-weight: bold;" if is_folder else "")
        
        top_row.addWidget(icon_label)
        top_row.addWidget(name_label)
        top_row.addStretch()
        layout.addLayout(top_row)
        
        # Bottom Row: Progress Bar (Only for folders with content)
        if is_folder and total > 0:
            self.total = total
            self.prog = in_progress
            self.comp = completed
            
            # Stats Text
            stats_txt = f"Done: {completed} | Active: {in_progress} | Total: {total}"
            lbl_stats = QLabel(stats_txt)
            lbl_stats.setStyleSheet("color: gray; font-size: 10px;")
            layout.addWidget(lbl_stats)
            
            # Custom Painted Bar
            self.bar = HeightProgressBar(total, in_progress, completed)
            layout.addWidget(self.bar)

# Helper for the colored bar
class HeightProgressBar(QWidget):
    def __init__(self, total, prog, comp):
        super().__init__()
        self.total = total
        self.prog = prog
        self.comp = comp
        self.setFixedHeight(6) # Thin bar
    
    def paintEvent(self, event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()
        
        # Background (Gray/Pending)
        painter.fillRect(0, 0, w, h, QColor("#e0e0e0"))
        
        if self.total > 0:
            # Widths
            w_comp = (self.comp / self.total) * w
            w_prog = (self.prog / self.total) * w
            
            # Draw Finished (Green)
            painter.fillRect(0, 0, int(w_comp), h, QColor("#28a745"))
            
            # Draw In Progress (Orange) - Starts where Green ends
            painter.fillRect(int(w_comp), 0, int(w_prog), h, QColor("#ffc107"))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ChronoRoot Annotation Suite")
        self.resize(1600, 900)
        
        # Initialize pure Python Backend Model
        self.global_model = PlantImageModel()
        self.current_base_name = ""
        self.current_task_path = ""
        
        # Navigation State
        self.root_dir = DATABASE_ROOT # Default "Test"
        self.current_dir = self.root_dir
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        ml = QHBoxLayout(main_widget)
        splitter = QSplitter(Qt.Horizontal)
        ml.addWidget(splitter)
        
        # --- LEFT PANEL (Browser) ---
        lp = QWidget()
        lpl = QVBoxLayout(lp)
        
        # 1. Path Selector
        path_layout = QHBoxLayout()
        self.lbl_path = QLineEdit(self.root_dir)
        self.lbl_path.setReadOnly(True)
        self.btn_browse = QPushButton("...")
        self.btn_browse.setFixedWidth(30)
        self.btn_browse.clicked.connect(self.change_root_directory)
        path_layout.addWidget(self.lbl_path)
        path_layout.addWidget(self.btn_browse)
        lpl.addLayout(path_layout)
        
        # 2. Navigation Buttons
        nav_layout = QHBoxLayout()
        self.btn_up = QPushButton("⬆ Up Level")
        self.btn_up.clicked.connect(self.navigate_up)
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.populate_browser)
        nav_layout.addWidget(self.btn_up)
        nav_layout.addWidget(self.btn_refresh)
        lpl.addLayout(nav_layout)
        
        lpl.addWidget(QLabel("CONTENTS:"))
        self.task_list = QListWidget()
        self.task_list.itemDoubleClicked.connect(self.on_item_double_clicked) # Navigation
        self.task_list.itemClicked.connect(self.on_item_clicked)              # Preview/Select
        lpl.addWidget(self.task_list)
        
        # 3. Save Actions
        action_layout = QHBoxLayout()
        self.btn_save_progress = QPushButton("Save Progress")
        self.btn_save_progress.setStyleSheet("background-color: #fff3cd; color: #856404;") # Yellowish
        self.btn_save_progress.clicked.connect(lambda: self.save_task(mark_finished=False))
        
        self.btn_finish = QPushButton("Finish Annotation")
        self.btn_finish.setStyleSheet("background-color: #d4edda; color: #155724; font-weight: bold;") # Green
        self.btn_finish.clicked.connect(lambda: self.save_task(mark_finished=True))
        
        action_layout.addWidget(self.btn_save_progress)
        action_layout.addWidget(self.btn_finish)
        lpl.addLayout(action_layout)
                
        splitter.addWidget(lp)
        
        self.tabs = QTabWidget()
        self.tab_review = ReviewTab(self.global_model)   
        self.tab_guidelines = GuidelinesTab() 
        self.tab_about = AboutTab()    
        self.tabs.addTab(self.tab_review, "Annotation Tool")
        self.tabs.addTab(self.tab_guidelines, "Guidelines")
        self.tabs.addTab(self.tab_about, "About")
        
        splitter.addWidget(self.tabs)
        splitter.setSizes([350, 1250])
        
        self.apply_tooltips()
        self.populate_browser()
        
    def apply_tooltips(self):
        """Centralized location for all Main Window tooltips."""
        # File Browser
        self.btn_browse.setToolTip("Browse system to select a different root database folder.")
        self.btn_up.setToolTip("Navigate up one directory level.")
        self.btn_refresh.setToolTip("Reload the folder contents from disk.")
        self.task_list.setToolTip("<b>File Browser</b><br>• <i>Double-click folders</i> to navigate.<br>• <i>Click files</i> to load them into the annotation canvas.")
        self.lbl_path.setToolTip("Current directory path.")
        
        # Saves
        self.btn_save_progress.setToolTip("<b>Save Progress</b><br>Saves your current mask data to a JSON file. Use this if you want to take a break and resume later.")
        self.btn_finish.setToolTip("<b>Finish Annotation</b><br>Saves the data and permanently bakes the result into a final NIfTI file for training. Marks the file as Completed in the browser.")
        
    # --- NAVIGATION & STATISTICS ---

    def change_root_directory(self):
        d = QFileDialog.getExistingDirectory(self, "Select Root Folder", self.root_dir)
        if d:
            self.root_dir = d
            self.current_dir = d
            self.lbl_path.setText(d)
            self.populate_browser()
            
            # 1. Read existing config 
            config = {}
            if os.path.exists(GLOBAL_CONFIG_FILE):
                try:
                    with open(GLOBAL_CONFIG_FILE, 'r') as f:
                        config = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    config = {}

            # 2. Update the value
            config["database_root"] = d

            # 3. Write back the updated config
            with open(GLOBAL_CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
                
    def navigate_up(self):
        # 1. Normalize both paths to ensure the comparison is accurate
        current_abs = os.path.abspath(self.current_dir)
        root_abs = os.path.abspath(self.root_dir)

        # 2. Check if we are already at or above the root
        if current_abs == root_abs or len(current_abs) <= len(root_abs):
            QMessageBox.information(self, "Navigation Limit", "You are already at the project root directory.")
            return

        # 3. If safe, move up one level
        parent = os.path.dirname(current_abs)
        self.current_dir = parent
        self.populate_browser()

    def populate_browser(self):
        """Delegates completely to the backend model to retrieve folder stats and contents."""
        self.task_list.clearSelection() 
        self.task_list.clear()
        self.lbl_path.setText(self.current_dir)
        
        try:
            # Replaced all the nested os.walk/json code with a single isolated API call!
            contents = self.global_model.scan_directory(self.current_dir)
        except Exception as e:
            self.task_list.addItem(f"Error reading directory: {e}")
            return

        for item in contents:
            if item["type"] == "dir":
                list_item = QListWidgetItem()
                list_item.setData(Qt.UserRole, {"type": "dir", "path": item["path"]})
                
                widget = FolderStatsWidget(item["name"], item["total"], item["in_progress"], item["completed"], is_folder=True)
                list_item.setSizeHint(widget.sizeHint())
                
                self.task_list.addItem(list_item)
                self.task_list.setItemWidget(list_item, widget)

            elif item["type"] == "file":
                list_item = QListWidgetItem()
                list_item.setData(Qt.UserRole, {"type": "file", "path": item["path"]})
                
                display_text = f"{item['name']}   [{item['status']}]"
                list_item.setText(display_text)
                
                if item["status"] == "Completed":
                    icon = self.style().standardIcon(QStyle.SP_DialogApplyButton)
                    list_item.setForeground(QBrush(QColor("green")))
                elif item["status"] == "In Progress":
                    icon = self.style().standardIcon(QStyle.SP_FileIcon)
                    list_item.setForeground(QBrush(QColor("#d68910")))
                else:
                    icon = self.style().standardIcon(QStyle.SP_FileIcon)
                    list_item.setForeground(QBrush(Qt.gray))
                    
                list_item.setIcon(icon)
                self.task_list.addItem(list_item)

    def on_item_double_clicked(self, item):
        data = item.data(Qt.UserRole)
        if data["type"] == "dir":
            self.current_dir = data["path"]
            self.populate_browser()

    # --- UI FREEZE PREVENTION HELPERS ---
    def show_loading(self, message):
        """Shows a blocking, spinning dialog so the user knows it's working."""
        self.progress = QProgressDialog(message, None, 0, 0, self)
        self.progress.setWindowTitle("Please Wait")
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.setCancelButton(None) # Remove cancel button
        self.progress.show()

    def hide_loading(self):
        if hasattr(self, 'progress') and self.progress:
            self.progress.close()
            self.progress.deleteLater() # Safely destroy the old loading bar
            self.progress = None

    # --- ASYNC LOADING ---
    def on_item_clicked(self, item):
        data = item.data(Qt.UserRole)
        if data["type"] != "file": return

        if self.global_model.dirty:
            reply = QMessageBox.question(self, 'Save?', 'Save changes?', QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                # Store the RAW DATA, not the UI item
                self._pending_load_data = data 
                self.save_task(mark_finished=False)
                return 
            elif reply == QMessageBox.Cancel:
                return

        # If no save is needed, load immediately
        self._execute_load(data)

    def _execute_load(self, data):
        """Now accepts the raw data dictionary instead of the QListWidgetItem."""
        nii_path = data["path"]
        self.current_task_path = os.path.dirname(nii_path)
        self.current_base_name = os.path.basename(nii_path).replace('.nii.gz', '')
        
        self.tab_review.unselect_instance()
        self.tab_review.set_mode("SELECT")
        self.global_model.callbacks_muted = True
        
        self.show_loading(f"Loading {self.current_base_name}...\n(Parsing masks and NIfTI data)")
        
        self.worker = ModelWorker(self.global_model.load_task, self.current_task_path, self.current_base_name)
        self.worker.finished.connect(self._on_load_finished)
        self.worker.error.connect(self._on_thread_error)
        self.worker.start()

    def _on_load_finished(self, _):
        self.hide_loading()
        self.global_model.callbacks_muted = False
        
        # Safely detach and schedule the thread for deletion
        if hasattr(self, 'worker') and self.worker:
            self.worker.deleteLater()
            self.worker = None
            
        self.global_model._notify_changed() 
        self.setWindowTitle(f"ChronoRoot Annotation Suite | {self.current_base_name} [{self.global_model.status.upper()}]")
        
    # --- ASYNC SAVING ---
    def save_task(self, mark_finished=False):
        if not self.global_model.masks: return
        
        self.global_model.callbacks_muted = True
        self.show_loading("Saving progress...\n(Flattening multi-class masks and generating NIfTI)")
        
        # Store state to use in the callback
        self._pending_mark_finished = mark_finished 
        
        self.worker = ModelWorker(self.global_model.save_current_task, self.current_task_path, self.current_base_name, mark_finished=mark_finished)
        self.worker.finished.connect(self._on_save_finished)
        self.worker.error.connect(self._on_thread_error)
        self.worker.start()

    def _on_save_finished(self, mapping):
        self.hide_loading()
        self.global_model.callbacks_muted = False
        
        # Safely detach and schedule the thread for deletion
        if hasattr(self, 'worker') and self.worker:
            self.worker.deleteLater()
            self.worker = None
        
        # If mapping is a dictionary, the save was successful
        if mapping:
            # 1. Translate the GUI's selection using the new IDs
            self.tab_review.update_selection_from_mapping(mapping)
            
            # 2. Force the model to broadcast the changes to trigger a tree rebuild
            self.global_model._notify_changed()
            
            self.setWindowTitle(f"ChronoRoot Annotation Suite | {self.current_base_name} [{self.global_model.status.upper()}]")
            self.populate_browser() 
            
            if getattr(self, '_is_closing', False):
                self.close() 
            elif getattr(self, '_pending_load_data', None):
                self._execute_load(self._pending_load_data)
                self._pending_load_data = None

    def _on_thread_error(self, err_msg):
        self.hide_loading()
        self.global_model.callbacks_muted = False
        
        # Safely detach and schedule the thread for deletion
        if hasattr(self, 'worker') and self.worker:
            self.worker.deleteLater()
            self.worker = None
            
        QMessageBox.critical(self, "Processing Error", err_msg)
        self._is_closing = False
        self._pending_load_data = None
    
    def closeEvent(self, event):
        if getattr(self, '_is_closing', False):
            event.accept()
            return

        if self.global_model.dirty:
            reply = QMessageBox.question(
                self, 
                'Unsaved Changes', 
                "You have unsaved changes.\nDo you want to save your progress before exiting?", 
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, 
                QMessageBox.Save
            )

            if reply == QMessageBox.Save:
                # 2. Flag that we want to close AFTER saving
                self._is_closing = True 
                self.save_task(mark_finished=False)
                
                # 3. IGNORE the close event for now so the background thread can run!
                event.ignore()  
            elif reply == QMessageBox.Discard:
                event.accept() 
            else:
                event.ignore()
        else:
            event.accept()

# ==========================================
# BACKGROUND THREAD WORKER
# ==========================================
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
            # Run the heavy model function in the background
            result = self.func(*self.args, **self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
            
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())