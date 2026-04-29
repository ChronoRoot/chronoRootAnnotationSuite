import os
import json
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QApplication,
                            QHBoxLayout, QLineEdit, QFileDialog, QListWidget, QListWidgetItem, 
                            QSplitter, QTabWidget, QStyle, QMainWindow, QProgressDialog,
                            QTreeWidget, QHeaderView, QFrame, QStackedWidget, QTreeWidgetItem)
                            
from PyQt5.QtGui import QPainter, QColor, QBrush, QPixmap, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class SortableTreeItem(QTreeWidgetItem):
    def __lt__(self, other):
        col = self.treeWidget().sortColumn()
        # Sort numerically using the hidden UserRole data (either ID or Area)
        return self.data(col, Qt.UserRole) < other.data(col, Qt.UserRole)
    
from model import PlantImageModel
from review_tab import ReviewCanvasTab, ReviewToolPanel
from guidelines_tab import GuidelinesTab
from about_tab import AboutTab
from frangi_tab import FrangiCanvasTab, FrangiToolPanel
from graph_tab import GraphCanvasTab, GraphToolPanel

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

class FileStatsWidget(QWidget):
    def __init__(self, name, status, plant_count):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        
        # Top Row: Icon + Name
        top_row = QHBoxLayout()
        icon_label = QLabel()
        
        # Set icon based on status
        if status == "Completed":
            icon = QApplication.style().standardIcon(QStyle.SP_DialogApplyButton)
        else:
            icon = QApplication.style().standardIcon(QStyle.SP_FileIcon)
            
        icon_label.setPixmap(icon.pixmap(16, 16))
        
        name_label = QLabel(name)
        
        top_row.addWidget(icon_label)
        top_row.addWidget(name_label)
        top_row.addStretch()
        layout.addLayout(top_row)
        
        # Bottom Row: Stats Text
        stats_txt = f"Status: {status} | Nº Plants: {plant_count}"
        lbl_stats = QLabel(stats_txt)
        
        # Color code the stats text based on the status
        if status == "Completed":
            lbl_stats.setStyleSheet("color: #28a745; font-size: 10px; font-weight: bold;") # Green
        elif status == "In Progress":
            lbl_stats.setStyleSheet("color: #d68910; font-size: 10px; font-weight: bold;") # Orange
        else:
            lbl_stats.setStyleSheet("color: gray; font-size: 10px;") # Gray for Pending
            
        layout.addWidget(lbl_stats)
        
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
        
        self.global_model = PlantImageModel()
        self.current_base_name = ""
        self.current_task_path = ""
        self.active_workers = set()
        
        self.root_dir = DATABASE_ROOT 
        self.current_dir = self.root_dir
        self.folder_cache = {}
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        ml = QHBoxLayout(main_widget)
        
        # We now use a splitter with 3 sections!
        splitter = QSplitter(Qt.Horizontal)
        ml.addWidget(splitter)
        
        # ==========================================
        # PANE 1: LEFT PANEL (Browser)
        # ==========================================
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
        self.btn_refresh.clicked.connect(lambda: self.populate_browser(force_refresh=True))
        nav_layout.addWidget(self.btn_up)
        nav_layout.addWidget(self.btn_refresh)
        lpl.addLayout(nav_layout)
        
        lpl.addWidget(QLabel("CONTENTS:"))
        self.task_list = QListWidget()
        self.task_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.task_list.itemClicked.connect(self.on_item_clicked)
        lpl.addWidget(self.task_list)
        
        # 3. Save Actions
        action_layout = QHBoxLayout()
        self.btn_save_progress = QPushButton("Save Progress")
        self.btn_save_progress.setStyleSheet("background-color: #fff3cd; color: #856404;") 
        self.btn_save_progress.clicked.connect(lambda: self.save_task(mark_finished=False))
        
        self.btn_finish = QPushButton("Finish Annotation")
        self.btn_finish.setStyleSheet("background-color: #d4edda; color: #155724; font-weight: bold;") 
        self.btn_finish.clicked.connect(lambda: self.save_task(mark_finished=True))
        
        action_layout.addWidget(self.btn_save_progress)
        action_layout.addWidget(self.btn_finish)
        lpl.addLayout(action_layout)
                
        splitter.addWidget(lp)

        # ==========================================
        # PANE 2: MIDDLE PANEL (Context Sidebar)
        # ==========================================
        mp = QWidget()
        mpl = QVBoxLayout(mp)
        
        # --- Top Half: Shared Plant List ---
        self.lbl_instance_count = QLabel("<b>PLANT INSTANCES (0 Total):</b>")
        mpl.addWidget(self.lbl_instance_count)
        
        self.list_instances = QTreeWidget()
        self.list_instances.setHeaderLabels(["Plant ID", "Area (px)"])
        self.list_instances.setSortingEnabled(True)
        self.list_instances.setRootIsDecorated(False)
        self.list_instances.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.list_instances.setFixedHeight(300) # Keep it compact
        
        self.list_instances.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.list_instances.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.list_instances.itemSelectionChanged.connect(self.on_global_list_selection_changed)
        self.list_instances.itemClicked.connect(self.on_global_list_item_clicked)
        mpl.addWidget(self.list_instances)
        
        # Shared Lifecycle Buttons
        btn_layout = QHBoxLayout()
        self.btn_new = QPushButton("+ New")
        self.btn_new.clicked.connect(self.action_new_instance)
        
        self.btn_merge = QPushButton("Merge")
        self.btn_merge.clicked.connect(self.action_merge)
        
        self.btn_del = QPushButton("Delete")
        self.btn_del.clicked.connect(self.action_delete)
        
        btn_layout.addWidget(self.btn_new)
        btn_layout.addWidget(self.btn_merge)
        btn_layout.addWidget(self.btn_del)
        mpl.addLayout(btn_layout)
        
        mpl.addSpacing(10)
        mpl.addWidget(QFrame(frameShape=QFrame.HLine, frameShadow=QFrame.Sunken))
        mpl.addSpacing(10)
        
        # --- Bottom Half: Dynamic Tool Stack ---
        mpl.addWidget(QLabel("<b>Control Panel:</b>"))
        self.tool_stack = QStackedWidget()        

        # ==========================================
        # PANE 3: RIGHT PANEL (Canvases / Viewport)
        # ==========================================
        self.tabs = QTabWidget()
        
        self.canvas_review = ReviewCanvasTab(self.global_model) 
        self.tool_panel_review = ReviewToolPanel(self.global_model, self.canvas_review)
        
        self.canvas_frangi = FrangiCanvasTab(self.global_model)
        self.tool_panel_frangi = FrangiToolPanel(self.global_model, self.canvas_frangi)

        self.canvas_graph = GraphCanvasTab(self.global_model)
        self.tool_panel_graph = GraphToolPanel(self.global_model, self.canvas_graph)
        
        self.tab_guidelines = GuidelinesTab() # Kept as is
        self.tab_about = AboutTab()           # Kept as is
        
        
        self.tool_stack.addWidget(self.tool_panel_review)
        self.tool_stack.addWidget(self.tool_panel_frangi)
        self.tool_stack.addWidget(self.tool_panel_graph)
        
        mpl.addWidget(self.tool_stack)
        splitter.addWidget(mp)
        
        self.tabs.addTab(self.canvas_review, "Annotation Tool")
        self.tabs.addTab(self.canvas_frangi, "Centerline Refinement")  
        self.tabs.addTab(self.canvas_graph, "Graph Visualization & Correction")
        self.tabs.addTab(self.tab_guidelines, "Guidelines")
        self.tabs.addTab(self.tab_about, "About")
        
        # === THE SMART TAB ROUTER ===
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        splitter.addWidget(self.tabs)
        
        # Proportions: Browser (300px), Sidebar (300px), Canvas (1000px)
        splitter.setSizes([300, 300, 1000]) 
        
        self.apply_tooltips()
        self.populate_browser()
        
        # Register MainWindow to listen to model data changes to rebuild the tree
        self.global_model.register_data_callback(self.populate_global_list)
        
        # Register MainWindow to listen to selection changes from the canvas ---
        self.global_model.register_selection_callback(self.sync_global_list_selection)
    
    def on_tab_changed(self, index):
        """Handles QoL state changes when navigating between tabs."""
        
        # 1. Always sync the sidebar tool stack with the main canvas tab
        self.tool_stack.setCurrentIndex(index)
        
        # 2. Moving TO Frangi (1) or Graph (2)
        if index in [1, 2]:
            # If no plant is selected, auto-select the first available one
            if not self.global_model.active_uid and self.global_model.masks:
                first_uid = sorted(self.global_model.masks.keys())[0]
                self.global_model.set_selection([first_uid])
                
        # 3. Moving TO Annotation Tool (0)
        elif index == 0:
            # Drop whatever tool they were using and return to normal pointer
            self.tool_panel_review.force_mode("SELECT")
            
            # If a plant is currently active, snap the camera to it
            if self.global_model.active_uid:
                self.canvas_review.zoom_to_plant(self.global_model.active_uid)
    
    def sync_global_list_selection(self):
        """Visually updates the lateral bar to match the model's selection state."""
        self.list_instances.blockSignals(True)
        
        for i in range(self.list_instances.topLevelItemCount()):
            item = self.list_instances.topLevelItem(i)
            uid = item.data(0, Qt.UserRole)
            
            # Highlight the item if it exists in the model's selected set
            item.setSelected(uid in self.global_model.selected_uids)
            
            # Automatically scroll to the active plant so it's always in view
            if uid == self.global_model.active_uid:
                self.list_instances.scrollToItem(item)
                
        self.list_instances.blockSignals(False)
          
    def on_global_list_selection_changed(self):
        """Updates the Model's single source of truth when the user clicks the list."""
        selected_items = self.list_instances.selectedItems()
        
        # --- SAFE CHECK: Block multi-selection if not in Annotation Tab ---
        if len(selected_items) > 1 and self.tabs.currentIndex() != 0:
            QMessageBox.warning(
                self, 
                "Multi-Selection Disabled", 
                "Multi-selection is only supported in the 'Annotation Tool' tab.\n\nPlease switch tabs to select multiple plants."
            )
            
            # Force the UI back to a single selection without triggering infinite loops
            self.list_instances.blockSignals(True)
            for item in selected_items[1:]:
                item.setSelected(False)
            self.list_instances.blockSignals(False)
            
            # Proceed with only the first clicked item
            selected_items = [selected_items[0]]
        # ------------------------------------------------------------------
        
        selected_uids = [item.data(0, Qt.UserRole) for item in selected_items]
        self.global_model.set_selection(selected_uids)

    def on_global_list_item_clicked(self, item, column):
        """Manually trigger the zoom ONLY when the user explicitly clicks the sidebar list."""
        uid = item.data(0, Qt.UserRole)
        
        # Check if we are currently looking at the Annotation Tool (Tab 0)
        if self.tabs.currentIndex() == 0:
            # We only zoom if it is a single selection to avoid erratic jumping
            if len(self.global_model.selected_uids) == 1:
                self.canvas_review.zoom_to_plant(uid)
                
    def populate_global_list(self):
        """Rebuilds or updates the shared tree view efficiently."""
        self.lbl_instance_count.setText(f"<b>PLANT INSTANCES ({len(self.global_model.masks)} Total):</b>")
        
        current_uids = set(self.global_model.masks.keys())
        existing_uids = {self.list_instances.topLevelItem(i).data(0, Qt.UserRole) for i in range(self.list_instances.topLevelItemCount())}
        
        if current_uids == existing_uids:
            self.list_instances.setSortingEnabled(False)
            for i in range(self.list_instances.topLevelItemCount()):
                item = self.list_instances.topLevelItem(i)
                uid = item.data(0, Qt.UserRole)
                area = self.global_model.areas.get(uid, 0)
                item.setText(1, f"{area:,}")
                item.setData(1, Qt.UserRole, area)
                
            self.list_instances.setSortingEnabled(True)
            return

        self.list_instances.blockSignals(True)
        self.list_instances.clearSelection()
        self.list_instances.setSortingEnabled(False)
        self.list_instances.clear()
        
        for uid in sorted(self.global_model.masks.keys()):
            area = self.global_model.areas.get(uid, 0)
            item = SortableTreeItem([f"Plant {uid}", f"{area:,}"])
            item.setData(0, Qt.UserRole, uid)   
            item.setData(1, Qt.UserRole, area)  
            
            if uid in self.global_model.color_map:
                c = self.global_model.color_map[uid]
                pix = QPixmap(16, 16); pix.fill(QColor(c[0], c[1], c[2]))
                item.setIcon(0, QIcon(pix))
                
            self.list_instances.addTopLevelItem(item)
            
            if uid in self.global_model.selected_uids:
                item.setSelected(True)
            
        self.list_instances.setSortingEnabled(True)
        self.list_instances.blockSignals(False)
        self.list_instances.viewport().update()

    def action_new_instance(self):
        """Prepares a new ID and signals the tools to enter paint mode."""
        if self.tabs.currentIndex() != 0:
            QMessageBox.warning(self, "Action Restricted", "Please switch to the 'Annotation Tool' tab to create new plants.")
            return
            
        self.global_model.set_selection([])
        new_uid = self.global_model.prepare_new_uid()
        self.global_model.active_uid = new_uid
        
        # Automatically switch to the Paint tool when creating a new plant
        self.tool_panel_review.force_mode("PAINT")
        
    def action_merge(self):
        if self.tabs.currentIndex() != 0:
            QMessageBox.warning(self, "Action Restricted", "Please switch to the 'Annotation Tool' tab to merge plants.")
            return
            
        if len(self.global_model.selected_uids) < 2: 
            return
            
        new_id = self.global_model.merge_instances(list(self.global_model.selected_uids))
        if new_id is not None:
            self.global_model.set_selection([new_id])
        
    def action_delete(self):
        if self.global_model.selected_uids:
            # 1. Capture the IDs we want to delete
            uids_to_delete = list(self.global_model.selected_uids)
            
            # 2. Clear the UI selection FIRST to prevent ghosting
            self.global_model.set_selection([])
            
            # 3. Destroy the data and trigger the tree rebuild
            self.global_model.delete_instances(uids_to_delete)
        
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
            self.folder_cache = {} # Clear cache when changing root
            self.lbl_path.setText(d)
            self.populate_browser(force_refresh=True)
            
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

    def populate_browser(self, force_refresh=False):
        """Checks cache or delegates to background thread to retrieve folder stats."""
        self.lbl_path.setText(self.current_dir)
        
        if force_refresh:
            # Clear cache for the current directory AND all subdirectories recursively
            keys_to_delete = [k for k in self.folder_cache if k.startswith(self.current_dir)]
            for k in keys_to_delete:
                del self.folder_cache[k]

        if self.current_dir in self.folder_cache:
            # Load instantly from cache
            self._render_browser_contents(self.folder_cache[self.current_dir])
        else:
            # Clear UI while loading
            self.task_list.clearSelection() 
            self.task_list.clear()
            
            # Spin up the background thread
            self.show_loading(f"Scanning directory stats...\n{self.current_dir}")
            
            worker = ModelWorker(self.global_model.scan_directory, self.current_dir)
            self.active_workers.add(worker)
            worker.finished.connect(self._on_scan_finished)
            worker.error.connect(self._on_thread_error)
            worker.start()

    def _on_scan_finished(self, contents):
        """Receives data from thread, caches it, and triggers render."""
        self.hide_loading()
        
        # Clean up the worker
        worker = self.sender()
        if worker in getattr(self, 'active_workers', set()):
            self.active_workers.remove(worker)
            worker.deleteLater()
            
        # Save to memory cache
        self.folder_cache[self.current_dir] = contents
        self._render_browser_contents(contents)

    def update_file_status_in_cache(self, file_path, new_status, new_plant_count=None):
        """Surgically updates the cache for a single file and propagates the changes upwards instantly."""
        file_dir = os.path.dirname(file_path)
        
        # 1. Update the actual file entry and determine the progress deltas
        delta_in_progress = 0
        delta_completed = 0
        
        if file_dir in self.folder_cache:
            for item in self.folder_cache[file_dir]:
                if item["type"] == "file" and item["path"] == file_path:
                    old_status = item["status"]
                    old_plant_count = item.get("plant_count", 0)
                    
                    # Check if anything actually changed
                    status_changed = (old_status != new_status)
                    count_changed = (new_plant_count is not None and old_plant_count != new_plant_count)
                    
                    if not status_changed and not count_changed:
                        return # Nothing changed, do nothing
                    
                    # Apply updates to the cache
                    item["status"] = new_status
                    if new_plant_count is not None:
                        item["plant_count"] = new_plant_count
                    
                    # Calculate the math for the progress bars (Only if status changed)
                    if status_changed:
                        if old_status == "Pending" and new_status == "In Progress":
                            delta_in_progress = 1
                        elif old_status == "Pending" and new_status == "Completed":
                            delta_completed = 1
                        elif old_status == "In Progress" and new_status == "Completed":
                            delta_in_progress = -1
                            delta_completed = 1
                        elif old_status == "Completed" and new_status == "In Progress":
                            delta_completed = -1
                            delta_in_progress = 1
                    break
            else:
                return # File not found in cache
        else:
            return # Directory not cached

        # 2. Propagate deltas upwards ONLY if there is a change in status
        if delta_in_progress != 0 or delta_completed != 0:
            current_iter_dir = file_dir
            while True:
                parent_dir = os.path.dirname(current_iter_dir)
                if parent_dir in self.folder_cache:
                    for item in self.folder_cache[parent_dir]:
                        if item["type"] == "dir" and item["path"] == current_iter_dir:
                            item["in_progress"] += delta_in_progress
                            item["completed"] += delta_completed
                            break
                
                # Stop if we've reached the user's root or the system root
                if current_iter_dir == self.root_dir or parent_dir == current_iter_dir:
                    break
                current_iter_dir = parent_dir
            
        # 3. Instantly re-render the current view from the modified cache (No rescan!)
        self.populate_browser(force_refresh=False)
        
    def _render_browser_contents(self, contents):
        """Handles the actual UI widget creation (must run on main thread)."""
        self.task_list.clearSelection() 
        self.task_list.clear()

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
                
                # Use .get() to safely grab the plant count, defaulting to 0 if missing
                plant_count = item.get("plant_count", 0) 
                
                # Use our new custom widget for files
                widget = FileStatsWidget(item["name"], item["status"], plant_count)
                list_item.setSizeHint(widget.sizeHint())
                
                self.task_list.addItem(list_item)
                self.task_list.setItemWidget(list_item, widget)

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
        file_path = data["path"]
        
        # Add this line so we remember the exact file path for the fast-save patch
        self.current_file_path = file_path  
        
        self.current_task_path = os.path.dirname(file_path)
        self.current_base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        self.global_model.set_selection([])
        self.tool_panel_review.force_mode("SELECT")        
        self.global_model.callbacks_muted = True
        
        self.show_loading(f"Loading {self.current_base_name}...\n(Parsing masks and NIfTI data)")
        
        worker = ModelWorker(self.global_model.load_task, self.current_task_path, self.current_base_name)
        self.active_workers.add(worker)
        worker.finished.connect(self._on_load_finished)
        worker.error.connect(self._on_thread_error)
        worker.start()
        
    def _on_load_finished(self, _):
        self.hide_loading()
        self.global_model.callbacks_muted = False
        
        worker = self.sender()
        if worker in getattr(self, 'active_workers', set()):
            self.active_workers.remove(worker)
            worker.deleteLater()
            
        self.global_model._notify_data_changed() 
        self.setWindowTitle(f"ChronoRoot Annotation Suite | {self.current_base_name} [{self.global_model.status.upper()}]")
        
        self.populate_browser(force_refresh=True)
        
    # --- ASYNC SAVING ---
    def save_task(self, mark_finished=False):
        if not self.global_model.masks: return
        
        self.global_model.callbacks_muted = True
        self.show_loading("Saving progress...\n(Flattening multi-class masks and generating NIfTI)")
        
        # Store state to use in the callback
        self._pending_mark_finished = mark_finished 
        
        worker = ModelWorker(self.global_model.save_current_task, self.current_task_path, self.current_base_name, mark_finished=mark_finished)
        self.active_workers.add(worker)
        worker.finished.connect(self._on_save_finished)
        worker.error.connect(self._on_thread_error)
        worker.start()

    def _on_save_finished(self, mapping):
        self.hide_loading()
        self.global_model.callbacks_muted = False
        
        worker = self.sender()
        if worker in getattr(self, 'active_workers', set()):
            self.active_workers.remove(worker)
            worker.deleteLater()
        
        if mapping is not None:
            new_selection = [mapping[uid] for uid in self.global_model.selected_uids if uid in mapping]
            self.global_model.set_selection(new_selection)
            
            self.global_model._notify_data_changed()
            self.setWindowTitle(f"ChronoRoot Annotation Suite | {self.current_base_name} [{self.global_model.status.upper()}]")

            if getattr(self, '_is_closing', False):
                self.close() 
            elif getattr(self, '_pending_load_data', None):
                data_to_load = self._pending_load_data
                self._pending_load_data = None
                self._execute_load(data_to_load)
            else:
                if hasattr(self, 'current_file_path'):
                    new_status = "Completed" if getattr(self, '_pending_mark_finished', False) else "In Progress"
                    
                    # Grab the exact number of plants currently in the model
                    current_plant_count = len(self.global_model.masks)
                    
                    # Pass the count to the cache updater
                    self.update_file_status_in_cache(
                        self.current_file_path, 
                        new_status, 
                        new_plant_count=current_plant_count
                    )
                else:
                    # Fallback just in case
                    self.populate_browser(force_refresh=True)

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