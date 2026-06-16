import os
import json

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QListWidget, QListWidgetItem, QCheckBox,
                             QFileDialog, QStyle, QApplication, QMessageBox)
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt, pyqtSignal


# ==========================================
# CUSTOM LIST ITEM WIDGETS (Annotator)
# ==========================================
class HeightProgressBar(QWidget):
    def __init__(self, total, prog, comp):
        super().__init__()
        self.total = total
        self.prog = prog
        self.comp = comp
        self.setFixedHeight(6)

    def paintEvent(self, event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()

        painter.fillRect(0, 0, w, h, QColor("#e0e0e0"))

        if self.total > 0:
            w_comp = (self.comp / self.total) * w
            w_prog = (self.prog / self.total) * w

            painter.fillRect(0, 0, int(w_comp), h, QColor("#28a745"))
            painter.fillRect(int(w_comp), 0, int(w_prog), h, QColor("#ffc107"))


class FolderStatsWidget(QWidget):
    def __init__(self, text, total, in_progress, completed, is_folder=True):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

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

        if is_folder and total > 0:
            self.total = total
            self.prog = in_progress
            self.comp = completed

            stats_txt = f"Done: {completed} | Active: {in_progress} | Total: {total}"
            lbl_stats = QLabel(stats_txt)
            lbl_stats.setStyleSheet("color: gray; font-size: 10px;")
            layout.addWidget(lbl_stats)

            self.bar = HeightProgressBar(total, in_progress, completed)
            layout.addWidget(self.bar)


class FileStatsWidget(QWidget):
    def __init__(self, name, status, plant_count):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        top_row = QHBoxLayout()
        icon_label = QLabel()

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

        stats_txt = f"Status: {status} | Nº Plants: {plant_count}"
        lbl_stats = QLabel(stats_txt)

        if status == "Completed":
            lbl_stats.setStyleSheet("color: #28a745; font-size: 10px; font-weight: bold;")
        elif status == "In Progress":
            lbl_stats.setStyleSheet("color: #d68910; font-size: 10px; font-weight: bold;")
        else:
            lbl_stats.setStyleSheet("color: gray; font-size: 10px;")

        layout.addWidget(lbl_stats)


# ==========================================
# CUSTOM LIST ITEM WIDGETS (Analyzer)
# ==========================================
class AnalyzerFileStatsWidget(QWidget):
    def __init__(self, name, annotator_status, analyzer_status, plant_count):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        top_row = QHBoxLayout()
        icon_label = QLabel()

        if analyzer_status == "Analyzed":
            icon = QApplication.style().standardIcon(QStyle.SP_DialogApplyButton)
        elif analyzer_status == "Ready":
            icon = QApplication.style().standardIcon(QStyle.SP_FileIcon)
        else:
            icon = QApplication.style().standardIcon(QStyle.SP_MessageBoxWarning)

        icon_label.setPixmap(icon.pixmap(16, 16))
        top_row.addWidget(icon_label)
        top_row.addWidget(QLabel(name))
        top_row.addStretch()
        layout.addLayout(top_row)

        lbl_stats = QLabel(f"Status: {analyzer_status} | Num Plants: {plant_count}")
        if analyzer_status == "Analyzed":
            lbl_stats.setStyleSheet("color: #28a745; font-size: 10px; font-weight: bold;")
        elif analyzer_status == "Ready":
            lbl_stats.setStyleSheet("color: #007bff; font-size: 10px; font-weight: bold;")
        else:
            lbl_stats.setStyleSheet("color: #dc3545; font-size: 10px;")
        layout.addWidget(lbl_stats)


class UnifiedFileStatsWidget(QWidget):
    """Combines Annotator and Analyzer status into a single visual."""

    def __init__(self, name, annotator_status, analyzer_status, plant_count):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        top_row = QHBoxLayout()
        icon_label = QLabel()

        if analyzer_status == "Analyzed":
            icon = QApplication.style().standardIcon(QStyle.SP_DialogApplyButton)
        elif annotator_status == "Completed":
            icon = QApplication.style().standardIcon(QStyle.SP_FileIcon)
        else:
            icon = QApplication.style().standardIcon(QStyle.SP_MessageBoxWarning)

        icon_label.setPixmap(icon.pixmap(16, 16))
        top_row.addWidget(icon_label)
        top_row.addWidget(QLabel(name))
        top_row.addStretch()
        layout.addLayout(top_row)

        if analyzer_status == "Analyzed":
            stat_str = f"Fully Analyzed | Plants: {plant_count}"
            color = "#28a745"
        elif annotator_status == "Completed":
            stat_str = f"Ready for Extraction | Plants: {plant_count}"
            color = "#007bff"
        elif annotator_status == "In Progress":
            stat_str = f"Annotation Active | Plants: {plant_count}"
            color = "#d68910"
        else:
            stat_str = f"Pending Annotation | Plants: {plant_count}"
            color = "gray"

        lbl_stats = QLabel(stat_str)
        lbl_stats.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold;")
        layout.addWidget(lbl_stats)


# ==========================================
# MAIN BROWSER PANEL
# ==========================================
class UnifiedFileBrowser(QWidget):
    file_selected = pyqtSignal(dict)
    refresh_requested = pyqtSignal(str)
    save_progress_requested = pyqtSignal()
    finish_annotation_requested = pyqtSignal()
    input_dir_changed = pyqtSignal(str)
    output_dir_changed = pyqtSignal(str)
    output_mode_changed = pyqtSignal(str)

    def __init__(self, default_in_dir, default_out_dir, config_file=None, show_save_actions=True,
                 unified_file_stats=True, output_mode="task_folder"):
        super().__init__()
        self.in_dir = default_in_dir
        self.out_dir = default_out_dir
        self.root_dir = default_in_dir
        self.current_dir = default_in_dir
        self.config_file = config_file
        self.show_save_actions = show_save_actions
        self.unified_file_stats = unified_file_stats
        self.output_mode = output_mode
        self.folder_cache = {}

        self.init_ui()
        self.apply_tooltips()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("<b>Project Root (Input):</b>"))
        in_layout = QHBoxLayout()
        self.lbl_in_path = QLineEdit(self.in_dir)
        self.lbl_in_path.setReadOnly(True)
        self.btn_browse_in = QPushButton("...")
        self.btn_browse_in.setFixedWidth(30)
        self.btn_browse_in.clicked.connect(self.change_input_dir)
        in_layout.addWidget(self.lbl_in_path)
        in_layout.addWidget(self.btn_browse_in)
        layout.addLayout(in_layout)

        self.lbl_current_dir = QLabel(f"Browsing: {self.current_dir}")
        self.lbl_current_dir.setWordWrap(True)
        self.lbl_current_dir.setStyleSheet("color: #555; font-size: 11px;")
        layout.addWidget(self.lbl_current_dir)

        layout.addWidget(QLabel("<b>Output Directory:</b>"))
        self.chk_output_with_input = QCheckBox("Store outputs with input image folder")
        self.chk_output_with_input.setChecked(self.output_mode == "task_folder")
        self.chk_output_with_input.toggled.connect(self._on_output_mode_toggled)
        layout.addWidget(self.chk_output_with_input)

        self.lbl_output_hint = QLabel("Exports are saved beside each loaded image.")
        self.lbl_output_hint.setWordWrap(True)
        self.lbl_output_hint.setStyleSheet("color: #666; font-size: 11px;")

        self.out_path_container = QWidget()
        out_layout = QHBoxLayout(self.out_path_container)
        out_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_out_path = QLineEdit(self.out_dir)
        self.lbl_out_path.setReadOnly(True)
        self.btn_browse_out = QPushButton("...")
        self.btn_browse_out.setFixedWidth(30)
        self.btn_browse_out.clicked.connect(self.change_output_dir)
        out_layout.addWidget(self.lbl_out_path)
        out_layout.addWidget(self.btn_browse_out)
        layout.addWidget(self.lbl_output_hint)
        layout.addWidget(self.out_path_container)
        self._apply_output_mode_ui()

        nav_layout = QHBoxLayout()
        self.btn_up = QPushButton("⬆ Up Level")
        self.btn_up.clicked.connect(self.navigate_up)
        self.btn_refresh = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.request_refresh)
        nav_layout.addWidget(self.btn_up)
        nav_layout.addWidget(self.btn_refresh)
        layout.addLayout(nav_layout)

        layout.addWidget(QLabel("CONTENTS:"))
        self.task_list = QListWidget()
        self.task_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        self.task_list.itemClicked.connect(self.on_item_clicked)
        layout.addWidget(self.task_list)

        if self.show_save_actions:
            action_layout = QHBoxLayout()
            self.btn_save_progress = QPushButton("Save Progress")
            self.btn_save_progress.setStyleSheet("background-color: #fff3cd; color: #856404;")
            self.btn_save_progress.clicked.connect(self.save_progress_requested.emit)

            self.btn_finish = QPushButton("Finish Annotation")
            self.btn_finish.setStyleSheet("background-color: #d4edda; color: #155724; font-weight: bold;")
            self.btn_finish.clicked.connect(self.finish_annotation_requested.emit)

            action_layout.addWidget(self.btn_save_progress)
            action_layout.addWidget(self.btn_finish)
            layout.addLayout(action_layout)
        else:
            self.btn_save_progress = None
            self.btn_finish = None

    def apply_tooltips(self):
        self.btn_browse_in.setToolTip("Browse system to select a different input database folder.")
        self.btn_browse_out.setToolTip("Browse system to select a different output folder.")
        self.btn_up.setToolTip("Navigate up one directory level.")
        self.btn_refresh.setToolTip("Reload the folder contents from disk.")
        self.task_list.setToolTip(
            "<b>File Browser</b><br>• <i>Double-click folders</i> to navigate.<br>"
            "• <i>Click files</i> to load them into the annotation canvas."
        )
        self.lbl_in_path.setToolTip("Fixed project root directory. Browse here to change the dataset root.")
        self.lbl_current_dir.setToolTip("Current folder being browsed within the project root.")
        self.lbl_out_path.setToolTip("Fixed output directory when not storing with input images.")
        self.chk_output_with_input.setToolTip(
            "If enabled, exports are saved beside each loaded image. "
            "Disable to use a fixed output folder."
        )
        if self.btn_save_progress:
            self.btn_save_progress.setToolTip(
                "<b>Save Progress</b><br>Saves your current mask data to a JSON file. "
                "Use this if you want to take a break and resume later."
            )
        if self.btn_finish:
            self.btn_finish.setToolTip(
                "<b>Finish Annotation</b><br>Saves the data and permanently bakes the result "
                "into a final NIfTI file for training. Marks the file as Completed in the browser."
            )

    def request_refresh(self, force_refresh=True):
        if force_refresh:
            keys_to_delete = [k for k in self.folder_cache if k.startswith(self.current_dir)]
            for k in keys_to_delete:
                del self.folder_cache[k]
        self.refresh_requested.emit(self.current_dir)

    def change_input_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Input Folder", self.in_dir)
        if not d:
            return

        self.in_dir = d
        self.root_dir = d
        self.current_dir = d
        self.folder_cache = {}
        self.lbl_in_path.setText(d)
        self._update_current_dir_label()
        self._persist_database_root(d)
        self.input_dir_changed.emit(d)
        self.refresh_requested.emit(d)

    def change_output_dir(self):
        if self.chk_output_with_input.isChecked():
            QMessageBox.information(
                self,
                "Output Folder Locked",
                "Disable 'Store outputs with input image folder' to select a custom output folder.",
            )
            return

        d = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.out_dir)
        if not d:
            return

        self.out_dir = d
        self.lbl_out_path.setText(d)
        self.output_mode = "fixed"
        self.chk_output_with_input.blockSignals(True)
        self.chk_output_with_input.setChecked(False)
        self.chk_output_with_input.blockSignals(False)
        self._apply_output_mode_ui()
        self.output_dir_changed.emit(d)
        self.output_mode_changed.emit("fixed")

    def _apply_output_mode_ui(self):
        task_folder = self.output_mode == "task_folder"
        self.lbl_output_hint.setVisible(task_folder)
        self.out_path_container.setVisible(not task_folder)
        self.btn_browse_out.setEnabled(not task_folder)

    def _on_output_mode_toggled(self, checked):
        self.output_mode = "task_folder" if checked else "fixed"
        self._apply_output_mode_ui()
        self.output_mode_changed.emit(self.output_mode)

    def _update_current_dir_label(self):
        self.lbl_current_dir.setText(f"Browsing: {self.current_dir}")

    def get_effective_output_dir(self, task_path=None):
        if self.output_mode == "task_folder":
            return task_path or self.out_dir
        return self.out_dir

    def navigate_up(self):
        current_abs = os.path.abspath(self.current_dir)
        root_abs = os.path.abspath(self.root_dir)

        if current_abs == root_abs:
            QMessageBox.information(self, "Navigation Limit", "You are already at the project root directory.")
            return

        parent = os.path.dirname(current_abs)
        try:
            rel = os.path.relpath(parent, root_abs)
            if rel.startswith(".."):
                parent = root_abs
        except ValueError:
            parent = root_abs

        self.current_dir = parent
        self._update_current_dir_label()
        self.refresh_requested.emit(self.current_dir)

    def _persist_database_root(self, path):
        if not self.config_file:
            return

        config = {}
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
            except (json.JSONDecodeError, ValueError):
                config = {}

        config["database_root"] = path
        config["input_root"] = path

        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)

    def cache_contents(self, directory, contents):
        self.folder_cache[directory] = contents

    def get_cached_contents(self, directory):
        return self.folder_cache.get(directory)

    def update_file_status_in_cache(self, file_path, new_status, new_plant_count=None):
        """Surgically updates the cache for a single file and propagates folder stats upward."""
        file_dir = os.path.dirname(file_path)

        delta_in_progress = 0
        delta_completed = 0

        if file_dir in self.folder_cache:
            for item in self.folder_cache[file_dir]:
                if item["type"] == "file" and item["path"] == file_path:
                    old_status = item["status"]
                    old_plant_count = item.get("plant_count", 0)

                    status_changed = (old_status != new_status)
                    count_changed = (new_plant_count is not None and old_plant_count != new_plant_count)

                    if not status_changed and not count_changed:
                        return

                    item["status"] = new_status
                    if new_plant_count is not None:
                        item["plant_count"] = new_plant_count

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
                return
        else:
            return

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

                if current_iter_dir == self.root_dir or parent_dir == current_iter_dir:
                    break
                current_iter_dir = parent_dir

        if self.current_dir in self.folder_cache:
            self.render_contents(self.folder_cache[self.current_dir])

    def render_contents(self, contents, use_cache=True):
        if use_cache:
            self.folder_cache[self.current_dir] = contents

        self._update_current_dir_label()
        self.task_list.clearSelection()
        self.task_list.clear()

        for item in contents:
            if item["type"] == "dir":
                list_item = QListWidgetItem()
                list_item.setData(Qt.UserRole, {"type": "dir", "path": item["path"]})

                widget = FolderStatsWidget(
                    item["name"], item["total"], item["in_progress"], item["completed"], is_folder=True
                )
                list_item.setSizeHint(widget.sizeHint())
                self.task_list.addItem(list_item)
                self.task_list.setItemWidget(list_item, widget)

            elif item["type"] == "file":
                list_item = QListWidgetItem()
                list_item.setData(Qt.UserRole, {
                    "type": "file",
                    "path": item["path"],
                    "status": item.get("status", "Pending"),
                })

                plant_count = item.get("plant_count", 0)
                annotator_status = item.get("status", "Pending")
                analyzer_status = item.get("analyzer_status", "Not Analyzed")

                if self.unified_file_stats:
                    widget = UnifiedFileStatsWidget(
                        item["name"], annotator_status, analyzer_status, plant_count
                    )
                else:
                    widget = FileStatsWidget(item["name"], annotator_status, plant_count)

                list_item.setSizeHint(widget.sizeHint())
                self.task_list.addItem(list_item)
                self.task_list.setItemWidget(list_item, widget)

    def render_analyzer_contents(self, contents):
        """Renders the analyzer-style browser (simple folder icons, analyzer file widgets)."""
        self._update_current_dir_label()
        self.task_list.clearSelection()
        self.task_list.clear()

        for item in contents:
            if item["type"] == "dir":
                icon = QApplication.style().standardIcon(QStyle.SP_DirIcon)
                list_item = QListWidgetItem(icon, item["name"])
                list_item.setData(Qt.UserRole, {"type": "dir", "path": item["path"]})
                self.task_list.addItem(list_item)

            elif item["type"] == "file":
                list_item = QListWidgetItem()
                list_item.setData(Qt.UserRole, {
                    "type": "file",
                    "path": item["path"],
                    "status": item.get("status", "Pending"),
                })
                widget = AnalyzerFileStatsWidget(
                    item["name"],
                    item.get("status", "Pending"),
                    item.get("analyzer_status", "Not Annotated"),
                    item.get("plant_count", 0),
                )
                list_item.setSizeHint(widget.sizeHint())
                self.task_list.addItem(list_item)
                self.task_list.setItemWidget(list_item, widget)

    def on_item_double_clicked(self, item):
        data = item.data(Qt.UserRole)
        if data["type"] == "dir":
            self.current_dir = data["path"]
            self._update_current_dir_label()
            self.refresh_requested.emit(self.current_dir)

    def on_item_clicked(self, item):
        data = item.data(Qt.UserRole)
        if data["type"] == "file":
            if data.get("status") == "Not Annotated":
                QMessageBox.warning(
                    self, "Skip",
                    "This file has not been completed in the Annotation Suite yet."
                )
                return
            self.file_selected.emit(data)
