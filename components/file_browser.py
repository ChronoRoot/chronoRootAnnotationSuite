import os
import json

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QLineEdit, QPushButton, QListWidget, QListWidgetItem, QCheckBox,
                             QFileDialog, QStyle, QApplication, QMessageBox)
from PyQt5.QtGui import QPainter, QColor
from PyQt5.QtCore import Qt, pyqtSignal

from core.model import annotation_status_display, analysis_status_display, normalize_annotation_status


# ==========================================
# CUSTOM LIST ITEM WIDGETS
# ==========================================
class SegmentedProgressBar(QWidget):
    def __init__(self, segments, colors, height=6):
        super().__init__()
        self.segments = segments
        self.colors = colors
        self.setFixedHeight(height)

    def paintEvent(self, event):
        painter = QPainter(self)
        w = self.width()
        h = self.height()
        total = sum(self.segments) or 1
        x = 0
        painter.fillRect(0, 0, w, h, QColor("#e0e0e0"))
        for count, color in zip(self.segments, self.colors):
            if count <= 0:
                continue
            seg_w = int((count / total) * w)
            painter.fillRect(x, 0, seg_w, h, QColor(color))
            x += seg_w


class FolderStatsWidget(QWidget):
    def __init__(self, text, total, missing, in_progress, completed,
                 analyzed=0, not_analyzed=0, is_folder=True):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        top_row = QHBoxLayout()
        icon_label = QLabel()
        icon = QApplication.style().standardIcon(
            QStyle.SP_DirIcon if is_folder else QStyle.SP_FileIcon
        )
        icon_label.setPixmap(icon.pixmap(16, 16))

        name_label = QLabel(text)
        name_label.setStyleSheet("font-weight: bold;" if is_folder else "")

        top_row.addWidget(icon_label)
        top_row.addWidget(name_label)
        top_row.addStretch()
        layout.addLayout(top_row)

        if is_folder and total > 0:
            ann_txt = (
                f"Annotation — Missing: {missing} | Active: {in_progress} | "
                f"Done: {completed} | Total: {total}"
            )
            lbl_ann = QLabel(ann_txt)
            lbl_ann.setStyleSheet("color: gray; font-size: 10px;")
            layout.addWidget(lbl_ann)

            layout.addWidget(SegmentedProgressBar(
                [completed, in_progress, missing],
                ["#28a745", "#ffc107", "#bdbdbd"],
            ))

            if completed > 0:
                anal_txt = (
                    f"Analysis — Analyzed: {analyzed} | Ready: {not_analyzed} | "
                    f"Completed tasks: {completed}"
                )
                lbl_anal = QLabel(anal_txt)
                lbl_anal.setStyleSheet("color: #666; font-size: 10px;")
                layout.addWidget(lbl_anal)
                layout.addWidget(SegmentedProgressBar(
                    [analyzed, not_analyzed],
                    ["#007bff", "#d6e4ff"],
                ))


def _annotation_row_style(display_status):
    if display_status == "Completed":
        return "color: #28a745; font-size: 10px; font-weight: bold;"
    if display_status == "In Progress":
        return "color: #d68910; font-size: 10px; font-weight: bold;"
    return "color: gray; font-size: 10px;"


def _analysis_row_style(analysis_status):
    if analysis_status == "analyzed":
        return "color: #28a745; font-size: 10px; font-weight: bold;"
    if analysis_status == "not_analyzed":
        return "color: #007bff; font-size: 10px; font-weight: bold;"
    return "color: #999; font-size: 10px;"


def _annotation_icon(display_status):
    if display_status == "Completed":
        return QApplication.style().standardIcon(QStyle.SP_DialogApplyButton)
    if display_status == "In Progress":
        return QApplication.style().standardIcon(QStyle.SP_FileIcon)
    return QApplication.style().standardIcon(QStyle.SP_MessageBoxWarning)


class FileStatsWidget(QWidget):
    """Annotator-only: annotation row only."""

    def __init__(self, name, annotation_status, plant_count):
        super().__init__()
        display = annotation_status_display(annotation_status)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        top_row = QHBoxLayout()
        icon_label = QLabel()
        icon_label.setPixmap(_annotation_icon(display).pixmap(16, 16))
        top_row.addWidget(icon_label)
        top_row.addWidget(QLabel(name))
        top_row.addStretch()
        layout.addLayout(top_row)

        lbl_stats = QLabel(f"Annotation: {display} | Plants: {plant_count}")
        lbl_stats.setStyleSheet(_annotation_row_style(display))
        layout.addWidget(lbl_stats)


class DualRowFileStatsWidget(QWidget):
    """Unified browser: separate annotation and analysis rows."""

    def __init__(self, name, annotation_status, analysis_status, plant_count,
                 plants_analyzed=None):
        super().__init__()
        ann_display = annotation_status_display(annotation_status)
        anal_display = analysis_status_display(analysis_status)
        if analysis_status == "analyzed" and plants_analyzed is not None:
            anal_line = f"Analysis: {anal_display} | Plants: {plants_analyzed}"
        elif analysis_status == "not_analyzed":
            anal_line = f"Analysis: {anal_display} | Plants: {plant_count}"
        else:
            anal_line = f"Analysis: {anal_display}"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)

        top_row = QHBoxLayout()
        icon_label = QLabel()
        icon_label.setPixmap(_annotation_icon(ann_display).pixmap(16, 16))
        top_row.addWidget(icon_label)
        top_row.addWidget(QLabel(name))
        top_row.addStretch()
        layout.addLayout(top_row)

        lbl_ann = QLabel(f"Annotation: {ann_display} | Plants: {plant_count}")
        lbl_ann.setStyleSheet(_annotation_row_style(ann_display))
        layout.addWidget(lbl_ann)

        lbl_anal = QLabel(anal_line)
        lbl_anal.setStyleSheet(_analysis_row_style(analysis_status))
        layout.addWidget(lbl_anal)


class AnalyzerFileStatsWidget(DualRowFileStatsWidget):
    """Analyzer browser reuses dual-row layout."""
    pass


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

    def update_file_status_in_cache(
        self,
        file_path,
        new_status=None,
        new_plant_count=None,
        new_analysis_status=None,
    ):
        """Update cached file entry and propagate folder annotation/analysis aggregates."""
        file_dir = os.path.dirname(file_path)

        delta_missing = delta_in_progress = delta_completed = 0
        delta_analyzed = delta_not_analyzed = 0

        def ann_key(display_or_raw):
            if display_or_raw in ("Missing", "missing", "Pending", "pending"):
                return "missing"
            if display_or_raw in ("Completed", "completed"):
                return "completed"
            if display_or_raw in ("In Progress", "in_progress"):
                return "in_progress"
            return normalize_annotation_status(display_or_raw)

        if file_dir not in self.folder_cache:
            return

        target = None
        for item in self.folder_cache[file_dir]:
            if item["type"] == "file" and item["path"] == file_path:
                target = item
                break
        if target is None:
            return

        old_ann = ann_key(target.get("annotation_status") or target.get("status"))
        old_analysis = target.get("analysis_status", "not_applicable")
        old_plant_count = target.get("plant_count", 0)

        new_ann = ann_key(new_status) if new_status is not None else old_ann
        new_analysis = new_analysis_status if new_analysis_status is not None else old_analysis

        ann_changed = new_ann != old_ann
        analysis_changed = new_analysis != old_analysis
        count_changed = new_plant_count is not None and old_plant_count != new_plant_count

        if not ann_changed and not analysis_changed and not count_changed:
            return

        target["annotation_status"] = new_ann
        target["status"] = annotation_status_display(new_ann)
        if new_plant_count is not None:
            target["plant_count"] = new_plant_count
        if new_analysis_status is not None:
            target["analysis_status"] = new_analysis
            target["analyzer_status"] = analysis_status_display(new_analysis)

        if ann_changed:
            for bucket, delta_name in (
                ("missing", "delta_missing"),
                ("in_progress", "delta_in_progress"),
                ("completed", "delta_completed"),
            ):
                if old_ann == bucket:
                    if bucket == "missing":
                        delta_missing -= 1
                    elif bucket == "in_progress":
                        delta_in_progress -= 1
                    elif bucket == "completed":
                        delta_completed -= 1
                if new_ann == bucket:
                    if bucket == "missing":
                        delta_missing += 1
                    elif bucket == "in_progress":
                        delta_in_progress += 1
                    elif bucket == "completed":
                        delta_completed += 1

        if analysis_changed and old_ann == "completed":
            if old_analysis == "analyzed":
                delta_analyzed -= 1
            elif old_analysis == "not_analyzed":
                delta_not_analyzed -= 1
        if analysis_changed and new_ann == "completed":
            if new_analysis == "analyzed":
                delta_analyzed += 1
            elif new_analysis == "not_analyzed":
                delta_not_analyzed += 1

        if any([delta_missing, delta_in_progress, delta_completed, delta_analyzed, delta_not_analyzed]):
            current_iter_dir = file_dir
            while True:
                parent_dir = os.path.dirname(current_iter_dir)
                if parent_dir in self.folder_cache:
                    for item in self.folder_cache[parent_dir]:
                        if item["type"] == "dir" and item["path"] == current_iter_dir:
                            item["missing"] = item.get("missing", 0) + delta_missing
                            item["in_progress"] = item.get("in_progress", 0) + delta_in_progress
                            item["completed"] = item.get("completed", 0) + delta_completed
                            item["analyzed"] = item.get("analyzed", 0) + delta_analyzed
                            item["not_analyzed"] = item.get("not_analyzed", 0) + delta_not_analyzed
                            break

                if current_iter_dir == self.root_dir or parent_dir == current_iter_dir:
                    break
                current_iter_dir = parent_dir

        if self.current_dir in self.folder_cache:
            self.render_contents(self.folder_cache[self.current_dir], use_cache=False)

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
                    item["name"],
                    item["total"],
                    item.get("missing", 0),
                    item.get("in_progress", 0),
                    item.get("completed", 0),
                    analyzed=item.get("analyzed", 0),
                    not_analyzed=item.get("not_analyzed", 0),
                    is_folder=True,
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
                annotation_status = item.get("annotation_status") or item.get("status", "missing")
                analysis_status = item.get("analysis_status", "not_applicable")

                if self.unified_file_stats:
                    widget = DualRowFileStatsWidget(
                        item["name"],
                        annotation_status,
                        analysis_status,
                        plant_count,
                    )
                else:
                    widget = FileStatsWidget(item["name"], annotation_status, plant_count)

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
                    item.get("annotation_status") or item.get("status", "missing"),
                    item.get("analysis_status", "not_applicable"),
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
            ann = normalize_annotation_status(data.get("status") or data.get("annotation_status"))
            if ann != "completed":
                QMessageBox.warning(
                    self, "Skip",
                    "This file has not been completed in the Annotation Suite yet."
                )
                return
            self.file_selected.emit(data)
