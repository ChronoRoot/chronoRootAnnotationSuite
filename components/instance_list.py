from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QTreeWidget, QHeaderView, QTreeWidgetItem,
                             QMessageBox, QSizePolicy)
from PyQt5.QtGui import QIcon, QPixmap, QColor
from PyQt5.QtCore import Qt, pyqtSignal


class SortableTreeItem(QTreeWidgetItem):
    def __lt__(self, other):
        col = self.treeWidget().sortColumn()
        return self.data(col, Qt.UserRole) < other.data(col, Qt.UserRole)


class InstanceListPanel(QWidget):
    zoom_requested = pyqtSignal(int)

    def __init__(self, shared_model, workspaces, review_tool_panel=None, annotation_tab_index=0):
        super().__init__()
        self.model = shared_model
        self.workspaces = workspaces
        self.review_tool_panel = review_tool_panel
        self.annotation_tab_index = annotation_tab_index
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.init_ui()

        self.model.register_data_callback(self.populate_list)
        self.model.register_selection_callback(self.sync_selection)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.lbl_instance_count = QLabel("<b>Plants (0 total):</b>")
        layout.addWidget(self.lbl_instance_count)

        self.list_instances = QTreeWidget()
        self.list_instances.setHeaderLabels(["Plant ID", "Area (px)"])
        self.list_instances.setSortingEnabled(True)
        self.list_instances.setRootIsDecorated(False)
        self.list_instances.setSelectionMode(QTreeWidget.ExtendedSelection)
        self.list_instances.setMinimumHeight(120)
        self.list_instances.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.list_instances.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.list_instances.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.list_instances.itemSelectionChanged.connect(self.on_selection_changed)
        self.list_instances.itemClicked.connect(self.on_item_clicked)
        layout.addWidget(self.list_instances, stretch=1)

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
        layout.addLayout(btn_layout)

        self.apply_tooltips()

    def apply_tooltips(self):
        self.btn_new.setToolTip("Create a new plant and enter paint mode.")
        self.btn_merge.setToolTip(
            "Select 2+ plants with Shift+click in the list, then merge into one plant."
        )
        self.btn_del.setToolTip("Delete the selected plant(s).")
        self.list_instances.setToolTip(
            "Select plants to edit. Multi-selection is only available in the Annotation tab."
        )

    def _is_annotation_tab_active(self):
        if hasattr(self.workspaces, "currentIndex"):
            return self.workspaces.currentIndex() == self.annotation_tab_index
        return True

    def sync_toolbar_for_tab(self, workspace_index):
        enabled = workspace_index == self.annotation_tab_index
        self.btn_new.setEnabled(enabled)
        self.btn_merge.setEnabled(enabled)
        self.btn_del.setEnabled(enabled)

    def populate_list(self):
        plant_uids = self.model.plant_uids()
        self.lbl_instance_count.setText(f"<b>Plants ({len(plant_uids)} total):</b>")

        current_uids = set(plant_uids)
        existing_uids = {
            self.list_instances.topLevelItem(i).data(0, Qt.UserRole)
            for i in range(self.list_instances.topLevelItemCount())
        }

        if current_uids == existing_uids:
            self.list_instances.setSortingEnabled(False)
            for i in range(self.list_instances.topLevelItemCount()):
                item = self.list_instances.topLevelItem(i)
                uid = item.data(0, Qt.UserRole)
                area = self.model.areas.get(uid, 0)
                # Plant numbers are edited in the metadata table, so refresh the label too.
                number = (self.model.plants_meta.get(uid) or {}).get("plant_num") or uid
                kind = "Seed" if self.model.is_seed_placeholder(uid) else "Plant"
                item.setText(0, f"{kind} {number}")
                item.setText(1, f"{area:,}")
                item.setData(1, Qt.UserRole, area)
                if uid in self.model.color_map:
                    c = self.model.color_map[uid]
                    pix = QPixmap(16, 16)
                    pix.fill(QColor(c[0], c[1], c[2]))
                    item.setIcon(0, QIcon(pix))

            self.list_instances.setSortingEnabled(True)
            return

        self.list_instances.blockSignals(True)
        self.list_instances.clearSelection()
        self.list_instances.setSortingEnabled(False)
        self.list_instances.clear()

        for uid in plant_uids:
            area = self.model.areas.get(uid, 0)
            is_seed = self.model.is_seed_placeholder(uid)
            number = (self.model.plants_meta.get(uid) or {}).get("plant_num") or uid
            item = SortableTreeItem([
                f"{'Seed' if is_seed else 'Plant'} {number}", f"{area:,}"
            ])
            # The uid stays the sort key, so rows keep their left-to-right plate order.
            item.setData(0, Qt.UserRole, uid)
            item.setData(1, Qt.UserRole, area)
            if is_seed:
                item.setForeground(0, QColor("#888888"))
                item.setForeground(1, QColor("#888888"))

            if uid in self.model.color_map:
                c = self.model.color_map[uid]
                pix = QPixmap(16, 16)
                pix.fill(QColor(c[0], c[1], c[2]))
                item.setIcon(0, QIcon(pix))

            self.list_instances.addTopLevelItem(item)

            if uid in self.model.selected_uids:
                item.setSelected(True)

        self.list_instances.setSortingEnabled(True)
        self.list_instances.blockSignals(False)
        self.list_instances.viewport().update()

    def sync_selection(self):
        self.list_instances.blockSignals(True)

        for i in range(self.list_instances.topLevelItemCount()):
            item = self.list_instances.topLevelItem(i)
            uid = item.data(0, Qt.UserRole)
            item.setSelected(uid in self.model.selected_uids)

            if uid == self.model.active_uid:
                self.list_instances.scrollToItem(item)

        self.list_instances.blockSignals(False)

    def on_selection_changed(self):
        selected_items = self.list_instances.selectedItems()

        if len(selected_items) > 1 and not self._is_annotation_tab_active():
            QMessageBox.warning(
                self,
                "Multi-Selection Disabled",
                "Multi-selection is only supported on the Annotation tab.\n\n"
                "Please switch to that tab to select multiple plants."
            )

            self.list_instances.blockSignals(True)
            for item in selected_items[1:]:
                item.setSelected(False)
            self.list_instances.blockSignals(False)

            selected_items = [selected_items[0]]

        selected_uids = [item.data(0, Qt.UserRole) for item in selected_items]
        self.model.set_selection(selected_uids)

    def on_item_clicked(self, item, column):
        uid = item.data(0, Qt.UserRole)

        if self._is_annotation_tab_active() and len(self.model.selected_uids) == 1:
            if hasattr(self.workspaces, "canvas_review"):
                self.workspaces.canvas_review.zoom_to_plant(uid)
            self.zoom_requested.emit(uid)

    def action_new_instance(self):
        if not self._is_annotation_tab_active():
            QMessageBox.warning(
                self, "Action Restricted",
                "Please switch to the Annotation tab to create new plants."
            )
            return

        self.model.set_selection([])
        new_uid = self.model.prepare_new_uid()
        self.model.active_uid = new_uid

        if self.review_tool_panel is not None:
            self.review_tool_panel.force_mode("PAINT")

    def action_merge(self):
        if not self._is_annotation_tab_active():
            QMessageBox.warning(
                self, "Action Restricted",
                "Please switch to the Annotation tab to merge plants."
            )
            return

        if len(self.model.selected_uids) < 2:
            return

        new_id = self.model.merge_instances(list(self.model.selected_uids))
        if new_id is not None:
            self.model.set_selection([new_id])

    def action_delete(self):
        if self.model.selected_uids:
            uids_to_delete = list(self.model.selected_uids)
            self.model.set_selection([])
            self.model.delete_instances(uids_to_delete)

    def set_review_tool_panel(self, panel):
        self.review_tool_panel = panel
