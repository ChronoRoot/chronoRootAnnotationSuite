import json
import os

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QFormLayout, QTableWidget,
                             QTableWidgetItem, QHeaderView, QComboBox,
                             QMessageBox, QInputDialog, QFileDialog)
from PyQt5.QtGui import QRegularExpressionValidator
from PyQt5.QtCore import Qt, pyqtSignal, QRegularExpression, QItemSelectionModel, QTimer

from core.model import normalize_plant_meta
from components.genotype_manager import GenotypeHelper, GenotypeManagerDialog
from components.ui_help import HELP_METADATA, show_help

BULK_ENTER_NEW_LABEL = "Enter new genotype…"
BULK_ENTER_NEW_ROLE = "enter_new_genotype"

# The Plant # cell carries the row identity: Qt.UserRole holds the model UID, which the
# user never sees, and ROLE_SEED_POS the plate position of a seed that never germinated.
COL_PLANT_NUM, COL_GENOTYPE, COL_GERMINATED, COL_IGNORE = range(4)
ROLE_SEED_POS = Qt.UserRole + 1


class PhenomicsControlPanel(QWidget):
    set_calib_ruler_requested = pyqtSignal()
    test_ruler_toggled = pyqtSignal(bool)
    calibration_changed = pyqtSignal()
    measure_requested = pyqtSignal(list, float)
    export_requested = pyqtSignal(dict)
    selection_changed = pyqtSignal(list)
    manage_genotypes_requested = pyqtSignal()
    overlay_labels_changed = pyqtSignal()

    metadata_dirty_changed = pyqtSignal()
    add_non_germinated_requested = pyqtSignal(bool)

    def __init__(self, global_config, config_file=None):
        super().__init__()
        self.config = global_config
        self.config_file = config_file
        self.current_cm_per_px = 1.0
        self.current_task_dir = ""
        self._external_selection_sync = False
        self._last_bulk_row_selection = set()
        self._table_selection_timer = QTimer(self)
        self._table_selection_timer.setSingleShot(True)
        self._table_selection_timer.timeout.connect(self._emit_table_selection)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        meta_header = QHBoxLayout()
        meta_header.addWidget(QLabel("<b>Plate information &amp; scale</b>"))
        meta_header.addStretch()
        btn_help = QPushButton("Help")
        btn_help.setToolTip("Open measurement and metadata help")
        btn_help.clicked.connect(lambda: show_help(self, "Metadata Help", HELP_METADATA))
        meta_header.addWidget(btn_help)
        layout.addLayout(meta_header)

        form = QFormLayout()

        self.in_plate_id = QLineEdit("Plate_01")
        self.in_condition = QLineEdit("Control")
        self.in_timepoint = QLineEdit("Day_07")

        self.in_plate_id.editingFinished.connect(lambda: self._on_plate_field_edited(self.in_plate_id, "Plate ID"))
        self.in_condition.editingFinished.connect(lambda: self._on_plate_field_edited(self.in_condition, "Condition"))
        self.in_timepoint.editingFinished.connect(lambda: self._on_plate_field_edited(self.in_timepoint, "Timepoint"))

        self.cb_calibration = QComboBox()
        self.cb_calibration.addItems([
            "Scanner DPI", "Known Image Height (cm)", "Known Image Width (cm)", "Custom Ratio (px/cm)"
        ])
        self.cb_calibration.setCurrentText(self.config.get("calib_mode", "Scanner DPI"))
        self.cb_calibration.setToolTip(
            "How pixel size is converted to cm. Use Scanner DPI from your scanner documentation, "
            "or measure a known distance on the plate."
        )
        self.cb_calibration.currentIndexChanged.connect(self.calibration_changed.emit)

        self.in_calib_val = QLineEdit(self.config.get("calib_val", "600"))
        reg_ex = QRegularExpression(r"^[0-9]+[.,]?[0-9]*$")
        self.in_calib_val.setValidator(QRegularExpressionValidator(reg_ex, self))
        self.in_calib_val.editingFinished.connect(self.calibration_changed.emit)

        calib_layout = QHBoxLayout()
        calib_layout.addWidget(self.in_calib_val)

        self.btn_set_calib = QPushButton("Set via Measurement")
        self.btn_set_calib.setToolTip(
            "Draw a line on the image for a known real-world distance to set scale."
        )
        self.btn_set_calib.clicked.connect(self.set_calib_ruler_requested.emit)
        calib_layout.addWidget(self.btn_set_calib)

        form.addRow("Plate ID:", self.in_plate_id)
        form.addRow("Condition:", self.in_condition)
        form.addRow("Timepoint:", self.in_timepoint)
        form.addRow("Calibration:", self.cb_calibration)
        form.addRow("Value:", calib_layout)
        layout.addLayout(form)

        self.btn_measure_tool = QPushButton("Check scale on image")
        self.btn_measure_tool.setToolTip("Test the ruler on the image without changing calibration.")
        self.btn_measure_tool.setCheckable(True)
        self.btn_measure_tool.setStyleSheet("background-color: #ffc107; color: black; font-weight: bold;")
        self.btn_measure_tool.toggled.connect(self._on_test_ruler_toggled)
        layout.addWidget(self.btn_measure_tool)

        self.btn_measure_tool.setEnabled(False)
        self.btn_set_calib.setEnabled(False)

        layout.addSpacing(12)

        id_header = QHBoxLayout()
        id_header.setSpacing(8)
        lbl_plant_id = QLabel("<b>Plant Identification</b>")
        id_header.addWidget(lbl_plant_id)
        id_header.addStretch()
        self.btn_manage_genos = QPushButton("Genotype List…")
        self.btn_manage_genos.setToolTip("Add, edit, or remove genotypes in the saved list.")
        self.btn_manage_genos.clicked.connect(self.manage_genotypes_requested.emit)
        id_header.addWidget(self.btn_manage_genos)
        layout.addLayout(id_header)

        id_actions = QHBoxLayout()
        id_actions.setSpacing(8)
        self.btn_add_non_germinated = QPushButton("Add \nNon-Germinated")
        self.btn_add_non_germinated.setToolTip(
            "Mark a seed that never germinated: one click on its position on the plate. "
            "It keeps its place in the left-to-right order but creates no annotation."
        )
        self.btn_add_non_germinated.setCheckable(True)
        self.btn_add_non_germinated.toggled.connect(self.add_non_germinated_requested.emit)
        self.btn_add_non_germinated.setEnabled(False)
        self.btn_import_genos = QPushButton("Import \nGenotypes")
        self.btn_import_genos.setToolTip("Load a plate genotype file (plant number to genotype).")
        self.btn_import_genos.clicked.connect(self.import_genotype_metadata)
        self.btn_export_genos = QPushButton("Export \nGenotypes")
        self.btn_export_genos.setToolTip("Save this plate's genotypes as a plant number to genotype file.")
        self.btn_export_genos.clicked.connect(self.export_genotype_metadata)
        for btn in (self.btn_add_non_germinated, self.btn_import_genos, self.btn_export_genos):
            id_actions.addWidget(btn)
        layout.addLayout(id_actions)

        self.lbl_bulk_hint = QLabel(
            "Select table rows (Ctrl/Shift+click), choose a genotype (or Enter new…), then Apply."
        )
        self.lbl_bulk_hint.setWordWrap(True)
        self.lbl_bulk_hint.setStyleSheet("color: #555; font-size: 12px; padding: 2px 0;")
        layout.addWidget(self.lbl_bulk_hint)

        bulk_layout = QHBoxLayout()
        bulk_layout.setSpacing(10)
        bulk_label = QLabel("Bulk genotype:")
        bulk_label.setMinimumWidth(95)
        self.cb_bulk_genotype = QComboBox()
        self.cb_bulk_genotype.setEditable(False)
        self.cb_bulk_genotype.setMinimumContentsLength(18)
        self.cb_bulk_genotype.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self._populate_bulk_genotype_combo(self.config.get("saved_genotypes", []))
        self.cb_bulk_genotype.activated.connect(self._on_bulk_genotype_activated)

        self.btn_apply_bulk = QPushButton("Apply")
        self.btn_apply_bulk.setToolTip("Apply the selected genotype to highlighted table rows.")
        self.btn_apply_bulk.setMinimumWidth(72)
        self.btn_apply_bulk.clicked.connect(self._apply_bulk_genotype_to_selection)

        bulk_layout.addWidget(bulk_label)
        bulk_layout.addWidget(self.cb_bulk_genotype, stretch=1)
        bulk_layout.addWidget(self.btn_apply_bulk)
        layout.addLayout(bulk_layout)

        layout.addSpacing(4)

        self.table_plants = QTableWidget(0, 4)
        self.table_plants.setHorizontalHeaderLabels(
            ["Plant #", "Genotype", "Germinated", "Ignore"]
        )
        self.table_plants.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for col in (COL_PLANT_NUM, COL_GERMINATED, COL_IGNORE):
            self.table_plants.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents
            )
        self.table_plants.setSelectionMode(QTableWidget.ExtendedSelection)
        self.table_plants.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_plants.itemSelectionChanged.connect(self._on_table_selection)
        self.table_plants.itemChanged.connect(self._on_table_item_changed)
        layout.addWidget(self.table_plants)

        btn_layout = QHBoxLayout()
        self.btn_measure = QPushButton("Run Measurements")
        self.btn_measure.setToolTip(
            "Compute traits for this plate. Unsaved changes must be saved first."
        )
        self.btn_measure.setStyleSheet("background-color: #007bff; color: white; font-weight: bold; padding: 10px;")
        self.btn_measure.clicked.connect(self._on_measure_clicked)
        self.btn_measure.setEnabled(False)

        self.btn_export = QPushButton("Save Results")
        self.btn_export.setToolTip(
            "Write trait outputs for this plate. Unfinished plates will be prompted to finish annotation first."
        )
        self.btn_export.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 10px;")
        self.btn_export.clicked.connect(self._on_export_clicked)
        self.btn_export.setEnabled(False)

        btn_layout.addWidget(self.btn_measure)
        btn_layout.addWidget(self.btn_export)
        layout.addLayout(btn_layout)

    def _on_plate_field_edited(self, line_edit, field_name):
        self._sanitize_input(line_edit, field_name)
        self.metadata_dirty_changed.emit()

    def _sanitize_input(self, line_edit, field_name):
        raw_text = line_edit.text()
        if " " in raw_text:
            line_edit.blockSignals(True)
            line_edit.setText(raw_text.replace(" ", "_"))
            line_edit.blockSignals(False)
            QMessageBox.warning(
                self, "Naming Convention Warning",
                f"Spaces are not allowed in the '{field_name}' field to prevent "
                "downstream file parsing and directory path breaks.\n\n"
                "Spaces have been automatically replaced with underscores (_)."
            )

    def _on_test_ruler_toggled(self, checked):
        self.btn_measure_tool.setText("Stop measuring" if checked else "Check scale on image")
        self.test_ruler_toggled.emit(checked)

    def _on_table_selection(self):
        if self._external_selection_sync:
            return
        self._table_selection_timer.start(0)

    def _emit_table_selection(self):
        if self._external_selection_sync:
            return

        selected_rows = sorted({idx.row() for idx in self.table_plants.selectionModel().selectedRows()})
        self._last_bulk_row_selection = set(selected_rows)
        if not selected_rows:
            return

        selected_uids = []
        for row in selected_rows:
            item = self.table_plants.item(row, COL_PLANT_NUM)
            if item:
                selected_uids.append(item.data(Qt.UserRole))
        if selected_uids:
            self.selection_changed.emit(selected_uids)

    def set_table_selection(self, uids):
        """Programmatically sync table rows to model selection without feedback loops."""
        self._external_selection_sync = True
        self.table_plants.blockSignals(True)

        selection_model = self.table_plants.selectionModel()
        selection_model.clearSelection()
        uid_set = set(uids) if uids else set()
        bulk_rows = set()

        if uid_set:
            for row in range(self.table_plants.rowCount()):
                item = self.table_plants.item(row, COL_PLANT_NUM)
                if item and item.data(Qt.UserRole) in uid_set:
                    index = self.table_plants.model().index(row, COL_PLANT_NUM)
                    selection_model.select(
                        index,
                        QItemSelectionModel.Select | QItemSelectionModel.Rows,
                    )
                    bulk_rows.add(row)

        self._last_bulk_row_selection = bulk_rows
        self.table_plants.blockSignals(False)
        self._external_selection_sync = False

    def _get_bulk_target_rows(self):
        """Rows to update for bulk genotype; survives focus loss when clicking Apply."""
        live_rows = {idx.row() for idx in self.table_plants.selectionModel().selectedRows()}
        if live_rows:
            self._last_bulk_row_selection = live_rows
        elif self._last_bulk_row_selection:
            return sorted(self._last_bulk_row_selection)
        return sorted(live_rows)

    def _apply_genotype_to_rows(self, genotype, rows):
        for row in rows:
            combo = self.table_plants.cellWidget(row, COL_GENOTYPE)
            if combo:
                combo.blockSignals(True)
                self._select_genotype_in_combo(combo, genotype)
                combo.blockSignals(False)
        self.update_overlay_labels()
        self.metadata_dirty_changed.emit()

    def _on_table_item_changed(self, item):
        GenotypeHelper.sanitize_table_cell(self.table_plants, item)
        if item.column() in (COL_GENOTYPE, COL_PLANT_NUM, COL_GERMINATED, COL_IGNORE):
            self.update_overlay_labels()
            self.metadata_dirty_changed.emit()

    @staticmethod
    def _make_flag_item(checked, tooltip, editable=True):
        """Checkbox cell used by the Germinated and Ignore columns."""
        item = QTableWidgetItem()
        flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if editable:
            flags |= Qt.ItemIsUserCheckable
        item.setFlags(flags)
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        item.setToolTip(tooltip)
        return item

    def _flag_from_row(self, row, col, default):
        item = self.table_plants.item(row, col)
        return default if item is None else item.checkState() == Qt.Checked

    def _populate_bulk_genotype_combo(self, genotypes, select_text=None):
        self.cb_bulk_genotype.blockSignals(True)
        self.cb_bulk_genotype.clear()
        self.cb_bulk_genotype.addItems(genotypes)
        if genotypes:
            self.cb_bulk_genotype.insertSeparator(self.cb_bulk_genotype.count())
        enter_idx = self.cb_bulk_genotype.count()
        self.cb_bulk_genotype.addItem(BULK_ENTER_NEW_LABEL)
        self.cb_bulk_genotype.setItemData(enter_idx, BULK_ENTER_NEW_ROLE, Qt.UserRole)

        if select_text and select_text in genotypes:
            self.cb_bulk_genotype.setCurrentText(select_text)
        elif genotypes:
            self.cb_bulk_genotype.setCurrentIndex(0)
        self.cb_bulk_genotype.blockSignals(False)

    def _bulk_item_role(self, index=None):
        if index is None:
            index = self.cb_bulk_genotype.currentIndex()
        if index < 0:
            return None
        return self.cb_bulk_genotype.itemData(index, Qt.UserRole)

    def _is_bulk_enter_new_index(self, index=None):
        return self._bulk_item_role(index) == BULK_ENTER_NEW_ROLE

    def _is_bulk_enter_new_selected(self):
        return self._is_bulk_enter_new_index()

    def _revert_bulk_genotype_selection(self):
        genotypes = self.config.get("saved_genotypes", [])
        self.cb_bulk_genotype.blockSignals(True)
        if genotypes:
            self.cb_bulk_genotype.setCurrentIndex(0)
        self.cb_bulk_genotype.blockSignals(False)

    def _on_bulk_genotype_activated(self, index):
        if not self._is_bulk_enter_new_index(index):
            return
        name = self._prompt_and_add_genotype()
        if not name:
            self._revert_bulk_genotype_selection()
            return
        rows = self._get_bulk_target_rows()
        if rows:
            self._apply_genotype_to_rows(name, rows)

    def _prompt_and_add_genotype(self):
        text, ok = QInputDialog.getText(self, "New Genotype", "Enter genotype name:")
        if not ok or not text.strip():
            return None
        return self._register_genotype(GenotypeManagerDialog.sanitize_genotype_name(text))

    def _register_genotype(self, name):
        """Add `name` to the saved list if it is new, then show it in the combos."""
        genotypes = self.config.get("saved_genotypes", [])
        if name in genotypes:
            self._populate_bulk_genotype_combo(genotypes, select_text=name)
            return name

        genotypes.append(name)
        self.config["saved_genotypes"] = genotypes
        GenotypeHelper._persist_genotypes(
            genotypes,
            config_file=self.config_file,
            config_dict=self.config,
        )
        self.refresh_genotype_combos(genotypes, select_genotype=name)
        return name

    def _resolve_bulk_genotype_for_apply(self):
        if self._is_bulk_enter_new_selected():
            genotype = self._prompt_and_add_genotype()
            if not genotype:
                self._revert_bulk_genotype_selection()
            return genotype

        genotype = self.cb_bulk_genotype.currentText().strip()
        if not genotype or self.cb_bulk_genotype.currentIndex() < 0:
            return None
        if self._is_bulk_enter_new_selected():
            return None
        return genotype

    def get_table_uids(self):
        uids = set()
        for row in range(self.table_plants.rowCount()):
            item = self.table_plants.item(row, COL_PLANT_NUM)
            if item:
                uids.add(item.data(Qt.UserRole))
        return uids

    def capture_metadata(self):
        """Snapshot the metadata table per UID before a table rebuild."""
        metadata = {}
        for row in range(self.table_plants.rowCount()):
            num_item = self.table_plants.item(row, COL_PLANT_NUM)
            if not num_item:
                continue
            uid = num_item.data(Qt.UserRole)
            metadata[uid] = normalize_plant_meta({
                "genotype": self._genotype_from_combo(self.table_plants.cellWidget(row, COL_GENOTYPE)),
                "plant_num": num_item.text(),
                "germinated": self._flag_from_row(row, COL_GERMINATED, True),
                "ignore": self._flag_from_row(row, COL_IGNORE, False),
                "seed_pos": num_item.data(ROLE_SEED_POS),
            }, uid)
        return metadata

    def _genotype_file_path(self, save):
        """Ask for a genotype file, defaulting to <plate id>.json beside the plate."""
        default = os.path.join(
            self.current_task_dir, f"{self.in_plate_id.text().strip() or 'Plate'}.json"
        )
        dialog = QFileDialog.getSaveFileName if save else QFileDialog.getOpenFileName
        title = "Export Plate Genotypes" if save else "Import Plate Genotypes"
        path, _ = dialog(self, title, default, "JSON Files (*.json)")
        return path

    def export_genotype_metadata(self):
        """Write {plant number: genotype} for every row, non-germinated plants included."""
        if not self.table_plants.rowCount():
            QMessageBox.information(self, "No Plants", "Open a plate before exporting genotypes.")
            return

        path = self._genotype_file_path(save=True)
        if not path:
            return

        genotypes = {
            meta["plant_num"]: meta["genotype"]
            for meta in self.capture_metadata().values()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(genotypes, f, indent=2)
        QMessageBox.information(
            self, "Genotypes Exported",
            f"Wrote {len(genotypes)} plant(s) to:\n{path}"
        )

    def import_genotype_metadata(self):
        """Apply a {plant number: genotype} file to the table, matching on Plant #."""
        if not self.table_plants.rowCount():
            QMessageBox.information(self, "No Plants", "Open a plate before importing genotypes.")
            return

        path = self._genotype_file_path(save=False)
        if not path:
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                imported = json.load(f)
            if not isinstance(imported, dict):
                raise ValueError("Expected an object mapping plant numbers to genotype names.")
        except (OSError, ValueError) as e:
            QMessageBox.critical(self, "Import Failed", f"Could not read the genotype file:\n{e}")
            return

        rows_by_plant_num = {}
        for row in range(self.table_plants.rowCount()):
            num_item = self.table_plants.item(row, COL_PLANT_NUM)
            if num_item:
                rows_by_plant_num.setdefault(num_item.text().strip(), row)

        rows_by_genotype, unmatched = {}, []
        for plant_num, genotype in imported.items():
            row = rows_by_plant_num.get(str(plant_num).strip())
            if row is None:
                unmatched.append(str(plant_num))
                continue
            name = GenotypeManagerDialog.sanitize_genotype_name(str(genotype))
            if name:
                rows_by_genotype.setdefault(name, []).append(row)

        applied = 0
        for name, rows in rows_by_genotype.items():
            self._register_genotype(name)
            self._apply_genotype_to_rows(name, rows)
            applied += len(rows)

        message = f"Applied genotypes to {applied} of {len(imported)} plant(s)."
        if unmatched:
            QMessageBox.warning(
                self, "Unmatched Plant Numbers",
                f"{message}\n\nNo plant on this plate carries these numbers:\n"
                + ", ".join(unmatched)
                + "\n\nAdd the missing plants (use + Non-Germinated for seeds that never "
                "grew), fix their Plant # values, then import again."
            )
        else:
            QMessageBox.information(self, "Genotypes Imported", message)

    def capture_plate_meta(self, raw_img_shape=None):
        """Snapshot plate-level metadata and calibration for persistence."""
        scale = self.get_cm_per_px(raw_img_shape)
        return {
            "plate_id": self.in_plate_id.text().strip(),
            "condition": self.in_condition.text().strip(),
            "timepoint": self.in_timepoint.text().strip(),
            "calibration_mode": self.cb_calibration.currentText(),
            "calibration_val": self.in_calib_val.text(),
            "scale_cm_px": scale,
        }

    def restore_plate_meta(self, plate_meta):
        """Restore plate fields and calibration from persisted metadata."""
        if not plate_meta:
            return
        self.in_plate_id.blockSignals(True)
        self.in_condition.blockSignals(True)
        self.in_timepoint.blockSignals(True)
        self.cb_calibration.blockSignals(True)
        self.in_calib_val.blockSignals(True)

        if plate_meta.get("plate_id"):
            self.in_plate_id.setText(str(plate_meta["plate_id"]))
        if plate_meta.get("condition"):
            self.in_condition.setText(str(plate_meta["condition"]))
        if plate_meta.get("timepoint"):
            self.in_timepoint.setText(str(plate_meta["timepoint"]))
        if plate_meta.get("calibration_mode"):
            self.cb_calibration.setCurrentText(str(plate_meta["calibration_mode"]))
        if plate_meta.get("calibration_val") is not None:
            self.in_calib_val.setText(str(plate_meta["calibration_val"]))

        self.in_plate_id.blockSignals(False)
        self.in_condition.blockSignals(False)
        self.in_timepoint.blockSignals(False)
        self.cb_calibration.blockSignals(False)
        self.in_calib_val.blockSignals(False)
        self.calibration_changed.emit()

    @staticmethod
    def remap_metadata(metadata, uid_mapping):
        """Re-key preserved metadata after model UID remapping (e.g. on save)."""
        if not uid_mapping:
            return metadata
        remapped = {}
        for old_uid, meta in metadata.items():
            new_uid = uid_mapping.get(old_uid)
            if new_uid is not None:
                remapped[new_uid] = normalize_plant_meta(meta, new_uid)
        return remapped

    def clear_table(self):
        self.table_plants.blockSignals(True)
        self.table_plants.setRowCount(0)
        self.table_plants.blockSignals(False)
        self._populate_bulk_genotype_combo(self.config.get("saved_genotypes", []))
        self.btn_measure.setEnabled(False)
        self.set_seed_mode(False)
        self.btn_add_non_germinated.setEnabled(False)
        self.reset_measurements()
        self.update_overlay_labels()

    def _apply_bulk_genotype_to_selection(self):
        selected_rows = self._get_bulk_target_rows()
        if not selected_rows:
            QMessageBox.information(
                self, "No Rows Selected",
                "Select one or more plants in the table first (Ctrl/Shift+click), "
                "then choose a genotype and click Apply."
            )
            return

        genotype = self._resolve_bulk_genotype_for_apply()
        if not genotype:
            QMessageBox.warning(
                self, "No Genotype",
                "Choose a genotype from the dropdown, or select "
                f"“{BULK_ENTER_NEW_LABEL}” to add one."
            )
            return

        self._apply_genotype_to_rows(genotype, selected_rows)

    def _on_genotype_line_edited(self, combo):
        GenotypeHelper.check_and_add_genotype(
            combo, self.config.get("saved_genotypes", []),
            on_genotypes_changed=self._on_genotypes_list_changed,
            on_text_resolved=self.update_overlay_labels,
            config_file=self.config_file,
            config_dict=self.config,
        )
        self.metadata_dirty_changed.emit()

    def _on_genotypes_list_changed(self, genotypes):
        self.config["saved_genotypes"] = genotypes
        self.refresh_genotype_combos(genotypes)

    def _genotype_from_combo(self, combo, default=""):
        """Resolved genotype of a row combo; `default` is used when nothing is assigned."""
        if not combo:
            return default
        genotypes = self.config.get("saved_genotypes", [])
        resolved = GenotypeHelper.resolve_genotype_text(combo.currentText(), genotypes)
        return resolved or combo.currentText().strip() or default

    def _on_measure_clicked(self):
        cm_per_px = self.get_cm_per_px()
        if cm_per_px is None:
            QMessageBox.warning(self, "Error", "Invalid Calibration Value.")
            return

        if not self.table_plants.selectedItems() and self.table_plants.rowCount() > 0:
            self.table_plants.selectRow(0)

        self.measure_requested.emit(self.build_plants_meta_list(), cm_per_px)

    def build_plants_meta_list(self):
        """Table snapshot as the uid-carrying list the analysis engine expects."""
        plants_meta = []
        for uid, meta in self.capture_metadata().items():
            plants_meta.append({
                "uid": uid,
                "genotype": meta["genotype"] or "Unknown",
                "plant_num": meta["plant_num"],
                "germinated": meta["germinated"],
                "ignore": meta["ignore"],
            })
        return plants_meta

    def _on_export_clicked(self):
        plate_meta = {
            "plate_id": self.in_plate_id.text().strip(),
            "condition": self.in_condition.text().strip(),
            "timepoint": self.in_timepoint.text().strip(),
            "calibration_mode": self.cb_calibration.currentText(),
            "calibration_val": self.in_calib_val.text(),
            "scale_cm_px": self.get_cm_per_px(),
        }
        self.export_requested.emit(plate_meta)

    def populate_table(self, uids, genotypes, preserved_metadata=None):
        preserved_metadata = preserved_metadata or {}
        self.table_plants.blockSignals(True)
        self.table_plants.setRowCount(0)

        self._populate_bulk_genotype_combo(genotypes)

        for row, uid in enumerate(sorted(uids)):
            saved = normalize_plant_meta(preserved_metadata.get(uid), uid)
            self.table_plants.insertRow(row)

            item_num = QTableWidgetItem(saved["plant_num"])
            item_num.setData(Qt.UserRole, uid)
            item_num.setData(ROLE_SEED_POS, saved["seed_pos"])
            self.table_plants.setItem(row, COL_PLANT_NUM, item_num)

            combo = QComboBox()
            combo.setEditable(True)
            combo.addItems(genotypes)
            self._select_genotype_in_combo(combo, saved["genotype"])
            combo.lineEdit().editingFinished.connect(
                lambda c=combo: self._on_genotype_line_edited(c)
            )
            combo.currentIndexChanged.connect(self.update_overlay_labels)
            combo.currentIndexChanged.connect(self.metadata_dirty_changed.emit)
            self.table_plants.setCellWidget(row, COL_GENOTYPE, combo)

            # A seed placeholder has no mask to measure, so its state is fixed.
            is_seed = bool(saved["seed_pos"])
            self.table_plants.setItem(row, COL_GERMINATED, self._make_flag_item(
                saved["germinated"] and not is_seed,
                "Seed marked on the plate; delete it in Annotation to remove it."
                if is_seed else
                "Uncheck for a seed that never germinated. It keeps its plate position and "
                "plant number, and exports with every trait at 0.",
                editable=not is_seed,
            ))
            self.table_plants.setItem(row, COL_IGNORE, self._make_flag_item(
                saved["ignore"],
                "Check for a plant that germinated but cannot be analyzed, for example "
                "because of contamination. It is left out of the results entirely.",
            ))

        self.table_plants.blockSignals(False)
        self._last_bulk_row_selection = set()
        self.btn_export.setEnabled(False)
        self.update_overlay_labels()

    def set_seed_mode(self, active):
        """Keep the + Non-Germinated button in sync when seed placement ends or is cancelled."""
        if self.btn_add_non_germinated.isChecked() != active:
            self.btn_add_non_germinated.blockSignals(True)
            self.btn_add_non_germinated.setChecked(active)
            self.btn_add_non_germinated.blockSignals(False)

    def refresh_genotype_combos(self, genotypes, select_genotype=None):
        previous = select_genotype or self.cb_bulk_genotype.currentText()
        if previous == BULK_ENTER_NEW_LABEL or self._is_bulk_enter_new_selected():
            previous = select_genotype
        self._populate_bulk_genotype_combo(
            genotypes,
            select_text=previous if previous in genotypes else select_genotype,
        )

        for row in range(self.table_plants.rowCount()):
            combo = self.table_plants.cellWidget(row, COL_GENOTYPE)
            if combo:
                current = combo.currentText()
                resolved_row = (
                    GenotypeHelper.resolve_genotype_text(current, genotypes) if current else ""
                )
                combo.blockSignals(True)
                combo.clear()
                combo.addItems(genotypes)
                self._select_genotype_in_combo(combo, resolved_row)
                combo.blockSignals(False)

        self.update_overlay_labels()

    @staticmethod
    def _select_genotype_in_combo(combo, genotype):
        """
        Show `genotype` in a row combo, keeping names that are absent from the saved list.
        An empty value leaves the combo blank instead of silently falling back to the
        first saved genotype, which would rewrite every unassigned plant.
        """
        if not genotype:
            combo.setCurrentIndex(-1)
            return
        if combo.findText(genotype) < 0:
            combo.addItem(genotype)
        combo.setCurrentText(genotype)

    def build_overlay_labels(self):
        label_data = {}
        genotypes = self.config.get("saved_genotypes", [])

        for uid, meta in self.capture_metadata().items():
            genotype = meta["genotype"]
            geno_text = genotype
            if not meta["germinated"]:
                geno_text = f"{genotype}\n(not germinated)".strip()
            elif meta["ignore"]:
                geno_text = f"{genotype}\n(ignored)".strip()
            label_data[uid] = {
                "plant_num": meta["plant_num"],
                "geno_text": geno_text,
                "geno_num": str(genotypes.index(genotype) + 1) if genotype in genotypes else "?",
            }
        return label_data

    def update_overlay_labels(self):
        self.overlay_labels_changed.emit()

    def get_cm_per_px(self, raw_img_shape=None):
        calib_mode = self.cb_calibration.currentText()
        try:
            val = float(self.in_calib_val.text().replace(",", "."))
            if val <= 0:
                raise ValueError
        except ValueError:
            return None

        if "DPI" in calib_mode:
            return 2.54 / val
        if "Custom Ratio" in calib_mode:
            return 1.0 / val
        if raw_img_shape:
            h_px, w_px = raw_img_shape[:2]
            if "Height" in calib_mode:
                return val / h_px
            if "Width" in calib_mode:
                return val / w_px
        return None

    def validate_calib_field(self):
        text_val = self.in_calib_val.text().strip()
        if not text_val:
            text_val = "1.0"

        if "," in text_val:
            text_val = text_val.replace(",", ".")

        if text_val.count(".") > 1:
            parts = text_val.split(".")
            text_val = parts[0] + "." + "".join(parts[1:])
            QMessageBox.warning(
                self, "Invalid Format",
                "Multiple decimal points detected. The value has been truncated to a valid number."
            )

        self.in_calib_val.blockSignals(True)
        self.in_calib_val.setText(text_val)
        self.in_calib_val.blockSignals(False)

    def enable_tools(self, enabled):
        self.btn_measure_tool.setEnabled(enabled)
        self.btn_set_calib.setEnabled(enabled)
        self.btn_measure.setEnabled(enabled)
        self.btn_add_non_germinated.setEnabled(enabled)

    def reset_measurements(self):
        self.btn_export.setEnabled(False)
