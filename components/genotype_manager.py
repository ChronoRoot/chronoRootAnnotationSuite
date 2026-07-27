import json
import os
import re

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
                             QPushButton, QTableWidgetItem, QHeaderView, QAbstractItemView,
                             QMessageBox)
from PyQt5.QtCore import Qt, pyqtSignal


class GenotypeManagerDialog(QDialog):
    genotypes_updated = pyqtSignal(list)

    def __init__(self, current_genotypes, parent=None, config_file=None):
        super().__init__(parent)
        self.current_genotypes = list(current_genotypes)
        self.config_file = config_file
        self.setWindowTitle("Manage Saved Genotypes")
        self.resize(350, 400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["ID", "Genotype Name"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.populate_table()
        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        btn_add = QPushButton("+ Add Row")
        btn_add.clicked.connect(self.add_row)
        btn_remove = QPushButton("- Remove Selected")
        btn_remove.clicked.connect(self.remove_row)

        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_remove)
        layout.addLayout(btn_layout)

        btn_save = QPushButton("Save & Close")
        btn_save.setStyleSheet("background-color: #28a745; color: white; font-weight: bold;")
        btn_save.clicked.connect(self.accept)
        layout.addWidget(btn_save)

    def populate_table(self):
        self.table.setRowCount(0)
        for i, geno in enumerate(self.current_genotypes):
            self.table.insertRow(i)
            id_item = QTableWidgetItem(str(i + 1))
            id_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
            self.table.setItem(i, 0, id_item)
            self.table.setItem(i, 1, QTableWidgetItem(geno))

    def add_row(self):
        row = self.table.rowCount()
        self.table.insertRow(row)
        id_item = QTableWidgetItem(str(row + 1))
        id_item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
        self.table.setItem(row, 0, id_item)
        self.table.setItem(row, 1, QTableWidgetItem("New_Genotype"))
        self.table.editItem(self.table.item(row, 1))

    def remove_row(self):
        rows = sorted(set(item.row() for item in self.table.selectedItems()), reverse=True)
        for row in rows:
            self.table.removeRow(row)
        for i in range(self.table.rowCount()):
            self.table.item(i, 0).setText(str(i + 1))

    def accept(self):
        genos = []
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 1)
            if item and item.text().strip():
                genos.append(self.sanitize_genotype_name(item.text()))

        if self.config_file:
            GenotypeManagerDialog._persist_genotypes(genos, self.config_file)

        self.genotypes_updated.emit(genos)
        super().accept()

    @staticmethod
    def sanitize_genotype_name(raw_text):
        """Replaces spaces with underscores to keep downstream file paths safe."""
        return raw_text.strip().replace(" ", "_")

    @staticmethod
    def _persist_genotypes(genotypes, config_file=None, config_dict=None):
        if config_dict is not None:
            config_dict["saved_genotypes"] = genotypes
            if config_file:
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=4)
            return config_dict

        if not config_file:
            return genotypes

        config = {}
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
            except (json.JSONDecodeError, ValueError):
                config = {}

        config["saved_genotypes"] = genotypes
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)
        return config


class GenotypeHelper:
    """Shared genotype combobox logic extracted from the Analyzer."""

    @staticmethod
    def resolve_genotype_text(raw_text, genotypes):
        """
        Resolves user input to a canonical genotype name.
        Supports: plain name, list index (e.g. '1'), or numbered label ('1. Col-0').
        """
        raw_text = raw_text.strip()
        if not raw_text:
            return None

        if raw_text.isdigit():
            idx = int(raw_text) - 1
            if 0 <= idx < len(genotypes):
                return genotypes[idx]

        numbered = re.match(r'^(\d+)\.\s*(.+)$', raw_text)
        if numbered:
            idx = int(numbered.group(1)) - 1
            name = GenotypeManagerDialog.sanitize_genotype_name(numbered.group(2))
            if 0 <= idx < len(genotypes):
                return genotypes[idx]
            if name in genotypes:
                return name
            return name

        if raw_text in genotypes:
            return raw_text

        return GenotypeManagerDialog.sanitize_genotype_name(raw_text)

    @staticmethod
    def refresh_combobox(combo, genotypes, preserve_text=True):
        current = combo.currentText() if preserve_text else ""
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(genotypes)
        if preserve_text:
            resolved = GenotypeHelper.resolve_genotype_text(current, genotypes)
            if resolved and resolved in genotypes:
                combo.setCurrentText(resolved)
        combo.blockSignals(False)

    @staticmethod
    def refresh_all_comboboxes(combos, genotypes):
        for combo in combos:
            if combo is not None:
                GenotypeHelper.refresh_combobox(combo, genotypes)

    @staticmethod
    def check_and_add_genotype(combo, genotypes, on_genotypes_changed=None, on_text_resolved=None,
                               config_file=None, config_dict=None, allow_create=True):
        """
        Resolves combo text to a saved genotype. Optionally adds new names to the list.
        Returns the resolved genotype name, or None if the field was empty.
        """
        raw_text = combo.currentText().strip()
        if not raw_text:
            return None

        resolved = GenotypeHelper.resolve_genotype_text(raw_text, genotypes)
        if resolved is None:
            return None

        if allow_create and resolved not in genotypes:
            genotypes.append(resolved)
            if config_file or config_dict is not None:
                GenotypeManagerDialog._persist_genotypes(genotypes, config_file, config_dict)
            if on_genotypes_changed:
                on_genotypes_changed(genotypes)

        combo.blockSignals(True)
        combo.setCurrentText(resolved)
        combo.blockSignals(False)

        if on_text_resolved:
            on_text_resolved()
        return resolved

    @staticmethod
    def sanitize_table_cell(table, item):
        """Sanitizes manual edits in genotype/plant-number table cells."""
        raw_text = item.text()
        if " " in raw_text:
            table.blockSignals(True)
            item.setText(raw_text.replace(" ", "_"))
            table.blockSignals(False)

            QMessageBox.warning(
                table, "Naming Convention Warning",
                "Spaces are not permitted in Genotype or Plant Number identifiers to maintain "
                "clean metrics maps.\n\nSpaces have been automatically converted to underscores (_)."
            )
