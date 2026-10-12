"""Unexpected-error dialog and process-wide hooks for the GUI."""

import os
import platform
import sys
import threading
import traceback
from datetime import datetime

from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QFontDatabase, QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
)

REPORT_DIR = os.path.expanduser("~/.config/chronoRootAnnotationSuite/error_reports")

_lock = threading.Lock()
_bridge = None
_open_dialog = None
_session_path = None


def _format_report(exc_text):
    python = sys.version.replace("\n", " ")
    body = exc_text if exc_text.endswith("\n") else exc_text + "\n"
    stamp = datetime.now().isoformat(timespec="seconds")
    return (
        "ChronoRoot Annotation Suite — Error Report\n"
        f"Time: {stamp}\n"
        f"Python: {python}\n"
        f"Platform: {platform.platform()}\n"
        "\n"
        f"{body}"
    )


def _write_new_file(report):
    os.makedirs(REPORT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = os.path.join(REPORT_DIR, f"error-{stamp}.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(report)
    return path


def _append_to_file(path, report):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write(report)


def _persist(exc_text):
    """Print the report and store it. Returns (report, path). Never raises."""
    global _session_path
    report = _format_report(exc_text)
    try:
        sys.stderr.write(report)
        sys.stderr.flush()
    except Exception:
        pass
    path = None
    try:
        with _lock:
            if _open_dialog is not None and _session_path:
                _append_to_file(_session_path, report)
                path = _session_path
            else:
                path = _write_new_file(report)
                _session_path = path
    except Exception:
        traceback.print_exc()
    return report, path


class ErrorReportDialog(QDialog):
    def __init__(self, report, saved_path, parent=None, still_open=True):
        super().__init__(parent)
        self.setWindowTitle("Unexpected Error")
        self.setWindowModality(Qt.ApplicationModal)
        self.resize(720, 480)

        layout = QVBoxLayout(self)
        if still_open:
            lead = "The application hit an unexpected error and is still open."
        else:
            lead = "The application hit an unexpected error while starting and will close when you continue."
        if saved_path:
            location = f"A copy of this report was saved to:\n{saved_path}"
        else:
            location = "The report could not be saved to disk. Copy or save it from this window."
        note = QLabel(lead + "\n" + location)
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(note)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self._text.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self._text.setPlainText(report)
        layout.addWidget(self._text, stretch=1)

        buttons = QHBoxLayout()
        buttons.addStretch()
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(self._copy)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_as)
        continue_btn = QPushButton("Continue")
        continue_btn.setDefault(True)
        continue_btn.clicked.connect(self.accept)
        buttons.addWidget(copy_btn)
        buttons.addWidget(save_btn)
        buttons.addWidget(continue_btn)
        layout.addLayout(buttons)

    def append_report(self, report):
        current = self._text.toPlainText().rstrip()
        extra = report.strip("\n")
        self._text.setPlainText(current + "\n\n" + extra + "\n")
        cursor = self._text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._text.setTextCursor(cursor)

    def _copy(self):
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._text.toPlainText())

    def _save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Error Report",
            os.path.join(REPORT_DIR, "error-report.txt"),
            "Text Files (*.txt)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self._text.toPlainText())
        except OSError as exc:
            QMessageBox.warning(self, "Save Failed", f"Could not save the report:\n{exc}")


def _dialog_parent(parent):
    if parent is not None:
        return parent
    app = QApplication.instance()
    if app is None:
        return None
    return app.activeWindow()


def _show_saved_report(report, path, parent=None, still_open=True):
    """Show a report that was already written. Must run on the GUI thread."""
    global _open_dialog
    try:
        with _lock:
            existing = _open_dialog
        if existing is not None:
            existing.append_report(report)
            return

        dialog = ErrorReportDialog(
            report, path, parent=_dialog_parent(parent), still_open=still_open,
        )
        with _lock:
            if _open_dialog is not None:
                existing = _open_dialog
            else:
                existing = None
                _open_dialog = dialog
        if existing is not None:
            existing.append_report(report)
            return
        try:
            dialog.exec_()
        finally:
            with _lock:
                if _open_dialog is dialog:
                    _open_dialog = None
    except Exception:
        traceback.print_exc()


def report_error_text(exc_text, parent=None, still_open=True):
    """Record a traceback and show it. Call this from the GUI thread."""
    try:
        report, path = _persist(exc_text)
        _show_saved_report(report, path, parent=parent, still_open=still_open)
    except Exception:
        traceback.print_exc()


def _deliver_report(exc_text):
    report, path = _persist(exc_text)
    app = QApplication.instance()
    on_gui = app is not None and QThread.currentThread() == app.thread()
    if on_gui:
        QTimer.singleShot(0, lambda report=report, path=path: _show_saved_report(report, path))
        return
    if _bridge is not None:
        _bridge.report_ready.emit(report, path)


def _excepthook(exc_type, exc_value, exc_tb):
    if exc_type is not None and issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    try:
        _deliver_report("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
    except Exception:
        traceback.print_exception(exc_type, exc_value, exc_tb)


def _threading_excepthook(args):
    exc_type = getattr(args, "exc_type", None)
    if exc_type is not None and issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
        original = getattr(threading, "__excepthook__", None)
        if original is not None:
            original(args)
        return
    try:
        text = "".join(traceback.format_exception(
            args.exc_type, args.exc_value, args.exc_traceback
        ))
        _deliver_report(text)
    except Exception:
        traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback)


class _ReportBridge(QObject):
    report_ready = pyqtSignal(str, str)


def install_gui_error_handler():
    """Install hooks that keep the process alive and show an error report."""
    global _bridge
    if QApplication.instance() is not None and _bridge is None:
        _bridge = _ReportBridge()
        _bridge.report_ready.connect(_show_saved_report, Qt.QueuedConnection)
    sys.excepthook = _excepthook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _threading_excepthook
