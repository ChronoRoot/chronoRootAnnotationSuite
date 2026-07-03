from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QFrame, QSizePolicy
from PyQt5.QtCore import Qt, QTimer


class LoadingOverlay(QWidget):
    """Full-area in-place loading indicator; blocks interaction while visible."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 0.35);")
        self.hide()

        card = QFrame(self)
        card.setStyleSheet(
            "QFrame { background-color: #ffffff; border-radius: 6px; padding: 12px; }"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(10)

        self.message_label = QLabel()
        self.message_label.setWordWrap(True)
        self.message_label.setAlignment(Qt.AlignCenter)
        self.message_label.setStyleSheet("color: #333; font-size: 13px;")
        self.message_label.setMaximumWidth(460)
        self.message_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
        card_layout.addWidget(self.message_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        card_layout.addWidget(self.progress_bar)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addStretch()
        outer.addWidget(card, alignment=Qt.AlignCenter)
        outer.addStretch()

        self._card = card
        self._card.setMinimumWidth(300)
        self._card.setMaximumWidth(500)

    def resizeEvent(self, event):
        if self.parentWidget():
            self.setGeometry(self.parentWidget().rect())
        super().resizeEvent(event)

    def show_message(self, text):
        self.message_label.setMinimumHeight(0)
        self.message_label.setText(text)
        if self.parentWidget():
            self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        QTimer.singleShot(0, self._refresh_layout)

    def _refresh_layout(self):
        self.message_label.adjustSize()
        self._card.adjustSize()
        self.updateGeometry()

    def hide_overlay(self):
        self.hide()
