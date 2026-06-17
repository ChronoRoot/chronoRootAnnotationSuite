from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox, QApplication
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import subprocess

class AboutTab(QWidget):
    def __init__(self):
        super().__init__()
        self.setAutoFillBackground(True)
        self.setStyleSheet("background-color: white;")
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        self.logo_label = QLabel()
        ico_path = "logo.ico"
        self.logo_label.setPixmap(QIcon(ico_path).pixmap(150, 150))
        
        self.logo_label.setStyleSheet("background-color: transparent;")
        layout.addWidget(self.logo_label, alignment=Qt.AlignCenter)

        title = QLabel("ChronoRoot Annotation Suite")
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #2c3e50; background-color: transparent;")
        layout.addWidget(title, alignment=Qt.AlignCenter)

        description = QLabel(
            "Correct root segmentations; optionally measure and compare traits."
        )
        description.setStyleSheet("font-size: 14px; color: #34495e; background-color: transparent; margin-bottom: 5px;")
        layout.addWidget(description, alignment=Qt.AlignCenter)

        self.update_btn = QPushButton("Check for Updates")
        self.update_btn.setFixedWidth(250)
        self.update_btn.setCursor(Qt.PointingHandCursor)
        self.update_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db; color: white; border-radius: 5px;
                padding: 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: #2980b9; }
        """)
        self.update_btn.clicked.connect(self.update_software)
        layout.addWidget(self.update_btn, alignment=Qt.AlignCenter)

        self.commit_label = QLabel(f"Last update: {self.get_last_commit_time()}")
        self.commit_label.setStyleSheet("color: #95a5a6; background-color: transparent; margin-top: 15px;")
        layout.addWidget(self.commit_label, alignment=Qt.AlignCenter)

    def get_git_hash(self):
        try:
            return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        except:
            return None

    def get_last_commit_time(self):
        try:
            cmd = ["git", "log", "-1", "--format=%cd", "--date=short"]
            return subprocess.check_output(cmd).decode().strip()
        except:
            return "Unknown"

    def update_software(self):
        try:
            self.update_btn.setText("Checking...")
            self.update_btn.setEnabled(False)
            QApplication.processEvents()

            old_hash = self.get_git_hash()
            subprocess.check_call(["git", "pull"], stderr=subprocess.STDOUT)
            new_hash = self.get_git_hash()

            if old_hash == new_hash:
                QMessageBox.information(self, "Update", "ChronoRoot is already up to date!")
            else:
                QMessageBox.information(self, "Update Success", 
                    "Update downloaded successfully!\nPlease restart the application to apply changes.")
                self.commit_label.setText(f"Last update: {self.get_last_commit_time()}")

        except Exception as e:
            QMessageBox.critical(self, "Update Error", 
                "Failed to update. Make sure you have an internet connection and 'git' is installed.")
        
        finally:
            self.update_btn.setText("Check for Updates")
            self.update_btn.setEnabled(True)
