from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ui.worker import ModelGenerationWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ChromaStep 3D")
        self.resize(960, 640)

        self.image_path: str | None = None
        self.worker: ModelGenerationWorker | None = None
        self._original_pixmap: QPixmap | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)

        control_row = QHBoxLayout()
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self._choose_image)

        self.generate_btn = QPushButton("Generate 3D")
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self._generate_model)

        control_row.addWidget(self.load_btn)
        control_row.addWidget(self.generate_btn)
        control_row.addStretch(1)

        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(360)
        self.image_label.setStyleSheet("border: 1px solid #555; background: #f4f4f4;")

        self.status_label = QLabel("Idle")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        layout.addLayout(control_row)
        layout.addWidget(self.image_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)

    def _choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return

        self.image_path = path
        self._set_image_preview(path)
        self.status_label.setText(f"Loaded: {Path(path).name}")
        self.progress_bar.setValue(0)
        self.generate_btn.setEnabled(True)

    def _set_image_preview(self, path: str) -> None:
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.image_label.setText("Failed to load image preview")
            self._original_pixmap = None
            return

        self._original_pixmap = pixmap
        self._render_scaled_pixmap()

    def _render_scaled_pixmap(self) -> None:
        if not self._original_pixmap:
            return
        scaled = self._original_pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._render_scaled_pixmap()

    def _generate_model(self) -> None:
        if not self.image_path:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        self.generate_btn.setEnabled(False)
        self.load_btn.setEnabled(False)

        self.worker = ModelGenerationWorker(self.image_path)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self._on_generation_complete)
        self.worker.error.connect(self._on_generation_error)
        self.worker.start()

    def _on_generation_complete(self, stl_path: str, step_path: str) -> None:
        self.generate_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        QMessageBox.information(
            self,
            "Completed",
            f"3D model exported successfully.\n\nSTL: {stl_path}\nSTEP: {step_path}",
        )

    def _on_generation_error(self, message: str) -> None:
        self.generate_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.status_label.setText("Error")
        QMessageBox.critical(self, "Generation Error", message)
