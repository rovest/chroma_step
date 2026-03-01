from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import cv2
import numpy as np
import pyvista as pv
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QDesktopServices, QImage, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import QUrl
from pyvistaqt import QtInteractor

from core.color_processor import ColorProcessor, ColorTarget
from ui.worker import CADWorker


class PreviewLabel(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._placeholder = "No image"
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(210)
        self.setStyleSheet(
            "QLabel { border: 1px solid #3a4450; background-color: #0f1520; color: #7f8a99; }"
        )
        self.setText(self._placeholder)

    def set_preview_pixmap(self, pixmap: QPixmap) -> None:
        self._pixmap = pixmap
        self._render_scaled()

    def clear_preview(self) -> None:
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText(self._placeholder)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._render_scaled()

    def _render_scaled(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.FastTransformation)
        self.setPixmap(scaled)


class VisualSection(QFrame):
    def __init__(self, title: str, footer: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(3)

        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.preview_label = PreviewLabel()
        self.footer_label = QLabel(footer)
        self.footer_label.setAlignment(Qt.AlignCenter)
        self.footer_label.setStyleSheet("color: #b6bfcc;")

        layout.addWidget(self.title_label)
        layout.addWidget(self.preview_label, 1)
        layout.addWidget(self.footer_label)


@dataclass
class LayerState:
    target_index: int
    target: ColorTarget
    layer_title: str
    initial_tolerance: int = 35


class LayerWidget(QFrame):
    tolerance_released = pyqtSignal()
    HEIGHT_STEP_MM = 1.2

    def __init__(self, state: LayerState, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.state = state
        self._build_ui()

    def _build_ui(self) -> None:
        self.setFrameShape(QFrame.StyledPanel)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        title = QLabel(self.state.layer_title)
        title.setStyleSheet("font-weight: 600;")

        row = QHBoxLayout()
        row.setSpacing(8)

        color_text = QLabel("Color:")
        self.color_swatch = QLabel()
        self.color_swatch.setFixedSize(20, 20)
        bgr = self.state.target.swatch_bgr
        self.color_swatch.setStyleSheet(
            f"border: 1px solid #7f8a99; background-color: rgb({bgr[2]}, {bgr[1]}, {bgr[0]});"
        )

        height_text = QLabel("Height:")
        self.z_spin = QDoubleSpinBox()
        self.z_spin.setRange(0.0, 100.0)
        self.z_spin.setDecimals(1)
        self.z_spin.setSingleStep(self.HEIGHT_STEP_MM)
        self.z_spin.setSuffix(" mm")
        self.z_spin.setValue(float(self.state.target.default_z))
        self.z_spin.setButtonSymbols(QDoubleSpinBox.UpDownArrows)

        row.addWidget(color_text)
        row.addWidget(self.color_swatch)
        row.addStretch(1)
        row.addWidget(height_text)
        row.addWidget(self.z_spin)

        tol_label = QLabel("Color Tolerance:")
        slider_row = QHBoxLayout()
        slider_row.setSpacing(8)
        narrow = QLabel("Narrow")
        wide = QLabel("Wide")
        narrow.setStyleSheet("color: #9ca7b6;")
        wide.setStyleSheet("color: #9ca7b6;")

        self.tolerance_slider = QSlider(Qt.Horizontal)
        self.tolerance_slider.setRange(0, 255)
        self.tolerance_slider.setValue(int(np.clip(self.state.initial_tolerance, 0, 255)))
        self.tolerance_slider.sliderReleased.connect(self.tolerance_released.emit)

        slider_row.addWidget(narrow)
        slider_row.addWidget(self.tolerance_slider, 1)
        slider_row.addWidget(wide)

        root.addWidget(title)
        root.addLayout(row)
        root.addWidget(tol_label)
        root.addLayout(slider_row)

    def current_tolerance(self) -> int:
        return int(self.tolerance_slider.value())


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ChromaStep 3D - Color Relief Studio [v0.1]")
        self.resize(1280, 760)

        self.processor = ColorProcessor()
        self.layer_widgets: List[LayerWidget] = []
        self.image_path: str | None = None
        self.worker: CADWorker | None = None
        self.recent_export_dirs: List[str] = []

        self._build_ui()
        self._init_default_layers()
        self._apply_styles()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, 1)

        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(8)
        left_header = QLabel("Image Visualization")
        left_layout.addWidget(left_header)

        self.original_section = VisualSection("Original Image", "No file loaded")
        self.mask_section = QFrame()
        self.mask_section.setFrameShape(QFrame.StyledPanel)
        mask_layout = QVBoxLayout(self.mask_section)
        mask_layout.setContentsMargins(6, 4, 6, 6)
        mask_layout.setSpacing(3)
        self.mask_title_label = QLabel("Live Segmentation Map (Mask) / 3D Viewer")
        self.mask_title_label.setAlignment(Qt.AlignCenter)
        self.view_stack = QStackedWidget()
        self.mask_preview_label = PreviewLabel()
        self.viewer = QtInteractor(self)
        self.viewer.set_background("#0f1520")
        self.view_stack.addWidget(self.mask_preview_label)  # index 0: 2D mask
        self.view_stack.addWidget(self.viewer)  # index 1: 3D model
        self.view_stack.setCurrentIndex(0)
        self.mask_footer_label = QLabel("Mask Preview")
        self.mask_footer_label.setAlignment(Qt.AlignCenter)
        self.mask_footer_label.setStyleSheet("color: #b6bfcc;")
        mask_layout.addWidget(self.mask_title_label)
        mask_layout.addWidget(self.view_stack, 1)
        mask_layout.addWidget(self.mask_footer_label)
        left_layout.addWidget(self.original_section, 1)
        left_layout.addWidget(self.mask_section, 1)

        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(8)
        right_layout.addWidget(QLabel("ChromaStep Settings"))

        self.load_btn = QPushButton("Load Image...")
        self.load_btn.clicked.connect(self._choose_image)
        right_layout.addWidget(self.load_btn)

        self.layer_scroll = QScrollArea()
        self.layer_scroll.setWidgetResizable(True)
        self.layer_container = QWidget()
        self.layer_layout = QVBoxLayout(self.layer_container)
        self.layer_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_layout.setSpacing(6)
        self.layer_layout.addStretch(1)
        self.layer_scroll.setWidget(self.layer_container)
        right_layout.addWidget(self.layer_scroll, 1)

        self.generate_btn = QPushButton("Generate 3D Solid Model")
        self.generate_btn.setEnabled(False)
        self.generate_btn.clicked.connect(self.start_generation)
        self.export_stl_btn = QPushButton("Export as STL")
        self.export_stl_btn.setEnabled(False)
        self.export_step_btn = QPushButton("Export as STEP")
        self.export_step_btn.setEnabled(False)

        right_layout.addWidget(self.generate_btn)
        right_layout.addWidget(self.export_stl_btn)
        right_layout.addWidget(self.export_step_btn)

        self.recent_label = QLabel("Recent Results")
        self.recent_list = QListWidget()
        self.recent_list.setMaximumHeight(120)
        self.recent_list.itemDoubleClicked.connect(self._open_recent_result)
        right_layout.addWidget(self.recent_label)
        right_layout.addWidget(self.recent_list)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 7)
        splitter.setStretchFactor(1, 3)

        bottom_bar = QHBoxLayout()
        self.status_label = QLabel("Status: Ready. Load an image.")
        self.progress_caption = QLabel("QProgressBar:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_percent = QLabel("0%")
        self.progress_bar.valueChanged.connect(self._on_progress_changed)

        bottom_bar.addWidget(self.status_label, 1)
        bottom_bar.addWidget(self.progress_caption)
        bottom_bar.addWidget(self.progress_bar)
        bottom_bar.addWidget(self.progress_percent)
        root_layout.addLayout(bottom_bar)

    def _init_default_layers(self) -> None:
        defaults = [
            "Layer 1: Base (Gray)",
            "Layer 2: Middle (Light Blue)",
            "Layer 3: Top (Dark Navy)",
        ]
        for i, title in enumerate(defaults):
            target = self.processor.color_targets[i % len(self.processor.color_targets)]
            layer = LayerWidget(LayerState(i, target, title, initial_tolerance=35))
            layer.tolerance_released.connect(self._on_tolerance_adjusted)
            self.layer_layout.insertWidget(self.layer_layout.count() - 1, layer)
            self.layer_widgets.append(layer)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background-color: #1f252d;
                color: #d8dee9;
                font-size: 12px;
            }
            QFrame {
                border: 1px solid #343f4d;
                background-color: #252d38;
            }
            QLabel {
                background: transparent;
                border: none;
            }
            QPushButton {
                border: 1px solid #4b586b;
                border-radius: 4px;
                padding: 7px;
                background-color: #2f3948;
                color: #d8dee9;
            }
            QPushButton:disabled {
                color: #7f8a99;
                background-color: #293240;
            }
            QPushButton#primary {
                background-color: #276aa0;
                border: 1px solid #4aa0df;
                font-weight: 600;
            }
            QProgressBar {
                border: 1px solid #4b586b;
                border-radius: 3px;
                text-align: center;
                background-color: #111722;
            }
            QProgressBar::chunk {
                background-color: #4aa0df;
            }
            QScrollArea {
                border: 1px solid #343f4d;
                background-color: #202a36;
            }
            QSlider::groove:horizontal {
                border: 1px solid #4b586b;
                height: 6px;
                background: #1a2230;
                margin: 2px 0;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4aa0df;
                border: 1px solid #9dcff2;
                width: 12px;
                margin: -5px 0;
                border-radius: 6px;
            }
            """
        )
        self.generate_btn.setObjectName("primary")
        # Re-apply style on primary button after objectName set
        self.generate_btn.style().unpolish(self.generate_btn)
        self.generate_btn.style().polish(self.generate_btn)

    def _choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return

        image_bgr = cv2.imread(path)
        if image_bgr is None:
            self.image_path = None
            self.original_section.preview_label.clear_preview()
            self.mask_preview_label.clear_preview()
            self.status_label.setText("Status: Failed to load image.")
            return

        height, width = image_bgr.shape[:2]
        if width > 640 or height > 640:
            answer = QMessageBox.question(
                self,
                "Image Too Large",
                "Image is larger than 640x640. Do you want to automatically downscale it to fit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.No:
                return

            scale = 640.0 / float(max(width, height))
            new_w = max(1, int(round(width * scale)))
            new_h = max(1, int(round(height * scale)))
            image_bgr = cv2.resize(
                image_bgr,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA,
            )

        color_clusters = self.processor.analyze_image_colors(image_bgr, k_clusters=5)
        if not color_clusters:
            QMessageBox.warning(
                self,
                "Color Analysis Failed",
                "No valid dominant color clusters were detected in this image.",
            )
            return

        self.processor.set_image(image_bgr)

        self.image_path = path
        self._rebuild_layers_from_clusters(color_clusters)
        self.generate_btn.setEnabled(True)
        self._set_original_preview(image_bgr, path.split("/")[-1])
        self.view_stack.setCurrentIndex(0)
        self.mask_footer_label.setText("Mask Preview")
        self._update_mask_preview()
        self.status_label.setText("Status: Image loaded. Configure layers and click Generate.")

    def _clear_layer_widgets(self) -> None:
        for layer in self.layer_widgets:
            self.layer_layout.removeWidget(layer)
            layer.deleteLater()
        self.layer_widgets.clear()

    def _rebuild_layers_from_clusters(self, clusters: List[dict[str, object]]) -> None:
        self._clear_layer_widgets()
        self.processor.color_targets = []

        for idx, cluster in enumerate(clusters):
            color_bgr = tuple(int(v) for v in cluster["color_bgr"])
            color_hsv = tuple(int(v) for v in cluster["color_hsv"])
            tolerance = int(cluster["tolerance"])
            z_height = float(idx * LayerWidget.HEIGHT_STEP_MM)

            target = ColorTarget(
                name=f"Cluster {idx + 1}",
                base_hsv=color_hsv,
                default_z=z_height,
                swatch_bgr=color_bgr,
            )
            self.processor.color_targets.append(target)

            layer = LayerWidget(
                LayerState(
                    target_index=idx,
                    target=target,
                    layer_title=f"Layer {idx + 1}: {target.name}",
                    initial_tolerance=tolerance,
                )
            )
            layer.tolerance_released.connect(self._on_tolerance_adjusted)
            self.layer_layout.insertWidget(self.layer_layout.count() - 1, layer)
            self.layer_widgets.append(layer)

    def _set_original_preview(self, image_bgr: np.ndarray, file_name: str) -> None:
        pixmap = self._mask_to_pixmap(image_bgr)
        if pixmap.isNull():
            self.original_section.preview_label.clear_preview()
            self.original_section.footer_label.setText("Invalid image")
            return
        self.original_section.preview_label.set_preview_pixmap(pixmap)
        self.original_section.footer_label.setText(file_name)

    def _collect_layer_settings(self) -> List[dict[str, int]]:
        return [
            {"target_index": layer.state.target_index, "tolerance": layer.current_tolerance()}
            for layer in self.layer_widgets
        ]

    def _update_mask_preview(self) -> None:
        if not self.processor.has_image:
            return
        combined = self.processor.build_combined_mask(self._collect_layer_settings())
        if combined is None:
            return
        pixmap = self._mask_to_pixmap(combined)
        self.mask_preview_label.set_preview_pixmap(pixmap)

    def _on_tolerance_adjusted(self) -> None:
        self.view_stack.setCurrentIndex(0)
        self.mask_footer_label.setText("Mask Preview")
        self._update_mask_preview()

    def _collect_generation_layers(self) -> list[dict[str, Any]]:
        return [
            {
                "layer_name": layer.state.layer_title,
                "z_height": float(layer.z_spin.value()),
                "target_index": layer.state.target_index,
                "tolerance": layer.current_tolerance(),
            }
            for layer in self.layer_widgets
        ]

    def start_generation(self) -> None:
        if not self.processor.has_image:
            QMessageBox.warning(self, "No Image", "Load an image before generating a 3D solid.")
            return
        if self.worker is not None and self.worker.isRunning():
            return

        self.generate_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Status: Starting CAD generation...")

        try:
            generation_layers = self._collect_generation_layers()
            worker_payload = self.processor.extract_contours_for_layers(generation_layers)
        except Exception as exc:
            self.generate_btn.setEnabled(True)
            QMessageBox.critical(self, "Preparation Error", str(exc))
            return

        self.worker = CADWorker(
            payload=worker_payload,
            image_path=self.image_path,
            parent=self,
        )
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.status_signal.connect(self._on_worker_status)
        self.worker.finished_signal.connect(self._on_worker_finished)
        self.worker.error_signal.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_status(self, message: str) -> None:
        self.status_label.setText(f"Status: {message}")

    def _on_worker_finished(self, success: bool, message: str) -> None:
        self.generate_btn.setEnabled(self.processor.has_image)
        if success:
            self.status_label.setText("Status: CAD generation complete.")
            self._render_stl_in_viewer(message)
            export_dir = self._extract_export_dir(message)
            if export_dir:
                self._add_recent_export(export_dir)
            self._show_generation_complete_dialog(message, export_dir)
        else:
            self.status_label.setText("Status: CAD generation canceled.")
            QMessageBox.warning(self, "Generation Canceled", message)

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    def _on_worker_error(self, error_text: str) -> None:
        self.generate_btn.setEnabled(self.processor.has_image)
        self.status_label.setText("Status: CAD generation error.")
        QMessageBox.critical(self, "CAD Worker Error", error_text)
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

    def _on_progress_changed(self, value: int) -> None:
        self.progress_percent.setText(f"{value}%")

    def _render_stl_in_viewer(self, worker_message: str) -> None:
        stl_path = self._extract_stl_path(worker_message)
        if not stl_path:
            return
        try:
            self.viewer.clear()
            mesh = pv.read(stl_path)
            cleaned_mesh = mesh.clean(point_merging=True, tolerance=1e-5).triangulate()
            if cleaned_mesh.n_points == 0:
                QMessageBox.warning(
                    self,
                    "3D Preview Warning",
                    "Loaded STL contains no valid points after cleanup.",
                )
                return
            self.viewer.add_mesh(
                cleaned_mesh,
                color="lightblue",
                show_edges=False,
                smooth_shading=True,
                split_sharp_edges=True,
            )
            self.viewer.reset_camera()
            self.mask_footer_label.setText(Path(stl_path).name)
            self.view_stack.setCurrentIndex(1)
        except Exception as exc:
            QMessageBox.warning(self, "3D Preview Error", f"Failed to load STL preview:\n{exc}")

    def _show_generation_complete_dialog(self, message: str, export_dir: str | None) -> None:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Generation Complete")
        dialog.setIcon(QMessageBox.Information)
        dialog.setText("3D model files were generated successfully.")
        dialog.setDetailedText(message)
        open_btn = dialog.addButton("Open Folder", QMessageBox.ActionRole)
        dialog.addButton(QMessageBox.Ok)
        dialog.exec_()
        if dialog.clickedButton() is open_btn and export_dir:
            QDesktopServices.openUrl(QUrl.fromLocalFile(export_dir))

    def _add_recent_export(self, export_dir: str) -> None:
        if export_dir in self.recent_export_dirs:
            self.recent_export_dirs.remove(export_dir)
        self.recent_export_dirs.insert(0, export_dir)
        self.recent_export_dirs = self.recent_export_dirs[:5]

        self.recent_list.clear()
        for path in self.recent_export_dirs:
            item = QListWidgetItem(Path(path).name)
            item.setData(Qt.UserRole, path)
            self.recent_list.addItem(item)

    def _open_recent_result(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    @staticmethod
    def _extract_stl_path(text: str) -> str | None:
        candidate_lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in candidate_lines:
            lower = line.lower()
            if lower.startswith("stl:"):
                path = line.split(":", 1)[1].strip()
                return path if path else None
            if lower.endswith(".stl"):
                return line
        return None

    @staticmethod
    def _extract_export_dir(text: str) -> str | None:
        candidate_lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in candidate_lines:
            lower = line.lower()
            if lower.startswith("export_dir:"):
                path = line.split(":", 1)[1].strip()
                return path if path else None
        return None

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(2000)
        event.accept()

    @staticmethod
    def _mask_to_pixmap(mask: np.ndarray) -> QPixmap:
        if mask.ndim == 2:
            rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        return QPixmap.fromImage(image)
