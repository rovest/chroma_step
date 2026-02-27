from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

from core.cad_builder import CADBuilder
from core.color_processor import ColorProcessor


class ModelGenerationWorker(QThread):
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)

    def __init__(self, image_path: str, output_dir: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self.image_path = image_path
        self.output_dir = Path(output_dir) if output_dir else Path(image_path).resolve().parent

    def run(self) -> None:
        try:
            self.progress.emit(5)
            self.status.emit("Reading image and extracting contours...")

            processor = ColorProcessor()
            result = processor.process_image(self.image_path)
            polygons = result["polygons"]

            self.progress.emit(35)
            self.status.emit(f"Detected {len(polygons)} polygon regions. Building CAD model...")

            builder = CADBuilder()
            solid = builder.build_solid(polygons)

            self.progress.emit(75)
            self.status.emit("Exporting STL and STEP files...")

            stem = Path(self.image_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stl_path = self.output_dir / f"{stem}_{timestamp}.stl"
            step_path = self.output_dir / f"{stem}_{timestamp}.step"

            stl_file, step_file = builder.export_model(solid, stl_path, step_path)

            self.progress.emit(100)
            self.status.emit("3D model generation complete.")
            self.finished.emit(stl_file, step_file)
        except Exception as exc:
            self.error.emit(str(exc))
