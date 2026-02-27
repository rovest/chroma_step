from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QThread, pyqtSignal

# Headless FreeCAD binary/lib directory. Update this path per environment.
# Example (Windows): r"C:\Program Files\FreeCAD 0.21\bin"
FREECAD_BIN_PATH = os.environ.get("FREECAD_BIN_PATH", "/usr/lib/freecad-python3/lib")


class CADWorker(QThread):
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        payload: list[dict[str, Any]],
        image_path: str | None = None,
        output_dir: str | None = None,
        scale_factor: float = 0.1,
        parent=None,
    ) -> None:
        super().__init__(parent)
        # Unified DTO: [{"layer_name": str, "z_height": float, "contours": list}]
        self.payload = [dict(item) for item in payload]
        self.image_path = image_path
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.scale_factor = float(scale_factor)

    def run(self) -> None:
        freecad = None
        part = None
        doc = None

        try:
            self._ensure_freecad_path()
            import FreeCAD as freecad  # type: ignore
            import MeshPart  # type: ignore
            import Part as part  # type: ignore

            self._check_interruption()
            self.status_signal.emit("Initializing FreeCAD document...")
            self.progress_signal.emit(3)
            doc = freecad.newDocument("ChromaStep")

            if not self.payload:
                raise ValueError("Worker payload is empty.")

            total_contours = sum(len(layer.get("contours", [])) for layer in self.payload)
            if total_contours == 0:
                raise ValueError("No contours provided for CAD generation.")

            solids: list[Any] = []
            processed_contours = 0

            for layer in self.payload:
                self._check_interruption()

                layer_name = str(layer.get("layer_name", "Layer"))
                z_height = float(layer.get("z_height", 0.0))
                contours = layer.get("contours", [])

                # Absolute stepped rule: every layer extrudes from Z=0.0 to target z_height.
                base_z = 0.0
                thickness = z_height
                if thickness <= 0:
                    processed_contours += len(contours)
                    continue

                self.status_signal.emit(f"Extruding {layer_name}...")
                for contour in contours:
                    self._check_interruption()
                    vectors = self._contour_to_vectors(freecad, contour, base_z)
                    if len(vectors) < 4:
                        processed_contours += 1
                        continue

                    wire = part.makePolygon(vectors)
                    if wire.isNull() or not wire.isClosed():
                        processed_contours += 1
                        continue

                    face = part.Face(wire)
                    if face.isNull():
                        processed_contours += 1
                        continue

                    extruded = face.extrude(freecad.Vector(0, 0, thickness))
                    if not extruded.isNull():
                        solids.append(extruded)

                    processed_contours += 1
                    progress = 5 + int((processed_contours / total_contours) * 75)
                    self.progress_signal.emit(min(progress, 80))

            if not solids:
                raise ValueError("No valid solids could be generated from contours.")

            self._check_interruption()
            self.status_signal.emit("Applying Boolean Union...")
            self.progress_signal.emit(85)
            fused_solid = self._fuse_solids(part, solids)

            self._check_interruption()
            obj = doc.addObject("Part::Feature", "FinalRelief")
            obj.Shape = fused_solid
            doc.recompute()

            stem = Path(self.image_path).stem if self.image_path else "chromastep_model"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            step_path = self.output_dir / f"{stem}_{timestamp}.step"
            stl_path = self.output_dir / f"{stem}_{timestamp}.stl"

            self.status_signal.emit("Exporting STEP...")
            self.progress_signal.emit(92)
            fused_solid.exportStep(str(step_path))

            self._check_interruption()
            self.status_signal.emit("Exporting STL...")
            self.progress_signal.emit(97)
            mesh = MeshPart.meshFromShape(Shape=fused_solid, MaxLength=1.0)
            mesh.write(str(stl_path))

            self.progress_signal.emit(100)
            self.status_signal.emit("CAD generation complete.")
            self.finished_signal.emit(True, f"STEP: {step_path}\nSTL: {stl_path}")

        except InterruptedError as exc:
            self.status_signal.emit("Generation canceled by user.")
            self.finished_signal.emit(False, str(exc))
        except Exception as exc:
            self.error_signal.emit(str(exc))
        finally:
            if freecad is not None and doc is not None:
                try:
                    freecad.closeDocument(doc.Name)
                except Exception:
                    pass

    def _ensure_freecad_path(self) -> None:
        if FREECAD_BIN_PATH and FREECAD_BIN_PATH not in sys.path:
            sys.path.append(FREECAD_BIN_PATH)

    def _check_interruption(self) -> None:
        if self.isInterruptionRequested():
            raise InterruptedError("User cancelled operation.")

    def _fuse_solids(self, part: Any, solids: list[Any]):
        if len(solids) == 1:
            return solids[0]

        try:
            fused = part.MultiFuse(solids)
            return fused.removeSplitter()
        except Exception:
            fused = solids[0]
            for next_solid in solids[1:]:
                self._check_interruption()
                fused = fused.fuse(next_solid)
            return fused.removeSplitter() if hasattr(fused, "removeSplitter") else fused

    def _contour_to_vectors(self, freecad: Any, contour: Any, base_z: float) -> list[Any]:
        points: list[tuple[float, float]] = []
        for item in contour:
            if len(item) == 1:
                x_raw, y_raw = item[0]
            else:
                x_raw, y_raw = item
            x = float(x_raw) * self.scale_factor
            y = float(y_raw) * self.scale_factor
            points.append((x, y))

        # Remove consecutive duplicates.
        normalized: list[tuple[float, float]] = []
        for p in points:
            if not normalized or normalized[-1] != p:
                normalized.append(p)

        # Handle OpenCV-style already-closed contours and close exactly once.
        if normalized and normalized[0] == normalized[-1]:
            normalized.pop()
        if len(normalized) < 3:
            return []
        normalized.append(normalized[0])

        return [freecad.Vector(x, y, base_z) for x, y in normalized]

