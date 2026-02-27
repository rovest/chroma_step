from __future__ import annotations

import json
import os
import re
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
        # Unified DTO: [{"layer_name": str, "z_height": float, "shapes": list}]
        self.payload = [dict(item) for item in payload]
        self.image_path = image_path
        default_root = Path.home() / "ChromaStep" / "exports"
        self.output_dir = Path(output_dir) if output_dir else default_root
        self.scale_factor = float(scale_factor)

    def run(self) -> None:
        freecad = None
        part = None
        doc = None
        export_dir: Path | None = None

        try:
            self._ensure_freecad_path()
            import FreeCAD as freecad  # type: ignore
            import Mesh  # type: ignore
            import MeshPart  # type: ignore
            import Part as part  # type: ignore

            self._check_interruption()
            self.status_signal.emit("Initializing FreeCAD document...")
            self.progress_signal.emit(3)
            doc = freecad.newDocument("ChromaStep")

            if not self.payload:
                raise ValueError("Worker payload is empty.")

            total_shapes = sum(len(layer.get("shapes", [])) for layer in self.payload)
            if total_shapes == 0:
                raise ValueError("No shapes provided for CAD generation.")

            all_solids: list[Any] = []
            processed_shapes = 0

            for layer in self.payload:
                self._check_interruption()

                layer_name = str(layer.get("layer_name", "Layer"))
                z_height = float(layer.get("z_height", 0.0))
                shapes = layer.get("shapes", [])

                # Absolute stepped rule: every layer extrudes from Z=0.0 to target z_height.
                base_z = 0.0
                thickness = z_height
                if thickness <= 0:
                    processed_shapes += len(shapes)
                    continue

                self.status_signal.emit(f"Extruding {layer_name}...")
                for shape in shapes:
                    self._check_interruption()
                    outer_wire = self._points_to_wire(
                        freecad,
                        part,
                        shape.get("outer", []),
                        base_z,
                    )
                    if outer_wire is None:
                        processed_shapes += 1
                        continue

                    hole_wires: list[Any] = []
                    for hole_points in shape.get("holes", []):
                        hole_wire = self._points_to_wire(freecad, part, hole_points, base_z)
                        if hole_wire is not None:
                            hole_wires.append(hole_wire)

                    try:
                        face = part.Face([outer_wire] + hole_wires)
                        if not face.isValid() or face.Area < 1.0:
                            raise ValueError("Invalid face with holes")
                    except Exception:
                        try:
                            face = part.Face(outer_wire)
                        except Exception:
                            processed_shapes += 1
                            continue

                    if face.isNull() or not face.isValid() or face.Area < 1.0:
                        processed_shapes += 1
                        continue

                    solid = face.extrude(freecad.Vector(0, 0, thickness))
                    try:
                        if solid.isValid() and solid.Volume > 1.0:
                            all_solids.append(solid)
                    except Exception:
                        pass

                    processed_shapes += 1
                    progress = 5 + int((processed_shapes / total_shapes) * 75)
                    self.progress_signal.emit(min(progress, 80))

            if not all_solids:
                raise ValueError("No valid solids could be generated from contours.")

            self._check_interruption()
            self.status_signal.emit("Preparing export solids...")
            self.progress_signal.emit(85)

            stem = Path(self.image_path).stem if self.image_path else "chromastep_model"
            safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "chromastep_model"
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            export_dir = self.output_dir / f"{timestamp}_{safe_stem}"
            export_dir.mkdir(parents=True, exist_ok=True)

            step_path = export_dir / "final_relief.step"
            stl_path = export_dir / "final_relief.stl"
            metadata_path = export_dir / "metadata.json"

            self.status_signal.emit("Exporting STEP...")
            self.progress_signal.emit(92)
            solid_features: list[Any] = []
            for index, solid in enumerate(all_solids):
                self._check_interruption()
                feature = doc.addObject("Part::Feature", f"Relief_{index:04d}")
                feature.Shape = solid
                solid_features.append(feature)
            doc.recompute()
            part.export(solid_features, str(step_path))

            self._check_interruption()
            self.status_signal.emit("Exporting STL...")
            self.progress_signal.emit(97)
            final_mesh = Mesh.Mesh()
            meshed_solid_count = 0
            for solid in all_solids:
                self._check_interruption()
                try:
                    temp_mesh = MeshPart.meshFromShape(Shape=solid, MaxLength=0.5)
                    final_mesh.addMesh(temp_mesh)
                    meshed_solid_count += 1
                except Exception:
                    continue

            if meshed_solid_count == 0:
                raise ValueError("No valid solid could be meshed into STL.")

            try:
                final_mesh.removeDuplicatedPoints()
            except Exception:
                pass
            try:
                final_mesh.removeDuplicatedFacets()
            except Exception:
                pass
            try:
                final_mesh.removeDegeneratedFacets()
            except Exception:
                pass
            try:
                final_mesh.removeNonManifolds()
            except Exception:
                pass
            try:
                final_mesh.fillupHoles()
            except Exception:
                pass
            final_mesh.write(str(stl_path))

            metadata = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "image_path": self.image_path,
                "scale_factor": self.scale_factor,
                "output_directory": str(export_dir),
                "outputs": {
                    "step": str(step_path),
                    "stl": str(stl_path),
                },
                "layers": [
                    {
                        "layer_name": str(layer.get("layer_name", "")),
                        "z_height": float(layer.get("z_height", 0.0)),
                        "shape_count": len(layer.get("shapes", [])),
                    }
                    for layer in self.payload
                ],
                "valid_solid_count": len(all_solids),
                "meshed_solid_count": meshed_solid_count,
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            self.progress_signal.emit(100)
            self.status_signal.emit("CAD generation complete.")
            self.finished_signal.emit(
                True,
                (
                    f"EXPORT_DIR: {export_dir}\n"
                    f"STEP: {step_path}\n"
                    f"STL: {stl_path}\n"
                    f"METADATA: {metadata_path}"
                ),
            )

        except InterruptedError as exc:
            self.status_signal.emit("Generation canceled by user.")
            self.finished_signal.emit(False, str(exc))
        except Exception as exc:
            if export_dir is not None:
                try:
                    error_log = export_dir / "error.log"
                    error_log.write_text(
                        f"{datetime.now().isoformat(timespec='seconds')}\n{exc}\n",
                        encoding="utf-8",
                    )
                except Exception:
                    pass
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

    def _points_to_wire(self, freecad: Any, part: Any, points: Any, base_z: float):
        if not points:
            return None

        vectors: list[Any] = []
        for item in points:
            x_raw, y_raw = item
            x = float(x_raw) * self.scale_factor
            y = float(y_raw) * self.scale_factor
            vec = freecad.Vector(x, y, base_z)
            if not vectors or (vec.x != vectors[-1].x or vec.y != vectors[-1].y):
                vectors.append(vec)

        if len(vectors) < 3:
            return None
        if vectors[0] == vectors[-1]:
            vectors.pop()
        if len(vectors) < 3:
            return None
        vectors.append(vectors[0])

        wire = part.makePolygon(vectors)
        if wire.isNull() or not wire.isClosed():
            return None
        if hasattr(wire, "isValid") and not wire.isValid():
            return None
        return wire
