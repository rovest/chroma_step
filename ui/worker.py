from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QThread, pyqtSignal

# Fallback FreeCAD library directory used when no environment variable is provided.
DEFAULT_FREECAD_LIB_PATH = "/usr/lib/freecad-python3/lib"


class CADWorker(QThread):
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        payload: list[dict[str, Any]],
        image_path: str | None = None,
        output_path: str | None = None,
        job_mode: str = "preview",
        scale_factor: float = 0.1,
        parent=None,
    ) -> None:
        super().__init__(parent)
        # Unified DTO: [{"layer_name": str, "z_height": float, "shapes": list}]
        self.payload = [dict(item) for item in payload]
        self.image_path = image_path
        self.output_path = Path(output_path).expanduser() if output_path else None
        self.job_mode = str(job_mode).strip().lower()
        self.scale_factor = float(scale_factor)

    def run(self) -> None:
        freecad = None
        part = None
        doc = None

        try:
            self._ensure_freecad_path()
            import FreeCAD as freecad  # type: ignore
            import Mesh as mesh_module  # type: ignore
            import MeshPart as mesh_part_module  # type: ignore
            import Part as part  # type: ignore

            self._check_interruption()
            self.status_signal.emit("Initializing FreeCAD...")
            self.progress_signal.emit(3)

            all_solids = self._build_solids(freecad, part)

            if self.job_mode == "preview":
                self._check_interruption()
                self.status_signal.emit("Building preview mesh...")
                self.progress_signal.emit(90)
                preview_path = self._write_stl(mesh_module, mesh_part_module, all_solids)

                self.progress_signal.emit(100)
                self.status_signal.emit("3D preview is ready.")
                self.finished_signal.emit(
                    True,
                    (
                        "MODE: preview\n"
                        f"PREVIEW_STL: {preview_path}\n"
                        f"SOLID_COUNT: {len(all_solids)}"
                    ),
                )
                return

            if self.output_path is None:
                raise ValueError("Output path is required for export mode.")

            output_path = self.output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if self.job_mode == "step":
                self._check_interruption()
                self.status_signal.emit("Exporting STEP...")
                self.progress_signal.emit(92)

                doc = freecad.newDocument("ChromaStepExport")
                solid_features: list[Any] = []
                for index, solid in enumerate(all_solids):
                    self._check_interruption()
                    feature = doc.addObject("Part::Feature", f"Relief_{index:04d}")
                    feature.Shape = solid
                    solid_features.append(feature)
                doc.recompute()
                part.export(solid_features, str(output_path))

                self.progress_signal.emit(100)
                self.status_signal.emit("STEP export complete.")
                self.finished_signal.emit(
                    True,
                    f"MODE: step\nSTEP: {output_path}",
                )
                return

            if self.job_mode == "stl":
                self._check_interruption()
                self.status_signal.emit("Exporting STL...")
                self.progress_signal.emit(92)
                self._write_stl(mesh_module, mesh_part_module, all_solids, output_path)

                self.progress_signal.emit(100)
                self.status_signal.emit("STL export complete.")
                self.finished_signal.emit(
                    True,
                    f"MODE: stl\nSTL: {output_path}",
                )
                return

            raise ValueError(f"Unsupported worker job mode: {self.job_mode}")

        except InterruptedError as exc:
            self.status_signal.emit("Operation canceled by user.")
            self.finished_signal.emit(False, str(exc))
        except Exception as exc:
            self.error_signal.emit(str(exc))
        finally:
            if freecad is not None and doc is not None:
                try:
                    freecad.closeDocument(doc.Name)
                except Exception:
                    pass

    def _build_solids(self, freecad: Any, part: Any) -> list[Any]:
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

        return all_solids

    def _write_stl(
        self,
        mesh_module: Any,
        mesh_part_module: Any,
        solids: list[Any],
        output_path: Path | None = None,
    ) -> Path:
        final_mesh = mesh_module.Mesh()
        meshed_solid_count = 0

        for solid in solids:
            self._check_interruption()
            try:
                temp_mesh = mesh_part_module.meshFromShape(
                    Shape=solid,
                    LinearDeflection=0.1,
                    AngularDeflection=0.1,
                )
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

        if output_path is None:
            with tempfile.NamedTemporaryFile(
                prefix="chromastep_preview_",
                suffix=".stl",
                delete=False,
            ) as temp_file:
                target_path = Path(temp_file.name)
        else:
            target_path = output_path

        final_mesh.write(str(target_path))
        return target_path

    def _ensure_freecad_path(self) -> None:
        # Ignore PYTHONPATH for FreeCAD resolution to avoid mixed Qt runtime imports.
        raw_pythonpath = os.environ.get("PYTHONPATH", "").strip()
        ignored_paths = {
            os.path.abspath(p)
            for p in raw_pythonpath.split(os.pathsep)
            if p.strip()
        }
        if ignored_paths:
            sys.path = [
                p for p in sys.path if os.path.abspath(p or os.curdir) not in ignored_paths
            ]

        # Also drop known AppImage/squashfs FreeCAD paths that can conflict with PyQt5 Qt libs.
        sys.path = [
            p
            for p in sys.path
            if "squashfs-root" not in os.path.abspath(p or os.curdir).lower()
        ]

        preferred_paths: list[str] = []
        freecad_bin_path = os.environ.get("FREECAD_BIN_PATH", "").strip()
        if freecad_bin_path:
            preferred_paths.extend(p for p in freecad_bin_path.split(os.pathsep) if p.strip())
        preferred_paths.append(DEFAULT_FREECAD_LIB_PATH)

        seen_paths: set[str] = set()
        resolved_paths: list[str] = []
        for raw_path in preferred_paths:
            resolved = os.path.abspath(raw_path)
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            if os.path.isdir(resolved):
                resolved_paths.append(resolved)

        # Put FreeCAD path(s) first so import resolution is deterministic.
        for resolved in reversed(resolved_paths):
            if resolved in sys.path:
                sys.path.remove(resolved)
            sys.path.insert(0, resolved)

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
