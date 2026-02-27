from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


class CADBuilder:
    """Build and export a fused 2.5D model from polygon layers using FreeCAD."""

    def __init__(self) -> None:
        try:
            import FreeCAD  # noqa: F401
            import Mesh  # noqa: F401
            import Part  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "FreeCAD Python modules are not available. Ensure FreeCAD is installed "
                "and PYTHONPATH includes FreeCAD libraries."
            ) from exc

        import FreeCAD
        import Part

        self.FreeCAD = FreeCAD
        self.Part = Part

    def build_solid(self, polygon_layers: Iterable[Dict[str, object]]):
        solids: List[object] = []

        for item in polygon_layers:
            z_height = float(item["z_height"])
            if z_height <= 0:
                continue

            vertices = item["vertices"]
            solid = self._extrude_polygon(vertices, z_height)
            solids.append(solid)

        if not solids:
            raise ValueError("No valid extrusions created. Check color detection thresholds.")

        if len(solids) == 1:
            return solids[0]

        fused = solids[0].multiFuse(solids[1:])
        fused = fused.removeSplitter()
        return fused

    def export_model(self, solid, stl_path: str | Path, step_path: str | Path) -> Tuple[str, str]:
        import Mesh

        stl_path = str(Path(stl_path).resolve())
        step_path = str(Path(step_path).resolve())

        self.Part.export([solid], step_path)
        Mesh.export([solid], stl_path)
        return stl_path, step_path

    def _extrude_polygon(self, vertices: Sequence[Sequence[float]], height: float):
        points_2d = [(float(x), float(y)) for x, y in vertices]
        if len(points_2d) < 3:
            raise ValueError("Polygon must have at least 3 vertices.")

        if points_2d[0] != points_2d[-1]:
            points_2d.append(points_2d[0])

        vectors = [self.FreeCAD.Vector(x, y, 0.0) for x, y in points_2d]
        wire = self.Part.makePolygon(vectors)
        face = self.Part.Face(wire)
        return face.extrude(self.FreeCAD.Vector(0, 0, height))
