from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ColorTarget:
    name: str
    base_hsv: Tuple[int, int, int]
    default_z: float
    swatch_bgr: Tuple[int, int, int]


@dataclass
class LayerSpec:
    name: str
    z_height: float
    lower_hsv: Tuple[int, int, int]
    upper_hsv: Tuple[int, int, int]


class ColorProcessor:
    """Extract color-based polygons from an image and assign target Z heights."""

    def __init__(self, min_area: float = 100.0, approx_factor: float = 0.01) -> None:
        self.min_area = min_area
        self.approx_factor = approx_factor
        self.image_bgr: np.ndarray | None = None
        self.image_hsv: np.ndarray | None = None

        self.color_targets: List[ColorTarget] = [
            ColorTarget("Gray", (0, 0, 130), 0.0, (130, 130, 130)),
            ColorTarget("Light Blue", (98, 125, 200), 1.2, (220, 180, 90)),
            ColorTarget("Dark Navy", (112, 200, 55), 2.4, (60, 30, 10)),
        ]

        self.layer_specs: List[LayerSpec] = [
            LayerSpec("gray_base", 0.0, (0, 0, 40), (179, 50, 220)),
            LayerSpec("light_blue_mid", 5.0, (85, 40, 60), (110, 255, 255)),
            LayerSpec("dark_top", 10.0, (0, 0, 0), (179, 255, 45)),
        ]

    @property
    def has_image(self) -> bool:
        return self.image_hsv is not None

    def load_image(self, image_path: str) -> None:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")
        self.set_image(image_bgr)

    def set_image(self, image_bgr: np.ndarray) -> None:
        self.image_bgr = image_bgr
        self.image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    def build_mask_for_target(self, target_index: int, tolerance: int) -> np.ndarray:
        if self.image_hsv is None:
            raise ValueError("No image is loaded. Call load_image first.")
        if not self.color_targets:
            raise ValueError("No color targets are available for masking.")
        target = self.color_targets[target_index % len(self.color_targets)]
        return self._mask_from_hsv(target.base_hsv, tolerance)

    def build_combined_mask(self, layer_settings: Sequence[Dict[str, int]]) -> np.ndarray | None:
        if self.image_hsv is None:
            return None
        if not layer_settings:
            h, w = self.image_hsv.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

        combined: np.ndarray | None = None
        for layer in layer_settings:
            target_index = int(layer["target_index"])
            tolerance = int(layer["tolerance"])
            mask = self.build_mask_for_target(target_index, tolerance)
            combined = mask if combined is None else cv2.bitwise_or(combined, mask)
        return combined

    def extract_contours_for_layers(
        self,
        layer_settings: Sequence[Dict[str, Any]],
        min_area: float | None = None,
    ) -> List[Dict[str, Any]]:
        """Return worker DTO payload with OpenCV contours per layer.

        DTO schema:
        [{"layer_name": str, "z_height": float, "shapes": list}]
        """
        if self.image_hsv is None:
            raise ValueError("No image is loaded. Call load_image first.")

        contour_min_area = 50.0 if min_area is None else float(min_area)
        hole_min_area = 100.0
        upscale_factor = 2
        payload: List[Dict[str, Any]] = []

        for layer in layer_settings:
            target_index = int(layer["target_index"])
            tolerance = int(layer["tolerance"])
            layer_name = str(layer["layer_name"])
            z_height = float(layer["z_height"])

            mask = self.build_mask_for_target(target_index, tolerance)
            mask = cv2.resize(
                mask,
                None,
                fx=upscale_factor,
                fy=upscale_factor,
                interpolation=cv2.INTER_LINEAR,
            )
            close_kernel = np.ones((3, 3), dtype=np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            _, contour_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(
                contour_mask,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            shapes: List[Dict[str, Any]] = []
            if hierarchy is not None and len(contours) > 0:
                h = hierarchy[0]
                for idx, contour in enumerate(contours):
                    parent = int(h[idx][3])
                    if parent != -1:
                        continue

                    outer_points = self._simplify_contour(
                        contour,
                        contour_min_area * (upscale_factor**2),
                        upscale_factor,
                    )
                    if outer_points is None:
                        continue

                    hole_points_list: List[List[Tuple[int, int]]] = []
                    child_idx = int(h[idx][2])
                    while child_idx != -1:
                        hole = contours[child_idx]
                        hole_points = self._simplify_contour(
                            hole,
                            hole_min_area * (upscale_factor**2),
                            upscale_factor,
                        )
                        if hole_points is not None:
                            hole_points_list.append(hole_points)
                        child_idx = int(h[child_idx][0])

                    shapes.append({"outer": outer_points, "holes": hole_points_list})

            payload.append(
                {
                    "layer_name": layer_name,
                    "z_height": z_height,
                    "shapes": shapes,
                }
            )

        return payload

    def analyze_image_colors(
        self,
        image_bgr: np.ndarray,
        k_clusters: int = 5,
    ) -> List[Dict[str, object]]:
        """Analyze dominant image colors without downsampling.

        Returns:
            [
                {
                    "color_bgr": (b, g, r),
                    "color_hsv": (h, s, v),
                    "tolerance": int,
                },
                ...
            ]
            Sorted by cluster area (largest first).
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Empty image provided for color analysis.")

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        pixels_hsv = hsv.reshape((-1, 3)).astype(np.float32)

        total_pixels = pixels_hsv.shape[0]
        if total_pixels == 0:
            return []

        k = max(1, min(int(k_clusters), total_pixels))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
        _compactness, labels, centers = cv2.kmeans(
            pixels_hsv,
            k,
            None,
            criteria,
            5,
            cv2.KMEANS_PP_CENTERS,
        )

        labels = labels.flatten()
        min_area_pixels = int(total_pixels * 0.02)

        results: List[Dict[str, object]] = []
        for idx in range(k):
            cluster_mask = labels == idx
            area = int(np.count_nonzero(cluster_mask))
            if area < min_area_pixels:
                continue

            cluster_hsv = pixels_hsv[cluster_mask]
            if cluster_hsv.shape[0] == 0:
                continue

            std_h = float(np.std(cluster_hsv[:, 0]))
            std_s = float(np.std(cluster_hsv[:, 1]))
            tolerance = int(np.clip(max(std_h, std_s), 10, 60))

            center_hsv = centers[idx]
            h = int(np.clip(round(center_hsv[0]), 0, 179))
            s = int(np.clip(round(center_hsv[1]), 0, 255))
            v = int(np.clip(round(center_hsv[2]), 0, 255))
            color_hsv = (h, s, v)

            hsv_patch = np.uint8([[[h, s, v]]])
            bgr_patch = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2BGR)[0][0]
            color_bgr = (int(bgr_patch[0]), int(bgr_patch[1]), int(bgr_patch[2]))

            results.append(
                {
                    "area": area,
                    "color_bgr": color_bgr,
                    "color_hsv": color_hsv,
                    "tolerance": tolerance,
                }
            )

        results.sort(key=lambda item: int(item["area"]), reverse=True)
        return [
            {
                "color_bgr": item["color_bgr"],
                "color_hsv": item["color_hsv"],
                "tolerance": int(item["tolerance"]),
            }
            for item in results
        ]

    def _simplify_contour(
        self,
        contour: np.ndarray,
        min_area: float,
        scale_divisor: int = 1,
    ) -> List[Tuple[int, int]] | None:
        area = abs(cv2.contourArea(contour))
        if area < min_area:
            return None

        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None

        epsilon = max(0.5, 0.001 * perimeter)
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        if len(simplified) < 3:
            return None

        points = [
            (
                int(round(float(pt[0][0]) / float(scale_divisor))),
                int(round(float(pt[0][1]) / float(scale_divisor))),
            )
            for pt in simplified
        ]
        normalized: List[Tuple[int, int]] = []
        for p in points:
            if not normalized or normalized[-1] != p:
                normalized.append(p)
        points = normalized
        if len(points) >= 2 and points[0] == points[-1]:
            points.pop()
        if len(points) < 3:
            return None
        return points

    def process_image(self, image_path: str) -> Dict[str, object]:
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Unable to read image: {image_path}")

        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        polygons: List[Dict[str, object]] = []

        for layer in self.layer_specs:
            mask = cv2.inRange(image_hsv, np.array(layer.lower_hsv), np.array(layer.upper_hsv))
            mask = self._clean_mask(mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area:
                    continue

                epsilon = self.approx_factor * cv2.arcLength(contour, True)
                simplified = cv2.approxPolyDP(contour, epsilon, True)
                points = [(int(pt[0][0]), int(pt[0][1])) for pt in simplified]

                if len(points) < 3:
                    continue

                polygons.append(
                    {
                        "layer": layer.name,
                        "z_height": layer.z_height,
                        "vertices": points,
                    }
                )

        polygons.sort(key=lambda p: float(p["z_height"]))
        h, w = image_bgr.shape[:2]
        return {"image_size": (w, h), "polygons": polygons}

    def _mask_from_hsv(self, base_hsv: Tuple[int, int, int], tolerance: int) -> np.ndarray:
        if self.image_hsv is None:
            raise ValueError("No image is loaded. Call load_image first.")

        h, s, v = base_hsv
        tolerance = max(0, min(255, int(tolerance)))

        h_tol = max(2, min(89, tolerance // 2))
        sv_tol = tolerance

        low_s = max(0, s - sv_tol)
        low_v = max(0, v - sv_tol)
        high_s = min(255, s + sv_tol)
        high_v = min(255, v + sv_tol)

        low_h = h - h_tol
        high_h = h + h_tol

        if 0 <= low_h and high_h <= 179:
            lower = np.array([low_h, low_s, low_v], dtype=np.uint8)
            upper = np.array([high_h, high_s, high_v], dtype=np.uint8)
            return cv2.inRange(self.image_hsv, lower, upper)

        # Hue wraps at 0 and 179 in OpenCV HSV, so split into two inRange masks.
        mask = np.zeros(self.image_hsv.shape[:2], dtype=np.uint8)
        if low_h < 0:
            lower_a = np.array([0, low_s, low_v], dtype=np.uint8)
            upper_a = np.array([high_h, high_s, high_v], dtype=np.uint8)
            lower_b = np.array([180 + low_h, low_s, low_v], dtype=np.uint8)
            upper_b = np.array([179, high_s, high_v], dtype=np.uint8)
            mask = cv2.bitwise_or(
                cv2.inRange(self.image_hsv, lower_a, upper_a),
                cv2.inRange(self.image_hsv, lower_b, upper_b),
            )
        elif high_h > 179:
            lower_a = np.array([low_h, low_s, low_v], dtype=np.uint8)
            upper_a = np.array([179, high_s, high_v], dtype=np.uint8)
            lower_b = np.array([0, low_s, low_v], dtype=np.uint8)
            upper_b = np.array([high_h - 180, high_s, high_v], dtype=np.uint8)
            mask = cv2.bitwise_or(
                cv2.inRange(self.image_hsv, lower_a, upper_a),
                cv2.inRange(self.image_hsv, lower_b, upper_b),
            )
        return mask

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask
