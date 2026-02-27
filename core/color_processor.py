from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

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
            ColorTarget("Light Blue", (98, 125, 200), 5.0, (220, 180, 90)),
            ColorTarget("Dark Navy", (112, 200, 55), 10.0, (60, 30, 10)),
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
