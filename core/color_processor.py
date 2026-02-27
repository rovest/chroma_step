from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


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

        self.layer_specs: List[LayerSpec] = [
            LayerSpec("gray_base", 0.0, (0, 0, 40), (179, 50, 220)),
            LayerSpec("light_blue_mid", 5.0, (85, 40, 60), (110, 255, 255)),
            LayerSpec("dark_top", 10.0, (0, 0, 0), (179, 255, 45)),
        ]

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

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask
