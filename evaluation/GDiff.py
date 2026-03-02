import cv2
import numpy as np
from PIL import Image
from collections import defaultdict

from .Metric import Metric

class GDiff(Metric):
    def __init__(self, device="cuda", edge_thickness: int = 5, eps: float = 1e-6):
        super().__init__(device=device)
        self.edge_thickness = edge_thickness
        self.eps = eps
        self.samples: dict[str, list] = defaultdict(list)
        self.REGIONS = ("full_edge", "inner_edge", "outer_edge")
        self.name = "GDiff"

    def update(self,
               original: Image.Image,
               mask: Image.Image,
               prompt: str,
               output: Image.Image):
        src_grad    = self._gradient_magnitude(original)  # (H, W)
        output_grad = self._gradient_magnitude(output)    # (H, W)

        binary_mask = (np.array(mask.convert("L")) > 127).astype(np.float32)  # (H, W)

        full_edge  = self._get_edge_mask(binary_mask, self.edge_thickness)
        inner_edge = full_edge * binary_mask
        outer_edge = full_edge * (1.0 - binary_mask)

        for region_name, edge_region in zip(
            self.REGIONS, [full_edge, inner_edge, outer_edge]
        ):
            n = edge_region.sum()
            if n == 0:
                continue

            grad_diff   = np.abs(output_grad - src_grad) * edge_region
            grad_orig   = np.abs(src_grad) * edge_region

            numerator   = grad_diff.sum() / n
            denominator = grad_orig.sum() / n + self.eps

            self.samples[region_name].append(float(numerator / denominator))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gradient_magnitude(img: Image.Image) -> np.ndarray:
        gray   = np.array(img.convert("L"))
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(grad_x ** 2 + grad_y ** 2)

    @staticmethod
    def _get_edge_mask(binary_mask: np.ndarray, thickness: int) -> np.ndarray:
        kernel  = np.ones((thickness, thickness), dtype=np.uint8)
        dilated = cv2.dilate(binary_mask, kernel)
        eroded  = cv2.erode(binary_mask, kernel)
        return (dilated - eroded).clip(0, 1)
