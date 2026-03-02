import torch
import torchvision.transforms as T
from PIL import Image
from abc import ABC, abstractmethod


class Metric(ABC):
    """
    Base class for all metrics.
    Each metric is computed over 4 image regions:
      - full:    entire image
      - bbox:    image cropped to the mask's bounding box
      - masked:  only the masked (inpainted) area
      - unmasked: only the unmasked (preserved) area
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.transform = T.Compose([
            T.ToTensor(),          # [0,1] float32
        ])
        self.REGIONS = ("full", "bbox", "masked", "unmasked")


    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @abstractmethod
    def get_name(self) -> str: ...

    @abstractmethod
    def update(self, original: Image.Image, mask: Image.Image,
               prompt: str, output: Image.Image): ...

    @abstractmethod
    def compute(self) -> dict: ...

    @abstractmethod
    def reset(self): ...

    # ------------------------------------------------------------------
    # Shared region-extraction helpers
    # ------------------------------------------------------------------

    def _extract_regions(self, src_tensor: torch.Tensor,
                         output_tensor: torch.Tensor,
                         binary_mask: torch.Tensor):
        """
        Returns a dict of (src_region, output_region, weight) tuples.
        All tensors are (1, C, H, W).  weight is a float or 1-d mask.

        Regions
        -------
        full      – entire image, no masking
        bbox      – tight bounding box around the mask (crop both images)
        masked    – pixels where binary_mask == 0
        unmasked  – pixels where binary_mask == 1
        """
        regions = {}

        # --- full image ---
        regions["full"] = (src_tensor, output_tensor, None)

        # --- bounding box ---
        bbox = self._mask_bbox(binary_mask)
        if bbox is not None:
            r0, r1, c0, c1 = bbox
            bounded_src = src_tensor[:, :, r0:r1, c0:c1] if src_tensor is not None else None
            bounded_output = output_tensor[:, :, r0:r1, c0:c1]
            regions["bbox"] = (
                bounded_src,
                bounded_output,
                None,
            )
        else:
            regions["bbox"] = (src_tensor, output_tensor, None)

        # --- masked area ---
        inv_mask = 1.0 - binary_mask
        regions["masked"] = (src_tensor, output_tensor, inv_mask)

        # --- unmasked area ---
        regions["unmasked"] = (src_tensor, output_tensor, binary_mask)

        return regions

    @staticmethod
    def _mask_bbox(binary_mask: torch.Tensor):
        """Return (row_min, row_max, col_min, col_max) or None."""
        # binary_mask: (1, 1, H, W)
        m = binary_mask.squeeze()  # (H, W)
        rows = torch.any(m > 0.5, dim=1)
        cols = torch.any(m > 0.5, dim=0)
        if not rows.any():
            return None
        r0, r1 = rows.nonzero(as_tuple=False)[[0, -1]].flatten().tolist()
        c0, c1 = cols.nonzero(as_tuple=False)[[0, -1]].flatten().tolist()
        return int(r0), int(r1) + 1, int(c0), int(c1) + 1

    # ------------------------------------------------------------------
    # Statistics helper
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stats(samples: list) -> dict:
        import numpy as np
        if not samples:
            return {}
        arr = np.array(samples, dtype=np.float64)
        return {
            "mean":   float(np.mean(arr)),
            "std":    float(np.std(arr)),
            "median": float(np.median(arr)),
            "min":    float(np.min(arr)),
            "max":    float(np.max(arr)),
            "n":      int(len(arr)),
        }
