import torch
import numpy as np
from PIL import Image
from collections import defaultdict

from .Metric import Metric
from utils.regions import extract_regions

class PSNR(Metric):
    """
    Peak Signal-to-Noise Ratio (dB).
    Higher is better.  Computed per sample, then aggregated.

    PSNR = 10 * log10(MAX² / MSE)

    For [0,1] tensors MAX = 1.0.
    We track per-sample values so we can report full statistics.
    """
    def __init__(self, device="cuda", max_val: float = 1.0):
        super().__init__(device=device)
        self.max_val = max_val
        self.samples: dict[str, list] = defaultdict(list)

    def get_name(self):
        return "PSNR"

    def reset(self):
        self.samples = defaultdict(list)

    def compute(self):
        return {region: self._compute_stats(self.samples[region])
                for region in self.REGIONS}

    @torch.no_grad()
    def update(self,
               original: Image.Image,
               mask: Image.Image,
               prompt: str,
               output: Image.Image):
        src_t    = self.transform(original.convert("RGB")).unsqueeze(0).to(self.device)
        mask_t   = self.transform(mask.convert("L")).unsqueeze(0).to(self.device)
        output_t = self.transform(output.convert("RGB")).unsqueeze(0).to(self.device)

        binary_mask = (mask_t > 0.5).float()
        regions = extract_regions(src_t, output_t, binary_mask)

        for region_name, (src_r, out_r, w) in regions.items():
            se = (out_r - src_r) ** 2

            if w is not None:
                w_r = w.expand_as(se)
                n   = w_r.sum()
                mse = (se * w_r).sum() / n if n > 0 else torch.tensor(0.0)
            else:
                mse = se.mean()

            mse_val = mse.cpu().item() if isinstance(mse, torch.Tensor) else float(mse)

            if mse_val <= 0:
                # Perfect reconstruction → infinite PSNR; use a sentinel
                psnr = float("inf")
            else:
                psnr = 10.0 * np.log10((self.max_val ** 2) / mse_val)

            self.samples[region_name].append(psnr)

    # ------------------------------------------------------------------
    # Override stats to handle inf values gracefully
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_stats(samples: list) -> dict:
        if not samples:
            return {}
        finite = [s for s in samples if np.isfinite(s)]
        arr    = np.array(finite, dtype=np.float64) if finite else np.array([])
        stats  = {
            "mean":     float(np.mean(arr))   if arr.size else float("inf"),
            "std":      float(np.std(arr))    if arr.size else 0.0,
            "median":   float(np.median(arr)) if arr.size else float("inf"),
            "min":      float(np.min(arr))    if arr.size else float("inf"),
            "max":      float(np.max(arr))    if arr.size else float("inf"),
            "n":        int(len(samples)),
            "n_inf":    int(len(samples) - len(finite)),
        }
        return stats
