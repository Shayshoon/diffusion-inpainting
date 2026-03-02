import torch
import numpy as np
from PIL import Image
from collections import defaultdict

from .Metric import Metric


class MSE(Metric):
    REGIONS = ("full", "bbox", "masked", "unmasked")

    def __init__(self, device="cuda"):
        super().__init__(device=device)
        self.samples: dict[str, list] = defaultdict(list)

    def get_name(self):
        return "MSE"

    def reset(self):
        self.samples = defaultdict(list)

    def compute(self):
        return {region: self._compute_stats(self.samples[region])
                for region in self.REGIONS}

    def update(self,
               original: Image.Image,
               mask: Image.Image,
               prompt: str,
               output: Image.Image):
        src_t    = self.transform(original.convert("RGB")).unsqueeze(0).to(self.device)
        mask_t   = self.transform(mask.convert("L")).unsqueeze(0).to(self.device)
        output_t = self.transform(output.convert("RGB")).unsqueeze(0).to(self.device)

        binary_mask = (mask_t > 0.5).float()
        regions = self._extract_regions(src_t, output_t, binary_mask)

        for region_name, (src_r, out_r, w) in regions.items():
            se = (out_r - src_r) ** 2          # squared error per pixel/channel
            if w is not None:
                # broadcast weight over channel dim
                w_r = w.expand_as(se)
                n   = w_r.sum()
                val = (se * w_r).sum() / n if n > 0 else 0.0
            else:
                val = se.mean()
            self.samples[region_name].append(
                val.cpu().item() if isinstance(val, torch.Tensor) else float(val)
            )
