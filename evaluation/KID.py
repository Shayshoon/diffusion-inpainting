import torch
import numpy as np
from PIL import Image
from collections import defaultdict

from torchmetrics.image.kid import KernelInceptionDistance

from .Metric import Metric

class KID(Metric):
    def __init__(self, device="cuda", subset_size=50):
        super().__init__(device=device)
        self.subset_size = subset_size
        self.kids = {
            region: KernelInceptionDistance(
                subset_size=self.subset_size, normalize=True
            ).to(self.device)
            for region in self.REGIONS
        }

    def get_name(self):
        return "KID"

    def reset(self):
        for kid in self.kids.values():
            kid.reset()

    def compute(self):
        results = {}
        for region, kid in self.kids.items():
            mean, std = kid.compute()
            results[region] = {
                "mean": mean.cpu().item(),
                "std":  std.cpu().item(),
            }
        return results

    def update(self,
               original: Image.Image,
               mask: Image.Image,
               prompt: str,
               output: Image.Image):
        src_tensor    = self.transform(original.convert("RGB")).unsqueeze(0).to(self.device)
        mask_tensor   = self.transform(mask.convert("L")).unsqueeze(0).to(self.device)
        output_tensor = self.transform(output.convert("RGB")).unsqueeze(0).to(self.device)

        binary_mask = (mask_tensor > 0.5).float()
        regions = self._extract_regions(src_tensor, output_tensor, binary_mask)

        for region_name, (src_region, out_region, mask) in regions.items():
            if mask is not None:
                src_region = src_region * mask.expand_as(src_region)
                out_region = out_region * mask.expand_as(out_region)
            # KID expects (N, C, H, W) with 3 channels
            self.kids[region_name].update(src_region, real=True)
            self.kids[region_name].update(out_region, real=False)
