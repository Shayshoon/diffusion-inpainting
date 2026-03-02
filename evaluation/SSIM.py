import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from torchmetrics import StructuralSimilarityIndexMeasure

from .Metric import Metric
from .utils.regions import extract_regions

class SSIM(Metric):
    def __init__(self, device="cuda"):
        super().__init__(device=device)
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        self.samples: dict[str, list] = defaultdict(list)

    def get_name(self):
        return "SSIM"

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
        src_tensor    = self.transform(original.convert("RGB")).unsqueeze(0).to(self.device)
        mask_tensor   = self.transform(mask.convert("L")).unsqueeze(0).to(self.device)
        output_tensor = self.transform(output.convert("RGB")).unsqueeze(0).to(self.device)

        binary_mask = (mask_tensor > 0.5).float()
        regions = extract_regions(src_tensor, output_tensor, binary_mask)

        for region_name, (src_region, out_region, mask_region) in regions.items():
            if mask_region is not None:
                mask_region   = mask_region.expand_as(src_region)
                src_region = src_region * mask_region
                out_region = out_region * mask_region

            score = self.ssim(src_region, out_region)
            self.samples[region_name].append(score.cpu().item())
