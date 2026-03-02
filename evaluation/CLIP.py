import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from torchmetrics.multimodal.clip_score import CLIPScore

from .Metric import Metric
from .utils.regions import extract_regions

class CLIP(Metric):
    def __init__(self, device="cuda"):
        super().__init__(device=device)
        self.clip = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
        self.samples: dict[str, list] = defaultdict(list)

    def get_name(self):
        return "CLIP"

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
        mask_tensor   = mask.convert("L").unsqueeze(0).to(self.device)
        output_tensor = output.convert("RGB")

        binary_mask = (mask_tensor > 0.5).float()
        regions = extract_regions(None, output_tensor, binary_mask)

        for region_name, (src_region, out_region, mask_region) in regions.items():
            if mask_region is not None:
                mask_region   = mask_region.expand_as(src_region)
                out_region = out_region * mask_region

            score = self.clip(out_region, prompt)
            score.detach().round()
            self.samples[region_name].append(score.cpu().item())
