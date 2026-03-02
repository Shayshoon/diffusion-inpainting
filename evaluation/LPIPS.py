import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from .Metric import Metric


class LPIPS(Metric):
    """
    Learned Perceptual Image Patch Similarity via torchmetrics.
    Lower is more similar (perceptually).

    torchmetrics' LPIPS expects tensors in [-1, 1].
    """
    def __init__(self, device="cuda", net="alex"):
        """
        Parameters
        ----------
        net : str
            Backbone network. Options: 'alex' (default), 'vgg', 'squeeze'.
            'alex' is fastest and generally recommended.
        """
        super().__init__(device=device)
        self.net = net
        self.loss_fn = LearnedPerceptualImagePatchSimilarity(
            net_type=net, normalize=False  # expects [-1, 1] input
        ).to(device)
        self.loss_fn.eval()
        self.samples: dict[str, list] = defaultdict(list)

    def get_name(self):
        return f"LPIPS_{self.net}"

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
        # LPIPS expects tensors in [-1, 1]
        src_t    = self._to_lpips(original.convert("RGB"))
        mask_t   = self.transform(mask.convert("L")).unsqueeze(0).to(self.device)
        output_t = self._to_lpips(output.convert("RGB"))

        binary_mask = (mask_t > 0.5).float()
        regions = self._extract_regions(src_t, output_t, binary_mask)

        for region_name, (src_r, out_r, w) in regions.items():
            if w is not None:
                # Mask pixels; LPIPS will still process the full spatial map
                w_r   = w.expand_as(src_r)
                src_r = src_r * w_r
                out_r = out_r * w_r

            # torchmetrics LPIPS returns a scalar averaged over the batch
            score = self.loss_fn(src_r, out_r)
            self.samples[region_name].append(score.cpu().item())

    # ------------------------------------------------------------------

    def _to_lpips(self, img: Image.Image) -> torch.Tensor:
        """Convert PIL image to a [-1,1] tensor (1,C,H,W)."""
        t = self.transform(img).unsqueeze(0).to(self.device)  # [0,1]
        return t * 2.0 - 1.0                                  # [-1,1]
