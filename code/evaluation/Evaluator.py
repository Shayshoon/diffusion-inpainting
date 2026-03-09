import os
import json
from typing import List
from tqdm import tqdm

import pandas as pd
from PIL import Image

from .Metric import Metric
from .utils.directory_iterator import mask_pair_generator
from .CLIP import CLIP
from .KID import KID
from .LPIPS import LPIPS
from .MSE import MSE
from .PSNR import PSNR
from .SSIM import SSIM
from .GDiff import GDiff

class Evaluator:
    def __init__(self, metrics: List[Metric]):
        self.metrics  = metrics

    def run(self, src, dst, pipeline_name):
        for metric in self.metrics:
            metric.reset()

        for target, mask, prompt, filename in tqdm(
            mask_pair_generator(src),
            desc=f"Evaluating {pipeline_name}:",
        ):
            output_path = os.path.join(dst, filename)
            
            for metric in self.metrics:
                metric.update(
                    target, mask, prompt, Image.open(output_path)
                )

        row = {}
        for metric in self.metrics:
            row.update(metric.compute())
        return pd.DataFrame(row, index=[pipeline_name])
