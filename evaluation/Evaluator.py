import os
import json
from typing import List
from tqdm import tqdm

from PIL import Image

from .Metric import Metric
from utils.directory_iterator import mask_pair_generator

class Evaluator:
    def __init__(self, metrics: List[Metric]):
        self.metrics  = metrics

    def run(self, src, dst):
        for metric in self.metrics:
            metric.reset()

        for target, mask, prompt, filename in tqdm(
            mask_pair_generator(src),
            desc=f"Evaluating {type(self.pipeline).__name__} pipeline:",
        ):
            sample_name, extension = os.path.splitext(filename)
            output_path = os.path.join(dst, filename)
            
            for metric in self.metrics:
                metric.update(
                    target, mask, prompt, Image.open(output_path)
                )
                
        return {metric.get_name(): metric.compute() for metric in self.metrics}
