import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

from abc import ABC, abstractmethod

class Metric(ABC):
    def __init__(self, device="cuda"):
        self.device = device
        self.transform = transforms.Compose([ transforms.ToTensor() ])
        self.REGIONS = ("full", "bbox", "masked", "unmasked")
        self.name = ""

    @abstractmethod
    def update(self, original: Image.Image, mask: Image.Image,
               prompt: str, output: Image.Image): 
        pass

    def get_name(self) -> str: 
        return self.name
    
    def reset(self):
        self.samples = defaultdict(list)

    def compute(self) -> dict:
        flat = {}
        for region in self.REGIONS:
            for stat_name, value in self.get_stats(self.samples[region]).items():
                flat[f"{self.get_name()}/{region}/{stat_name}"] = value
        return flat

    @staticmethod
    def get_stats(samples: list) -> dict:
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
