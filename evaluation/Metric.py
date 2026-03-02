import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image

from abc import ABC, abstractmethod

class Metric(ABC):
    def __init__(self, device="cuda"):
        self.device = device
        self.transform = T.Compose([ T.ToTensor() ])
        self.REGIONS = ("full", "bbox", "masked", "unmasked")

    @abstractmethod
    def get_name(self) -> str: 
        pass

    @abstractmethod
    def update(self, original: Image.Image, mask: Image.Image,
               prompt: str, output: Image.Image): 
        pass

    @abstractmethod
    def compute(self) -> dict: 
        pass

    @abstractmethod
    def reset(self): 
        pass

    @staticmethod
    def _compute_stats(samples: list) -> dict:
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
