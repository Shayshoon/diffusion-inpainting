import torch
import numpy as np
import torchvision.transforms as transforms
from torchmetrics.image.kid import KernelInceptionDistance

from abc import ABC, abstractmethod
from PIL import Image
import os

class Metric(ABC):
    def __init__(self,
                device="cuda"):
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
