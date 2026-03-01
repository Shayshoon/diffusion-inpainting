import torch
import numpy as np
import torchvision.transforms as transforms

from abc import ABC, abstractmethod

class Metric(ABC):
    def __init__(self,
                device="cuda",
                transform=None):
        self.device = device
        self.transform = transform or transforms.Compose([transforms.ToTensor()])
    
    def get_name(self):
        pass

    @abstractmethod
    def update(self, original, mask, prompt, output):
        pass

    @abstractmethod
    def compute(self) -> dict:
        pass

    @abstractmethod
    def reset(self):
        pass
