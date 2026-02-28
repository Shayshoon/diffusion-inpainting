import torch
import numpy as np
from torchmetrics.image.kid import KernelInceptionDistance
from PIL import Image

from Metric import Metric

class KID(Metric):
    def __init__(self, device="cuda", subset_size=50):
        super().__init__(device=device)
        self.kid = KernelInceptionDistance(subset_size=subset_size, normalize=True).to(device)

    def update(self, 
               original: Image.Image, 
               mask: Image.Image, 
               prompt: str, 
               output: Image.Image):
        
        src_tensor = self.transform(original.convert("RGB")).unsqueeze(0).to(self.device)
        mask_tensor = self.transform(mask.convert("L")).unsqueeze(0).to(self.device)
        output_tensor = self.transform(output.convert("RGB")).unsqueeze(0).to(self.device)
        
        binary_mask = (mask_tensor < 0.5).float()
        
        masked_src = src_tensor * binary_mask
        masked_output = output_tensor * binary_mask
        self.kid.update(masked_src, real=True)
        self.kid.update(masked_output, real=False)
    
    def compute(self):
        return self.kid.compute()
    
    # TODO: call dataset and shit
    def run(self):
        pass
