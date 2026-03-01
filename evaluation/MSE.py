import torch
import numpy as np

from .Metric import Metric
from PIL import Image

class MSE(Metric):
    def __init__(self, device="cuda"):
        super().__init__(device=device)
        self.samples = np.array([])

    def get_name(self):
        return "MSE"
    
    def compute(self):
        return {"mean": np.mean(self.samples), "std": np.std(self.samples)}

    def reset(self):
        self.samples = np.array([])

    def update(self, 
                original: Image.Image, 
                mask: Image.Image, 
                prompt: str, 
                output: Image.Image):
        src_tensor = self.transform(original.convert("RGB")).unsqueeze(0).to(self.device)
        mask_tensor = self.transform(mask.convert("L")).unsqueeze(0).to(self.device)
        output_tensor = self.transform(output.convert("RGB")).unsqueeze(0).to(self.device)

        binary_mask = (mask_tensor > 0.5).float()
        
        masked_src = src_tensor * binary_mask
        masked_output = output_tensor * binary_mask

        unmasked_pixels = binary_mask.sum()
        squared_error = (masked_output - masked_src) ** 2
        
        MSE = (squared_error.sum() / unmasked_pixels).cpu().item() if unmasked_pixels != 0 else 0.0
        
        self.samples = np.insert(self.samples, 0, MSE)
    

