import torch
import numpy as np
import torchvision.transforms as transforms

from Metric import Metric
from PIL import Image
import os

class MSE(Metric):
    def __init__(self, device="cuda"):
        super().__init__(device=device)

    def compute(self, 
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
        
        MSE = squared_error.sum() / unmasked_pixels if unmasked_pixels != 0 else 0.0
        
        return MSE.item()
    
    # TODO: call dataset and shit
    def run(self):
        pass

# ~~~~~~~~~~~~~~~~~~~~~ BELOW HERE IS CODE FOR TESTING ONLY ~~~~~~~~~~~~~~~~~~~~~

def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        return "Error: The file was not found."
    except Exception as e:
        return f"An unexpected error occurred: {e}"
        
def mask_pair_generator(directory):
    """
    Yields (source_pil, mask_pil, prompt, filename) for every pair found.
    """
    files = os.listdir(directory)
    # Filter for base images only
    base_images = [f for f in files if ".mask." not in f and f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for filename in base_images:
        name_part, ext = os.path.splitext(filename)
        mask_filename = f"{name_part}.mask{ext}"
        prompt_filename = f"{name_part}.txt"
        
        src_path = os.path.join(directory, filename)
        mask_path = os.path.join(directory, mask_filename)
        prompt_path = os.path.join(directory, prompt_filename)

        src_obj = Image.open(src_path)
        mask_obj = Image.open(mask_path)
        prompt = read_file(prompt_path)
            
        yield src_obj, mask_obj, prompt, filename

def main():
    for source, mask, prompt, file_name in mask_pair_generator('./test/'):
        metric = MSE()
        print(metric.compute(source, mask, prompt, Image.open(os.path.join('./test/result/generations', 'inside.jpg'))))
        print(metric.compute(source, mask, prompt, Image.open(os.path.join('./test/result/generations', 'outside.jpg'))))
        print(metric.compute(source, mask, prompt, Image.open(os.path.join('./test/result/generations', 'original.jpg'))))

if __name__ == "__main__":
    main()