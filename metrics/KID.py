import torch
import numpy as np
import torchvision.transforms as transforms
from torchmetrics.image.kid import KernelInceptionDistance

from PIL import Image
import os

class KID:
    def __init__(self):
        self.kid = KernelInceptionDistance(subset_size=50, normalize=True)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def update(self, original: Image.Image, mask: Image.Image, prompt: str, output: Image.Image):
        # TODO: crop
        # TODO: to.device("cuda")

        src_tensor = self.transform(original).unsqueeze(0)
        mask_tensor = self.transform(mask).unsqueeze(0)
        output_tensor = self.transform(output).unsqueeze(0)

        binary_mask = (mask_tensor < 0.5).float()
        
        masked_src = src_tensor * binary_mask
        masked_output = output_tensor * binary_mask
        self.kid.update(masked_src, real=True)
        self.kid.update(masked_output, real=False)
    
    def compute(self):
        return self.kid.compute()


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
    Yields (source_pil, mask_pil, filename) for every pair found.
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
        compare(source, mask, prompt, Image.open(os.path.join('./test/result/generations', 'outside.jpg')))
        compare(source, mask, prompt, Image.open(os.path.join('./test/result/generations', 'inside.jpg')))
        compare(source, mask, prompt, Image.open(os.path.join('./test/result/generations', 'original.jpg')))

if __name__ == "__main__":
    main()