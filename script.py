import os
import argparse
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

import torch
from diffusers import DDPMScheduler

from VanillaPipeline import VanillaPipeline

from mask_utils import get_square, overlay_mask, get_grid, create_comparison_canvas

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
        mask_obj = Image.open(mask_path) if os.path.exists(mask_path) else get_square()
        prompt = read_file(prompt_path) if os.path.exists(prompt_path) else ""
            
        yield src_obj, mask_obj, prompt, filename

def run_pipeline(pipeline_name, src="./media", dst="./results"):
    model_id = "sd2-community/stable-diffusion-2-base"

    pipelines = {"vanilla": VanillaPipeline}
    
    pipeline = pipelines[pipeline_name]
    
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = pipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    
    for source, mask, prompt, file_name in mask_pair_generator(src):
        print(f"Processing {file_name}...")
        
        output = pipe(
            prompt=prompt,
            num_inference_steps=50,
            image=source,
            mask=mask,
            guidance_scale=7.5,
        )
        result = output.images[0]
    
        final_strip = create_comparison_canvas(
            source, mask, result,
            text_label=prompt, alpha=0.35
        )
        
        Path(dst).mkdir(parents=True, exist_ok=True)
        
        final_strip.convert('RGB').save(f"{dst}/{file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline on all samples")
    parser.add_argument("--pipeline", 
                        type=str, 
                        default="vanilla", 
                        help="The pipeline to run")
    parser.add_argument("--src", type=str, default="./media", help="Source media directory")
    parser.add_argument("--dst", type=str, default="./results", help="Destination media directory")
    args = parser.parse_args()
    
    run_pipeline(args.pipeline, args.src, args.dst)
