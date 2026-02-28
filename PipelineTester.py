import os
import argparse
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

import torch
from diffusers import DDPMScheduler
from huggingface_hub import login
import matplotlib.pyplot as plt

from pipelines.VanillaPipeline import VanillaPipeline
from pipelines.TDPaint import TDPaint
from utils.image import get_square, create_comparison_canvas
from utils.interactive import make_callback
# from metrics.Metrics import metrics

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
    Yields (source_pil, mask_pil, prompt, filename) for every sample.
    """
    files = os.listdir(directory)
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

def run_pipeline(pipeline_name, src="./media", dst="./results", interactive=False):
    model_id = "sd2-community/stable-diffusion-2-base"

    pipelines = {"vanilla": VanillaPipeline, "TDPaint": TDPaint}
    
    pipeline = pipelines[pipeline_name]
    
    scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = pipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float32)
    pipe = pipe.to("cuda")
    pipe.vae.to(dtype=torch.float32)
    if interactive:
        plt.ion()
        
    kwargs = {
        "callback": make_callback(pipe, display_every_n_steps=1),
        "callback_steps": 1,
    } if interactive else {}

    for source, mask, prompt, file_name in mask_pair_generator(src):
        print(f"Processing {file_name}...")
        
        output = pipe(
            prompt=prompt,
            num_inference_steps=50,
            image=source,
            mask=mask,
            guidance_scale=7.5,
            **kwargs
        )
        
        result = output.images[0]
        final_strip = create_comparison_canvas(
            source, mask, result,
            text_label=prompt, alpha=0.35
        )

        Path(os.path.join(dst, pipeline_name)).mkdir(parents=True, exist_ok=True)
        name, extension = os.path.splitext(file_name)
        
        result.convert('RGB').save(os.path.join(dst, pipeline_name, file_name))
        final_strip.convert('RGB').save(os.path.join(dst, pipeline_name, f'{name}.compare{extension}'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline on all samples")
    parser.add_argument("--pipeline", 
                        type=str, 
                        default="vanilla", 
                        help="The pipeline to run")
    parser.add_argument("--src", type=str, default="./media", help="Source media directory")
    parser.add_argument("--dst", type=str, default="./results", help="Destination media directory")
    parser.add_argument("--metric", type=str, help="Run metrics (KID|MSE)")
    parser.add_argument("--interactive", action=argparse.BooleanOptionalAction, help="Use this flag to watch diffusion process (slows performance)")
    args = parser.parse_args()
    
    load_dotenv()
    token = os.getenv('HF_TOKEN')
    
    if args.metric:
        # TODO: run metrics
        for source, mask, prompt, file_name in mask_pair_generator(args.src):
            print(metrics[args.metric].compare(source, mask, prompt, Image.open(os.path.join(args.dst, file_name))))
    else:
        if token:
            login(token)
        run_pipeline(args.pipeline, args.src, args.dst, args.interactive)
