import os
import argparse
from pathlib import Path
from typing import List
from PIL import Image
from dotenv import load_dotenv

import torch
from huggingface_hub import login

from evaluation import KID, MSE
from evaluation.Dataset import PipeDataset
from pipelines.Vanilla import Vanilla
from pipelines.BackgroundReconstruction import BackgroundReconstruction
from pipelines.TDPaint import TDPaint
from pipelines.BackgroundCopy import BackgroundCopy
from utils.image import get_square, create_comparison_canvas
from utils.interactive import init_callback
from evaluation.Evaluator import Evaluator

pipelines: dict[str, Vanilla] = {
    "vanilla": Vanilla,
    "BackgroundReconstruction": BackgroundReconstruction, 
    "BackgroundCopy": BackgroundCopy,
    "TDPaint": TDPaint, 
    }

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

def run_pipeline(
        pipeline_name="vanilla", 
        src="./media", 
        dst="./results", 
        interactive=False, 
        model_id = "sd2-community/stable-diffusion-2-base"):
    
    pipeline = pipelines[pipeline_name](model_id, dtype=torch.float16)

    kwargs = {
        "callback": init_callback(pipeline),
        "callback_steps": 1,
    } if interactive else {}

    for source, mask, prompt, file_name in mask_pair_generator(src):
        print(f"Processing {file_name}...")
        result = pipeline.inpaint(source, mask, prompt, num_inference_steps=50, **kwargs)
        
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
    parser.add_argument("--evaluate", action="store_true", help="Run metrics")
    parser.add_argument("--interactive", 
                        action=argparse.BooleanOptionalAction, 
                        help="Use this flag to watch diffusion process (slows performance)")
    args = parser.parse_args()
    
    load_dotenv()
    token = os.getenv('HF_TOKEN')
    
    if args.evaluate:
        evaluation_pipelines: List[Vanilla] = [ Vanilla, BackgroundReconstruction,  BackgroundCopy ]
        for pipeline in evaluation_pipelines:
            evaluator = Evaluator(pipeline(), [MSE.MSE(), KID.KID()])
            evaluation_results = evaluator.run(PipeDataset())
            with open("evaluation.txt", 'a+') as file:
                file.write(evaluation_results)
                file.write('\n')
    else:
        if token:
            login(token)
        run_pipeline(pipeline_name=args.pipeline, 
                     src=args.src, 
                     dst=args.dst, 
                     interactive=args.interactive)
