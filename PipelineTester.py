import os
import argparse
from pathlib import Path
from typing import List
from PIL import Image
from dotenv import load_dotenv

import torch
from huggingface_hub import login

from evaluation import KID, MSE
from pipelines.Vanilla import Vanilla
from pipelines.BackgroundReconstruction import BackgroundReconstruction
from pipelines.TDPaint import TDPaint
from pipelines.BackgroundCopy import BackgroundCopy
from utils.image import get_square, create_comparison_canvas
from utils.interactive import init_callback
from evaluation.Evaluator import Evaluator
from evaluations.utils.directory_iterator import mask_pair_generator

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
        
def run_pipeline(
        pipeline_name="vanilla", 
        src="./media", 
        dst="./results", 
        interactive=False, 
        model_id = "sd2-community/stable-diffusion-2-base",
        skip_existing = False):
    
    pipeline = pipelines[pipeline_name](model_id, dtype=torch.float16)
    
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
        print(f"GPU: {gpu.name} ({gpu.total_memory / 1024**3:.1f} GB)")
    else:
        print("No GPU available, running on CPU")
    

    kwargs = {
        "callback": init_callback(pipeline),
        "callback_steps": 1,
    } if interactive else {}

    for source, mask, prompt, file_name in mask_pair_generator(src):
        output_path = os.path.join(dst, pipeline_name, file_name)
        if skip_existing and os.path.exists(output_path):
            print(f"Skipping {file_name} (already exists)...")
            continue
        
        print(f"Processing {file_name}...")
        result = pipeline.inpaint(source, mask, prompt, num_inference_steps=100, **kwargs)
        
        final_strip = create_comparison_canvas(
            source, mask, result,
            text_label=prompt, alpha=0.35
        )

        Path(os.path.join(dst, pipeline_name)).mkdir(parents=True, exist_ok=True)
        name, extension = os.path.splitext(file_name)
        
        result.convert('RGB').save(output_path)
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
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that have already been generated")
    args = parser.parse_args()
    
    load_dotenv()
    token = os.getenv('HF_TOKEN')
    
    if args.evaluate:
        evaluation_pipelines: List[Vanilla] = [ BackgroundCopy, BackgroundReconstruction ]
        for pipeline in evaluation_pipelines:
            evaluator = Evaluator(pipeline(), [MSE.MSE(), KID.KID()])
            evaluation_results = evaluator.run()
            with open("evaluation.txt", 'a+') as file:
                file.write(str(evaluation_results))
                file.write('\n')
    else:
        if token:
            login(token)
        run_pipeline(pipeline_name=args.pipeline, 
                     src=args.src, 
                     dst=args.dst, 
                     interactive=args.interactive,
                     skip_existing=args.skip_existing)
