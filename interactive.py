import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from IPython.display import display, clear_output

def decode_latents(pipeline, latents):
    """Decode latents to a PIL image for display."""
    latents = latents / pipeline.vae.config.scaling_factor
    with torch.no_grad():
        image = pipeline.vae.decode(latents, return_dict=False)[0]
    # [-1, 1] → [0, 1] → [0, 255]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image[0] * 255).astype(np.uint8)
    return Image.fromarray(image)

def make_callback(pipeline, display_every_n_steps=1):
    """
    Returns a callback function to pass into pipeline.__call__.
    
    display_every_n_steps: decode and show image every N steps (decoding is slow,
                           so you may want every 5 steps rather than every 1).
    """
    def callback(step: int, timestep: int, latents: torch.FloatTensor):
        if step % display_every_n_steps != 0:
            return
        
        img = decode_latents(pipeline, latents)
        
        # --- Jupyter ---
        clear_output(wait=True)
        display(img)
        print(f"Step {step} | timestep {timestep}")
        
        # --- Matplotlib (works outside Jupyter too) ---
        axes.clear()
        axes.imshow(img)
        axes.set_title(f"Step {step} / timestep {timestep}")
        axes.axis("off")
        plt.pause(0.01)

    return callback