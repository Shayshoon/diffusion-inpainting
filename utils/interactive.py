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
    plt.ion() 
    fig, ax = plt.subplots(figsize=(5, 5))

    def callback(step: int, timestep: int, latents: torch.FloatTensor):
        if step % display_every_n_steps != 0:
            return
        
        img = decode_latents(pipeline, latents)
        
        ax.clear()
        ax.imshow(img)
        ax.set_title(f"Step {step} | Timestep {timestep}")
        ax.axis("off")
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

    return callback