import torch
import matplotlib.pyplot as plt

def init_callback():
    plt.ion() 
    fig, ax = plt.subplots(figsize=(5, 5))

    def callback(step: int, timestep: int, ps_image: torch.FloatTensor):     
        img = ps_image.squeeze(0).permute(1, 2, 0).clamp(0, 1).float().cpu().numpy()
        
        ax.clear()
        ax.imshow(img)
        ax.set_title(f"Step {step} | Timestep {timestep}")
        ax.axis("off")
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

    return callback