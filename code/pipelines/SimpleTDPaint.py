import torch
import numpy as np

from pipelines.Vanilla import Vanilla

class SimpleTDPaint(Vanilla):
    def __init__(self, model_id="sd2-community/stable-diffusion-2-base", dtype=torch.float32, device="cuda"):
        super().__init__(model_id=model_id, dtype=dtype, device=device)
        self.known_noise = 0.2
        