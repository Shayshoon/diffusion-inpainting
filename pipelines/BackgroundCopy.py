import torch
import numpy as np

from pipelines.Vanilla import Vanilla

class BackgroundCopy(Vanilla):
    def postprocess_output(self, output, ps_mask, ps_image):
        return ((1 - ps_mask) * output) + (ps_mask * ps_image)
    
