import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Callable, Optional, Union, Tuple

from pipelines.Vanilla import Vanilla
from .utils.conversions import image_to_tensor, mask_to_tensor

class MaskBlur(Vanilla):
    def preprocess_image_and_mask(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        mask: Union[Image.Image, np.ndarray, torch.Tensor],
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ps_image = image_to_tensor(image).to(device=device, dtype=dtype)
        ps_mask  = mask_to_tensor(mask).to(device=device, dtype=dtype)

        blur = transforms.GaussianBlur(kernel_size=(13, 13), sigma=(10, 10))
        
        return ps_image, blur(ps_mask)
