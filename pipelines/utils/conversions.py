import torch
import numpy as np
from PIL import Image

from typing import Union

def image_to_tensor(image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        t = torch.from_numpy(arr).permute(2, 0, 1)
    elif isinstance(image, np.ndarray):
        t = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
    elif isinstance(image, torch.Tensor):
        t = image.float()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    t = (t / 127.5) - 1.0
    return t.unsqueeze(0)

def mask_to_tensor(mask: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(mask, Image.Image):
        arr = np.array(mask.convert("L"), dtype=np.float32)
        t = torch.from_numpy(arr)
    elif isinstance(mask, np.ndarray):
        t = torch.from_numpy(mask.astype(np.float32))
    elif isinstance(mask, torch.Tensor):
        t = mask.float().squeeze()
    else:
        raise TypeError(f"Unsupported mask type: {type(mask)}")

    t = (t > 127.0).float()
    return t.unsqueeze(0).unsqueeze(0)
