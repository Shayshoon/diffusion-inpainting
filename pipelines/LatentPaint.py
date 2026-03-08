from typing import Dict, Tuple, Union

import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from diffusers import UNet2DConditionModel

from pipelines.Vanilla import Vanilla
from pipelines.utils.conversions import image_to_tensor, mask_to_tensor

class ExplicitPropogation(nn.Module):
    def __init__(self, channels, mask, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.mask = mask
        self.phi = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.ReLU(),
            nn.Linear(channels * 4, channels * 2),
        ).to(device=device, dtype=dtype)

    def forward(self, h: torch.Tensor):
        B, C, H, W = h.shape
        h_cond = h * self.mask

        h_cond_pooled  = torch.amax(h_cond, (2,3)) # (1, C)
        
        mlp_output = self.phi(h_cond_pooled).unsqueeze(2).unsqueeze(3) # (1, C*2, 1, 1)
        h_infr_unpooled = torch.broadcast_to(mlp_output[:, :C], (B, C, H,W)) # (1, C, H, W)
        h_cond_unpooled = torch.broadcast_to(mlp_output[:, C:], (B, C, H,W)) # (1, C, H, W)
        
        h_hat = h_infr_unpooled * (1.0 - self.mask) + h_cond_unpooled * self.mask
        return h_hat

class UnetWrapper(nn.Module):
    def __init__(self, unet: UNet2DConditionModel, device="cuda", dtype=torch.float32):
        super().__init__()
        # from the stable diffussion 2 unet config
        # https://huggingface.co/sd2-community/stable-diffusion-2-base/blob/main/unet/config.json 
        self.channels = [320, 640, 1280, 1280]
        self.shapes = [64, 32, 16, 8]
        self.unet = unet
        self.exp_prop = nn.ModuleDict()
        self.masks: Dict[str, torch.Tensor] = {}
        self.device = device
        self.dtype = dtype
        self.hooks = []
        self.attach_hooks()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def attach_hooks(self):
        self.remove_hooks()

        def make_hook(self_ref):
            def hook(module, input, output):
                # unwrap output
                is_dataclass = hasattr(output, "sample")
                h = output.sample if is_dataclass else output
                if isinstance(h, tuple):
                    h = h[0]

                res = str(h.shape[-1])
                if res not in self_ref.exp_prop:
                    return output  # pass through unchanged

                mask = self_ref.masks[res]  # (B, 1, H, W)
                h_cond_latent = self_ref.h_cond_latents.get(res)

                # --- Step 1: Latent Space Conditioning (Eq. 11) ---
                if h_cond_latent is not None:
                    h_star = h * (1.0 - mask) + h_cond_latent * mask
                else:
                    h_star = h

                # --- Step 2: Explicit Propagation (Eq. 12) ---
                h_hat = self_ref.exp_prop[res](h_star, mask)

                if is_dataclass:
                    output.sample = h_hat
                    return output
                return h_hat

            return hook

        for name, module in self.unet.named_children():
            if any(t in name for t in ['down_blocks', 'mid_block', 'up_blocks']):
                h = module.register_forward_hook(make_hook(self))
                self.hooks.append(h)
     
    def predict(self, ls_unet_input, t, prompt_embeddings, h_cond_latents):
        self.h_cond_latents = h_cond_latents
        self.exp_prop = nn.ModuleDict({
            str(s): ExplicitPropogation(c, self.masks[s], device=self.device, dtype=self.dtype)
            for s, c in zip(self.shapes, self.channels)
        })

        return self.unet(
                ls_unet_input,
                t,
                encoder_hidden_states=prompt_embeddings)

    def prepare_masks(self, ps_mask):
        self.masks = {
            s: nn.AdaptiveAvgPool2d(s)(ps_mask).to(device=self.device, dtype=self.dtype)
                for s in self.shapes
        }
        

class LatentPaint(Vanilla):
    def __init__(self, model_id="sd2-community/stable-diffusion-2-base", dtype=torch.float32, device="cuda"):
        super().__init__(model_id=model_id, dtype=dtype, device=device)
        self.unet = UnetWrapper(self.pipe.unet, device=device, dtype=torch.float32)

    def diffuse(self, ls_image, ls_mask, prompt_embeddings, ls_result, guidance_scale, callback, callback_steps):
        device = self.device

        for i, t in enumerate(tqdm(self.pipe.scheduler.timesteps, desc="Inpainting")):
            # generate noise for known region
            noise = torch.randn_like(ls_image)
        
            scaled_t = int(t * self.known_noise_multiplier) if self.known_noise_multiplier != 0.0 else t
            t_tensor = torch.tensor([scaled_t], device=device)
            ls_image_noised = self.pipe.scheduler.add_noise(ls_image, noise, t_tensor)
            
            ls_unet_input = torch.cat([ls_result] * 2)
            ls_unet_input = self.pipe.scheduler.scale_model_input(ls_unet_input, t)

            t_tensor = torch.Tensor([t])
            # predict noise
            # this is a problematic operation that is likely to produce NaN values if not handled correctly
            # thats the reason we're using it with float32
            noise_pred = self.unet.predict(
                ls_unet_input.float(),
                t_tensor,
                prompt_embeddings.float(),
                ls_image_noised.float()
            ).sample.to(self.dtype)

            # Classifier free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            ls_inferred = self.pipe.scheduler.step(noise_pred, t, ls_result).prev_sample
            
            # 1 -> keep original, 0 -> inpaint
            ls_result = ((1 - ls_mask) * ls_inferred) + (ls_mask * ls_image_noised)
            
            # callback used for interactive mode
            if callback is not None and i % callback_steps == 0:
                callback(i, t, self.decode_latents(ls_result))
        return ls_result

    def preprocess_image_and_mask(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        mask: Union[Image.Image, np.ndarray, torch.Tensor],
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ps_image = image_to_tensor(image).to(device=device, dtype=dtype)
        ps_mask  = mask_to_tensor(mask).to(device=device, dtype=dtype)
        
        self.unet.prepare_masks(ps_mask)
        return ps_image, ps_mask
