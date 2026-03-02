import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import DiffusionPipeline, DDIMScheduler, AutoencoderKL

from typing import Callable, Optional, Union, Tuple

from .utils.conversions import image_to_tensor, mask_to_tensor

class Vanilla:
    def __init__(self, model_id="sd2-community/stable-diffusion-2-base", dtype=torch.float16, device="cuda"):
        self.dtype = dtype
        self.device = device
        self.pipe: DiffusionPipeline = DiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=dtype, 
                device_map=device)
        self.pipe.unet = self.pipe.unet.to(torch.float32)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.known_noise_multiplier = 0.0
    
    def preprocess_image_and_mask(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        mask: Union[Image.Image, np.ndarray, torch.Tensor],
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ps_image = image_to_tensor(image).to(device=device, dtype=dtype)
        ps_mask  = mask_to_tensor(mask).to(device=device, dtype=dtype)
        return ps_image, ps_mask
    
    def decode_latents(self, latents: torch.Tensor):
        output = self.pipe.vae.decode(latents.to(self.pipe.vae.dtype) / self.pipe.vae.config.scaling_factor).sample.clamp(-1, 1)
        return output

    # defined empty function we can inherit, this will simplify implementation of other pipelines
    def postprocess(self,
                    z_0:    torch.Tensor,
                    x:      torch.Tensor,
                    m:      torch.Tensor) -> None:
        pass

    def postprocess_output(self, output, ps_mask, ps_image):
        return output

    def noise_image(self, ls_image, t):
        device = self.device

        noise = torch.randn_like(ls_image)
        
        scaled_t = int(t * self.known_noise_multiplier) if self.known_noise_multiplier != 0.0 else t
        t_tensor = torch.tensor([scaled_t], device=device)
        ls_image_noised = self.pipe.scheduler.add_noise(ls_image, noise, t_tensor)

        return ls_image_noised

    def predict_noise(self, ls_input, t, prompt_embeddings):
        # this is a problematic operation that is likely to produce NaN values if not handled correctly
        # thats the reason we're using it with float32
        return self.pipe.unet(
                ls_input.float(),
                t,
                encoder_hidden_states=prompt_embeddings.float()
            ).sample.to(self.dtype)

    def diffuse(self, ls_image, ls_mask, prompt_embeddings, guidance_scale, callback, callback_steps):
        device = self.device

        # initialize result as noise in latent space
        ls_result = torch.randn_like(ls_image) * self.pipe.scheduler.init_noise_sigma

        for i, t in enumerate(tqdm(self.pipe.scheduler.timesteps, desc="Inpainting")):
            # generate noise for known region
            ls_image_noised = self.noise_image(ls_image, t)

            # 1 -> keep original, 0 -> inpaint
            ls_result = ((1 - ls_mask) * ls_result) + (ls_mask * ls_image_noised)

            ls_input = torch.cat([ls_result] * 2)
            ls_input = self.pipe.scheduler.scale_model_input(ls_input, t)

            # predict noise
            noise_pred = self.predict_noise(ls_input, t, prompt_embeddings)

            # Classifier free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            ls_result = self.pipe.scheduler.step(noise_pred, t, ls_result).prev_sample
            
            # callback used for interactive mode
            if callback is not None and i % callback_steps == 0:
                callback(i, t, self.decode_latents(ls_result))

        return ls_result

    @torch.no_grad()
    def inpaint(self, 
                image:              Union[Image.Image, np.ndarray, torch.Tensor], 
                mask:               Union[Image.Image, np.ndarray, torch.Tensor], 
                prompt:             str, 
                num_inference_steps = 50,
                guidance_scale      = 10.0,
                callback:           Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
                callback_steps:     int = 10) -> Image.Image:
        device = self.device
        dtype = self.dtype

        # pixel-space image and mask
        ps_image, ps_mask = self.preprocess_image_and_mask(image, mask, device, dtype)
    
        # encode image
        ls_image = self.pipe.vae.encode(ps_image).latent_dist.mode()
        ls_image = self.pipe.vae.config.scaling_factor * ls_image

        # encode mask
        ls_mask = torch.nn.functional.interpolate(
            ps_mask, 
            size=ls_image.shape[2:], 
            mode="nearest"
            ).clamp(0, 1)
        
        self.pipe.scheduler.set_timesteps(num_inference_steps)

        # encode prompt
        positive_embeddings, negative_embeddings = self.pipe.encode_prompt(prompt, device, 1, True)
        prompt_embeddings = torch.cat([negative_embeddings, positive_embeddings]).to(dtype)

        ls_result = self.diffuse(ls_image, ls_mask, prompt_embeddings, guidance_scale, callback, callback_steps)

        ls_result = (1 - ls_mask) * ls_result + ls_mask * ls_image

        self.postprocess(ls_result, ps_image, ps_mask)
        output = self.decode_latents(ls_result)
        output = self.postprocess_output(output, ps_mask, ps_image)

        # return as pillow image 
        output = ((output / 2) + 0.5)
        output = output.cpu().permute(0, 2, 3, 1).float().numpy()
        image_out = self.pipe.numpy_to_pil(output)[0]
        
        return image_out
    
