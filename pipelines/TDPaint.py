import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

logger = logging.get_logger(__name__)

class TDPaint(StableDiffusionPipeline):

    @staticmethod
    def prepare_mask(
        mask: Image.Image,
        batch_size: int,
        num_images_per_prompt: int,
        device: torch.device,
        dtype: torch.dtype,
        vae_scale_factor: int = 8,
    ) -> torch.Tensor:
        mask_np = np.array(mask.convert("L")).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np)[None, None]
        
        latent_h = mask_tensor.shape[2] // vae_scale_factor
        latent_w = mask_tensor.shape[3] // vae_scale_factor
        mask_latent = torch.nn.functional.interpolate(
            mask_tensor,
            size=(latent_h, latent_w),
            mode="nearest",
        )
        mask_latent = (mask_latent > 0.5).float()
        total = batch_size * num_images_per_prompt
        mask_latent = mask_latent.expand(total, 1, -1, -1).contiguous()
        return mask_latent.to(device=device, dtype=dtype).clamp(0.0, 1.0)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[torch.FloatTensor] = None,
        mask: Optional[Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
    ):
        # 1. Setup
        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # 2. Encode Prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, device, num_images_per_prompt,
            do_classifier_free_guidance, negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 3. Timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare Mask and Image Latents
        mask_latent = self.prepare_mask(mask, batch_size=1, num_images_per_prompt=num_images_per_prompt, device=device, dtype=prompt_embeds.dtype)
        mask_clamped = mask_latent.clamp(0.0, 1.0)

        # Encode reference image
        image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=prompt_embeds.dtype)
        image_tensor = image_tensor * 2.0 - 1.0
        image_latents = self.vae.encode(image_tensor).latent_dist.sample(generator) * self.vae.config.scaling_factor
        image_latents = image_latents.expand(num_images_per_prompt, -1, -1, -1)

        # 5. TD-PAINT INITIALIZATION
        # phi_minus is the standard schedule (T -> 0)
        # phi_plus is the "displaced" lower noise schedule for known pixels
        phi_minus = timesteps
        phi_plus = (timesteps * 0.2).long().clamp(0, 999) # Known regions start at ~20% noise

        # Start with pure noise for the unknown region
        init_noise = randn_tensor(image_latents.shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
        
        # Initial blend: Known regions start at phi_plus[0] noise level
        t_start_plus = phi_plus[0]
        alpha_bar_start = self.scheduler.alphas_cumprod[t_start_plus]
        
        noisy_known_init = (alpha_bar_start**0.5 * image_latents + (1 - alpha_bar_start)**0.5 * init_noise)
        latents = (mask_clamped * noisy_known_init) + ((1 - mask_clamped) * init_noise)

        # 6. Denoising loop
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # TD-Paint Step: Re-sync known pixels to their specific noise level phi_plus[i]
                t_plus = phi_plus[i]
                alpha_bar_plus = self.scheduler.alphas_cumprod[t_plus]
                
                noise_known = randn_tensor(image_latents.shape, generator=generator, device=device, dtype=prompt_embeds.dtype)
                noisy_known_step = (alpha_bar_plus**0.5 * image_latents + (1 - alpha_bar_plus)**0.5 * noise_known)
                
                # Blend: KEEP region gets re-noised to phi_plus, REPAINT region stays from diffusion
                latents = (mask_clamped * noisy_known_step) + ((1 - mask_clamped) * latents)

                # UNet Prediction
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Step
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)
                progress_bar.update()

        # 7. Final composite and decode
        latents = (mask_clamped * image_latents) + ((1 - mask_clamped) * latents)
        image_out = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        image_out = self.image_processor.postprocess(image_out, output_type=output_type)

        return StableDiffusionPipelineOutput(images=image_out, nsfw_content_detected=None)