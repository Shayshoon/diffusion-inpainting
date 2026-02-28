import torch
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from PIL import Image

from diffusers import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

logger = logging.get_logger(__name__)


class VanillaPipeline(StableDiffusionPipeline):

    @staticmethod
    def prepare_mask(
        mask: Image.Image,
        batch_size: int,
        num_images_per_prompt: int,
        device: torch.device,
        dtype: torch.dtype,
        vae_scale_factor: int = 8,
    ) -> torch.Tensor:
        """
        Converts a PIL mask (L-mode, 512x512) into a latent-space mask tensor.

        Convention (matches the input):
            1.0 (white) = KEEP original pixel   → we do NOT repaint here
            0.0 (black) = REPAINT               → we inject noise here

        The mask is downscaled from pixel space (H x W) to latent space
        (H/8 x W/8) using nearest-neighbour to preserve hard edges, then
        thresholded so every value is exactly 0 or 1.

        Returns
        -------
        mask_latent : torch.Tensor  shape (B, 1, H//8, W//8)
        """
        # --- 1. Pixel-space: float32 in [0, 1] ----------------------------
        mask_np = np.array(mask.convert("L")).astype(np.float32) / 255.0
        # shape: (H, W)

        # --- 2. Add batch + channel dims: (1, 1, H, W) --------------------
        mask_tensor = torch.from_numpy(mask_np)[None, None]   # (1, 1, H, W)

        # --- 3. Downscale to latent resolution with nearest-neighbour ------
        #        Nearest-neighbour keeps the mask binary (no blurring at edges)
        latent_h = mask_tensor.shape[2] // vae_scale_factor
        latent_w = mask_tensor.shape[3] // vae_scale_factor
        mask_latent = torch.nn.functional.interpolate(
            mask_tensor,
            size=(latent_h, latent_w),
            mode="nearest",
        )
        # shape: (1, 1, H//8, W//8)

        # --- 4. Hard threshold (safety net against any float drift) --------
        mask_latent = (mask_latent > 0.5).float()

        # --- 5. Expand to full batch size ----------------------------------
        total = batch_size * num_images_per_prompt
        # FIX 2: .contiguous() forces a real memory copy so that downstream
        # writes don't corrupt the original tensor through the shared view
        # that .expand() would otherwise return.
        mask_latent = mask_latent.expand(total, 1, -1, -1).contiguous()
        # shape: (B, 1, H//8, W//8)

        # FIX 3: clamp AFTER the dtype cast because converting to fp16 can
        # introduce tiny rounding errors that push values just outside [0, 1],
        # which would cause a leak of original-image information into the
        # repaint region during the blend step.
        return mask_latent.to(device=device, dtype=dtype).clamp(0.0, 1.0)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[torch.FloatTensor] = None,   # original image to preserve under mask
        mask: Optional[Image.Image] = None,           # PIL mask described above
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
        # ------------------------------------------------------------------ #
        # 0. Defaults
        # ------------------------------------------------------------------ #
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width  = width  or self.unet.config.sample_size * self.vae_scale_factor

        # ------------------------------------------------------------------ #
        # 1. Validate
        # ------------------------------------------------------------------ #
        self.check_inputs(
            prompt, height, width, callback_steps,
            negative_prompt, prompt_embeds, negative_prompt_embeds,
        )

        # ------------------------------------------------------------------ #
        # 2. Batch size & device
        # ------------------------------------------------------------------ #
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        # ------------------------------------------------------------------ #
        # 3. Encode prompt
        # ------------------------------------------------------------------ #
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, device, num_images_per_prompt,
            do_classifier_free_guidance, negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # ------------------------------------------------------------------ #
        # 4. Scheduler timesteps
        # ------------------------------------------------------------------ #
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # ------------------------------------------------------------------ #
        # 5. Prepare initial latents (pure noise)
        # ------------------------------------------------------------------ #
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents, height, width,
            prompt_embeds.dtype, device, generator, latents,
        )

        # ------------------------------------------------------------------ #
        # 6. Encode the original image into latent space (needed for blending)
        # ------------------------------------------------------------------ #
        # image_latents holds the VAE encoding of the unmasked original.
        # During the loop we will paste these back under the KEEP region.
        image_latents = None
        if image is not None and mask is not None:
            # Accept PIL Image or tensor
            if isinstance(image, Image.Image):
                image_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
                # Normalise to [-1, 1] and convert to (1, 3, H, W)
                image_tensor = (
                    torch.from_numpy(image_np)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device=device, dtype=prompt_embeds.dtype)
                )
                image_tensor = image_tensor * 2.0 - 1.0
            else:
                image_tensor = image.to(device=device, dtype=prompt_embeds.dtype)

            # Encode through the VAE encoder; the scaling factor matches decoding
            image_latents = self.vae.encode(image_tensor).latent_dist.sample(generator)
            image_latents = image_latents * self.vae.config.scaling_factor
            # Expand to match batch size
            image_latents = image_latents.expand(
                batch_size * num_images_per_prompt, -1, -1, -1
            )

        # ------------------------------------------------------------------ #
        # 7. Prepare the mask in latent space
        # ------------------------------------------------------------------ #
        mask_latent = None
        if mask is not None:
            mask_latent = self.prepare_mask(
                mask,
                batch_size=batch_size,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=prompt_embeds.dtype,
                vae_scale_factor=self.vae_scale_factor,
            )
            # mask_latent shape: (B, 1, H//8, W//8)
            # 1.0 = KEEP original   0.0 = REPAINT (free diffusion)

        # ------------------------------------------------------------------ #
        # 8. Extra scheduler kwargs (eta → DDIM / DDPM stochasticity)
        # ------------------------------------------------------------------ #
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # ------------------------------------------------------------------ #
        # 9. Denoising loop
        # ------------------------------------------------------------------ #
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if torch.isnan(latents).any():
                    print("NaNs detected in latents at step", i)
                # -- 9a. Blend: re-apply original image noise under KEEP mask --
                #
                # At timestep t the "clean" original image in latent space should
                # look like:  x_t = sqrt(ᾱ_t) * x_0  +  sqrt(1-ᾱ_t) * ε
                # We compute this noisy version and paste it back wherever mask=1
                # so the KEEP region always stays anchored to the original image.
                if mask_latent is not None and image_latents is not None:
                    # Retrieve ᾱ_t from the scheduler's cumulative alphas
                    alpha_bar_t = self.scheduler.alphas_cumprod[t]
                    sqrt_alpha_bar = alpha_bar_t ** 0.5
                    sqrt_one_minus_alpha_bar = (1 - alpha_bar_t) ** 0.5

                    # Sample fresh noise matching latent shape
                    noise = randn_tensor(
                        image_latents.shape, generator=generator,
                        device=device, dtype=prompt_embeds.dtype
                    )

                    # Noisy version of the original image at timestep t
                    noisy_image_latents = (
                        sqrt_alpha_bar * image_latents
                        + sqrt_one_minus_alpha_bar * noise
                    )

                    # FIX 4: clamp mask again at the blend site to guard against
                    # fp16 drift that accumulates across 50 loop iterations.
                    # Without this, border pixels can drift outside [0, 1] and
                    # the blend coefficients no longer sum to 1, leaking original
                    # image content into the repaint region.
                    mask_clamped = mask_latent.clamp(0.0, 1.0)

                    # Paste: KEEP region ← noisy original,  REPAINT region ← current diffusion latent
                    latents = (
                        mask_clamped       * noisy_image_latents   # keep
                        + (1 - mask_clamped) * latents             # repaint
                    )

                # -- 9b. Standard UNet forward pass --------------------------
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                noise_pred = self.unet(
                    latent_model_input, t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # -- 9c. Classifier-free guidance ----------------------------
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    noise_pred = self._rescale_noise_cfg(
                        noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                    )

                # -- 9d. Scheduler step: x_t → x_{t-1} ----------------------
                latents = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                )[0]

                # Progress + callback
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # ------------------------------------------------------------------ #
        # 10. Final paste: hard-composite KEEP region from clean original
        # ------------------------------------------------------------------ #
        # After the loop x_0 ≈ latents.  We do one last clean paste so the
        # KEEP region is pixel-perfect (no accumulated drift from re-noising).
        if mask_latent is not None and image_latents is not None:
            mask_clamped = mask_latent.clamp(0.0, 1.0)   # FIX 5: clamp here too for the final paste
            latents = (
                mask_clamped       * image_latents   # keep  ← clean original
                + (1 - mask_clamped) * latents       # repaint ← diffusion result
            )

        # ------------------------------------------------------------------ #
        # 11. Decode latents → pixels
        # ------------------------------------------------------------------ #
        if output_type != "latent":
            image_out = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
            image_out, has_nsfw_concept = self.run_safety_checker(
                image_out, device, prompt_embeds.dtype
            )
        else:
            image_out = latents
            has_nsfw_concept = None

        # ------------------------------------------------------------------ #
        # 12. Post-process
        # ------------------------------------------------------------------ #
        do_denormalize = (
            [True] * image_out.shape[0]
            if has_nsfw_concept is None
            else [not c for c in has_nsfw_concept]
        )
        image_out = self.image_processor.postprocess(
            image_out, output_type=output_type, do_denormalize=do_denormalize
        )

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image_out, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image_out, nsfw_content_detected=has_nsfw_concept
        )