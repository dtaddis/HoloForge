from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image

from .image_utils import feather_mask


class Inpainter(ABC):
    @abstractmethod
    def inpaint(
        self,
        image: Image.Image,
        mask: np.ndarray,
        *,
        prompt: str,
        negative_prompt: str,
        seed: int,
    ) -> Image.Image: ...


class FastInpainter(Inpainter):
    """Deterministic narrow-gap fallback; useful for previews and tests."""

    def inpaint(
        self,
        image: Image.Image,
        mask: np.ndarray,
        *,
        prompt: str = "",
        negative_prompt: str = "",
        seed: int = 0,
    ) -> Image.Image:
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        result = cv2.inpaint(bgr, mask.astype(np.uint8), 7, cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


class DiffusionInpainter(Inpainter):
    """SDXL inpainting constrained to hidden regions."""

    def __init__(
        self,
        model_id: str,
        *,
        steps: int = 18,
        strength: float = 0.98,
        guidance_scale: float = 6.0,
        device: str | None = None,
        max_side: int = 1024,
    ) -> None:
        self.model_id = model_id
        self.steps = steps
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_side = max_side
        self.pipe = None

    def _load(self) -> None:
        if self.pipe is not None:
            return
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        kwargs = {"dtype": dtype, "use_safetensors": True}
        try:
            self.pipe = AutoPipelineForInpainting.from_pretrained(
                self.model_id,
                variant="fp16" if self.device == "cuda" else None,
                **kwargs,
            )
        except (OSError, ValueError):
            self.pipe = AutoPipelineForInpainting.from_pretrained(self.model_id, **kwargs)
        if self.device == "cuda":
            self.pipe.to("cuda")
        else:
            self.pipe.to("cpu")
        self.pipe.set_progress_bar_config(disable=True)
        if hasattr(self.pipe, "enable_vae_tiling"):
            self.pipe.enable_vae_tiling()
        elif hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()

    @staticmethod
    def _generation_size(size: tuple[int, int], max_side: int) -> tuple[int, int]:
        width, height = size
        scale = min(1.0, max_side / max(width, height))
        return max(64, round(width * scale / 8) * 8), max(64, round(height * scale / 8) * 8)

    @torch.inference_mode()
    def inpaint(
        self,
        image: Image.Image,
        mask: np.ndarray,
        *,
        prompt: str,
        negative_prompt: str,
        seed: int,
    ) -> Image.Image:
        self._load()
        assert self.pipe is not None
        original_size = image.size
        generation_size = self._generation_size(original_size, self.max_side)
        source = image.resize(generation_size, Image.Resampling.LANCZOS)
        mask_image = Image.fromarray(mask.astype(np.uint8)).resize(
            generation_size, Image.Resampling.NEAREST
        )
        generator_device = "cuda" if self.device == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source,
            mask_image=mask_image,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale,
            strength=self.strength,
            generator=generator,
            width=generation_size[0],
            height=generation_size[1],
        ).images[0]
        result = result.resize(original_size, Image.Resampling.LANCZOS)

        # Some diffusion pipelines alter unmasked pixels slightly. Re-composite
        # so the visible source photograph remains pixel-faithful.
        feather = feather_mask(mask, radius=7)[..., None]
        src = np.asarray(image).astype(np.float32)
        gen = np.asarray(result).astype(np.float32)
        merged = src * (1.0 - feather) + gen * feather
        return Image.fromarray(np.clip(merged, 0, 255).astype(np.uint8))


def create_inpainter(
    mode: str,
    model_id: str,
    *,
    steps: int,
    strength: float,
    guidance_scale: float,
) -> Inpainter:
    if mode == "fast":
        return FastInpainter()
    if mode != "generative":
        raise ValueError(f"Unknown inpaint mode: {mode}")
    return DiffusionInpainter(
        model_id,
        steps=steps,
        strength=strength,
        guidance_scale=guidance_scale,
    )
