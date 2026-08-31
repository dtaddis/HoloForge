from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def safe_stem(path: str | Path) -> str:
    stem = Path(path).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "hologram"


def resize_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))


def prepare_frame(
    image: Image.Image,
    size: tuple[int, int],
    mode: str = "fit",
) -> tuple[Image.Image, np.ndarray]:
    """Place an input in the view frame and return a content alpha mask."""
    width, height = size
    if mode == "fill":
        return resize_cover(image, size), np.ones((height, width), dtype=np.float32)

    ratio = min(width / image.width, height / image.height)
    inner = (
        max(1, round(image.width * ratio)),
        max(1, round(image.height * ratio)),
    )
    resized = image.resize(inner, Image.Resampling.LANCZOS)
    x = (width - inner[0]) // 2
    y = (height - inner[1]) // 2

    if mode == "ambient":
        background = resize_cover(image, size)
        bg = cv2.cvtColor(np.asarray(background), cv2.COLOR_RGB2BGR)
        sigma = max(width, height) / 22
        bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=sigma, sigmaY=sigma)
        background = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
    else:
        background = Image.new("RGB", size, "black")

    background.paste(resized, (x, y))
    alpha = np.zeros((height, width), dtype=np.float32)
    alpha[y : y + inner[1], x : x + inner[0]] = 1.0
    return background, alpha


def robust_normalize(array: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    lo, hi = np.percentile(array[np.isfinite(array)], [low, high])
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def depth_preview(depth: np.ndarray) -> Image.Image:
    gray = np.clip(depth * 255, 0, 255).astype(np.uint8)
    colour = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return Image.fromarray(cv2.cvtColor(colour, cv2.COLOR_BGR2RGB))


def feather_mask(mask: np.ndarray, radius: int = 5) -> np.ndarray:
    mask = mask.astype(np.float32) / (255.0 if mask.max() > 1 else 1.0)
    if radius <= 0:
        return np.clip(mask, 0, 1)
    sigma = max(0.5, radius / 2)
    return np.clip(cv2.GaussianBlur(mask, (0, 0), sigma), 0, 1)
