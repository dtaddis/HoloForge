from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class BackplateRequest:
    # A wide object-removal mask used by the AI.
    mask: np.ndarray
    # The smaller parallax band that is safe to reveal in the renderer.
    alpha_mask: np.ndarray
    threshold: float
    fraction: float


@dataclass
class Backplate:
    image: Image.Image
    depth: np.ndarray
    alpha: np.ndarray
    mask: np.ndarray
    threshold: float


def make_backplate_alpha(mask: np.ndarray) -> np.ndarray:
    """Limit a generated plate to pixels that were actually reconstructed.

    Inpainting returns a full image whose unmasked pixels are copies of the source.
    Treating that full image as opaque creates a second, differently shifted copy of
    every face and silhouette. Only the reconstruction mask is valid hidden content.
    """
    return (mask > 0).astype(np.float32)


def make_backplate_requests(
    depth: np.ndarray,
    *,
    count: int = 2,
    foreground_fraction: float = 0.34,
    dilation: int = 13,
    content_alpha: np.ndarray | None = None,
) -> list[BackplateRequest]:
    if count <= 0:
        return []
    valid = np.ones_like(depth, dtype=bool)
    if content_alpha is not None:
        valid &= content_alpha > 0.5
    samples = depth[valid]
    if samples.size == 0:
        return []

    minimum = max(0.09, foreground_fraction / max(count, 1))
    fractions = np.linspace(minimum, foreground_fraction, count)
    requests: list[BackplateRequest] = []
    kernel = np.ones((3, 3), np.uint8)
    for fraction in fractions:
        threshold = float(np.quantile(samples, 1.0 - fraction))
        core = ((depth >= threshold) & valid).astype(np.uint8) * 255
        core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, kernel, iterations=2)
        core = cv2.morphologyEx(core, cv2.MORPH_OPEN, kernel, iterations=1)

        # Newly visible background is confined to a band around an occluder's
        # silhouette. Do not display the generated centre of a large removed
        # person: it is never needed at the configured view cone and is where
        # generative models are most likely to leave a ghostly duplicate.
        reveal_radius = max(3, int(dilation))
        distance_inside = cv2.distanceTransform((core > 0).astype(np.uint8), cv2.DIST_L2, 5)
        inner_band = (distance_inside <= 2.0 * reveal_radius) & (core > 0)
        # Generated pixels outside the original occluder would cover valid source
        # background and move at a slightly different disparity, creating a visible
        # cutout-shaped halo. Only truly hidden pixels inside the silhouette are valid.
        alpha_mask = inner_band.astype(np.uint8) * 255

        # Generate beyond the visible band. The extra margin absorbs LaMa/SDXL's
        # blend boundary so no source-coloured edge can leak into a disocclusion.
        safety_radius = max(9, reveal_radius // 2)
        generation_radius = reveal_radius + safety_radius
        generation_kernel = np.ones((2 * generation_radius + 1,) * 2, np.uint8)
        generation_mask = cv2.dilate(core, generation_kernel, iterations=1)
        requests.append(
            BackplateRequest(
                mask=generation_mask,
                alpha_mask=alpha_mask,
                threshold=threshold,
                fraction=float(fraction),
            )
        )
    return requests


def make_backplate_depth(depth: np.ndarray, mask: np.ndarray, threshold: float) -> np.ndarray:
    inpainted = cv2.inpaint(
        depth.astype(np.float32),
        mask,
        inpaintRadius=5,
        flags=cv2.INPAINT_TELEA,
    )
    # A generated pixel belongs behind the object that hid it.
    behind = max(0.0, threshold - 0.025)
    inpainted[mask > 0] = np.minimum(inpainted[mask > 0], behind)
    return np.clip(inpainted, 0, 1).astype(np.float32)
