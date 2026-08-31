from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class BackplateRequest:
    mask: np.ndarray
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
        mask = ((depth >= threshold) & valid).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        if dilation > 0:
            d = dilation if dilation % 2 == 1 else dilation + 1
            mask = cv2.dilate(mask, np.ones((d, d), np.uint8), iterations=1)
        requests.append(BackplateRequest(mask=mask, threshold=threshold, fraction=float(fraction)))
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
