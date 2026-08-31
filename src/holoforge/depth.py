from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from .image_utils import robust_normalize


@dataclass
class DepthResult:
    near_depth: np.ndarray
    raw_depth: np.ndarray


class DepthEstimator:
    """Lazy GPU-backed monocular depth estimator."""

    def __init__(self, model_id: str, device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None

    def _load(self) -> None:
        if self.model is not None:
            return
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            self.model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def estimate(
        self,
        image: Image.Image,
        *,
        edge_refine: float = 0.65,
        depth_gamma: float = 1.0,
    ) -> DepthResult:
        self._load()
        assert self.processor is not None and self.model is not None
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.device == "cuda":
            inputs = {k: v.half() if torch.is_floating_point(v) else v for k, v in inputs.items()}
        prediction = self.model(**inputs).predicted_depth.unsqueeze(1)
        prediction = F.interpolate(
            prediction,
            size=(image.height, image.width),
            mode="bicubic",
            align_corners=False,
        )[0, 0]
        raw = prediction.float().cpu().numpy()

        # Depth Anything V2 relative checkpoints encode nearer surfaces with
        # larger values. Bilateral smoothing removes transformer patch texture
        # without smearing silhouettes.
        near = robust_normalize(raw)
        refined = cv2.bilateralFilter(near, d=9, sigmaColor=0.055, sigmaSpace=7)
        near = near * (1.0 - edge_refine) + refined * edge_refine
        near = np.power(np.clip(near, 0, 1), max(depth_gamma, 1e-3))
        return DepthResult(near.astype(np.float32), raw.astype(np.float32))


def apply_face_relief(
    image: Image.Image,
    depth: np.ndarray,
    amount: float = 0.07,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Add a conservative convex face/nose prior to otherwise flat portraits."""
    if amount <= 0:
        return depth, []
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        return depth, []
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    min_side = max(28, min(image.size) // 35)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=4,
        minSize=(min_side, min_side),
    )
    result = depth.copy()
    yy, xx = np.mgrid[: depth.shape[0], : depth.shape[1]]
    found: list[tuple[int, int, int, int]] = []
    for x, y, w, h in faces:
        found.append((int(x), int(y), int(w), int(h)))
        nx = (xx - (x + 0.50 * w)) / max(0.5 * w, 1)
        ny = (yy - (y + 0.53 * h)) / max(0.55 * h, 1)
        oval = np.exp(-2.2 * (nx * nx + ny * ny))
        nose = np.exp(-18.0 * (nx * nx + ((ny - 0.02) * 1.25) ** 2))
        prior = amount * (0.48 * oval + 0.52 * nose)
        support = (nx * nx + ny * ny) <= 1.0
        result[support] += prior[support]
    return np.clip(result, 0, 1).astype(np.float32), found


def draw_face_diagnostics(
    image: Image.Image, faces: list[tuple[int, int, int, int]]
) -> Image.Image:
    array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    for x, y, w, h in faces:
        cv2.rectangle(array, (x, y), (x + w, y + h), (40, 220, 70), 2)
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
