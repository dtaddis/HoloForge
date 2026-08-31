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
    """Add a conservative spheroidal head and nose prior to flat portraits."""
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
    found: list[tuple[int, int, int, int]] = []
    for x, y, w, h in faces:
        found.append((int(x), int(y), int(w), int(h)))
    return sculpt_head_depth(depth, found, amount), found


def sculpt_head_depth(
    depth: np.ndarray,
    faces: list[tuple[int, int, int, int]],
    amount: float,
) -> np.ndarray:
    """Turn detected frontal faces into smoothly rounded head-depth islands.

    Face detectors tightly frame the visible features and omit much of the hair and
    cranium. A prior limited to that rectangle makes the face bulge while leaving a
    cardboard-flat head. This expanded ellipsoid softly pulls the hair/temples toward
    the face depth, then adds a smaller nose lobe. The prior reaches zero at the oval
    boundary, avoiding a new hard depth seam.
    """
    if amount <= 0 or not faces:
        return depth.astype(np.float32, copy=True)

    result = depth.astype(np.float32, copy=True)
    yy, xx = np.mgrid[: depth.shape[0], : depth.shape[1]]
    height, width = depth.shape
    for x, y, w, h in faces:
        x0 = max(0, round(x + 0.18 * w))
        x1 = min(width, round(x + 0.82 * w))
        y0 = max(0, round(y + 0.14 * h))
        y1 = min(height, round(y + 0.88 * h))
        core = depth[y0:y1, x0:x1]
        if core.size == 0:
            continue
        anchor = float(np.quantile(core, 0.62))

        # The expanded support includes hair, ears, and the jaw rather than only
        # the feature detector's rectangle.
        head_x = (xx - (x + 0.50 * w)) / max(0.67 * w, 1)
        head_y = (yy - (y + 0.38 * h)) / max(0.72 * h, 1)
        radius2 = head_x * head_x + head_y * head_y
        support = radius2 < 1.0
        shell = np.sqrt(np.clip(1.0 - radius2, 0.0, 1.0))
        blend = np.clip((1.0 - radius2) / 0.34, 0.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)

        # Keep the head a coherent near-depth surface, while retaining stronger
        # monocular detail where the estimator already found it.
        target = anchor - 0.18 * amount + 0.58 * amount * shell
        raised = np.maximum(result, target)
        result[support] = (
            result[support] * (1.0 - blend[support]) + raised[support] * blend[support]
        )

        face_x = (xx - (x + 0.50 * w)) / max(0.50 * w, 1)
        face_y = (yy - (y + 0.52 * h)) / max(0.56 * h, 1)
        oval = np.exp(-2.4 * (face_x * face_x + face_y * face_y))
        nose = np.exp(-18.0 * (face_x * face_x + ((face_y - 0.01) * 1.25) ** 2))
        result[support] += amount * (0.24 * oval[support] + 0.48 * nose[support])

    return np.clip(result, 0, 1).astype(np.float32)


def pin_noncontent_depth(
    depth: np.ndarray,
    content_alpha: np.ndarray,
    focus: float,
) -> np.ndarray:
    """Keep letterbox/ambient pixels stationary instead of tearing at view extremes."""
    result = depth.astype(np.float32, copy=True)
    result[content_alpha < 0.5] = float(np.clip(focus, 0.0, 1.0))
    return result


def make_head_support_mask(
    depth: np.ndarray,
    faces: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """Select near-depth head pixels for view-dependent side-surface extrusion."""
    mask = np.zeros_like(depth, dtype=np.float32)
    yy, xx = np.mgrid[: depth.shape[0], : depth.shape[1]]
    height, width = depth.shape
    for x, y, w, h in faces:
        x0 = max(0, round(x + 0.18 * w))
        x1 = min(width, round(x + 0.82 * w))
        y0 = max(0, round(y + 0.14 * h))
        y1 = min(height, round(y + 0.88 * h))
        core = depth[y0:y1, x0:x1]
        if core.size == 0:
            continue
        anchor = float(np.quantile(core, 0.55))
        nx = (xx - (x + 0.50 * w)) / max(0.64 * w, 1)
        ny = (yy - (y + 0.39 * h)) / max(0.70 * h, 1)
        oval = (nx * nx + ny * ny) < 1.0
        mask[oval & (depth >= anchor - 0.075)] = 1.0

    if np.any(mask):
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask.astype(np.float32)


def draw_face_diagnostics(
    image: Image.Image, faces: list[tuple[int, int, int, int]]
) -> Image.Image:
    array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    for x, y, w, h in faces:
        cv2.rectangle(array, (x, y), (x + w, y + h), (40, 220, 70), 2)
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
