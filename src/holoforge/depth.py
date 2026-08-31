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


def snap_depth_to_image_edges(
    image: Image.Image,
    depth: np.ndarray,
    *,
    radius: int = 4,
    depth_threshold: float = 0.045,
) -> np.ndarray:
    """Remove intermediate-depth halos where RGB and depth boundaries agree.

    Monocular depth upsampling often leaves a soft ramp between a foreground
    silhouette and its background. Those few pixels travel at an intermediate
    disparity and become detached copies at off-axis views. Strong photographic
    edges provide the missing boundary cue: within their neighbourhood, snap the
    ramp to whichever stable depth surface it is already closest to.
    """
    if radius <= 0:
        return depth.astype(np.float32, copy=True)

    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    median = float(np.median(gray))
    low = max(18, round(0.55 * median))
    high = max(low + 1, round(1.35 * median))
    photo_edges = cv2.Canny(gray, low, min(255, high), L2gradient=True)
    zone_kernel = np.ones((2 * radius + 1,) * 2, np.uint8)
    edge_zone = cv2.dilate(photo_edges, zone_kernel, iterations=1) > 0

    # Use a smaller depth neighbourhood than the RGB support so the operation
    # removes only the interpolation ramp rather than broadening either object.
    depth_radius = max(1, radius // 2)
    depth_kernel = np.ones((2 * depth_radius + 1,) * 2, np.uint8)
    local_far = cv2.erode(depth, depth_kernel)
    local_near = cv2.dilate(depth, depth_kernel)
    transition = (local_near - local_far) >= depth_threshold
    midpoint = (local_near + local_far) * 0.5
    snapped = np.where(depth >= midpoint, local_near, local_far)

    result = depth.astype(np.float32, copy=True)
    use_snap = edge_zone & transition
    result[use_snap] = snapped[use_snap]
    return np.clip(result, 0, 1).astype(np.float32)


def apply_face_relief(
    image: Image.Image,
    depth: np.ndarray,
    amount: float = 0.07,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Add subtle feature relief without guessing any hidden cranium geometry."""
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
    """Add a tapered cheek/nose prior strictly inside each detected face.

    A single photograph does not reveal the head's orientation or its hidden profile.
    Expanding a frontal detector box into a spherical cranium can therefore overwrite
    real background on three-quarter and profile portraits. This prior changes only
    visible facial-feature pixels and smoothly reaches zero at the face ellipse.
    """
    if amount <= 0 or not faces:
        return depth.astype(np.float32, copy=True)

    result = depth.astype(np.float32, copy=True)
    yy, xx = np.mgrid[: depth.shape[0], : depth.shape[1]]
    for x, y, w, h in faces:
        face_x = (xx - (x + 0.50 * w)) / max(0.48 * w, 1)
        face_y = (yy - (y + 0.52 * h)) / max(0.50 * h, 1)
        radius2 = face_x * face_x + face_y * face_y
        support = radius2 < 1.0
        taper = np.clip(1.0 - radius2, 0.0, 1.0) ** 1.5
        cheek = np.exp(-2.8 * radius2) * taper
        nose = np.exp(-20.0 * (face_x * face_x + ((face_y - 0.02) * 1.35) ** 2)) * taper
        result[support] += amount * (0.16 * cheek[support] + 0.50 * nose[support])

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


def draw_face_diagnostics(
    image: Image.Image, faces: list[tuple[int, int, int, int]]
) -> Image.Image:
    array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    for x, y, w, h in faces:
        cv2.rectangle(array, (x, y), (x + w, y + h), (40, 220, 70), 2)
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
