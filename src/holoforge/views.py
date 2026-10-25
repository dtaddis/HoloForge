"""Turn an orbit clip into evenly spaced light-field views.

H3 does not move the camera at a steady speed: it holds still for a moment after the first
keyframe, eases in, and eases out again before the last. It also does not always move the
camera at all. Some clips are a true camera orbit (the background swings past), others a
turntable (the subjects rotate in front of a still background). Frame numbers therefore say
little about viewing angle, and parallax measured over the whole frame is confounded by a
still background.

Two measurements that hold in both cases are used instead:

* Progress. The magnitude of optical flow between consecutive frames is accumulated along
  the clip. Small steps keep the flow reliable, and the sum is monotonic. The prompt asks
  for one full circle that ends where it began, so progress maps to an orbit angle: the
  fraction of total motion times 360 degrees, with the second half counted as negative
  (approaching the photographed viewpoint from the other side).
* Direction. Which side the clip leaves towards is read from epipolar geometry: an
  essential matrix fitted to the features that moved gives the camera centre's sideways
  offset for a few frames on each side (see ViewpointSide). Looking Glass quilts store the
  leftmost view first.

Views are placed at evenly spaced angles; where no frame lands close enough to a target,
the two bracketing frames are blended along their optical flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

ANALYSIS_WIDTH = 480
FIT_MARGIN = 0.06
# Assumed half-amplitude of the "sweep" motion prompt (right, back, left, back).
SWEEP_AMPLITUDE = 25.0


@dataclass
class Sample:
    frame: int
    angle: float  # degrees; positive = viewpoint left of the photograph


@dataclass
class ViewSource:
    frame_a: int
    frame_b: int
    weight: float
    target: float


@dataclass
class ViewPlan:
    sources: list[ViewSource]
    samples: list[Sample]
    span: tuple[float, float]
    requested: float
    direction_score: float = 0.0
    warnings: list[str] = field(default_factory=list)


def _gray(image: Image.Image, width: int = ANALYSIS_WIDTH) -> np.ndarray:
    scale = width / image.width
    size = (width, max(16, round(image.height * scale)))
    return cv2.cvtColor(
        np.asarray(image.resize(size, Image.Resampling.BILINEAR)), cv2.COLOR_RGB2GRAY
    )


def _texture_weights(gray: np.ndarray) -> np.ndarray:
    """Gradient energy inside a margin; textureless regions give arbitrary flow."""
    h, w = gray.shape
    mask = np.zeros((h, w), bool)
    my, mx = round(h * FIT_MARGIN), round(w * FIT_MARGIN)
    mask[my : h - my, mx : w - mx] = True
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    energy = cv2.GaussianBlur(gx * gx + gy * gy, (0, 0), 3)
    return np.where(mask, np.sqrt(energy), 0.0).astype(np.float64)


def _dis() -> cv2.DISOpticalFlow:
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


def motion_steps(frames: list[Image.Image]) -> np.ndarray:
    """Texture-weighted mean flow magnitude between consecutive frames (pixels at 480 wide)."""
    flow = _dis()
    grays = [_gray(frame) for frame in frames]
    steps = np.zeros(len(frames), np.float64)
    for i in range(1, len(frames)):
        f = flow.calc(grays[i - 1], grays[i], None).astype(np.float64)
        weights = _texture_weights(grays[i - 1])
        magnitude = np.hypot(f[..., 0], f[..., 1])
        steps[i] = (weights * magnitude).sum() / max(weights.sum(), 1e-9)
    # Sub-pixel jitter while H3 holds a keyframe is noise, not motion.
    noise = np.percentile(steps[1:], 10) if len(steps) > 2 else 0.0
    return np.clip(steps - noise, 0.0, None)


def paced_steps(steps: np.ndarray) -> np.ndarray:
    """Frame-to-frame progress for a constant-speed orbit with pauses at either end.

    The orbit LoRA cruises at constant speed, but H3 may still hold the first or last frames
    or ease in and out. Raw flow magnitude is a poor clock (nearby surroundings sweep past
    faster than distant ones), so each step counts as its share of the cruising speed: a
    pause counts as zero, easing as a fraction, and cruising frames count equally.
    """
    moving = steps[steps > 0]
    if moving.size == 0:
        return steps
    cruising = steps[steps > 0.25 * np.median(moving)]
    return np.clip(steps / max(float(np.median(cruising)), 1e-6), 0.0, 1.0)


def progress_angles(steps: np.ndarray, motion: str = "orbit") -> np.ndarray:
    """Signed angle per frame, positive on the side the clip leaves towards."""
    travel = np.cumsum(steps)
    if travel[-1] <= 1e-6:
        raise ValueError("The orbit shows no measurable motion; the camera did not move")
    q = travel / travel[-1]
    if motion == "sweep":
        a = SWEEP_AMPLITUDE
        return np.where(q < 0.25, 4 * q * a, np.where(q < 0.75, (2 - 4 * q) * a, (4 * q - 4) * a))
    theta = 360.0 * q
    return np.where(theta <= 180.0, theta, theta - 360.0)


class ViewpointSide:
    """Which side of the photographed viewpoint a frame was seen from, by epipolar geometry.

    Features are matched between the photograph's frame and a later one, and only those
    that moved are kept: a turntable clip's still background would otherwise make the
    essential matrix degenerate. The recovered camera centre's x is the side. A rotating
    subject in front of a still background gives the same answer as a camera orbiting it,
    because the moving features all belong to the subject.
    """

    def __init__(self, reference: Image.Image) -> None:
        self.sift = cv2.SIFT_create(4000)
        self.matcher = cv2.BFMatcher()
        ref = _gray(reference, 640)
        h, w = ref.shape
        f = float(max(w, h))  # ~50 mm equivalent; only the sign of the result is used
        self.camera = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        self.keypoints, self.descriptors = self.sift.detectAndCompute(ref, None)

    def measure(self, frame: Image.Image) -> tuple[float, int]:
        """(-1..1 camera-left amount, inlier count); (0, 0) when undetermined."""
        if self.descriptors is None:
            return 0.0, 0
        keypoints, descriptors = self.sift.detectAndCompute(_gray(frame, 640), None)
        if descriptors is None or len(keypoints) < 8:
            return 0.0, 0
        pairs = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        p0 = np.float32([self.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 2)
        p1 = np.float32([keypoints[m.trainIdx].pt for m in good]).reshape(-1, 2)
        moved = np.linalg.norm(p1 - p0, axis=1) > 1.0
        p0, p1 = p0[moved], p1[moved]
        if len(p0) < 30:
            return 0.0, 0
        essential, mask = cv2.findEssentialMat(p0, p1, self.camera, cv2.RANSAC, 0.999, 1.0)
        if essential is None or essential.shape != (3, 3):
            return 0.0, 0
        inliers, rotation, translation, _ = cv2.recoverPose(
            essential, p0, p1, self.camera, mask=mask
        )
        if inliers < 30:
            return 0.0, 0
        centre = (-rotation.T @ translation).ravel()
        return float(-centre[0]), int(inliers)


def direction_score(frames: list[Image.Image], angles: np.ndarray) -> float:
    """Inlier-weighted vote in [-2, 2]; > 0 when the clip leaves towards a viewpoint left of
    the photograph (and so returns from the right)."""
    meter = ViewpointSide(frames[0])
    half = len(frames) // 2

    def vote(indices: range) -> float:
        # Walk away from the photographed viewpoint and use the first frames that resolve:
        # features on small or plain sources stop matching after a few degrees, so the
        # nearest frames that have moved at all are the most reliable.
        results: list[tuple[float, int]] = []
        for i in indices:
            if abs(angles[i]) > 30.0 or len(results) >= 4:
                break
            side, inliers = meter.measure(frames[i])
            if inliers:
                results.append((side, inliers))
        weight = sum(n for _, n in results)
        return sum(side * n for side, n in results) / weight if weight else 0.0

    return vote(range(1, half)) - vote(range(len(frames) - 1, half - 1, -1))


def background_stillness(frames: list[Image.Image], motion: str = "orbit") -> float:
    """Share of matched features that stay put while the viewpoint is ~10 degrees off.

    Near 0 for a real camera orbit; near 1 when H3 turned the subjects on a turntable in
    front of a frozen background, which plays back with a flat, parallax-free background.
    """
    angles = progress_angles(motion_steps(frames), motion)
    sift = cv2.SIFT_create(3000)
    matcher = cv2.BFMatcher()
    reference = _gray(frames[0], 640)
    ref_kp, ref_desc = sift.detectAndCompute(reference, None)
    if ref_desc is None:
        return 0.0
    half = len(frames) // 2
    shares = []
    for want, indices in ((10.0, range(1, half)), (-10.0, range(half, len(frames)))):
        index = min(indices, key=lambda i: abs(angles[i] - want))
        kp, desc = sift.detectAndCompute(_gray(frames[index], 640), None)
        if desc is None:
            continue
        pairs = matcher.knnMatch(ref_desc, desc, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) < 30:
            continue
        moved = [np.hypot(*np.subtract(kp[m.trainIdx].pt, ref_kp[m.queryIdx].pt)) for m in good]
        shares.append(float(np.mean(np.asarray(moved) < 0.75)))
    return float(np.mean(shares)) if shares else 0.0


def plan_views(
    frames: list[Image.Image],
    views: int,
    sweep_degrees: float,
    motion: str = "orbit",
    *,
    constant_speed: bool = False,
) -> ViewPlan:
    """Choose a source (or flow blend of two) for each quilt view, leftmost camera first.

    sweep_degrees is the total viewing angle the quilt spans. constant_speed is for orbit-LoRA
    clips, trained on constant-speed renders: frames count evenly while the camera cruises
    and not at all while it pauses (see paced_steps), instead of by raw flow magnitude.
    """
    if len(frames) < 3:
        raise ValueError("An orbit needs at least three frames")
    steps = motion_steps(frames)
    angles = progress_angles(paced_steps(steps) if constant_speed else steps, motion)
    score = direction_score(frames, angles)
    sign = 1.0 if score >= 0 else -1.0
    warnings: list[str] = []
    if abs(score) < 0.5:
        warnings.append(
            f"Orbit direction is uncertain (score {score:+.5f}); check that near objects "
            "appear in front on the display"
        )

    # Only the outer quarters of the motion: frames near the photographed viewpoint.
    travel_q = np.cumsum(np.abs(np.diff(angles, prepend=0.0)))
    travel_q = travel_q / max(travel_q[-1], 1e-9)
    half = sweep_degrees / 2.0
    samples = [
        Sample(i, float(sign * angles[i]))
        for i in range(len(frames))
        if (travel_q[i] <= 0.25 or travel_q[i] >= 0.75) and abs(angles[i]) <= half * 1.5
    ]
    samples.sort(key=lambda s: -s.angle)
    top, bottom = samples[0].angle, samples[-1].angle
    hi, lo = min(half, top), max(-half, bottom)
    if hi - lo < sweep_degrees * 0.98:
        warnings.append(
            f"Clip covered {bottom:+.1f}..{top:+.1f} degrees; using {lo:+.1f}..{hi:+.1f}"
        )

    sources: list[ViewSource] = []
    spacing = (hi - lo) / max(views - 1, 1)
    for k in range(views):
        target = hi - k * spacing
        above = [s for s in samples if s.angle >= target]
        below = [s for s in samples if s.angle <= target]
        a = min(above, key=lambda s: s.angle - target) if above else samples[0]
        b = max(below, key=lambda s: s.angle - target) if below else samples[-1]
        gap = a.angle - b.angle
        weight = 0.0 if gap <= 1e-9 else (a.angle - target) / gap
        if abs(a.angle - target) <= 0.15 * spacing:
            b, weight = a, 0.0
        elif abs(b.angle - target) <= 0.15 * spacing:
            a, weight = b, 0.0
        sources.append(ViewSource(a.frame, b.frame, float(weight), float(target)))
    return ViewPlan(sources, samples, (lo, hi), sweep_degrees, score, warnings)


def flow_blend(a: Image.Image, b: Image.Image, weight: float) -> Image.Image:
    """Intermediate view between two frames, warping both along their optical flow."""
    if weight <= 0.0:
        return a
    if weight >= 1.0:
        return b
    ga = cv2.cvtColor(np.asarray(a), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(np.asarray(b), cv2.COLOR_RGB2GRAY)
    dis = _dis()
    f_ab = dis.calc(ga, gb, None)
    f_ba = dis.calc(gb, ga, None)
    h, w = ga.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    wa = cv2.remap(
        np.asarray(a),
        xx - weight * f_ab[..., 0],
        yy - weight * f_ab[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    wb = cv2.remap(
        np.asarray(b),
        xx - (1 - weight) * f_ba[..., 0],
        yy - (1 - weight) * f_ba[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    blended = wa.astype(np.float32) * (1 - weight) + wb.astype(np.float32) * weight
    return Image.fromarray(np.clip(blended + 0.5, 0, 255).astype(np.uint8))


def render_views(frames: list[Image.Image], plan: ViewPlan) -> list[Image.Image]:
    return [flow_blend(frames[s.frame_a], frames[s.frame_b], s.weight) for s in plan.sources]


def draw_parallax_plot(
    plan: ViewPlan, frame_count: int, size: tuple[int, int] = (900, 360)
) -> Image.Image:
    """Diagnostic: estimated viewing angle per frame with the chosen view targets."""
    width, height = size
    canvas = np.full((height, width, 3), 255, np.uint8)
    values = [s.angle for s in plan.samples] + list(plan.span)
    vmax = max(abs(v) for v in values) * 1.15 or 1.0

    def point(frame: float, value: float) -> tuple[int, int]:
        x = 40 + frame / max(frame_count - 1, 1) * (width - 60)
        y = height / 2 - value / vmax * (height / 2 - 20)
        return round(x), round(y)

    cv2.line(canvas, point(0, 0), point(frame_count - 1, 0), (180, 180, 180), 1)
    for value in plan.span:
        cv2.line(canvas, point(0, value), point(frame_count - 1, value), (120, 170, 255), 1)
    for source in plan.sources:
        cv2.line(canvas, point(0, source.target), point(6, source.target), (60, 120, 230), 1)
    for sample in plan.samples:
        cv2.circle(canvas, point(sample.frame, sample.angle), 3, (200, 40, 40), -1)
    cv2.putText(
        canvas,
        f"viewing angle per frame (red, +/-{vmax:.0f} deg), view targets (blue), "
        f"direction {plan.direction_score:+.4f}",
        (40, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (40, 40, 40),
        1,
        cv2.LINE_AA,
    )
    return Image.fromarray(canvas)
