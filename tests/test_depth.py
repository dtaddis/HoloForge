import numpy as np
from PIL import Image

from holoforge.depth import pin_noncontent_depth, sculpt_head_depth, snap_depth_to_image_edges


def test_face_sculpture_stays_inside_detector_box_and_rounds_to_zero() -> None:
    depth = np.full((100, 100), 0.35, dtype=np.float32)
    depth[36:62, 39:61] = 0.62
    sculpted = sculpt_head_depth(depth, [(35, 30, 30, 36)], amount=0.10)

    # Background and hair outside the detector rectangle must never be changed.
    assert sculpted[23, 50] == depth[23, 50]
    assert sculpted[49, 32] == depth[49, 32]
    # The visible nose/centre receives a small, tapered amount of relief.
    assert sculpted[49, 50] > depth[49, 50]


def test_noncontent_depth_is_pinned_to_focus() -> None:
    depth = np.full((10, 12), 0.2, dtype=np.float32)
    alpha = np.zeros_like(depth)
    alpha[2:8, 3:9] = 1.0
    pinned = pin_noncontent_depth(depth, alpha, focus=0.57)
    assert np.allclose(pinned[alpha < 0.5], 0.57)
    assert np.allclose(pinned[alpha > 0.5], 0.2)


def test_rgb_guided_depth_snap_removes_intermediate_silhouette_ramp() -> None:
    pixels = np.zeros((24, 40, 3), dtype=np.uint8)
    pixels[:, :20] = 240
    image = Image.fromarray(pixels)
    depth = np.full((24, 40), 0.2, dtype=np.float32)
    depth[:, :18] = 0.8
    depth[:, 18] = 0.68
    depth[:, 19] = 0.54
    depth[:, 20] = 0.36
    snapped = snap_depth_to_image_edges(image, depth, radius=4, depth_threshold=0.04)
    assert np.all(snapped[:, 19] > depth[:, 19])
    assert np.all(snapped[:, 20] < depth[:, 20])
    assert np.all(snapped[:, 10] == depth[:, 10])
