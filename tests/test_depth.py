import numpy as np

from holoforge.depth import make_head_support_mask, pin_noncontent_depth, sculpt_head_depth


def test_head_sculpture_extends_above_tight_face_box_and_rounds_to_zero() -> None:
    depth = np.full((100, 100), 0.35, dtype=np.float32)
    depth[36:62, 39:61] = 0.62
    sculpted = sculpt_head_depth(depth, [(35, 30, 30, 36)], amount=0.10)

    # Hair/cranium above the detector rectangle is pulled into the head surface.
    assert sculpted[23, 50] > depth[23, 50]
    # The nose/centre is nearer than the temples.
    assert sculpted[49, 50] > sculpted[49, 32]
    # Pixels beyond the expanded oval remain untouched.
    assert sculpted[15, 50] == depth[15, 50]


def test_noncontent_depth_is_pinned_to_focus() -> None:
    depth = np.full((10, 12), 0.2, dtype=np.float32)
    alpha = np.zeros_like(depth)
    alpha[2:8, 3:9] = 1.0
    pinned = pin_noncontent_depth(depth, alpha, focus=0.57)
    assert np.allclose(pinned[alpha < 0.5], 0.57)
    assert np.allclose(pinned[alpha > 0.5], 0.2)


def test_head_support_uses_near_pixels_inside_expanded_oval() -> None:
    depth = np.full((80, 80), 0.2, dtype=np.float32)
    depth[20:54, 25:55] = 0.7
    mask = make_head_support_mask(depth, [(28, 25, 24, 28)])
    assert mask[35, 40] == 1
    assert mask[10, 40] == 0
    assert mask[35, 10] == 0
