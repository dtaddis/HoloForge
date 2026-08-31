import numpy as np

from holoforge.layers import make_backplate_depth, make_backplate_requests


def test_backplate_masks_grow_deeper() -> None:
    depth = np.tile(np.linspace(0, 1, 100, dtype=np.float32), (60, 1))
    requests = make_backplate_requests(depth, count=2, foreground_fraction=0.4, dilation=0)
    assert len(requests) == 2
    assert np.count_nonzero(requests[0].mask) < np.count_nonzero(requests[1].mask)
    assert requests[0].threshold > requests[1].threshold


def test_inpainted_depth_is_behind_removed_surface() -> None:
    depth = np.tile(np.linspace(0, 1, 80, dtype=np.float32), (40, 1))
    mask = np.zeros_like(depth, dtype=np.uint8)
    mask[:, 55:] = 255
    plate = make_backplate_depth(depth, mask, threshold=0.7)
    assert plate.shape == depth.shape
    assert float(plate[mask > 0].max()) <= 0.6751
