import numpy as np

from holoforge.layers import make_backplate_alpha, make_backplate_depth, make_backplate_requests


def test_backplate_masks_grow_deeper() -> None:
    depth = np.tile(np.linspace(0, 1, 100, dtype=np.float32), (60, 1))
    requests = make_backplate_requests(depth, count=2, foreground_fraction=0.4, dilation=0)
    assert len(requests) == 2
    assert np.count_nonzero(requests[0].mask) < np.count_nonzero(requests[1].mask)
    assert requests[0].threshold > requests[1].threshold


def test_backplate_reveal_is_inset_from_generation_boundary_and_omits_centre() -> None:
    depth = np.zeros((100, 120), dtype=np.float32)
    depth[20:80, 25:95] = 1.0
    request = make_backplate_requests(
        depth, count=1, foreground_fraction=0.25, dilation=8
    )[0]
    assert np.all(request.mask[request.alpha_mask > 0] == 255)
    assert request.alpha_mask[50, 60] == 0
    assert request.alpha_mask[50, 25] == 255
    assert request.alpha_mask[50, 20] == 0
    assert request.mask[50, 15] == 255


def test_inpainted_depth_is_behind_removed_surface() -> None:
    depth = np.tile(np.linspace(0, 1, 80, dtype=np.float32), (40, 1))
    mask = np.zeros_like(depth, dtype=np.uint8)
    mask[:, 55:] = 255
    plate = make_backplate_depth(depth, mask, threshold=0.7)
    assert plate.shape == depth.shape
    assert float(plate[mask > 0].max()) <= 0.6751


def test_backplate_alpha_excludes_copied_source_pixels() -> None:
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[7:17, 11:21] = 255
    alpha = make_backplate_alpha(mask)
    assert np.all(alpha[7:17, 11:21] == 1)
    assert np.count_nonzero(alpha) == 100
    assert np.all(alpha[:6] == 0)
