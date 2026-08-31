import numpy as np
from PIL import Image

from holoforge.image_utils import prepare_frame


def test_fit_preserves_whole_image() -> None:
    source = Image.new("RGB", (100, 50), "white")
    frame, alpha = prepare_frame(source, (60, 100), "fit")
    assert frame.size == (60, 100)
    assert alpha.shape == (100, 60)
    assert int(np.count_nonzero(alpha)) == 60 * 30


def test_fill_covers_target() -> None:
    source = Image.new("RGB", (100, 50), "white")
    frame, alpha = prepare_frame(source, (60, 100), "fill")
    assert frame.size == (60, 100)
    assert np.all(alpha == 1)


def test_ambient_keeps_a_separate_content_mask() -> None:
    source = Image.new("RGB", (100, 50), "white")
    frame, alpha = prepare_frame(source, (60, 100), "ambient")
    assert frame.size == (60, 100)
    assert int(np.count_nonzero(alpha)) == 60 * 30
