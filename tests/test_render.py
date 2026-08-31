import numpy as np
from PIL import Image

from holoforge.render import LightFieldRenderer


def test_zero_view_preserves_source() -> None:
    image = Image.fromarray(np.random.default_rng(2).integers(0, 255, (24, 32, 3), dtype=np.uint8))
    depth = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (24, 1))
    renderer = LightFieldRenderer(
        image,
        depth,
        np.ones_like(depth),
        [],
        view_cone=0.08,
        focus=0.5,
        device="cpu",
    )
    rendered = np.asarray(renderer.render(0.0))
    error = np.abs(rendered.astype(int) - np.asarray(image).astype(int))
    assert float(error.mean()) < 1.1
