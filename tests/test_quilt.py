from PIL import Image

from holoforge.config import Preset
from holoforge.quilt import QuiltWriter, quilt_filename


def test_quilt_order_starts_bottom_left() -> None:
    preset = Preset("test", columns=2, rows=2, quilt_size=10, aspect=1.0)
    writer = QuiltWriter(preset)
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for index, colour in enumerate(colours):
        writer.add(index, Image.new("RGB", (5, 5), colour))
    assert writer.canvas.getpixel((1, 8)) == colours[0]
    assert writer.canvas.getpixel((8, 8)) == colours[1]
    assert writer.canvas.getpixel((1, 1)) == colours[2]
    assert writer.canvas.getpixel((8, 1)) == colours[3]


def test_quilt_name_encodes_layout() -> None:
    preset = Preset("test", 10, 6, 8192, 9 / 16)
    assert quilt_filename("photo", preset, ".jpg") == "photo_quilt_qs10x6a0.5625.jpg"
