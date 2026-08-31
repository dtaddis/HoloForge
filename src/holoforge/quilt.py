from __future__ import annotations

from pathlib import Path

from PIL import Image

from .config import Preset


def quilt_filename(stem: str, preset: Preset, extension: str = ".png") -> str:
    aspect = f"{preset.aspect:.6f}".rstrip("0").rstrip(".")
    return f"{stem}_quilt_qs{preset.columns}x{preset.rows}a{aspect}{extension}"


class QuiltWriter:
    def __init__(self, preset: Preset) -> None:
        self.preset = preset
        self.canvas = Image.new("RGB", (preset.quilt_size, preset.quilt_size), "black")

    def tile_box(self, view_index: int) -> tuple[int, int, int, int]:
        col = view_index % self.preset.columns
        row_from_bottom = view_index // self.preset.columns
        row_from_top = self.preset.rows - 1 - row_from_bottom
        x0 = round(col * self.preset.quilt_size / self.preset.columns)
        x1 = round((col + 1) * self.preset.quilt_size / self.preset.columns)
        y0 = round(row_from_top * self.preset.quilt_size / self.preset.rows)
        y1 = round((row_from_top + 1) * self.preset.quilt_size / self.preset.rows)
        return x0, y0, x1, y1

    def add(self, view_index: int, view: Image.Image) -> None:
        box = self.tile_box(view_index)
        target_size = (box[2] - box[0], box[3] - box[1])
        if view.size != target_size:
            view = view.resize(target_size, Image.Resampling.LANCZOS)
        self.canvas.paste(view, box[:2])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.canvas.save(path, compress_level=2)
        return path


def save_wiggle(views: list[Image.Image], path: str | Path, max_side: int = 720) -> Path:
    if not views:
        raise ValueError("No views supplied for wiggle preview")
    sequence = views + list(reversed(views[1:-1]))
    frames: list[Image.Image] = []
    for view in sequence:
        scale = min(1.0, max_side / max(view.size))
        size = (round(view.width * scale), round(view.height * scale))
        frames.append(view.resize(size, Image.Resampling.LANCZOS))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=85,
        loop=0,
        optimize=False,
    )
    return path
