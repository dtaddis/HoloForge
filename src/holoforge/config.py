from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Preset:
    name: str
    columns: int
    rows: int
    quilt_size: int
    aspect: float

    @property
    def views(self) -> int:
        return self.columns * self.rows

    @property
    def tile_size(self) -> tuple[int, int]:
        return self.quilt_size // self.columns, self.quilt_size // self.rows


PRESETS: dict[str, Preset] = {
    # musubi's published content aspect is 9:16. Looking Glass has not published a
    # final developer quilt grid for musubi, so the grid remains user-editable.
    "musubi": Preset("musubi", columns=10, rows=6, quilt_size=8192, aspect=9 / 16),
    "portrait": Preset("portrait", columns=8, rows=6, quilt_size=3360, aspect=3 / 4),
    "lg16": Preset("lg16", columns=5, rows=9, quilt_size=4096, aspect=16 / 9),
    "preview": Preset("preview", columns=10, rows=6, quilt_size=2040, aspect=9 / 16),
}


@dataclass
class ConversionConfig:
    preset: Preset = PRESETS["musubi"]
    depth_model: str = "depth-anything/Depth-Anything-V2-Large-hf"
    inpaint_model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    inpaint_mode: str = "generative"
    inpaint_steps: int = 18
    inpaint_strength: float = 0.98
    guidance_scale: float = 6.0
    backplate_count: int = 2
    foreground_fraction: float = 0.34
    mask_dilation: int = 13
    view_cone: float = 0.075
    depth_gamma: float = 1.0
    focus: float = 0.55
    face_relief: float = 0.07
    edge_refine: float = 0.65
    fill_mode: str = "ambient"
    seed: int = 12345
    prompt: str = (
        "faithfully continue the original photograph behind the masked foreground, "
        "same location, era, lighting, lens, grain, colours and photographic style, "
        "natural coherent background structure, no new prominent subject"
    )
    negative_prompt: str = (
        "text, watermark, frame, border, duplicate person, extra face, deformed person, "
        "modern objects, painting, illustration, blur, grey void, empty band"
    )
    output_dir: Path = Path("outputs")
    output_format: str = "jpg"
    keep_diagnostics: bool = True

    def with_preset_overrides(
        self,
        *,
        columns: int | None = None,
        rows: int | None = None,
        quilt_size: int | None = None,
        aspect: float | None = None,
    ) -> ConversionConfig:
        p = self.preset
        p = replace(
            p,
            columns=columns or p.columns,
            rows=rows or p.rows,
            quilt_size=quilt_size or p.quilt_size,
            aspect=aspect or p.aspect,
        )
        return replace(self, preset=p)
