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
class OrbitConfig:
    """Settings for the MiniMax H3 orbit method (see orbit.py and views.py)."""

    preset: Preset = PRESETS["portrait"]
    # Total orbit angle spanned from the leftmost to the rightmost view. The Portrait's
    # optical view cone is 40 degrees; a little less keeps backgrounds from smearing.
    sweep_degrees: float = 30.0
    # Re-render once with camera-insistent wording when H3 makes a turntable clip (subjects
    # rotate in front of a frozen background); keep whichever clip has more real parallax.
    flat_retry: bool = True
    flat_threshold: float = 0.4
    fill_mode: str = "ambient"
    output_dir: Path = Path("outputs")
    output_format: str = "jpg"
    portrait_recording: bool = True

    def with_preset_overrides(
        self,
        *,
        columns: int | None = None,
        rows: int | None = None,
        quilt_size: int | None = None,
        aspect: float | None = None,
    ) -> OrbitConfig:
        p = replace(
            self.preset,
            columns=columns or self.preset.columns,
            rows=rows or self.preset.rows,
            quilt_size=quilt_size or self.preset.quilt_size,
            aspect=aspect or self.preset.aspect,
        )
        return replace(self, preset=p)


@dataclass
class ConversionConfig:
    preset: Preset = PRESETS["musubi"]
    depth_model: str = "depth-anything/Depth-Anything-V2-Large-hf"
    inpaint_model: str = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    inpaint_mode: str = "generative"
    inpaint_steps: int = 28
    inpaint_strength: float = 0.55
    guidance_scale: float = 7.0
    backplate_count: int = 2
    foreground_fraction: float = 0.34
    mask_dilation: int = 17
    view_cone: float = 0.075
    depth_gamma: float = 1.0
    focus: float = 0.55
    face_relief: float = 0.10
    edge_refine: float = 0.65
    fill_mode: str = "ambient"
    seed: int = 12345
    prompt: str = (
        "object-removal photograph: reconstruct only the empty hidden scenery behind "
        "the masked foreground; completely remove masked people, faces, bodies, clothing, "
        "vehicles and foreground objects; faithfully continue the same architecture, ground, "
        "location, era, lighting, lens, grain, colours and photographic style"
    )
    negative_prompt: str = (
        "person, people, human, face, head, body, portrait, duplicate subject, mannequin, "
        "vehicle, text, watermark, frame, border, modern objects, painting, illustration, "
        "blur, grey void, empty band"
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
