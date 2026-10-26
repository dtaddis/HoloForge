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
}


@dataclass
class OrbitConfig:
    """Settings for the MiniMax H3 orbit method (see orbit.py and views.py)."""

    preset: Preset = PRESETS["portrait"]
    # Total orbit angle spanned from the leftmost to the rightmost view. Matching the display's
    # view cone (40 degrees on the Portrait, from its calibration) is orthoscopic: the scene
    # turns exactly as far as the viewer's eye moves.
    sweep_degrees: float = 40.0
    # Real orbit angle per timed degree. The LoRA's 360-degree timing runs about 10% short:
    # viewed on a Looking Glass Portrait, a 40-degree quilt timed at 1.0 turned ~10% further
    # than the eye moves, so 1.1 makes the default sweep orthoscopic.
    angle_calibration: float = 1.1
    # Re-render once with camera-insistent wording when H3 makes a turntable clip (subjects
    # rotate in front of a frozen background); keep whichever clip has more real parallax.
    flat_retry: bool = True
    flat_threshold: float = 0.4
    # Below this camera_turn_rate the subjects moved rather than the camera (measured ~0.04 for
    # such a clip against 0.3-1.5 for real orbits); also triggers the one re-render.
    min_turn_rate: float = 0.15
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
