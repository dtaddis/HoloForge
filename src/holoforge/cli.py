from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import PRESETS, ConversionConfig
from .pipeline import HoloForgePipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="holoforge",
        description="Convert a 2D photo into a generated multi-view Looking Glass quilt.",
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--preset", choices=PRESETS, default="musubi")
    parser.add_argument("--inpaint", choices=["generative", "fast"], default="generative")
    parser.add_argument("--layout", choices=["fit", "fill", "ambient"], default="ambient")
    parser.add_argument("--view-cone", type=float, default=0.075)
    parser.add_argument("--focus", type=float, default=0.55)
    parser.add_argument("--face-relief", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--columns", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--quilt-size", type=int)
    parser.add_argument("--aspect", type=float)
    parser.add_argument("--png", action="store_true", help="Write lossless PNG instead of JPEG")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cfg = ConversionConfig(
        preset=PRESETS[args.preset],
        inpaint_mode=args.inpaint,
        fill_mode=args.layout,
        view_cone=args.view_cone,
        focus=args.focus,
        face_relief=args.face_relief,
        seed=args.seed,
        output_dir=args.output,
        output_format="png" if args.png else "jpg",
    ).with_preset_overrides(
        columns=args.columns,
        rows=args.rows,
        quilt_size=args.quilt_size,
        aspect=args.aspect,
    )
    if args.prompt:
        cfg = replace(cfg, prompt=args.prompt)

    def progress(value: float, message: str) -> None:
        print(f"[{value * 100:5.1f}%] {message}", flush=True)

    result = HoloForgePipeline(cfg).convert(args.input, progress)
    print(f"Quilt: {result.quilt_path.resolve()}")
    print(f"Preview: {result.wiggle_path.resolve()}")
    print(f"Elapsed: {result.elapsed_seconds:.1f}s")


if __name__ == "__main__":
    main()
