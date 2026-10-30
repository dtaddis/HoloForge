from __future__ import annotations

import argparse
import re
import sys
from dataclasses import replace
from pathlib import Path

from .config import PRESETS, OrbitConfig
from .orbit import MOTION_PROMPTS, ORBIT_LORA, OrbitSettings

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def expand_inputs(paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for path in paths:
        if path.is_dir():
            images += sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        else:
            images.append(path)
    return images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="holoforge",
        description="Convert 2D photos into Looking Glass holograms (or, with --orbit-only, "
        "look-around videos and view images) with MiniMax H3 camera orbits rendered in a "
        "local ComfyUI.",
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="Photos or folders of photos")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--preset", choices=PRESETS, default="portrait")
    parser.add_argument("--layout", choices=["fit", "fill", "ambient"], default="ambient")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--prompt", default=None, help="Replace the whole generation prompt")
    parser.add_argument("--columns", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--quilt-size", type=int)
    parser.add_argument("--aspect", type=float)
    parser.add_argument("--png", action="store_true", help="Write lossless PNG instead of JPEG")
    parser.add_argument(
        "--orbit-only",
        action="store_true",
        help="No Looking Glass output: save the orbit video, each view as an image and a "
        "look-around video (for anyone without a Looking Glass)",
    )

    orbit = parser.add_argument_group("H3 orbit")
    orbit.add_argument("--motion", choices=MOTION_PROMPTS, default="orbit")
    orbit.add_argument(
        "--sweep",
        type=float,
        help="Degrees of orbit across the quilt (default 40, the Portrait's view cone)",
    )
    orbit.add_argument(
        "--angle-calibration",
        type=float,
        help="Real orbit degrees per timed degree (default 1.1; raise if views over-rotate)",
    )
    orbit.add_argument("--subject", default=None, help="Who/what is in the photo, for the prompt")
    orbit.add_argument("--style", default=None, help="e.g. 'Vintage film photograph'")
    orbit.add_argument(
        "--caption",
        choices=["auto", "off"],
        default="auto",
        help="Describe each photo with a local Ollama vision model for the prompt",
    )
    orbit.add_argument(
        "--frames", type=int, help="Clip length in frames (17k+5; default fits 24 GB)"
    )
    orbit.add_argument("--steps", type=int, default=28)
    orbit.add_argument(
        "--no-lora",
        action="store_true",
        help="Use the base H3 model with the descriptive prompt instead of the 360 orbit LoRA",
    )
    orbit.add_argument("--pdd", action="store_true", help="8-step PDD distillation (2.5x faster)")
    orbit.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    orbit.add_argument("--unet", default=None, help="H3 FL2VA model file in ComfyUI")
    orbit.add_argument("--clip", default=None, help="H3 Qwen3-VL text encoder file in ComfyUI")
    orbit.add_argument("--vae", default=None, help="H3 video VAE file in ComfyUI")
    orbit.add_argument(
        "--idle-seconds",
        type=float,
        default=180.0,
        help="Only submit after ComfyUI's queue has been empty this long (shares the GPU "
        "with other pipelines; 0 submits immediately)",
    )
    orbit.add_argument(
        "--no-flat-retry",
        action="store_true",
        help="Don't re-render clips where H3 turned the subjects instead of moving the camera",
    )
    orbit.add_argument("--push", action="store_true", help="Copy results to a connected Portrait")
    orbit.add_argument("--device", type=Path, help="Portrait drive (default: auto-detect)")

    return parser


def progress(value: float, message: str) -> None:
    print(f"[{value * 100:5.1f}%] {message}", flush=True)


def run_orbit(args: argparse.Namespace, images: list[Path]) -> None:
    from .orbit_pipeline import OrbitPipeline
    from .portrait import Calibration, find_portrait_drive, push_recordings

    if args.orbit_only and args.push:
        sys.exit("--orbit-only makes no Looking Glass output, so there is nothing to --push")
    device = None if args.orbit_only else args.device or find_portrait_drive()
    if args.push and device is None:
        sys.exit("--push: no Looking Glass Portrait drive found; connect it or pass --device")
    calibration = None
    if device is not None:
        calibration = Calibration.from_file(Path(device) / "LKG_calibration" / "visual.json")

    cfg = OrbitConfig(
        preset=PRESETS[args.preset],
        fill_mode=args.layout,
        output_dir=args.output,
        output_format="png" if args.png else "jpg",
        portrait_recording=calibration is not None,
        looking_glass=not args.orbit_only,
    ).with_preset_overrides(
        columns=args.columns, rows=args.rows, quilt_size=args.quilt_size, aspect=args.aspect
    )
    if args.sweep is not None:
        cfg = replace(cfg, sweep_degrees=args.sweep)
    if args.angle_calibration is not None:
        cfg = replace(cfg, angle_calibration=args.angle_calibration)
    if args.no_flat_retry:
        cfg = replace(cfg, flat_retry=False)
    settings = OrbitSettings(
        motion=args.motion,
        prompt=args.prompt,
        frames=args.frames,
        steps=args.steps,
        pdd=args.pdd,
        lora=None if args.no_lora else ORBIT_LORA,
        seed=args.seed,
        server=args.comfy_url,
        idle_seconds=args.idle_seconds,
    )
    for field, value in (
        ("unet_name", args.unet),
        ("clip_name", args.clip),
        ("vae_name", args.vae),
    ):
        if value:
            settings = replace(settings, **{field: value})
    if args.subject:
        settings = replace(settings, subject=args.subject)
    if args.style:
        settings = replace(settings, style=args.style)

    pipeline = OrbitPipeline(cfg, settings, calibration)
    per_image: dict[Path, OrbitSettings] = {}
    # The orbit LoRA wants its training caption verbatim, so captions only feed the base model.
    if args.caption == "auto" and args.no_lora and not (args.subject or args.prompt):
        from .caption import unload

        # Caption everything first so the vision model is out of VRAM before H3 runs.
        for image in images:
            caption = pipeline.caption(image)
            if caption is None:
                print(f"No caption for {image.name} (is Ollama running?); using generic wording")
                continue
            print(f"Caption {image.name}: {caption.subject} | {caption.style}", flush=True)
            per_image[image] = replace(
                settings, subject=caption.subject, style=args.style or caption.style
            )
        unload()

    failures: list[tuple[Path, str]] = []
    pushed: list[tuple[str, Path]] = []
    from .setup import gpu_session

    # Hold any GPU claim for the whole batch rather than handing it back between photos.
    with gpu_session():
        for number, image in enumerate(images, 1):
            print(f"=== {number}/{len(images)}: {image}", flush=True)
            try:
                result = pipeline.convert(image, progress, orbit=per_image.get(image))
                for warning in result.warnings:
                    print(f"Warning: {warning}")
                if result.quilt_path:
                    print(f"Quilt: {result.quilt_path.resolve()}")
                if result.orbit_video_path:
                    print(f"Orbit video: {result.orbit_video_path.resolve()}")
                if result.view_paths:
                    print(
                        f"Views: {len(result.view_paths)} in {result.view_paths[0].parent.resolve()}"
                    )
                if result.sweep_video_path:
                    print(f"Look-around video: {result.sweep_video_path.resolve()}")
                print(f"Preview: {result.wiggle_path.resolve()}")
                print(f"Elapsed: {result.elapsed_seconds:.1f}s")
                if args.push and result.recording_path:
                    entry = (result.quilt_path.name, result.recording_path)
                    push_recordings(Path(device), [entry])
                    pushed.append(entry)
                    # Keep the batch's own order at the front of the playlist.
                    push_recordings(Path(device), pushed, copy=False)
                    print(f"Pushed to Portrait: {result.recording_path.name}", flush=True)
            except Exception as exc:  # noqa: BLE001 - keep a long batch going past one bad photo
                failures.append((image, str(exc)))
                print(f"FAILED: {exc}", flush=True)
    if failures:
        print(f"{len(failures)} of {len(images)} photos failed:")
        for image, message in failures:
            print(f"  {image}: {message}")
        sys.exit(1)


def main() -> None:
    args = build_parser().parse_args()
    images = expand_inputs(args.inputs)
    if not images:
        sys.exit("No images found")
    run_orbit(args, images)


QUILT_PATTERN = re.compile(r"qs(\d+)x(\d+)a(\d+(?:\.\d+)?)", re.IGNORECASE)


def portrait_main() -> None:
    """holoforge-portrait: interlace existing quilts and copy them to a Portrait."""
    from .portrait import (
        Calibration,
        find_portrait_drive,
        prepare_recording,
        push_recordings,
        replace_playlist,
    )

    parser = argparse.ArgumentParser(
        prog="holoforge-portrait",
        description="Interlace quilts (named *_qs<cols>x<rows>a<aspect>.*) for a Looking Glass "
        "Portrait and add them to its standalone playlist.",
    )
    parser.add_argument("quilts", type=Path, nargs="+")
    parser.add_argument("--device", type=Path, help="Portrait drive (default: auto-detect)")
    parser.add_argument("--output", type=Path, help="Keep recordings here (default: beside quilt)")
    parser.add_argument("--append", action="store_true", help="Add to the end of the playlist")
    parser.add_argument("--no-push", action="store_true", help="Only render the recordings")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Make the Portrait play only these quilts, in this order, and delete every other "
        "*__lkgrec.mp4 recording on it",
    )
    args = parser.parse_args()
    if args.replace and (args.append or args.no_push):
        sys.exit("--replace cannot be combined with --append or --no-push")

    device = args.device or find_portrait_drive()
    if device is None:
        sys.exit("No Looking Glass Portrait drive found; connect it or pass --device")
    calibration = Calibration.from_file(Path(device) / "LKG_calibration" / "visual.json")
    entries = []
    for quilt in args.quilts:
        match = QUILT_PATTERN.search(quilt.name)
        if not match:
            sys.exit(f"{quilt.name}: name must contain qs<cols>x<rows>a<aspect>")
        columns, rows, aspect = int(match[1]), int(match[2]), float(match[3])
        recording, _ = prepare_recording(
            quilt, columns, rows, aspect, calibration, args.output or quilt.parent
        )
        print(f"Rendered {recording}")
        entries.append((quilt.name, recording))
    if args.replace:
        removed = replace_playlist(Path(device), entries)
        print(f"Playlist on {device} now has exactly {len(entries)} item(s)")
        for name in removed:
            print(f"  removed {name}")
    elif not args.no_push:
        push_recordings(Path(device), entries, first=not args.append)
        print(f"Added {len(entries)} item(s) to the playlist on {device}")


if __name__ == "__main__":
    main()
