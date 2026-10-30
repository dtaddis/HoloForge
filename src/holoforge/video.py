"""Save orbit clips as ordinary MP4s for viewing."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from .portrait import find_ffmpeg

H264_ENCODERS = (
    ["-c:v", "libx264", "-preset", "slow", "-crf", "16"],
    ["-c:v", "h264_nvenc", "-preset", "p7", "-cq", "16"],
    ["-c:v", "h264_qsv", "-global_quality", "18"],
    ["-c:v", "h264_amf", "-quality", "quality"],
)


def _encoders(ffmpeg: str) -> str:
    return subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
    ).stdout


def save_frames_video(frames_dir: str | Path, path: str | Path, fps: float = 24.0) -> Path:
    """Encode frame_0000.png... as an H.264 MP4 that browsers and phones play."""
    frames_dir, path = Path(frames_dir), Path(path)
    if not any(frames_dir.glob("frame_*.png")):
        raise FileNotFoundError(f"No frames in {frames_dir}")
    ffmpeg = find_ffmpeg()
    available = _encoders(ffmpeg)
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for args in H264_ENCODERS:
        if args[1] not in available:
            continue
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png"),
            *args, "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(path),
        ]  # fmt: skip
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            return path
        errors.append(f"{args[1]}: {result.stderr.strip()[-300:]}")
    raise RuntimeError("No working H.264 encoder in ffmpeg. " + " | ".join(errors))


def save_sweep_video(views: list[Image.Image], path: str | Path, fps: float = 24.0) -> Path:
    """Encode the views left to right and back again: the look-around a hologram shows."""
    sequence = views + list(reversed(views[1:-1]))
    with tempfile.TemporaryDirectory() as tmp:
        for index, view in enumerate(sequence):
            view.save(Path(tmp) / f"frame_{index:04d}.png", compress_level=1)
        return save_frames_video(tmp, path, fps)


def orbit_video_path(run_dir: Path, attempt_dir: str, lora: bool) -> Path:
    kind = "lora" if lora else "base"
    return run_dir / "videos" / f"{run_dir.name}_{attempt_dir}_{kind}.mp4"


def backfill_orbit_videos(root: str | Path) -> list[Path]:
    """Encode an MP4 for every finished orbit clip under root that lacks an up-to-date one."""
    written = []
    for meta in sorted(Path(root).glob("**/orbit*/orbit.json")):
        clip = meta.parent
        try:
            lora = bool(json.loads(meta.read_text(encoding="utf-8"))["signature"].get("lora"))
        except (OSError, ValueError, KeyError):
            continue
        video = orbit_video_path(clip.parent, clip.name, lora)
        if video.exists() and video.stat().st_mtime >= meta.stat().st_mtime:
            continue
        save_frames_video(clip, video)
        written.append(video)
    return written
