"""Looking Glass Portrait standalone playback.

In standalone mode the Portrait plays pre-rendered lenticular videos from its own USB
storage, not quilts. Looking Glass Studio renders each quilt through its quilt_to_lent
shader at twice the panel width (3072x2048, so 4:2:0 chroma keeps one sample per physical
pixel), encodes it as HEVC with a 3:4 display aspect, copies it to the device as
``<quilt name>__lkgrec.mp4`` and lists it in ``playlist.json`` / ``playlist.m3u``.

This module reproduces that path without Studio. The shader port below follows Studio
1.7's quilt_to_lent_{vert,frag}.glsl, and the uniforms are derived from the device's own
``LKG_calibration/visual.json`` exactly as Studio logs them (pitch, slope, center,
subpixelSize).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import string
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

STILL_FPS = 12
STILL_SECONDS = 10
RECORDING_SUFFIX = "__lkgrec.mp4"


@dataclass(frozen=True)
class Calibration:
    pitch: float
    slope: float
    center: float
    dpi: float
    screen_w: int
    screen_h: int
    inv_view: bool = True
    serial: str = ""

    @classmethod
    def from_file(cls, path: str | Path) -> Calibration:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))

        def value(key: str, default: float = 0.0) -> float:
            item = raw.get(key, default)
            return float(item["value"] if isinstance(item, dict) else item)

        return cls(
            pitch=value("pitch"),
            slope=value("slope"),
            center=value("center"),
            dpi=value("DPI"),
            screen_w=int(value("screenW")),
            screen_h=int(value("screenH")),
            inv_view=bool(value("invView", 1)),
            serial=str(raw.get("serial", "")),
        )

    @property
    def aspect(self) -> float:
        return self.screen_w / self.screen_h

    @property
    def shader_pitch(self) -> float:
        return self.pitch * self.screen_w / self.dpi * math.cos(math.atan(1.0 / self.slope))

    @property
    def shader_slope(self) -> float:
        return self.screen_h / (self.screen_w * self.slope)

    @property
    def subpixel_size(self) -> float:
        return 1.0 / (3 * self.screen_w)


def render_lenticular(
    quilt: Image.Image,
    columns: int,
    rows: int,
    aspect: float,
    calibration: Calibration,
    *,
    width: int | None = None,
    height: int | None = None,
) -> Image.Image:
    """Interlace a quilt for the panel, matching Studio's quilt_to_lent shader."""
    width = width or calibration.screen_w * 2
    height = height or calibration.screen_h
    views = columns * rows
    tex = np.asarray(quilt.convert("RGB"), dtype=np.float32) / 255.0
    tex_h, tex_w = tex.shape[:2]

    # Fragment centres in GL screen space (y up; Studio vflips the read-back).
    sx = (np.arange(width, dtype=np.float64) + 0.5) / width
    sy = 1.0 - (np.arange(height, dtype=np.float64) + 0.5) / height
    sx, sy = np.meshgrid(sx, sy)

    # Vertex shader: the quilt quad is scaled vertically by device_aspect / aspect.
    uv_x = sx
    uv_y = ((2.0 * sy - 1.0) * aspect / calibration.aspect) * 0.5 + 0.5
    covered = (uv_y >= 0.0) & (uv_y <= 1.0)

    out = np.zeros((height, width, 3), dtype=np.float32)
    tx = columns - 0.00001
    for channel in range(3):
        lerp = sx + calibration.subpixel_size * channel + sy * calibration.shader_slope
        lerp = lerp * calibration.shader_pitch - calibration.center + 0.5 / views
        lerp = np.mod(lerp, 1.0)
        if calibration.inv_view:
            lerp = 1.0 - lerp
        lerp = np.clip(lerp, 0.00001, 0.99999)
        view1 = np.floor(lerp * views)
        view1[view1 == views] = 0
        view2 = view1 + 1
        view2[view2 == views] = 0
        blend = (lerp * views - view1).astype(np.float32)
        samples = []
        for view in (view1, view2):
            u = (np.mod(view, tx) + uv_x) / tx
            v = (np.floor(view / tx) + uv_y) / rows
            # GL texture v runs up from the bottom of the quilt image.
            map_x = (u * tex_w - 0.5).astype(np.float32)
            map_y = ((1.0 - v) * tex_h - 0.5).astype(np.float32)
            samples.append(
                cv2.remap(
                    tex[..., channel],
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            )
        out[..., channel] = samples[0] * (1.0 - blend) + samples[1] * blend
    out[~covered] = 0.0
    return Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8))


def find_ffmpeg() -> str:
    """A free ffmpeg build: the one holoforge-setup installed, or one on PATH."""
    from .setup import ensure_ffmpeg

    return str(ensure_ffmpeg(download=False))


def encoder_args(ffmpeg: str) -> list[str]:
    listing = subprocess.run(
        [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
    ).stdout
    if "libx265" in listing:
        # CRF 18 keeps the subpixel lenticular pattern at least as faithfully as Looking Glass
        # Studio's kvazaar encode (measured 25.46 vs 25.30 dB PSNR; the 4:2:0 chroma that every
        # Portrait recording uses sets the ceiling), in a smaller file. hev1 tagging as Studio.
        # The 1 MB VBV buffer caps every frame at 1 MB: the Portrait shows green for videos whose
        # first frame is ~2.5 MB or more, and for some (Scan0141) at 1.26 MB, though 0.91 MB of
        # the same picture played. Later frames restore the detail the cap costs the first one
        # (within 0.1 dB of uncapped).
        return ["-c:v", "libx265", "-preset", "fast", "-crf", "18", "-tag:v", "hev1",
                "-x265-params",
                "log-level=error:vbv-bufsize=8000:vbv-maxrate=16000:vbv-init=0.9"]  # fmt: skip
    if "libkvazaar" in listing:
        return ["-c:v", "libkvazaar", "-kvazaar-params", "preset=fast"]
    raise RuntimeError(f"{ffmpeg} has no HEVC encoder (libx265); install a full ffmpeg build")


def encode_still(lenticular: Image.Image, path: str | Path, ffmpeg: str | None = None) -> Path:
    """Encode an interlaced frame as the 10 s, 12 fps HEVC loop Studio makes for stills."""
    ffmpeg = ffmpeg or find_ffmpeg()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        frame = Path(tmp) / "lenticular.png"
        lenticular.save(frame, compress_level=1)
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-loop", "1", "-framerate", str(STILL_FPS), "-i", str(frame),
            "-frames:v", str(STILL_FPS * STILL_SECONDS),
            *encoder_args(ffmpeg),
            "-aspect", f"{lenticular.width / 2 / lenticular.height:.6f}",
            "-pix_fmt", "yuv420p",
            str(path),
        ]  # fmt: skip
        # kvazaar writes its banner to stderr regardless of -loglevel; keep it for errors only.
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed ({result.returncode}): {result.stderr[-2000:]}")
    return path


def recording_name(quilt_name: str) -> str:
    return f"{quilt_name}{RECORDING_SUFFIX}"


def find_portrait_drive() -> Path | None:
    """Return the mounted Portrait's storage root (it carries LKG_calibration/visual.json)."""
    roots: list[Path] = []
    if os.name == "nt":
        roots = [Path(f"{letter}:\\") for letter in string.ascii_uppercase]
    else:
        for base in (Path("/media"), Path("/run/media"), Path("/Volumes")):
            if base.exists():
                roots += [p for p in base.glob("*") if p.is_dir()]
                roots += [p for p in base.glob("*/*") if p.is_dir()]
    for root in roots:
        try:
            if (root / "LKG_calibration" / "visual.json").exists() and (
                root / "playlist.m3u"
            ).exists():
                return root
        except OSError:
            continue
    return None


def _backup_once(path: Path) -> None:
    backup = path.with_name(path.name + ".holoforge-backup")
    if path.exists() and not backup.exists():
        shutil.copy2(path, backup)


def push_recordings(
    device: Path, entries: list[tuple[str, Path]], *, first: bool = True, copy: bool = True
) -> None:
    """Copy (quilt name, recording) pairs to the device and list them in its playlists.

    New entries go to the front of the playlist (or the end with first=False). An entry that
    is already listed is moved rather than duplicated; other items are never removed. The
    original playlist files are backed up once. copy=False only rewrites the playlists.
    """
    if copy:
        needed = sum(rec.stat().st_size for _, rec in entries)
        free = shutil.disk_usage(device).free
        if needed > free:
            raise OSError(
                f"Portrait storage has {free / 1e6:.0f} MB free; need {needed / 1e6:.0f} MB"
            )
        for quilt_name, recording in entries:
            shutil.copyfile(recording, device / recording_name(quilt_name))

    names = [quilt_name for quilt_name, _ in entries]
    json_path = device / "playlist.json"
    m3u_path = device / "playlist.m3u"
    _backup_once(json_path)
    _backup_once(m3u_path)

    playlist = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else []
    playlist = [item for item in playlist if item.get("filename") not in names]
    added = [{"filename": n, "last_updated": "never", "needs_sync": False} for n in names]
    playlist = added + playlist if first else playlist + added
    # Studio writes both playlists with CRLF line endings.
    json_path.write_text(json.dumps(playlist, indent=4), encoding="utf-8", newline="\r\n")

    lines = m3u_path.read_text(encoding="utf-8").splitlines() if m3u_path.exists() else []
    recordings = [recording_name(n) for n in names]
    lines = [line for line in lines if line.strip() and line.strip() not in recordings]
    lines = recordings + lines if first else lines + recordings
    m3u_path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8", newline="")


def replace_playlist(device: Path, entries: list[tuple[str, Path]]) -> list[str]:
    """Make the device play exactly these entries, in order.

    Copies the recordings, rewrites both playlists to list only them, and deletes every other
    ``*__lkgrec.mp4`` recording from the device root. Other files (calibration, system
    folders, playlist backups) are left alone. Returns the names of the deleted recordings.
    """
    keep = {recording_name(name) for name, _ in entries}
    if len(keep) != len(entries):
        raise ValueError("Duplicate quilt names in the new playlist")
    # Free space first: the replaced recordings may be what makes room for the new ones.
    removed = []
    for path in sorted(device.glob(f"*{RECORDING_SUFFIX}")):
        if path.name not in keep:
            path.unlink()
            removed.append(path.name)
    json_path = device / "playlist.json"
    m3u_path = device / "playlist.m3u"
    _backup_once(json_path)
    _backup_once(m3u_path)
    json_path.write_text("[]", encoding="utf-8", newline="\r\n")
    m3u_path.write_text("", encoding="utf-8", newline="")
    push_recordings(device, entries, first=False)
    return removed


def prepare_recording(
    quilt_path: str | Path,
    columns: int,
    rows: int,
    aspect: float,
    calibration: Calibration,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Render and encode one quilt; returns (recording, lenticular preview frame)."""
    quilt_path = Path(quilt_path)
    output_dir = Path(output_dir)
    with Image.open(quilt_path) as quilt:
        lenticular = render_lenticular(quilt, columns, rows, aspect, calibration)
    preview = output_dir / f"{quilt_path.stem}_lenticular.png"
    lenticular.save(preview, compress_level=1)
    recording = encode_still(lenticular, output_dir / recording_name(quilt_path.name))
    return recording, preview
