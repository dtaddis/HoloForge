"""Photograph -> MiniMax H3 orbit clip -> evenly spaced views -> quilt (-> Portrait recording)."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from PIL import Image

from .caption import Caption, caption_photo
from .config import OrbitConfig
from .image_utils import load_rgb, prepare_frame, safe_stem
from .orbit import OrbitSettings, generate_orbit
from .portrait import Calibration, find_portrait_drive, prepare_recording
from .quilt import QuiltWriter, quilt_filename, save_wiggle
from .setup import gpu_session
from .video import orbit_video_path, save_frames_video
from .views import (
    background_stillness,
    camera_turn_rate,
    draw_parallax_plot,
    plan_views,
    render_views,
)

ProgressCallback = Callable[[float, str], None]


@dataclass
class OrbitResult:
    quilt_path: Path
    wiggle_path: Path
    centre_view_path: Path
    parallax_plot_path: Path
    metadata_path: Path
    recording_path: Path | None
    warnings: list[str]
    elapsed_seconds: float
    orbit_video_path: Path | None = None


class OrbitPipeline:
    def __init__(
        self,
        config: OrbitConfig,
        orbit: OrbitSettings,
        calibration: Calibration | None = None,
    ) -> None:
        self.config = config
        self.orbit = orbit
        self.calibration = calibration

    @staticmethod
    def _notify(callback: ProgressCallback | None, value: float, message: str) -> None:
        if callback:
            callback(value, message)

    def _calibration(self) -> Calibration | None:
        if self.calibration is None:
            device = find_portrait_drive()
            if device is not None:
                self.calibration = Calibration.from_file(device / "LKG_calibration" / "visual.json")
        return self.calibration

    def run_dir(self, input_path: str | Path) -> Path:
        return self.config.output_dir / safe_stem(Path(input_path))

    def caption(self, input_path: str | Path) -> Caption | None:
        """Caption once per photo and keep it, so cached orbits stay valid on reruns."""
        path = self.run_dir(input_path) / "caption.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return Caption(**data) if data else None
        caption = caption_photo(load_rgb(input_path))
        if caption is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(asdict(caption), indent=2), encoding="utf-8")
        return caption

    def convert(
        self,
        input_path: str | Path,
        progress: ProgressCallback | None = None,
        *,
        orbit: OrbitSettings | None = None,
    ) -> OrbitResult:
        # One GPU claim covers a photo's render and any re-render.
        with gpu_session():
            return self._convert(input_path, progress, orbit=orbit)

    def _convert(
        self,
        input_path: str | Path,
        progress: ProgressCallback | None = None,
        *,
        orbit: OrbitSettings | None = None,
    ) -> OrbitResult:
        started = time.perf_counter()
        cfg = self.config
        orbit = orbit or self.orbit
        source_path = Path(input_path)
        stem = safe_stem(source_path)
        run_dir = self.run_dir(source_path)
        run_dir.mkdir(parents=True, exist_ok=True)
        source = load_rgb(source_path)

        self._notify(progress, 0.02, "Generating the H3 orbit (several minutes)")

        last_wait_report = [-1.0]

        def waiting(seconds: float) -> None:
            if seconds - last_wait_report[0] >= 60:
                last_wait_report[0] = seconds
                self._notify(progress, 0.05, f"H3 is rendering the orbit ({seconds / 60:.0f} min)")

        last_busy_report = [-1.0]

        def busy(seconds: float) -> None:
            if seconds - last_busy_report[0] >= 300:
                last_busy_report[0] = seconds
                self._notify(
                    progress,
                    0.02,
                    f"ComfyUI is busy with other work; waiting ({seconds / 60:.0f} min)",
                )

        frames, orbit_meta = generate_orbit(
            source, run_dir / "orbit", orbit, name=stem, on_wait=waiting, on_busy=busy
        )

        def assess(clip: list[Image.Image], settings: OrbitSettings, directory: str) -> dict:
            still = background_stillness(clip, settings.motion)
            turn = camera_turn_rate(clip, settings.motion) if settings.lora else None
            problem = None
            if still > cfg.flat_threshold:
                problem = (
                    f"H3 turned the subjects instead of the camera ({still:.0%} of the scene "
                    "stayed still): the background will look flat"
                )
            elif turn is not None and turn < cfg.min_turn_rate:
                problem = (
                    "H3 animated the subjects instead of moving the camera (the viewpoint "
                    f"barely turns: {turn:.2f} of the expected rate)"
                )
            return {"directory": directory, "motion": settings.motion, "stillness": still,
                    "turn_rate": turn, "problem": problem}  # fmt: skip

        attempts = [assess(frames, orbit, "orbit")]
        if (
            cfg.flat_retry
            and attempts[0]["problem"]
            and orbit.prompt is None
            and orbit.motion in ("orbit", "orbit_camera")
        ):
            self._notify(progress, 0.40, attempts[0]["problem"] + "; re-rendering with a new seed")
            # With the orbit LoRA the prompt is fixed, so only the seed changes.
            retry = replace(
                orbit,
                motion=orbit.motion if orbit.lora else "orbit_camera",
                seed=orbit.seed + 1,
            )
            retry_frames, retry_meta = generate_orbit(
                source, run_dir / "orbit_retry", retry, name=f"{stem}_retry", on_wait=waiting
            )
            attempts.append(assess(retry_frames, retry, "orbit_retry"))
            first, second = attempts

            def badness(attempt: dict) -> tuple:
                return (attempt["problem"] is not None, attempt["stillness"])

            if badness(second) < badness(first):
                frames, orbit_meta, orbit = retry_frames, retry_meta, retry
        chosen = min(attempts, key=lambda a: (a["problem"] is not None, a["stillness"]))
        stillness = chosen["stillness"]

        # Keep every generated clip as a viewable MP4 (frames are overwritten on re-render).
        chosen_video = None
        for attempt in attempts:
            video = orbit_video_path(run_dir, attempt["directory"], bool(orbit.lora))
            try:
                if (
                    not video.exists()
                    or video.stat().st_mtime
                    < (run_dir / attempt["directory"] / "orbit.json").stat().st_mtime
                ):
                    save_frames_video(run_dir / attempt["directory"], video)
                attempt["video"] = str(video)
                if attempt is chosen:
                    chosen_video = video
            except (OSError, RuntimeError) as exc:
                attempt["video_error"] = str(exc)

        self._notify(progress, 0.80, "Measuring camera motion")
        plan = plan_views(
            frames,
            cfg.preset.views,
            cfg.sweep_degrees,
            orbit.motion,
            constant_speed=bool(orbit.lora),
            angle_scale=cfg.angle_calibration if orbit.lora else 1.0,
        )
        if chosen["problem"]:
            plan.warnings.append(chosen["problem"])
        draw_parallax_plot(plan, len(frames)).save(run_dir / "parallax.png")
        parallax_plot_path = run_dir / "parallax.png"

        self._notify(progress, 0.86, "Building views and quilt")
        views = render_views(frames, plan)
        tile_size = (
            math.ceil(cfg.preset.quilt_size / cfg.preset.columns),
            math.ceil(cfg.preset.quilt_size / cfg.preset.rows),
        )
        writer = QuiltWriter(cfg.preset)
        framed: list[Image.Image] = []
        for index, view in enumerate(views):
            tile, _ = prepare_frame(view, tile_size, cfg.fill_mode)
            writer.add(index, tile)
            framed.append(tile)

        extension = ".jpg" if cfg.output_format.lower() in {"jpg", "jpeg"} else ".png"
        quilt_path = run_dir / quilt_filename(stem, cfg.preset, extension)
        if extension == ".jpg":
            writer.canvas.save(quilt_path, quality=96, subsampling=0, optimize=False)
        else:
            writer.save(quilt_path)
        sample = [framed[round(i)] for i in _linspace(0, len(framed) - 1, min(12, len(framed)))]
        wiggle_path = save_wiggle(sample, run_dir / f"{stem}_wiggle.gif")
        centre_view_path = run_dir / f"{stem}_centre.png"
        framed[len(framed) // 2].save(centre_view_path)

        recording_path = None
        calibration = self._calibration() if cfg.portrait_recording else None
        if calibration is not None:
            self._notify(progress, 0.92, "Interlacing for the Looking Glass Portrait")
            recording_path, _ = prepare_recording(
                quilt_path,
                cfg.preset.columns,
                cfg.preset.rows,
                cfg.preset.aspect,
                calibration,
                run_dir,
            )

        elapsed = time.perf_counter() - started
        metadata = asdict(cfg)
        metadata["preset"] = asdict(cfg.preset)
        metadata["output_dir"] = str(cfg.output_dir)
        metadata.update(
            {
                "input": str(source_path.resolve()),
                "orbit": orbit_meta,
                "frames": len(frames),
                "orbit_attempts": attempts,
                "background_stillness": stillness,
                "angle_span_degrees": plan.span,
                "direction_score": plan.direction_score,
                "side_scales": plan.side_scales,
                "samples": [asdict(s) for s in plan.samples],
                "views": [asdict(s) for s in plan.sources],
                "warnings": plan.warnings,
                "quilt": str(quilt_path.resolve()),
                "recording": str(recording_path.resolve()) if recording_path else None,
                "calibration": asdict(calibration) if calibration else None,
                "elapsed_seconds": elapsed,
            }
        )
        metadata_path = run_dir / "conversion.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._notify(progress, 1.0, f"Complete in {elapsed:.1f}s")
        return OrbitResult(
            quilt_path=quilt_path,
            wiggle_path=wiggle_path,
            centre_view_path=centre_view_path,
            parallax_plot_path=parallax_plot_path,
            metadata_path=metadata_path,
            recording_path=recording_path,
            warnings=plan.warnings,
            elapsed_seconds=elapsed,
            orbit_video_path=chosen_video,
        )


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [start]
    return [start + (stop - start) * i / (count - 1) for i in range(count)]
