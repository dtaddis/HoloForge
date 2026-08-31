from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .config import ConversionConfig
from .depth import DepthEstimator, apply_face_relief, draw_face_diagnostics
from .image_utils import depth_preview, load_rgb, prepare_frame, safe_stem
from .inpaint import create_inpainter
from .layers import Backplate, make_backplate_depth, make_backplate_requests
from .quilt import QuiltWriter, quilt_filename, save_wiggle
from .render import LightFieldRenderer

ProgressCallback = Callable[[float, str], None]


@dataclass
class ConversionResult:
    quilt_path: Path
    wiggle_path: Path
    depth_path: Path
    mask_path: Path | None
    centre_view_path: Path
    metadata_path: Path
    elapsed_seconds: float


class HoloForgePipeline:
    def __init__(self, config: ConversionConfig) -> None:
        self.config = config
        self.depth_estimator = DepthEstimator(config.depth_model)

    @staticmethod
    def _notify(callback: ProgressCallback | None, value: float, message: str) -> None:
        if callback:
            callback(value, message)

    def convert(
        self,
        input_path: str | Path,
        progress: ProgressCallback | None = None,
    ) -> ConversionResult:
        started = time.perf_counter()
        cfg = self.config
        source_path = Path(input_path)
        stem = safe_stem(source_path)
        run_dir = cfg.output_dir / stem
        run_dir.mkdir(parents=True, exist_ok=True)

        # Use the largest possible cell because a square quilt whose dimensions
        # are not divisible by its grid has one-pixel cell-size differences.
        tile_size = (
            math.ceil(cfg.preset.quilt_size / cfg.preset.columns),
            math.ceil(cfg.preset.quilt_size / cfg.preset.rows),
        )
        self._notify(progress, 0.02, "Preparing photograph")
        original = load_rgb(source_path)
        frame, content_alpha = prepare_frame(original, tile_size, cfg.fill_mode)
        frame.save(run_dir / "prepared.png")

        self._notify(progress, 0.06, "Estimating detailed depth")
        depth_result = self.depth_estimator.estimate(
            frame,
            edge_refine=cfg.edge_refine,
            depth_gamma=cfg.depth_gamma,
        )
        depth, faces = apply_face_relief(frame, depth_result.near_depth, cfg.face_relief)
        # Letterbox/ambient pixels should remain at the convergence plane or back.
        if cfg.fill_mode != "fill":
            depth[content_alpha < 0.5] = 0.0
        depth_path = run_dir / "depth.png"
        depth_preview(depth).save(depth_path)
        draw_face_diagnostics(frame, faces).save(run_dir / "faces.png")

        self._notify(progress, 0.14, "Finding occluded regions")
        requests = make_backplate_requests(
            depth,
            count=cfg.backplate_count,
            foreground_fraction=cfg.foreground_fraction,
            dilation=cfg.mask_dilation,
            content_alpha=content_alpha,
        )
        inpainter = create_inpainter(
            cfg.inpaint_mode,
            cfg.inpaint_model,
            steps=cfg.inpaint_steps,
            strength=cfg.inpaint_strength,
            guidance_scale=cfg.guidance_scale,
        )
        backplates: list[Backplate] = []
        mask_path: Path | None = None
        for index, request in enumerate(requests):
            fraction = index / max(len(requests), 1)
            self._notify(
                progress,
                0.16 + 0.46 * fraction,
                f"Generating hidden layer {index + 1}/{len(requests)}",
            )
            mask_path = run_dir / f"mask_{index + 1}.png"
            Image.fromarray(request.mask).save(mask_path)
            plate_image = inpainter.inpaint(
                frame,
                request.mask,
                prompt=cfg.prompt,
                negative_prompt=cfg.negative_prompt,
                seed=cfg.seed + index,
            )
            plate_image.save(run_dir / f"backplate_{index + 1}.png")
            plate_depth = make_backplate_depth(depth, request.mask, request.threshold)
            np.save(run_dir / f"backplate_depth_{index + 1}.npy", plate_depth)
            backplates.append(
                Backplate(
                    image=plate_image,
                    depth=plate_depth,
                    alpha=np.ones_like(depth, dtype=np.float32),
                    mask=request.mask,
                    threshold=request.threshold,
                )
            )

        self._notify(progress, 0.64, "Building the layered scene")
        renderer = LightFieldRenderer(
            frame,
            depth,
            np.ones_like(depth, dtype=np.float32),
            backplates,
            view_cone=cfg.view_cone,
            focus=cfg.focus,
        )
        writer = QuiltWriter(cfg.preset)
        positions = np.linspace(-1.0, 1.0, cfg.preset.views)
        sample_indices = {
            int(i) for i in np.linspace(0, cfg.preset.views - 1, min(9, cfg.preset.views))
        }
        wiggle_views: list[Image.Image] = []
        endpoint_views: list[Image.Image] = []
        for view_index, position in enumerate(positions):
            self._notify(
                progress,
                0.66 + 0.29 * view_index / max(cfg.preset.views - 1, 1),
                f"Rendering view {view_index + 1}/{cfg.preset.views}",
            )
            view = renderer.render(float(position))
            writer.add(view_index, view)
            if view_index in sample_indices:
                wiggle_views.append(view.copy())
            if view_index in {0, cfg.preset.views - 1}:
                endpoint_views.append(view.copy())

        extension = ".jpg" if cfg.output_format.lower() in {"jpg", "jpeg"} else ".png"
        quilt_path = run_dir / quilt_filename(stem, cfg.preset, extension)
        self._notify(progress, 0.96, "Writing quilt and preview")
        if extension == ".jpg":
            writer.canvas.save(quilt_path, quality=96, subsampling=0, optimize=False)
        else:
            writer.save(quilt_path)
        wiggle_path = save_wiggle(wiggle_views, run_dir / f"{stem}_wiggle.gif")
        centre_view = renderer.render(0.0)
        centre_view_path = run_dir / f"{stem}_centre.png"
        centre_view.save(centre_view_path)
        if len(endpoint_views) == 2:
            endpoint_views[0].save(run_dir / f"{stem}_left.png")
            endpoint_views[1].save(run_dir / f"{stem}_right.png")
            endpoints = Image.new(
                "RGB", (endpoint_views[0].width * 2, endpoint_views[0].height), "black"
            )
            endpoints.paste(endpoint_views[0], (0, 0))
            endpoints.paste(endpoint_views[1], (endpoint_views[0].width, 0))
            endpoints.save(run_dir / f"{stem}_endpoints.png")

        elapsed = time.perf_counter() - started
        metadata_path = run_dir / "conversion.json"
        metadata = asdict(cfg)
        metadata["preset"] = asdict(cfg.preset)
        metadata["output_dir"] = str(cfg.output_dir)
        metadata.update(
            {
                "input": str(source_path.resolve()),
                "elapsed_seconds": elapsed,
                "faces_detected": len(faces),
                "tile_size": tile_size,
                "quilt": str(quilt_path.resolve()),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._notify(progress, 1.0, f"Complete in {elapsed:.1f}s")
        return ConversionResult(
            quilt_path=quilt_path,
            wiggle_path=wiggle_path,
            depth_path=depth_path,
            mask_path=mask_path,
            centre_view_path=centre_view_path,
            metadata_path=metadata_path,
            elapsed_seconds=elapsed,
        )
