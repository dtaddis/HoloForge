from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from .layers import Backplate


@dataclass
class RenderLayer:
    colour: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor


class LightFieldRenderer:
    """GPU forward-splat renderer with an explicit z-buffer."""

    def __init__(
        self,
        source: Image.Image,
        depth: np.ndarray,
        source_alpha: np.ndarray,
        backplates: list[Backplate],
        *,
        view_cone: float,
        focus: float,
        head_alpha: np.ndarray | None = None,
        head_sidewall: float = 0.12,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.height, self.width = depth.shape
        self.view_cone = view_cone
        self.focus = focus
        self.head_sidewall = max(0.0, head_sidewall)
        self.source = self._make_layer(source, depth, source_alpha)
        self.head = (
            self._make_layer(source, depth, head_alpha)
            if head_alpha is not None and np.any(head_alpha > 0)
            else None
        )
        # Far plate first, near plate last.
        self.backplates = [
            self._make_layer(p.image, p.depth, p.alpha) for p in reversed(backplates)
        ]
        yy, xx = torch.meshgrid(
            torch.arange(self.height, device=self.device),
            torch.arange(self.width, device=self.device),
            indexing="ij",
        )
        self.xx = xx.reshape(-1).float()
        self.yy = yy.reshape(-1).long()

    def _make_layer(
        self,
        image: Image.Image,
        depth: np.ndarray,
        alpha: np.ndarray,
    ) -> RenderLayer:
        colour = torch.from_numpy(np.asarray(image, dtype=np.float32).copy() / 255.0).to(
            self.device
        )
        dep = torch.from_numpy(depth.astype(np.float32)).to(self.device)
        alp = torch.from_numpy(alpha.astype(np.float32)).to(self.device)
        return RenderLayer(colour=colour, depth=dep, alpha=alp)

    @torch.inference_mode()
    def _splat(self, layer: RenderLayer, view_position: float) -> tuple[torch.Tensor, torch.Tensor]:
        depth = layer.depth.reshape(-1)
        alpha = layer.alpha.reshape(-1)
        disparity = view_position * self.view_cone * self.width * (depth - self.focus)
        target_x = self.xx + disparity
        x0 = torch.floor(target_x).long()
        x1 = x0 + 1
        w1 = target_x - x0.float()
        w0 = 1.0 - w1

        candidates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for x, weight in ((x0, w0), (x1, w1)):
            valid = (x >= 0) & (x < self.width) & (alpha > 1e-4) & (weight > 1e-4)
            index = self.yy[valid] * self.width + x[valid]
            candidates.append((valid, index, weight[valid] * alpha[valid]))

        pixels = self.height * self.width
        zbuffer = torch.full((pixels,), -1e4, device=self.device)
        for valid, index, _ in candidates:
            zbuffer.scatter_reduce_(0, index, depth[valid], reduce="amax", include_self=True)

        colour = layer.colour.reshape(-1, 3)
        accum = torch.zeros((pixels, 3), device=self.device)
        coverage = torch.zeros((pixels,), device=self.device)
        for valid, index, weight in candidates:
            visible = depth[valid] >= (zbuffer[index] - 1e-3)
            idx = index[visible]
            weights = weight[visible]
            accum.index_add_(0, idx, colour[valid][visible] * weights[:, None])
            coverage.index_add_(0, idx, weights)
        output = accum / coverage.clamp_min(1e-6)[:, None]
        # Bilinear weights reconstruct colour; they are not material opacity.
        # Fractional coverage here lets generated backplates sparkle through
        # otherwise opaque faces and silhouettes.
        occupancy = (coverage > 1e-4).float()
        return output.reshape(self.height, self.width, 3), occupancy.reshape(
            self.height, self.width
        )

    @staticmethod
    def _close_single_pixel_cracks(
        colour: torch.Tensor, alpha: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill isolated raster cracks while retaining real disocclusions."""
        left_alpha = torch.roll(alpha, 1, dims=1)
        right_alpha = torch.roll(alpha, -1, dims=1)
        cracks = (alpha < 0.5) & (left_alpha > 0.5) & (right_alpha > 0.5)
        cracks[:, 0] = False
        cracks[:, -1] = False
        neighbour_colour = (
            torch.roll(colour, 1, dims=1) + torch.roll(colour, -1, dims=1)
        ) * 0.5
        colour = torch.where(cracks[..., None], neighbour_colour, colour)
        alpha = torch.where(cracks, torch.ones_like(alpha), alpha)
        return colour, alpha

    def _add_head_sidewall(
        self,
        colour: torch.Tensor,
        alpha: torch.Tensor,
        view_position: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extrude head-edge texture into narrow view-dependent disocclusions."""
        if self.head is None or abs(view_position) < 1e-6 or self.head_sidewall <= 0:
            return colour, alpha
        head_colour, head_alpha = self._splat(self.head, view_position)
        width = round(
            abs(view_position) * self.view_cone * self.width * self.head_sidewall
        )
        if width <= 0:
            return colour, alpha

        shift = -1 if view_position > 0 else 1
        carrier_colour = head_colour
        carrier_alpha = head_alpha
        output_colour = colour.clone()
        output_alpha = alpha.clone()
        for _ in range(width):
            candidate_colour = torch.roll(carrier_colour, shifts=shift, dims=1)
            candidate_alpha = torch.roll(carrier_alpha, shifts=shift, dims=1)
            if shift < 0:
                candidate_alpha[:, -1] = 0
            else:
                candidate_alpha[:, 0] = 0
            # A disocclusion is often already occupied by the farther surface,
            # not represented as an empty pixel. The newly exposed head side is
            # nearer and must therefore occlude that background strip.
            fill = (carrier_alpha < 0.5) & (candidate_alpha > 0.5)
            output_colour = torch.where(fill[..., None], candidate_colour, output_colour)
            output_alpha = torch.where(fill, torch.ones_like(output_alpha), output_alpha)
            carrier_colour = torch.where(fill[..., None], candidate_colour, carrier_colour)
            carrier_alpha = torch.where(fill, torch.ones_like(carrier_alpha), carrier_alpha)
        return output_colour, output_alpha

    @staticmethod
    def _over(
        foreground: torch.Tensor,
        foreground_alpha: torch.Tensor,
        background: torch.Tensor,
        background_alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fa = foreground_alpha.clamp(0, 1)
        ba = background_alpha.clamp(0, 1)
        out_alpha = fa + ba * (1 - fa)
        premul = foreground * fa[..., None] + background * ba[..., None] * (1 - fa[..., None])
        return premul / out_alpha.clamp_min(1e-6)[..., None], out_alpha

    @torch.inference_mode()
    def render(self, view_position: float) -> Image.Image:
        canvas = torch.zeros((self.height, self.width, 3), device=self.device)
        canvas_alpha = torch.zeros((self.height, self.width), device=self.device)
        for layer in self.backplates:
            colour, alpha = self._splat(layer, view_position)
            canvas, canvas_alpha = self._over(colour, alpha, canvas, canvas_alpha)
        colour, alpha = self._splat(self.source, view_position)
        colour, alpha = self._close_single_pixel_cracks(colour, alpha)
        colour, alpha = self._add_head_sidewall(colour, alpha, view_position)
        canvas, canvas_alpha = self._over(colour, alpha, canvas, canvas_alpha)

        array = (canvas.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        holes = (canvas_alpha.cpu().numpy() < 0.05).astype(np.uint8) * 255
        if holes.any():
            # Last-resort fill for subpixel cracks at the outer image edge. The
            # substantive disocclusions have already come from generated plates.
            bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            bgr = cv2.inpaint(bgr, holes, 3, cv2.INPAINT_TELEA)
            array = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(array)
