from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import gradio as gr

from .config import PRESETS, ConversionConfig
from .pipeline import HoloForgePipeline

DEFAULT_PROGRESS = gr.Progress()


def convert_from_ui(
    image_path: str,
    preset_name: str,
    inpaint_mode: str,
    layout: str,
    view_cone: float,
    focus: float,
    face_relief: float,
    seed: int,
    prompt: str,
    progress=DEFAULT_PROGRESS,
):
    if not image_path:
        raise gr.Error("Choose a photograph first.")
    cfg = ConversionConfig(
        preset=PRESETS[preset_name],
        inpaint_mode=inpaint_mode,
        fill_mode=layout,
        view_cone=view_cone,
        focus=focus,
        face_relief=face_relief,
        seed=int(seed),
        output_dir=Path("outputs"),
    )
    if prompt.strip():
        cfg = replace(cfg, prompt=prompt.strip())

    def report(value: float, message: str) -> None:
        progress(value, desc=message)

    result = HoloForgePipeline(cfg).convert(image_path, report)
    return (
        str(result.wiggle_path),
        str(result.centre_view_path),
        str(result.depth_path),
        str(result.mask_path) if result.mask_path else None,
        str(result.quilt_path),
        str(result.metadata_path),
        f"Finished in {result.elapsed_seconds:.1f} seconds (model download time excluded after first run).",
    )


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="HoloForge") as demo:
        gr.Markdown(
            """
            # HoloForge
            **Private, local 2D-photo to light-field conversion.** The centre view keeps the
            source photograph intact; AI only reconstructs surfaces exposed by parallax.

            The musubi preset uses its published 9:16 content aspect and a provisional 10×6
            quilt grid. The final grid remains editable in the CLI until Looking Glass publishes
            the musubi developer quilt specification.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="filepath", label="Source photograph")
                preset = gr.Dropdown(choices=list(PRESETS), value="musubi", label="Display preset")
                inpaint = gr.Radio(
                    choices=["generative", "fast"],
                    value="generative",
                    label="Hidden-detail reconstruction",
                    info="Generative is intended for a 24 GB RTX GPU; fast is a diagnostic fallback.",
                )
                layout = gr.Radio(
                    choices=["fit", "ambient", "fill"], value="ambient", label="Framing"
                )
                with gr.Accordion("Depth and camera controls", open=False):
                    view_cone = gr.Slider(0.02, 0.13, value=0.075, step=0.002, label="Parallax")
                    focus = gr.Slider(0.0, 1.0, value=0.55, step=0.01, label="Focus plane")
                    face_relief = gr.Slider(0.0, 0.15, value=0.07, step=0.005, label="Face relief")
                    seed = gr.Number(value=12345, precision=0, label="Generation seed")
                    prompt = gr.Textbox(
                        label="Optional reconstruction prompt",
                        placeholder="Leave blank for the photograph-preserving default",
                        lines=3,
                    )
                run = gr.Button("Forge hologram", variant="primary")
            with gr.Column(scale=1):
                status = gr.Markdown("Ready.")
                wiggle = gr.Image(label="Parallax preview", type="filepath")
                centre = gr.Image(label="Centre view", type="filepath")
                with gr.Row():
                    depth = gr.Image(label="Estimated depth", type="filepath")
                    mask = gr.Image(label="Deepest generated mask", type="filepath")
                quilt = gr.File(label="Quilt")
                metadata = gr.File(label="Conversion settings")
        run.click(
            convert_from_ui,
            inputs=[
                image,
                preset,
                inpaint,
                layout,
                view_cone,
                focus,
                face_relief,
                seed,
                prompt,
            ],
            outputs=[wiggle, centre, depth, mask, quilt, metadata, status],
        )
    return demo


def main() -> None:
    build_ui().launch(inbrowser=True)


if __name__ == "__main__":
    main()
