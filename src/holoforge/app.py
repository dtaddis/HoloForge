from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import gradio as gr

from .config import PRESETS, OrbitConfig
from .orbit import ORBIT_LORA, OrbitSettings

DEFAULT_PROGRESS = gr.Progress()


def clean_optional_prompt(prompt: str | None) -> str | None:
    """Normalise Gradio's untouched textbox value and whitespace-only input."""
    if not prompt:
        return None
    cleaned = prompt.strip()
    return cleaned or None


def comfy_status_line(comfy_url: str = "http://127.0.0.1:8188") -> str:
    from .setup import comfy_status

    return comfy_status(comfy_url.strip() or "http://127.0.0.1:8188")


def portrait_status() -> str:
    from .portrait import find_portrait_drive

    device = find_portrait_drive()
    if device is None:
        return (
            "No Looking Glass Portrait drive found. Connect it by USB-C to add holograms to "
            "its standalone playlist; quilts are still written either way."
        )
    return f"Looking Glass Portrait found at **{device}**."


def convert_orbit(
    image_path: str,
    preset_name: str,
    layout: str,
    sweep: float,
    use_lora: bool,
    steps: int,
    pdd: bool,
    comfy_url: str,
    push: bool,
    seed: int,
    prompt: str | None,
    report,
):
    from .orbit_pipeline import OrbitPipeline
    from .portrait import Calibration, find_portrait_drive, push_recordings

    device = find_portrait_drive()
    calibration = (
        Calibration.from_file(device / "LKG_calibration" / "visual.json") if device else None
    )
    cfg = OrbitConfig(
        preset=PRESETS[preset_name],
        sweep_degrees=float(sweep),
        fill_mode=layout,
        output_dir=Path("outputs"),
        portrait_recording=calibration is not None,
    )
    settings = OrbitSettings(
        lora=ORBIT_LORA if use_lora else None,
        steps=int(steps),
        pdd=pdd,
        seed=int(seed),
        server=comfy_url.strip() or "http://127.0.0.1:8188",
        # A person at the UI wants their job now, not after other queued work.
        idle_seconds=0,
        prompt=clean_optional_prompt(prompt),
    )
    pipeline = OrbitPipeline(cfg, settings, calibration)
    if not use_lora and settings.prompt is None:
        report(0.01, "Describing the photo")
        caption = pipeline.caption(image_path)
        if caption is not None:
            from .caption import unload

            unload()
            settings = replace(settings, subject=caption.subject, style=caption.style)

    result = pipeline.convert(image_path, report, orbit=settings)
    notes = [f"Finished in {result.elapsed_seconds / 60:.1f} minutes."]
    notes += [f"⚠️ {warning}" for warning in result.warnings]
    if push and result.recording_path and device is not None:
        push_recordings(device, [(result.quilt_path.name, result.recording_path)])
        notes.append(
            f"Added to the Portrait at {device}. Eject it and power it from USB-C alone to "
            "play the playlist in standalone mode."
        )
    elif push:
        notes.append("Not added to a Portrait: none is connected.")
    return (
        str(result.orbit_video_path) if result.orbit_video_path else None,
        str(result.wiggle_path),
        str(result.centre_view_path),
        str(result.parallax_plot_path),
        str(result.quilt_path),
        str(result.recording_path) if result.recording_path else None,
        str(result.metadata_path),
        "\n\n".join(notes),
    )


def convert_from_ui(
    image_path: str,
    preset_name: str,
    layout: str,
    sweep: float,
    use_lora: bool,
    steps: int,
    pdd: bool,
    comfy_url: str,
    push: bool,
    seed: int,
    prompt: str | None,
    progress=DEFAULT_PROGRESS,
):
    if not image_path:
        raise gr.Error("Choose a photograph first.")

    def report(value: float, message: str) -> None:
        progress(value, desc=message)

    try:
        return convert_orbit(
            image_path, preset_name, layout, sweep, use_lora, steps, pdd,
            comfy_url, push, seed, prompt, report,
        )  # fmt: skip
    except gr.Error:
        raise
    except Exception as exc:  # surface any pipeline failure in the UI
        raise gr.Error(str(exc)) from exc


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="HoloForge") as demo:
        gr.Markdown(
            """
            # HoloForge
            **Turn a 2D photograph into a Looking Glass hologram.** HoloForge asks MiniMax H3
            (running in your local ComfyUI) to fly a camera round the frozen scene,
            starting and ending on the photograph, then builds the quilt from the views either
            side of it. With a Looking Glass Portrait connected by USB, the result is added to
            its standalone playlist.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="filepath", label="Source photograph")
                preset = gr.Dropdown(
                    choices=list(PRESETS), value="portrait", label="Display preset"
                )
                layout = gr.Radio(
                    choices=["fit", "ambient", "fill"],
                    value="ambient",
                    label="Framing",
                    info="ambient: whole photo over a blurred fill; fill: crop to the display",
                )
                with gr.Group():
                    sweep = gr.Slider(
                        10,
                        50,
                        value=40,
                        step=1,
                        label="3D strength (degrees of orbit)",
                        info="40 matches the Portrait's view cone (true depth)",
                    )
                    use_lora = gr.Checkbox(
                        value=True,
                        label="360° orbit LoRA",
                        info="Keeps the scene frozen so backgrounds get real parallax",
                    )
                    push = gr.Checkbox(value=True, label="Add to the connected Portrait")
                    status_portrait = gr.Markdown(portrait_status())
                    status_comfy = gr.Markdown(comfy_status_line())
                    with gr.Accordion("H3 settings", open=False):
                        steps = gr.Slider(8, 40, value=28, step=1, label="Sampling steps")
                        pdd = gr.Checkbox(
                            value=False, label="PDD 8-step distillation (2.5× faster)"
                        )
                        comfy_url = gr.Textbox(
                            value="http://127.0.0.1:8188", label="ComfyUI server"
                        )
                seed = gr.Number(value=12345, precision=0, label="Seed")
                prompt = gr.Textbox(
                    label="Optional prompt override",
                    placeholder="Leave blank for the built-in prompt",
                    lines=3,
                )
                run = gr.Button("Forge hologram", variant="primary")
            with gr.Column(scale=1):
                status = gr.Markdown("Ready.")
                orbit_video = gr.Video(label="Generated orbit")
                wiggle = gr.Image(label="Parallax preview", type="filepath")
                centre = gr.Image(label="Centre view", type="filepath")
                diagnostic = gr.Image(label="Viewing angles", type="filepath")
                quilt = gr.File(label="Quilt")
                recording = gr.File(label="Portrait recording (__lkgrec.mp4)")
                metadata = gr.File(label="Conversion settings")

        run.click(
            convert_from_ui,
            inputs=[
                image, preset, layout, sweep, use_lora, steps, pdd, comfy_url, push,
                seed, prompt,
            ],
            outputs=[orbit_video, wiggle, centre, diagnostic, quilt, recording, metadata, status],
        ).then(portrait_status, outputs=status_portrait).then(
            comfy_status_line, inputs=comfy_url, outputs=status_comfy
        )  # fmt: skip
        # Refresh both on each page load: the Portrait or ComfyUI may have come and gone.
        demo.load(portrait_status, outputs=status_portrait)
        demo.load(comfy_status_line, inputs=comfy_url, outputs=status_comfy)
    return demo


def main() -> None:
    build_ui().launch(inbrowser=True)


if __name__ == "__main__":
    main()
