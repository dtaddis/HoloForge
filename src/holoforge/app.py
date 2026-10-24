from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import gradio as gr

from .config import PRESETS, ConversionConfig, OrbitConfig
from .orbit import ORBIT_LORA, OrbitSettings

DEFAULT_PROGRESS = gr.Progress()
ORBIT = "H3 orbit (MiniMax H3 via ComfyUI)"
LAYERED = "Layered (depth + inpainting)"


def clean_optional_prompt(prompt: str | None) -> str | None:
    """Normalise Gradio's untouched textbox value and whitespace-only input."""
    if not prompt:
        return None
    cleaned = prompt.strip()
    return cleaned or None


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


def convert_layered(
    image_path: str,
    preset_name: str,
    inpaint_mode: str,
    layout: str,
    view_cone: float,
    focus: float,
    face_relief: float,
    seed: int,
    prompt: str | None,
    report,
):
    from .pipeline import HoloForgePipeline

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
    cleaned_prompt = clean_optional_prompt(prompt)
    if cleaned_prompt:
        cfg = replace(cfg, prompt=cleaned_prompt)
    result = HoloForgePipeline(cfg).convert(image_path, report)
    return (
        None,
        str(result.wiggle_path),
        str(result.centre_view_path),
        str(result.depth_path),
        str(result.quilt_path),
        None,
        str(result.metadata_path),
        f"Finished in {result.elapsed_seconds:.1f} seconds.",
    )


def convert_from_ui(
    image_path: str,
    method: str,
    preset_name: str,
    layout: str,
    sweep: float,
    use_lora: bool,
    steps: int,
    pdd: bool,
    comfy_url: str,
    push: bool,
    inpaint_mode: str,
    view_cone: float,
    focus: float,
    face_relief: float,
    seed: int,
    prompt: str | None,
    progress=DEFAULT_PROGRESS,
):
    if not image_path:
        raise gr.Error("Choose a photograph first.")

    def report(value: float, message: str) -> None:
        progress(value, desc=message)

    try:
        if method == ORBIT:
            return convert_orbit(
                image_path, preset_name, layout, sweep, use_lora, steps, pdd,
                comfy_url, push, seed, prompt, report,
            )  # fmt: skip
        return convert_layered(
            image_path, preset_name, inpaint_mode, layout, view_cone, focus,
            face_relief, seed, prompt, report,
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
            **Turn a 2D photograph into a Looking Glass hologram.** The H3 orbit method asks
            MiniMax H3 (running in your local ComfyUI) to fly a camera round the frozen scene,
            starting and ending on the photograph, then builds the quilt from the views either
            side of it. With a Looking Glass Portrait connected by USB, the result is added to
            its standalone playlist.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="filepath", label="Source photograph")
                method = gr.Radio(choices=[ORBIT, LAYERED], value=ORBIT, label="Method")
                preset = gr.Dropdown(
                    choices=list(PRESETS), value="portrait", label="Display preset"
                )
                layout = gr.Radio(
                    choices=["fit", "ambient", "fill"],
                    value="ambient",
                    label="Framing",
                    info="ambient: whole photo over a blurred fill; fill: crop to the display",
                )
                with gr.Group(visible=True) as orbit_group:
                    sweep = gr.Slider(
                        10, 40, value=30, step=1, label="3D strength (degrees of orbit)"
                    )
                    use_lora = gr.Checkbox(
                        value=True,
                        label="360° orbit LoRA",
                        info="Keeps the scene frozen so backgrounds get real parallax",
                    )
                    push = gr.Checkbox(value=True, label="Add to the connected Portrait")
                    status_portrait = gr.Markdown(portrait_status())
                    with gr.Accordion("H3 settings", open=False):
                        steps = gr.Slider(8, 40, value=28, step=1, label="Sampling steps")
                        pdd = gr.Checkbox(
                            value=False, label="PDD 8-step distillation (2.5× faster)"
                        )
                        comfy_url = gr.Textbox(
                            value="http://127.0.0.1:8188", label="ComfyUI server"
                        )
                with gr.Accordion("Layered settings", open=False, visible=False) as layered:
                    inpaint = gr.Radio(
                        choices=["generative", "fast"],
                        value="generative",
                        label="Hidden-detail reconstruction",
                    )
                    view_cone = gr.Slider(0.02, 0.13, value=0.075, step=0.002, label="Parallax")
                    focus = gr.Slider(0.0, 1.0, value=0.55, step=0.01, label="Focus plane")
                    face_relief = gr.Slider(0.0, 0.18, value=0.10, step=0.005, label="Face relief")
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
                diagnostic = gr.Image(label="Viewing angles / depth", type="filepath")
                quilt = gr.File(label="Quilt")
                recording = gr.File(label="Portrait recording (__lkgrec.mp4)")
                metadata = gr.File(label="Conversion settings")

        def switch(selected: str):
            is_orbit = selected == ORBIT
            return gr.update(visible=is_orbit), gr.update(visible=not is_orbit)

        method.change(switch, inputs=method, outputs=[orbit_group, layered])
        run.click(
            convert_from_ui,
            inputs=[
                image, method, preset, layout, sweep, use_lora, steps, pdd, comfy_url,
                push, inpaint, view_cone, focus, face_relief, seed, prompt,
            ],
            outputs=[orbit_video, wiggle, centre, diagnostic, quilt, recording, metadata, status],
        ).then(portrait_status, outputs=status_portrait)  # fmt: skip
    return demo


def main() -> None:
    build_ui().launch(inbrowser=True)


if __name__ == "__main__":
    main()
