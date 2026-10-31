from __future__ import annotations

import argparse
import socket
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


def playlist_summary() -> str:
    from .portrait import find_portrait_drive, playlist_names

    device = find_portrait_drive()
    if device is None:
        return "No Looking Glass Portrait connected."
    names = playlist_names(device)
    lines = [f"**Playlist on {device}** ({len(names)} items):"]
    lines += [f"{index}. {name}" for index, name in enumerate(names, 1)]
    return "\n".join(lines)


def copy_to_portrait(files: list[str] | None, replace_all: bool, rerender: bool) -> str:
    """Copy uploaded quilts or recordings to the connected Portrait's playlist."""
    from .portrait import (
        Calibration,
        find_holograms,
        find_portrait_drive,
        prepare_holograms,
        push_recordings,
        replace_playlist,
    )

    if not files:
        raise gr.Error("Choose quilts (*_qs8x6a0.75.jpg) or Portrait recordings first.")
    device = find_portrait_drive()
    if device is None:
        raise gr.Error("No Looking Glass Portrait found. Connect it by USB-C and try again.")
    try:
        holograms = find_holograms(files)
        calibration = Calibration.from_file(device / "LKG_calibration" / "visual.json")
        entries = prepare_holograms(holograms, calibration, rerender=rerender, report=print)
        if replace_all:
            replace_playlist(device, entries)
        else:
            push_recordings(device, entries)
    except (OSError, ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc
    done = (
        f"Copied {len(entries)} item(s). Eject the drive and power the Portrait from USB-C "
        "alone to play them."
    )
    return done + "\n\n" + playlist_summary()


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
    looking_glass: bool,
    report,
    output_dir: Path = Path("outputs"),
):
    from .orbit_pipeline import OrbitPipeline
    from .portrait import Calibration, find_portrait_drive, push_recordings

    device = find_portrait_drive() if looking_glass else None
    calibration = (
        Calibration.from_file(device / "LKG_calibration" / "visual.json") if device else None
    )
    cfg = OrbitConfig(
        preset=PRESETS[preset_name],
        sweep_degrees=float(sweep),
        fill_mode=layout,
        output_dir=output_dir,
        portrait_recording=calibration is not None,
        looking_glass=looking_glass,
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
    if result.view_paths:
        notes.append(
            f"Saved {len(result.view_paths)} views in {result.view_paths[0].parent} and a "
            "look-around video."
        )
    if push and looking_glass:
        if result.quilt_path and result.recording_path and device is not None:
            push_recordings(device, [(result.quilt_path.name, result.recording_path)])
            notes.append(
                f"Added to the Portrait at {device}. Eject it and power it from USB-C alone to "
                "play the playlist in standalone mode."
            )
        else:
            notes.append("Not added to a Portrait: none is connected.")
    return (
        str(result.orbit_video_path) if result.orbit_video_path else None,
        str(result.wiggle_path),
        str(result.centre_view_path),
        str(result.parallax_plot_path),
        str(result.sweep_video_path) if result.sweep_video_path else None,
        [str(p) for p in result.view_paths] or None,
        str(result.quilt_path) if result.quilt_path else None,
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
    looking_glass: bool = True,
    progress=DEFAULT_PROGRESS,
    output_dir: Path = Path("outputs"),
):
    if not image_path:
        raise gr.Error("Choose a photograph first.")

    def report(value: float, message: str) -> None:
        progress(value, desc=message)

    try:
        return convert_orbit(
            image_path, preset_name, layout, sweep, use_lora, steps, pdd,
            comfy_url, push, seed, prompt, looking_glass, report, output_dir,
        )  # fmt: skip
    except gr.Error:
        raise
    except Exception as exc:  # surface any pipeline failure in the UI
        raise gr.Error(str(exc)) from exc


def build_ui(output_dir: Path = Path("outputs")) -> gr.Blocks:
    def convert(*inputs, progress=DEFAULT_PROGRESS):
        return convert_from_ui(*inputs, progress=progress, output_dir=output_dir)

    with gr.Blocks(title="HoloForge") as demo:
        gr.Markdown(
            """
            # HoloForge
            **Turn a 2D photograph into a Looking Glass hologram.** HoloForge asks MiniMax H3
            (running in your local ComfyUI) to fly a camera round the frozen scene,
            starting and ending on the photograph, then builds the quilt from the views either
            side of it. With a Looking Glass Portrait connected by USB, the result is added to
            its standalone playlist. No Looking Glass? Untick **Looking Glass output** to get
            the orbit video, the views as images and a look-around video instead.
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
                    looking_glass = gr.Checkbox(
                        value=True,
                        label="Looking Glass output",
                        info="Off: just the orbit video, each view as an image and a "
                        "look-around video",
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
                sweep_video = gr.Video(label="Look-around video (orbit-only)")
                views = gr.Gallery(label="Views, left to right (orbit-only)", columns=8)
                quilt = gr.File(label="Quilt")
                recording = gr.File(label="Portrait recording (__lkgrec.mp4)")
                metadata = gr.File(label="Conversion settings")

        with gr.Accordion("Copy existing holograms to the Portrait", open=False):
            gr.Markdown(
                "Add quilts (named like `photo_quilt_qs8x6a0.75.jpg`) or Portrait recordings "
                "(`*__lkgrec.mp4`) to the connected Portrait's playlist, without Looking Glass "
                "Studio or Bridge. Quilts are interlaced for this Portrait first."
            )
            holograms = gr.File(
                file_count="multiple",
                file_types=[".jpg", ".jpeg", ".png", ".webp", ".mp4"],
                label="Quilts or recordings",
            )
            with gr.Row():
                replace_all = gr.Checkbox(
                    value=False,
                    label="Replace the whole playlist",
                    info="Play only these, in this order, and delete every other recording",
                )
                rerender = gr.Checkbox(
                    value=False,
                    label="Interlace quilts again",
                    info="Even if a recording of them exists",
                )
            with gr.Row():
                send = gr.Button("Copy to the Portrait", variant="primary")
                refresh = gr.Button("Show playlist")
            playlist = gr.Markdown()
        send.click(copy_to_portrait, inputs=[holograms, replace_all, rerender], outputs=playlist)
        refresh.click(playlist_summary, outputs=playlist)

        run.click(
            convert,
            inputs=[
                image, preset, layout, sweep, use_lora, steps, pdd, comfy_url, push,
                seed, prompt, looking_glass,
            ],
            outputs=[
                orbit_video, wiggle, centre, diagnostic, sweep_video, views, quilt, recording,
                metadata, status,
            ],
        ).then(portrait_status, outputs=status_portrait).then(
            comfy_status_line, inputs=comfy_url, outputs=status_comfy
        )  # fmt: skip
        # Refresh both on each page load: the Portrait or ComfyUI may have come and gone.
        demo.load(portrait_status, outputs=status_portrait)
        demo.load(comfy_status_line, inputs=comfy_url, outputs=status_comfy)
    return demo


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="HoloForge web app")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="folder for results; point it at an earlier batch's folder to reuse its H3 clips",
    )
    args = parser.parse_args(argv)
    demo = build_ui(args.output)
    try:
        demo.launch(inbrowser=True)
    except OSError:
        # Gradio tries 7860-7959, which Windows sometimes reserves wholesale (Hyper-V/WSL
        # excluded port ranges); any port the OS hands out works just as well.
        demo.launch(inbrowser=True, server_port=free_port())


if __name__ == "__main__":
    main()
