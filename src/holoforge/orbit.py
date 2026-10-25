"""MiniMax H3 orbit generation through a local ComfyUI server.

The source photograph is given to H3's first-and-last-frame (FL2VA) mode as both the
opening and the closing frame, with a prompt asking for a camera orbit around subjects
frozen like statues. Because the clip starts and ends on the original viewpoint, frames
just after the start and just before the end look at the scene from either side of it,
which is exactly what a light-field quilt needs.

The open H3 weights are licensed under the MiniMax H3 Community License, which excludes
the EU, UK, South Korea and the USA without separate authorisation from MiniMax.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from .comfy import ComfyClient, save_frames

H3_FPS = 24
H3_DIFFUSION_GGUF = "MiniMax-H3-FL2VA-Pruned-Q5_K_M.gguf"
H3_TEXT_ENCODER_GGUF = "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf"
H3_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
H3_PDD_ACC = "MiniMax-H3-FL2VA-Acc-8Step.safetensors"
# H3's default output has a 768 px short side; 768x1024 is that area at the Portrait's 3:4.
H3_TARGET_PIXELS = 768 * 1024
# Pixel-frames that stay resident on a 24 GB card with the Q5 transformer: measured with
# the ai-remaster-pipeline H3 graph (124 frames at 1536x640). Longer passes start offloading
# weights to system RAM and slow down by an order of magnitude.
H3_MAX_PIXEL_FRAMES = 124 * 1536 * 640
PDD_STEPS = 8
SAVE_NODE = "30"
MODEL_FIELDS = ("unet_name", "clip_name", "vae_name")

# Pablo Dawson's 360° orbit LoRA for H3 FL2VA (huggingface.co/pablodawson/
# MiniMax-H3-360-Orbit-LoRA). Without it, giving FL2VA the same first and last frame often
# yields a turntable (subjects rotate in front of a frozen background) or barely any motion;
# it was trained on orbit renders of static Gaussian splats, so only the camera moves.
ORBIT_LORA = "minimax_h3_flf2v_lora_v1.safetensors"
# The LoRA's training caption; its card says to use it verbatim.
ORBIT_LORA_PROMPT = (
    "One frozen instant. Only the camera moves. In a continuous 360 orbit. Preserve every "
    "person and object in exactly the same world position, orientation, shape and pose "
    "throughout the shot. Airborne objects remain suspended at the captured height and angle: "
    "no wobbling, shaking, spinning, drifting, falling or continued action. Keep faces, hands, "
    "clothing, liquids and the background motionless while retaining their natural "
    "appearance. Camera parallax is the only source of apparent movement. No cuts, zoom, "
    "morphing or added objects."
)

MOTION_PROMPTS = {
    # The user's recipe: one full circle that ends where it began.
    "orbit": (
        "The camera performs one smooth, continuous Arc Shot at a steady, slow speed, "
        "orbiting a full circle around {subject} at the same height and distance, revealing "
        "them from every side, and arrives back at exactly the original viewpoint at the end."
    ),
    # Retry wording for clips where H3 turned the subjects on a turntable in front of a
    # frozen background instead of moving the camera.
    "orbit_camera": (
        "The camera itself physically travels one full circle around {subject}, like a drone "
        "orbiting a statue, at a steady, slow speed and the same height and distance. The "
        "subjects stay fixed in place and never turn; only the camera moves, so the background "
        "and surroundings visibly sweep past behind them with strong parallax and the scene "
        "behind and beside them is revealed. The camera arrives back at exactly the original "
        "viewpoint at the end."
    ),
    # Gentler alternative: swing to one side, back through the start, to the other side,
    # and home. Keeps every frame close to the photographed viewpoint.
    "sweep": (
        "The camera performs a smooth Arc Shot with small amplitude at slow speed: it arcs "
        "around {subject} to the right, swings back through the original viewpoint to the "
        "left at the same height and distance, and returns to exactly the original viewpoint "
        "at the end."
    ),
}

PROMPT_TEMPLATE = (
    "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns "
    "with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the "
    "{end:.2f}-second mark of the target video.\n\n"
    "integrated_multimodal_description: [Shot 1] {style}, a single continuous shot of a "
    "moment frozen in time, showing {subject}. Every person and object keeps exactly the "
    "same pose and expression as in the photograph and stays perfectly still like a statue: "
    "no blinking, no breathing, no movement of faces, hands, hair or clothing, and nothing "
    "else in the scene moves. "
    "{motion} The lighting, colours, focus, film grain and photographic style stay identical "
    "to the photograph throughout. No cuts, no zoom, no camera shake.\n\n"
    "overall_soundscape: Complete silence.\n\n"
    "non_diegetic_music: None."
)


@dataclass(frozen=True)
class OrbitSettings:
    motion: str = "orbit"
    subject: str = "the people in the photograph"
    style: str = "Live-action photograph"
    prompt: str | None = None
    frames: int | None = None
    target_pixels: int = H3_TARGET_PIXELS
    # ComfyUI model files: GGUF names use the ComfyUI-GGUF loaders, anything else the core
    # UNETLoader/CLIPLoader (e.g. Comfy-Org's int8 ConvRot and nvfp4 repacks).
    unet_name: str = H3_DIFFUSION_GGUF
    clip_name: str = H3_TEXT_ENCODER_GGUF
    vae_name: str = H3_VIDEO_VAE
    # The orbit LoRA, or None for the base model with the descriptive prompt template.
    lora: str | None = ORBIT_LORA
    lora_strength: float = 1.0
    steps: int = 28
    sampler: str = "res_multistep"
    scheduler: str = "simple"
    pdd: bool = False
    seed: int = 12345
    server: str = "http://127.0.0.1:8188"
    # Share the ComfyUI server politely: only submit after its queue has been empty this
    # long, so another pipeline that submits job after job keeps the GPU.
    idle_seconds: float = 180.0


def h3_valid_length(frame_count: int) -> int:
    """Snap down to H3's temporal grid of 17k + 5 frames."""
    count = max(5, int(frame_count))
    return count - (count - 5) % 17


def h3_size(aspect: float, target_pixels: int = H3_TARGET_PIXELS) -> tuple[int, int]:
    """Width and height near target_pixels with the image's aspect, on a 32 px grid."""
    aspect = min(max(aspect, 0.4), 2.5)
    height = math.sqrt(target_pixels / aspect)
    width = height * aspect
    return max(256, round(width / 32) * 32), max(256, round(height / 32) * 32)


def max_frames_for(width: int, height: int) -> int:
    return h3_valid_length(min(362, H3_MAX_PIXEL_FRAMES // (width * height)))


def build_prompt(settings: OrbitSettings, length: int) -> str:
    if settings.prompt:
        return settings.prompt
    if settings.lora:
        return ORBIT_LORA_PROMPT
    subject = settings.subject.strip() or "the people in the photograph"
    motion = MOTION_PROMPTS[settings.motion].format(subject=subject)
    return PROMPT_TEMPLATE.format(
        end=(length - 1) / H3_FPS, style=settings.style, subject=subject, motion=motion
    )


def _unet_loader(name: str) -> dict:
    if name.lower().endswith(".gguf"):
        return {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": name}}
    return {"class_type": "UNETLoader", "inputs": {"unet_name": name, "weight_dtype": "default"}}


def _clip_class(name: str) -> str:
    return "CLIPLoaderGGUF" if name.lower().endswith(".gguf") else "CLIPLoader"


def build_graph(
    image_name: str,
    width: int,
    height: int,
    length: int,
    prompt: str,
    settings: OrbitSettings,
    prefix: str,
) -> dict:
    graph: dict[str, dict] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "10": _unet_loader(settings.unet_name),
        "13": {
            "class_type": _clip_class(settings.clip_name),
            "inputs": {"clip_name": settings.clip_name, "type": "minimax"},
        },
        "14": {"class_type": "VAELoader", "inputs": {"vae_name": settings.vae_name}},
        "16": {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": {
                "clip": ["13", 0],
                "vae": ["14", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": length,
                "first_frame": ["1", 0],
                "last_frame": ["1", 0],
            },
        },
        "18": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler" if settings.pdd else settings.sampler},
        },
        "20": {"class_type": "RandomNoise", "inputs": {"noise_seed": settings.seed}},
        "22": {"class_type": "VAEDecode", "inputs": {"samples": ["21", 0], "vae": ["14", 0]}},
        SAVE_NODE: {
            "class_type": "SaveImage",
            "inputs": {"images": ["22", 0], "filename_prefix": prefix},
        },
    }
    model = ["10", 0]
    if settings.lora:
        graph["12"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model,
                "lora_name": settings.lora,
                "strength_model": settings.lora_strength,
            },
        }
        model = ["12", 0]
    if settings.pdd:
        graph["11"] = {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {"model": model, "shift_video": 12.0, "shift_audio": 3.0},
        }
        graph["24"] = {
            "class_type": "MiniMaxH3PDDAccApply",
            "inputs": {
                "model": ["11", 0],
                "pdd_file": H3_PDD_ACC,
                "nfe": str(PDD_STEPS),
                "lora_strength": 1.0,
                "head_strength": 1.0,
                "on_off_grid": "error",
            },
        }
        model, sigmas = ["24", 0], ["24", 1]
    else:
        graph["19"] = {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": model,
                "scheduler": settings.scheduler,
                "steps": settings.steps,
                "denoise": 1.0,
            },
        }
        sigmas = ["19", 0]
    graph["17"] = {
        "class_type": "BasicGuider",
        "inputs": {"model": model, "conditioning": ["16", 0]},
    }
    graph["21"] = {
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["20", 0],
            "guider": ["17", 0],
            "sampler": ["18", 0],
            "sigmas": sigmas,
            "latent_image": ["16", 1],
        },
    }
    return graph


def load_cached_frames(directory: Path, signature: dict) -> list[Image.Image] | None:
    meta = directory / "orbit.json"
    if not meta.exists():
        return None
    try:
        cached = json.loads(meta.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if cached.get("signature") != signature:
        return None
    paths = sorted(directory.glob("frame_*.png"))
    if len(paths) != cached.get("frame_count"):
        return None
    return [Image.open(p).convert("RGB") for p in paths]


def generate_orbit(
    source: Image.Image,
    directory: Path,
    settings: OrbitSettings,
    *,
    name: str = "holoforge",
    on_wait: Callable[[float], None] | None = None,
    on_busy: Callable[[float], None] | None = None,
) -> tuple[list[Image.Image], dict]:
    """Render (or reuse) the H3 orbit clip for one photograph. Returns frames and metadata.

    on_busy is called while waiting for other users of the ComfyUI server to finish.
    """
    width, height = h3_size(source.width / source.height, settings.target_pixels)
    length = h3_valid_length(settings.frames) if settings.frames else max_frames_for(width, height)
    prompt = build_prompt(settings, length)
    conditioning = source.resize((width, height), Image.Resampling.LANCZOS)
    # Model names enter as "model"/"text_encoder" (the keys caches were written with).
    fields = {k: v for k, v in asdict(settings).items() if k not in MODEL_FIELDS}
    signature = {
        **fields,
        "server": None,
        "idle_seconds": None,
        "prompt": prompt,
        "width": width,
        "height": height,
        "length": length,
        "model": settings.unet_name,
        "text_encoder": settings.clip_name,
        "source_size": list(source.size),
    }
    cached = load_cached_frames(directory, signature)
    if cached is not None:
        return cached, signature

    from .setup import start_comfyui_if_needed

    # Start the ComfyUI that holoforge-setup recorded, if nothing is listening yet.
    start_comfyui_if_needed(settings.server)
    client = ComfyClient(settings.server)
    unet_class = _unet_loader(settings.unet_name)["class_type"]
    clip_class = _clip_class(settings.clip_name)
    client.require_nodes(["MiniMaxH3ImageToVideo", unet_class, clip_class])
    client.require_choice(unet_class, "unet_name", settings.unet_name)
    client.require_choice(clip_class, "clip_name", settings.clip_name)
    client.require_choice("VAELoader", "vae_name", settings.vae_name)
    if settings.lora:
        client.require_choice("LoraLoaderModelOnly", "lora_name", settings.lora)
    client.wait_for_idle(settings.idle_seconds, on_busy=on_busy)
    image_name = client.upload_image(conditioning, f"holoforge_{name}.png")
    graph = build_graph(image_name, width, height, length, prompt, settings, f"holoforge/{name}")
    prompt_id = client.queue(graph)
    outputs = client.wait(prompt_id, on_wait=on_wait)
    frames = client.fetch_images(outputs, SAVE_NODE)

    for stale in directory.glob("frame_*.png"):
        stale.unlink()
    save_frames(frames, directory)
    conditioning.save(directory / "conditioning.png")
    (directory / "orbit.json").write_text(
        json.dumps({"signature": signature, "frame_count": len(frames)}, indent=2),
        encoding="utf-8",
    )
    return frames, signature
