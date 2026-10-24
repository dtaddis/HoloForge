"""Local-only Qwen-Image-Edit-2511 multi-view experiment.

This deliberately lives outside the packaged HoloForge source tree. It submits two
camera-orbit edits to an existing ComfyUI server, morphs the original photograph to
the generated anchors, and packs 60 views into a musubi quilt.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.request
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from holoforge.config import PRESETS
from holoforge.image_utils import load_rgb, prepare_frame, safe_stem
from holoforge.quilt import QuiltWriter, quilt_filename, save_wiggle

DEFAULT_COMFY_ROOT = Path(r"D:\dtaddis\ai-remaster-pipeline\tools\comfyui")
GGUF_MODEL = "qwen-image-edit-2511-Q4_K_M.gguf"
LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
CAMERA_LORA = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
TEXT_ENCODER = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"


def api_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def wait_for_idle(server: str, stable_seconds: int = 20) -> None:
    stable_since: float | None = None
    while True:
        queue = api_json(f"{server}/queue")
        busy = bool(queue.get("queue_running") or queue.get("queue_pending"))
        if busy:
            stable_since = None
            print("ComfyUI is busy with another job; waiting...", flush=True)
        else:
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= stable_seconds:
                return
        time.sleep(10)


def make_graph(
    input_name: str,
    prompt: str,
    prefix: str,
    seed: int,
    *,
    camera_lora: bool,
    camera_strength: float,
    input_name2: str | None = None,
    model_name: str = GGUF_MODEL,
) -> dict:
    model_source = ["2", 0]
    nodes: dict[str, dict] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": input_name}},
        "2": {
            "class_type": "UnetLoaderGGUF" if model_name.lower().endswith(".gguf") else "UNETLoader",
            "inputs": {
                "unet_name": model_name,
                **({} if model_name.lower().endswith(".gguf") else {"weight_dtype": "default"}),
            },
        },
        "4": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": ["3", 0] if camera_lora else model_source,
                "lora_name": LIGHTNING_LORA,
                "strength_model": camera_strength,
            },
        },
        "5": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": ["4", 0], "shift": 3.1},
        },
        "6": {"class_type": "CFGNorm", "inputs": {"model": ["5", 0], "strength": 1.0}},
        "7": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": TEXT_ENCODER, "type": "qwen_image", "device": "default"},
        },
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "9": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["1", 0]}},
        "10": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": ["7", 0],
                "vae": ["8", 0],
                "image1": ["9", 0],
                "prompt": prompt,
            },
        },
        "11": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "conditioning": ["10", 0],
                "reference_latents_method": "index_timestep_zero",
            },
        },
        "12": {
            "class_type": "TextEncodeQwenImageEditPlus",
            "inputs": {
                "clip": ["7", 0],
                "vae": ["8", 0],
                "image1": ["9", 0],
                "prompt": "",
            },
        },
        "13": {
            "class_type": "FluxKontextMultiReferenceLatentMethod",
            "inputs": {
                "conditioning": ["12", 0],
                "reference_latents_method": "index_timestep_zero",
            },
        },
        "14": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["9", 0], "vae": ["8", 0]},
        },
        "15": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["6", 0],
                "seed": seed,
                "steps": 4,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": ["11", 0],
                "negative": ["13", 0],
                "latent_image": ["14", 0],
                "denoise": 1.0,
            },
        },
        "16": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["15", 0], "vae": ["8", 0]},
        },
        "17": {
            "class_type": "SaveImage",
            "inputs": {"images": ["16", 0], "filename_prefix": prefix},
        },
    }
    if camera_lora:
        nodes["3"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_source,
                "lora_name": CAMERA_LORA,
                "strength_model": 1.0,
            },
        }
    if input_name2:
        nodes["18"] = {"class_type": "LoadImage", "inputs": {"image": input_name2}}
        nodes["19"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": ["18", 0]}}
        nodes["10"]["inputs"]["image2"] = ["19", 0]
        nodes["12"]["inputs"]["image2"] = ["19", 0]
    return nodes


def run_anchor(
    server: str,
    comfy_root: Path,
    input_name: str,
    prompt: str,
    prefix: str,
    seed: int,
    *,
    camera_lora: bool,
    camera_strength: float,
    input_name2: str | None = None,
    model_name: str = GGUF_MODEL,
) -> Path:
    graph = make_graph(
        input_name,
        prompt,
        prefix,
        seed,
        camera_lora=camera_lora,
        camera_strength=camera_strength,
        input_name2=input_name2,
        model_name=model_name,
    )
    response = api_json(
        f"{server}/prompt",
        {"prompt": graph, "client_id": str(uuid.uuid4())},
    )
    prompt_id = response["prompt_id"]
    print(f"Queued {prefix}: {prompt_id}", flush=True)
    while True:
        history = api_json(f"{server}/history/{prompt_id}")
        record = history.get(prompt_id)
        if record:
            status = record.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(json.dumps(status, indent=2))
            images = record.get("outputs", {}).get("17", {}).get("images", [])
            if images:
                item = images[0]
                return comfy_root / "output" / item.get("subfolder", "") / item["filename"]
        time.sleep(3)


def resize_pair(source: Image.Image, anchor: Image.Image, width: int = 820) -> tuple[np.ndarray, np.ndarray]:
    height = round(width * source.height / source.width)
    size = (width, height)
    source_array = np.asarray(source.resize(size, Image.Resampling.LANCZOS), dtype=np.uint8)
    anchor_array = np.asarray(anchor.resize(size, Image.Resampling.LANCZOS), dtype=np.uint8)
    return source_array, anchor_array


def align_anchor_to_source(source: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Remove Qwen's incidental zoom/crop while retaining its local viewpoint edit."""
    sift = cv2.SIFT_create(5000)
    source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    anchor_gray = cv2.cvtColor(anchor, cv2.COLOR_RGB2GRAY)
    source_points, source_desc = sift.detectAndCompute(source_gray, None)
    anchor_points, anchor_desc = sift.detectAndCompute(anchor_gray, None)
    if source_desc is None or anchor_desc is None:
        return anchor
    matches = cv2.BFMatcher().knnMatch(source_desc, anchor_desc, k=2)
    good = [first for first, second in matches if first.distance < 0.72 * second.distance]
    if len(good) < 20:
        return anchor
    source_xy = np.float32([source_points[item.queryIdx].pt for item in good])
    anchor_xy = np.float32([anchor_points[item.trainIdx].pt for item in good])
    transform, inliers = cv2.findHomography(source_xy, anchor_xy, cv2.USAC_MAGSAC, 3.0)
    if transform is None or inliers is None or int(inliers.sum()) < 20:
        return anchor
    inverse = np.linalg.inv(transform)
    height, width = source.shape[:2]
    print(f"Anchor alignment: {int(inliers.sum())}/{len(good)} projective inliers", flush=True)
    warped = cv2.warpPerspective(
        anchor,
        inverse,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    # A projective registration can expose pixels outside Qwen's generated
    # canvas. Reflecting that edge produces conspicuous horizontal/vertical
    # echoes in the quilt. Keep the generated view wherever it has coverage,
    # then feather only its uncovered perimeter back to the source photograph.
    coverage = cv2.warpPerspective(
        np.full((height, width), 255, dtype=np.uint8),
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    coverage = cv2.GaussianBlur(coverage, (0, 0), sigmaX=8.0, sigmaY=8.0)
    alpha = coverage.astype(np.float32)[..., None] / 255.0
    composite = warped.astype(np.float32) * alpha + source.astype(np.float32) * (1.0 - alpha)
    return np.clip(composite, 0, 255).astype(np.uint8)


def calculate_flows(source: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate a regularised correspondence field once per generated anchor."""
    gray_source = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    gray_anchor = cv2.cvtColor(anchor, cv2.COLOR_RGB2GRAY)
    flow_engine = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow_forward = flow_engine.calc(gray_source, gray_anchor, None)
    flow_backward = flow_engine.calc(gray_anchor, gray_source, None)
    # Raw dense flow follows every generated texture variation and visibly bends
    # door frames and other rigid architecture. A broad regulariser retains the
    # camera motion while suppressing those small non-geometric deviations.
    flow_forward = cv2.GaussianBlur(flow_forward, (0, 0), sigmaX=6.0, sigmaY=6.0)
    flow_backward = cv2.GaussianBlur(flow_backward, (0, 0), sigmaX=6.0, sigmaY=6.0)
    return flow_forward, flow_backward


def view_morph(
    source: np.ndarray,
    anchor: np.ndarray,
    amount: float,
    flows: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    if amount <= 1e-6:
        return source.copy()
    if amount >= 1.0 - 1e-6:
        return anchor.copy()
    flow_forward, flow_backward = flows
    height, width = source.shape[:2]
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    source_warp = cv2.remap(
        source,
        xx - amount * flow_forward[..., 0],
        yy - amount * flow_forward[..., 1],
        cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT,
    )
    anchor_weight = amount
    anchor_warp = cv2.remap(
        anchor,
        xx - (1.0 - amount) * flow_backward[..., 0],
        yy - (1.0 - amount) * flow_backward[..., 1],
        cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT,
    )
    blended = source_warp.astype(np.float32) * (1.0 - anchor_weight)
    blended += anchor_warp.astype(np.float32) * anchor_weight
    return np.clip(blended, 0, 255).astype(np.uint8)


def build_quilt(source_path: Path, left_path: Path, right_path: Path, output_dir: Path) -> None:
    source = load_rgb(source_path)
    left = load_rgb(left_path)
    right = load_rgb(right_path)
    source_array, left_array = resize_pair(source, left)
    _, right_array = resize_pair(source, right)
    left_array = align_anchor_to_source(source_array, left_array)
    right_array = align_anchor_to_source(source_array, right_array)
    Image.fromarray(left_array).save(output_dir / "qwen_left_aligned.png")
    Image.fromarray(right_array).save(output_dir / "qwen_right_aligned.png")
    print("Calculating regularised anchor correspondence", flush=True)
    left_flows = calculate_flows(source_array, left_array)
    right_flows = calculate_flows(source_array, right_array)

    preset = PRESETS["musubi"]
    tile_size = (
        int(np.ceil(preset.quilt_size / preset.columns)),
        int(np.ceil(preset.quilt_size / preset.rows)),
    )
    writer = QuiltWriter(preset)
    positions = np.linspace(-1.0, 1.0, preset.views)
    sample_indices = {int(i) for i in np.linspace(0, preset.views - 1, 9)}
    previews: list[Image.Image] = []
    endpoints: list[Image.Image] = []
    for index, position in enumerate(positions):
        if position < 0:
            pixels = view_morph(source_array, left_array, abs(float(position)), left_flows)
        else:
            pixels = view_morph(source_array, right_array, abs(float(position)), right_flows)
        view, _ = prepare_frame(Image.fromarray(pixels), tile_size, "ambient")
        writer.add(index, view)
        if index in sample_indices:
            previews.append(view.copy())
        if index in {0, preset.views - 1}:
            endpoints.append(view.copy())
        print(f"Morphing view {index + 1}/{preset.views}", flush=True)

    stem = safe_stem(source_path) + "_qwen"
    output_dir.mkdir(parents=True, exist_ok=True)
    quilt_path = output_dir / quilt_filename(stem, preset, ".jpg")
    writer.canvas.save(quilt_path, quality=96, subsampling=0, optimize=False)
    save_wiggle(previews, output_dir / f"{stem}_wiggle.gif")
    if len(endpoints) == 2:
        audit = Image.new("RGB", (endpoints[0].width * 2, endpoints[0].height), "black")
        audit.paste(endpoints[0], (0, 0))
        audit.paste(endpoints[1], (endpoints[0].width, 0))
        audit.save(output_dir / f"{stem}_endpoints.png")
    print(f"Quilt: {quilt_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--output", type=Path, default=Path("outputs/qwen_experiment"))
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--camera-lora", action="store_true")
    parser.add_argument("--model", default=GGUF_MODEL)
    parser.add_argument("--camera-strength", type=float, default=1.0)
    parser.add_argument(
        "--camera-pose",
        choices=("quarter", "side"),
        default="quarter",
        help="Camera-LoRA diagnostic pose; side should produce unmistakable rotation.",
    )
    parser.add_argument("--reuse-anchors", action="store_true")
    parser.add_argument("--anchors-only", action="store_true")
    args = parser.parse_args()

    output_dir = args.output / safe_stem(args.input)
    output_dir.mkdir(parents=True, exist_ok=True)
    left_path = output_dir / "qwen_left.png"
    right_path = output_dir / "qwen_right.png"
    if args.reuse_anchors:
        if not left_path.exists() or not right_path.exists():
            raise FileNotFoundError("--reuse-anchors requires qwen_left.png and qwen_right.png")
        build_quilt(args.input, left_path, right_path, output_dir)
        return

    wait_for_idle(args.server)
    input_name = f"holoforge_qwen_{uuid.uuid4().hex[:10]}{args.input.suffix.lower()}"
    input_copy = args.comfy_root / "input" / input_name
    shutil.copy2(args.input, input_copy)
    try:
        if args.camera_lora:
            if args.camera_pose == "side":
                left_prompt = "<sks> left side view eye-level shot medium shot"
                right_prompt = "<sks> right side view eye-level shot medium shot"
            else:
                left_prompt = "<sks> front-left quarter view eye-level shot medium shot"
                right_prompt = "<sks> front-right quarter view eye-level shot medium shot"
        else:
            preservation = (
                "Preserve the exact identities, faces, expressions, pose, clothing, objects, "
                "background, architecture, lighting, crop, lens, colours and photographic grain. "
                "Do not restyle, rearrange, remove or add anything. Reconstruct newly visible side "
                "surfaces and hidden background with physically plausible perspective."
            )
            left_prompt = (
                "Move the camera horizontally about 7 degrees to the left at the same eye-level "
                f"distance. {preservation}"
            )
            right_prompt = (
                "Move the camera horizontally about 7 degrees to the right at the same eye-level "
                f"distance. {preservation}"
            )
        left_generated = run_anchor(
            args.server,
            args.comfy_root,
            input_name,
            left_prompt,
            "HoloForge_Qwen_left",
            args.seed,
            camera_lora=args.camera_lora,
            camera_strength=args.camera_strength,
            model_name=args.model,
        )
        right_generated = run_anchor(
            args.server,
            args.comfy_root,
            input_name,
            right_prompt,
            "HoloForge_Qwen_right",
            args.seed + 1,
            camera_lora=args.camera_lora,
            camera_strength=args.camera_strength,
            model_name=args.model,
        )
        shutil.copy2(left_generated, left_path)
        shutil.copy2(right_generated, right_path)
        if not args.anchors_only:
            build_quilt(args.input, left_path, right_path, output_dir)
    finally:
        input_copy.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
