"""Generate Qwen midpoint views from an original and two side-view references."""

from __future__ import annotations

import argparse
import shutil
import uuid
from pathlib import Path

from qwen_multiview import DEFAULT_COMFY_ROOT, run_anchor, wait_for_idle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:10]
    inputs = {
        "source": (args.source, f"holoforge_bridge_source_{token}{args.source.suffix.lower()}"),
        "left": (args.left, f"holoforge_bridge_left_{token}{args.left.suffix.lower()}"),
        "right": (args.right, f"holoforge_bridge_right_{token}{args.right.suffix.lower()}"),
    }
    for path, name in inputs.values():
        shutil.copy2(path, args.comfy_root / "input" / name)

    shared = (
        "Both images show the same person, clothing, bouquet and location. Generate the exact "
        "camera viewpoint halfway between image 1 and image 2. Preserve the person's identity, "
        "expression, pose, proportions, dress, bouquet, door geometry, lighting, colours and old "
        "photographic texture. Use image 2 to resolve newly visible side surfaces and occluded "
        "background. Do not redesign, beautify, blur, add or remove objects."
    )
    try:
        wait_for_idle(args.server)
        left = run_anchor(
            args.server,
            args.comfy_root,
            inputs["source"][1],
            shared,
            "HoloForge_Qwen_bridge_left",
            20260921,
            camera_lora=False,
            camera_strength=0.0,
            input_name2=inputs["left"][1],
        )
        right = run_anchor(
            args.server,
            args.comfy_root,
            inputs["source"][1],
            shared,
            "HoloForge_Qwen_bridge_right",
            20260922,
            camera_lora=False,
            camera_strength=0.0,
            input_name2=inputs["right"][1],
        )
        shutil.copy2(left, args.output / "qwen_bridge_left.png")
        shutil.copy2(right, args.output / "qwen_bridge_right.png")
    finally:
        for _, name in inputs.values():
            (args.comfy_root / "input" / name).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
