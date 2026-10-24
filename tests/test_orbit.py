from dataclasses import replace

import numpy as np
from PIL import Image

from holoforge.orbit import (
    ORBIT_LORA,
    ORBIT_LORA_PROMPT,
    OrbitSettings,
    build_graph,
    build_prompt,
    h3_size,
    h3_valid_length,
)
from holoforge.views import plan_views


def test_h3_lengths_and_sizes() -> None:
    assert h3_valid_length(141) == 141
    assert h3_valid_length(150) == 141
    assert h3_valid_length(3) == 5
    width, height = h3_size(0.75)
    assert (width, height) == (768, 1024)
    width, height = h3_size(0.3)  # clamped to H3's 2:5 limit
    assert width % 32 == 0 and height % 32 == 0 and width / height >= 0.38


def test_base_prompt_aligns_both_keyframes_with_the_clip_ends() -> None:
    base = OrbitSettings(subject="a bride holding flowers", lora=None)
    prompt = build_prompt(base, 141)
    assert prompt.startswith("How the reference pictures align")
    assert "5.83-second mark" in prompt
    assert "showing a bride holding flowers." in prompt
    assert "orbiting a full circle around a bride holding flowers" in prompt
    graph = build_graph("in.png", 768, 1024, 141, prompt, base, "x")
    i2v = graph["16"]["inputs"]
    assert i2v["first_frame"] == i2v["last_frame"] == ["1", 0]
    assert "12" not in graph and "24" not in graph
    assert graph["17"]["inputs"]["model"] == ["10", 0]
    pdd = build_graph("in.png", 768, 1024, 141, prompt, replace(base, pdd=True), "x")
    assert pdd["21"]["inputs"]["sigmas"] == ["24", 1]


def test_orbit_lora_uses_its_training_caption_and_patches_the_model() -> None:
    settings = OrbitSettings(subject="ignored with the LoRA")
    prompt = build_prompt(settings, 141)
    assert prompt == ORBIT_LORA_PROMPT
    graph = build_graph("in.png", 768, 1024, 141, prompt, settings, "x")
    assert graph["12"]["class_type"] == "LoraLoaderModelOnly"
    assert graph["12"]["inputs"]["lora_name"] == ORBIT_LORA
    # Both the scheduler and the guider see the LoRA-patched model.
    assert graph["17"]["inputs"]["model"] == ["12", 0]
    assert graph["19"]["inputs"]["model"] == ["12", 0]


def synthetic_orbit(count: int, max_shift: float) -> list[Image.Image]:
    """A near card over a far textured wall; shift > 0 means the camera moved left.

    The card is the only thing with a saturated blue channel, so depth is easy to recover.
    """
    rng = np.random.default_rng(1)
    height, width = 120, 160
    wall = (rng.random((height, width * 2, 3)) * 200).astype(np.uint8)
    card = (rng.random((50, 50, 3)) * 200).astype(np.uint8)
    card[..., 2] = 255
    frames = []
    # Orbit: out to one side, round the back (ignored), back in from the other side.
    for i in range(count):
        phase = i / (count - 1)
        offset = np.sin(2 * np.pi * phase) * max_shift
        image = wall[:, 40 : 40 + width].copy()
        x = 55 + round(offset)
        image[35:85, x : x + 50] = card
        frames.append(Image.fromarray(image))
    return frames


def test_views_come_from_both_ends_of_the_clip() -> None:
    frames = synthetic_orbit(61, 24)
    plan = plan_views(frames, views=9, sweep_degrees=40)
    targets = [s.target for s in plan.sources]
    assert targets == sorted(targets, reverse=True)
    # A flat sliding card has no epipolar geometry, so the direction vote is undetermined
    # and defaults to "leaves towards the left"; the real-clip direction is checked on H3
    # footage. One end of the quilt comes from each end of the clip.
    first, last = plan.sources[0], plan.sources[-1]
    assert first.frame_a < 30 and last.frame_a > 30
    assert plan.span[0] < 0 < plan.span[1]
