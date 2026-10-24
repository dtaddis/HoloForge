"""Describe a photograph's subjects and look with a local Ollama vision model.

H3 is steered almost entirely by its prompt (guidance 1 runs no negative), so naming what
is actually in the picture keeps identities, clothing and the film look stable through
the orbit. Captioning is optional; without Ollama the generic subject wording is used.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from PIL import Image

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5vl:7b"

INSTRUCTION = (
    "You are writing part of a prompt for a video model that will orbit the camera around "
    "this exact photograph. Reply with JSON only, with two keys.\n"
    '"subject": one noun phrase naming everyone and every main object in the photo, with '
    'their visible clothing and pose, as they appear (for example: "a man in a grey suit and '
    'a girl in a yellow dress sitting together on a garden bench"). Start with a lowercase '
    "article. Do not guess names, ages or relationships.\n"
    '"style": a short phrase for the photographic look (for example: "faded 1970s colour '
    'film snapshot" or "modern digital phone photo").'
)


@dataclass(frozen=True)
class Caption:
    subject: str
    style: str


def _encode(image: Image.Image, max_side: int = 896) -> str:
    image = image.copy()
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def caption_photo(
    image: Image.Image,
    *,
    model: str = DEFAULT_MODEL,
    server: str = OLLAMA_URL,
    keep_alive: str | int = "5m",
    timeout: float = 300.0,
) -> Caption | None:
    """Return a caption, or None when Ollama or the model is unavailable."""
    payload = {
        "model": model,
        "prompt": INSTRUCTION,
        "images": [_encode(image)],
        "format": "json",
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"temperature": 0.1},
    }
    request = urllib.request.Request(
        f"{server}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
        parsed = json.loads(body.get("response", "{}"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None
    subject = str(parsed.get("subject", "")).strip().rstrip(".")
    style = str(parsed.get("style", "")).strip().rstrip(".")
    if not subject:
        return None
    return Caption(subject=subject[0].lower() + subject[1:], style=style or "Photograph")


def unload(model: str = DEFAULT_MODEL, server: str = OLLAMA_URL) -> None:
    """Free the vision model's VRAM before H3 needs it (a prompt-less request unloads)."""
    request = urllib.request.Request(
        f"{server}/api/generate",
        data=json.dumps({"model": model, "keep_alive": 0}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=60).close()
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
