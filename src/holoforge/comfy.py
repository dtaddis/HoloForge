"""Minimal client for a local ComfyUI server's HTTP API (stdlib only)."""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path

from PIL import Image

DEFAULT_SERVER = "http://127.0.0.1:8188"


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, server: str = DEFAULT_SERVER, timeout: float = 60.0) -> None:
        self.server = server.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    def _request(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.server}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise ComfyError(f"ComfyUI {path} failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise ComfyError(
                f"Cannot reach ComfyUI at {self.server} ({exc.reason}). Start it, or run "
                "`uv run holoforge-setup` so HoloForge can install and start it for you."
            ) from exc

    def object_info(self, node_class: str) -> dict:
        return self._request(f"/object_info/{node_class}").get(node_class, {})

    def require_nodes(self, node_classes: list[str]) -> None:
        missing = [name for name in node_classes if not self.object_info(name)]
        if missing:
            raise ComfyError(
                "ComfyUI is missing nodes: " + ", ".join(missing) + ". MiniMax H3 needs "
                "ComfyUI 0.30+ and the ComfyUI-GGUF custom nodes."
            )

    def require_choice(self, node_class: str, field: str, value: str) -> None:
        info = self.object_info(node_class)
        choices = info.get("input", {}).get("required", {}).get(field, [[]])[0]
        if isinstance(choices, list) and value not in choices:
            raise ComfyError(f"{node_class}.{field} has no '{value}'. Available: {choices}")

    def upload_image(self, image: Image.Image, name: str) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        boundary = uuid.uuid4().hex
        body = b"".join(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'.encode(),
                b"Content-Type: image/png\r\n\r\n",
                buffer.getvalue(),
                f"\r\n--{boundary}\r\n".encode(),
                b'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue',
                f"\r\n--{boundary}--\r\n".encode(),
            ]
        )
        request = urllib.request.Request(
            f"{self.server}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        subfolder = result.get("subfolder") or ""
        return f"{subfolder}/{result['name']}" if subfolder else result["name"]

    def wait_for_idle(
        self,
        stable_seconds: float,
        *,
        poll_seconds: float = 10.0,
        on_busy: Callable[[float], None] | None = None,
    ) -> None:
        """Yield to other users of the server: return once its queue has stayed empty
        (nothing running or pending) for stable_seconds."""
        if stable_seconds <= 0:
            return
        started = time.monotonic()
        idle_since: float | None = None
        while True:
            queue = self._request("/queue")
            if queue.get("queue_running") or queue.get("queue_pending"):
                idle_since = None
                if on_busy:
                    on_busy(time.monotonic() - started)
            else:
                idle_since = idle_since or time.monotonic()
                if time.monotonic() - idle_since >= stable_seconds:
                    return
            time.sleep(poll_seconds)

    def queue(self, graph: dict) -> str:
        result = self._request("/prompt", {"prompt": graph, "client_id": self.client_id})
        if result.get("node_errors"):
            raise ComfyError(f"ComfyUI rejected the graph: {result['node_errors']}")
        return result["prompt_id"]

    def wait(
        self,
        prompt_id: str,
        *,
        poll_seconds: float = 5.0,
        on_wait: Callable[[float], None] | None = None,
    ) -> dict:
        started = time.monotonic()
        while True:
            history = self._request(f"/history/{prompt_id}").get(prompt_id)
            if history:
                status = history.get("status", {})
                if status.get("status_str") == "error":
                    messages = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    raise ComfyError(f"ComfyUI execution failed: {messages or status}")
                if status.get("completed", True):
                    return history.get("outputs", {})
            if on_wait:
                on_wait(time.monotonic() - started)
            time.sleep(poll_seconds)

    def fetch_image(self, ref: dict) -> Image.Image:
        query = urllib.parse.urlencode(
            {
                "filename": ref["filename"],
                "subfolder": ref.get("subfolder", ""),
                "type": ref.get("type", "output"),
            }
        )
        with urllib.request.urlopen(f"{self.server}/view?{query}", timeout=self.timeout) as r:
            image = Image.open(io.BytesIO(r.read()))
            image.load()
        return image.convert("RGB")

    def fetch_images(self, outputs: dict, node_id: str) -> list[Image.Image]:
        refs = outputs.get(node_id, {}).get("images", [])
        if not refs:
            raise ComfyError(f"ComfyUI node {node_id} produced no images")
        return [self.fetch_image(ref) for ref in refs]


def save_frames(frames: list[Image.Image], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, frame in enumerate(frames):
        path = directory / f"frame_{index:04d}.png"
        frame.save(path, compress_level=1)
        paths.append(path)
    return paths
