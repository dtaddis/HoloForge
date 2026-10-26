"""One-time setup: ComfyUI with MiniMax H3, the model files, and a free ffmpeg.

``holoforge-setup`` either adopts an existing ComfyUI or clones a fresh one with its own
Python environment and CUDA PyTorch, adds the ComfyUI-GGUF custom nodes, downloads the four
model files after the user confirms the MiniMax H3 licence, and on Windows fetches the free
gyan.dev ffmpeg build. The choices are saved to ``~/.holoforge/config.json`` so HoloForge can
start ComfyUI on demand and find ffmpeg.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

COMFYUI_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
GGUF_NODES_REPO = "https://github.com/city96/ComfyUI-GGUF.git"
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu130"
H3_LICENSE_URL = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
DEFAULT_COMFY_URL = "http://127.0.0.1:8188"


@dataclass(frozen=True)
class ModelFile:
    repo: str
    path: str  # inside the Hugging Face repo
    folder: str  # ComfyUI models/ subfolder
    size_gb: float

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


MODELS = (
    ModelFile("Abiray/MiniMax-H3-Pruned-GGUF", "MiniMax-H3-FL2VA-Pruned-Q5_K_M.gguf",
              "diffusion_models", 14.1),
    ModelFile("realrebelai/MiniMax-H3_GGUFs", "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf",
              "text_encoders", 14.6),
    ModelFile("Comfy-Org/MiniMax-H3", "vae/minimax_h3_video_vae_fp16.safetensors", "vae", 5.2),
    ModelFile("pablodawson/MiniMax-H3-360-Orbit-LoRA", "minimax_h3_flf2v_lora_v1.safetensors",
              "loras", 0.2),
)  # fmt: skip


# --- configuration -------------------------------------------------------------------------


def home_dir() -> Path:
    return Path(os.environ.get("HOLOFORGE_HOME", Path.home() / ".holoforge"))


def config_path() -> Path:
    return home_dir() / "config.json"


def load_config() -> dict:
    try:
        return json.loads(config_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(**values: str | bool) -> dict:
    config = {**load_config(), **{k: v for k, v in values.items() if v is not None}}
    config_path().parent.mkdir(parents=True, exist_ok=True)
    config_path().write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def tools_dir() -> Path:
    """`tools/` beside a source checkout, else under ~/.holoforge (for wheel installs)."""
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").exists() and (checkout / "src" / "holoforge").exists():
        return checkout / "tools"
    return home_dir() / "tools"


# --- ffmpeg --------------------------------------------------------------------------------


def managed_ffmpeg() -> Path:
    return tools_dir() / "ffmpeg" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")


def ensure_ffmpeg(*, download: bool = True) -> Path:
    """A free ffmpeg with libx265/libx264: configured, managed, on PATH, or downloaded."""
    configured = load_config().get("ffmpeg")
    for candidate in (configured, str(managed_ffmpeg()), shutil.which("ffmpeg")):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    if not download or os.name != "nt":
        raise FileNotFoundError(
            "ffmpeg was not found. Install it (Windows: rerun holoforge-setup; macOS: "
            "brew install ffmpeg; Linux: sudo apt install ffmpeg) or put it on PATH."
        )
    target = managed_ffmpeg()
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = target.parent / "ffmpeg-release-essentials.zip"
    print(f"Downloading ffmpeg (about 115 MB) from {FFMPEG_URL}", flush=True)
    urllib.request.urlretrieve(FFMPEG_URL, archive)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            name = member.rsplit("/", 1)[-1]
            if name in {"ffmpeg.exe", "ffprobe.exe"} and "/bin/" in member:
                (target.parent / name).write_bytes(bundle.read(member))
    archive.unlink()
    if not target.exists():
        raise FileNotFoundError(f"The ffmpeg download did not contain ffmpeg.exe ({FFMPEG_URL})")
    save_config(ffmpeg=str(target))
    return target


# --- ComfyUI -------------------------------------------------------------------------------


def find_comfy_python(comfy_dir: Path) -> Path | None:
    """The Python a ComfyUI install runs with: its own venv, or the portable build's."""
    exe = "python.exe" if os.name == "nt" else "python"
    sub = "Scripts" if os.name == "nt" else "bin"
    for candidate in (
        comfy_dir / ".venv" / sub / exe,
        comfy_dir / "venv" / sub / exe,
        comfy_dir.parent / "python_embeded" / exe,  # Windows portable build
    ):
        if candidate.exists():
            return candidate
    return None


def run(command: list[str], cwd: Path | None = None) -> None:
    print("> " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def pip_install(python: Path, args: list[str]) -> None:
    uv = shutil.which("uv")
    if uv:
        run([uv, "pip", "install", "--python", str(python), *args])
    else:
        run([str(python), "-m", "pip", "install", *args])


def has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def install_comfyui(comfy_dir: Path) -> Path:
    """Clone ComfyUI into comfy_dir with its own venv and PyTorch; return its Python."""
    if not (comfy_dir / "main.py").exists():
        if shutil.which("git") is None:
            raise RuntimeError("git is needed to clone ComfyUI: https://git-scm.com/downloads")
        comfy_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", COMFYUI_REPO, str(comfy_dir)])
    else:
        run(["git", "-C", str(comfy_dir), "pull", "--ff-only"])
    python = find_comfy_python(comfy_dir)
    if python is None:
        uv = shutil.which("uv")
        if uv:
            run([uv, "venv", "--python", "3.12", str(comfy_dir / ".venv")])
        else:
            run([sys.executable, "-m", "venv", str(comfy_dir / ".venv")])
        python = find_comfy_python(comfy_dir)
        assert python is not None
        if not shutil.which("uv"):
            run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    torch = ["torch", "torchvision", "torchaudio"]
    if sys.platform != "darwin" and has_nvidia_gpu():
        pip_install(python, [*torch, "--index-url", TORCH_CUDA_INDEX])
    else:
        print("No NVIDIA GPU found: installing CPU PyTorch. H3 needs a 24 GB NVIDIA card "
              "to run at a usable speed.", flush=True)  # fmt: skip
        pip_install(python, torch)
    pip_install(python, ["-r", str(comfy_dir / "requirements.txt")])
    return python


def install_gguf_nodes(comfy_dir: Path, python: Path | None, *, managed: bool) -> None:
    """Add the ComfyUI-GGUF nodes. In a ComfyUI that HoloForge did not install, an existing
    copy (and that ComfyUI's Python environment) is the user's to maintain and is left alone."""
    nodes = comfy_dir / "custom_nodes" / "ComfyUI-GGUF"
    if nodes.exists():
        if not managed:
            print(f"ComfyUI-GGUF is already installed: {nodes}", flush=True)
            return
        if (nodes / ".git").exists():
            run(["git", "-C", str(nodes), "pull", "--ff-only"])
    else:
        run(["git", "clone", GGUF_NODES_REPO, str(nodes)])
    if python is not None:
        pip_install(python, ["-r", str(nodes / "requirements.txt")])
    else:
        print(f"Install the GGUF nodes' requirements into your ComfyUI environment: "
              f"pip install -r {nodes / 'requirements.txt'}", flush=True)  # fmt: skip


def download_models(comfy_dir: Path) -> None:
    from huggingface_hub import hf_hub_download

    for model in MODELS:
        destination = comfy_dir / "models" / model.folder / model.name
        if destination.exists():
            print(f"Already present: {destination}", flush=True)
            continue
        print(f"Downloading {model.repo}/{model.path} ({model.size_gb:.1f} GB)", flush=True)
        downloaded = Path(hf_hub_download(model.repo, model.path, local_dir=destination.parent))
        if downloaded != destination:  # files in a repo subfolder land in a matching subfolder
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded), destination)


def comfy_is_running(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/queue", timeout=3):
            return True
    except (urllib.error.URLError, OSError):
        return False


def start_comfyui_if_needed(url: str = DEFAULT_COMFY_URL, timeout: float = 300.0) -> bool:
    """Start the configured ComfyUI when nothing answers at url. True once it is reachable."""
    if comfy_is_running(url):
        return True
    config = load_config()
    comfy_dir = Path(config["comfy_dir"]) if config.get("comfy_dir") else None
    python = Path(config["comfy_python"]) if config.get("comfy_python") else None
    if not comfy_dir or not python or not (comfy_dir / "main.py").exists() or not python.exists():
        return False
    host_port = url.split("://", 1)[-1].rstrip("/")
    host, _, port = host_port.partition(":")
    command = [str(python), "main.py", "--listen", host or "127.0.0.1", "--port", port or "8188"]
    kwargs: dict = {"cwd": str(comfy_dir)}
    if os.name == "nt":
        # Its own console window shows model loading and sampling progress.
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_CONSOLE | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        kwargs["start_new_session"] = True
    print(f"Starting ComfyUI: {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, **kwargs)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if comfy_is_running(url):
            return True
        if process.poll() is not None:
            return False
        time.sleep(2)
    return False


def comfy_status(url: str = DEFAULT_COMFY_URL) -> str:
    """One line for the web app: running, startable, or not set up."""
    if comfy_is_running(url):
        return f"ComfyUI is running at {url}."
    config = load_config()
    if config.get("comfy_dir") and config.get("comfy_python"):
        return f"ComfyUI isn't running; HoloForge will start it from {config['comfy_dir']}."
    return "ComfyUI isn't set up yet: run `uv run holoforge-setup` (or rerun the installer)."


# --- sharing the GPU with another pipeline ------------------------------------------------
#
# Optional. With ``gpu_request_flag`` and ``gpu_go_flag`` set in the config, HoloForge creates
# the request file before it renders, waits for the other pipeline to write the go file (once
# it has paused and freed the GPU), and deletes the request file when done.


class _GpuClaim:
    depth = 0
    held = False


def _release_gpu() -> None:
    flag = load_config().get("gpu_request_flag")
    if _GpuClaim.held and flag:
        Path(flag).unlink(missing_ok=True)
        print("Released the GPU", flush=True)
    _GpuClaim.held = False


@contextlib.contextmanager
def gpu_session():
    """Hold any GPU claim made inside until the outermost session ends."""
    _GpuClaim.depth += 1
    try:
        yield
    finally:
        _GpuClaim.depth -= 1
        if _GpuClaim.depth == 0:
            _release_gpu()


def claim_gpu(timeout: float = 600.0) -> None:
    """Ask the configured pipeline for the GPU and wait until it hands it over."""
    config = load_config()
    request, go = config.get("gpu_request_flag"), config.get("gpu_go_flag")
    if not request or not go or _GpuClaim.held:
        return
    Path(request).parent.mkdir(parents=True, exist_ok=True)
    Path(request).touch()
    _GpuClaim.held = True
    atexit.register(_release_gpu)
    print("Waiting for the GPU to be handed over...", flush=True)
    deadline = time.monotonic() + timeout
    while not Path(go).exists():
        if time.monotonic() > deadline:
            _release_gpu()
            raise TimeoutError(f"No go signal ({go}) within {timeout:.0f} s")
        time.sleep(2)


# --- command line --------------------------------------------------------------------------


def ask(question: str, default: bool = True) -> bool:
    suffix = " [Y/n] " if default else " [y/N] "
    try:
        answer = input(question + suffix).strip().lower()
    except EOFError:
        return default
    return default if not answer else answer.startswith("y")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="holoforge-setup",
        description="Set up ComfyUI, the MiniMax H3 models and ffmpeg for HoloForge.",
    )
    parser.add_argument("--comfy-dir", type=Path, help="Existing ComfyUI, or where to clone one")
    parser.add_argument("--comfy-python", type=Path, help="Python that runs an existing ComfyUI")
    parser.add_argument("--comfy-url", default=DEFAULT_COMFY_URL)
    parser.add_argument("--accept-h3-license", action="store_true",
                        help="Confirm the MiniMax H3 Community License covers you")  # fmt: skip
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Accept the defaults without asking")
    args = parser.parse_args()
    confirm = (lambda question, default=True: True) if args.yes else ask

    print("HoloForge setup\n", flush=True)
    ffmpeg = ensure_ffmpeg()
    print(f"ffmpeg: {ffmpeg}\n", flush=True)

    default_dir = tools_dir() / "comfyui"
    comfy_dir = (args.comfy_dir or Path(load_config().get("comfy_dir", default_dir))).resolve()
    existing = (comfy_dir / "main.py").exists()
    config = load_config()
    # HoloForge manages a ComfyUI it cloned (now or on an earlier run), wherever it lives.
    managed = (
        not existing
        or comfy_dir == default_dir.resolve()
        or (bool(config.get("comfy_managed")) and config.get("comfy_dir") == str(comfy_dir))
    )
    if existing and not managed:
        print(f"Using your existing ComfyUI at {comfy_dir}", flush=True)
        python = args.comfy_python or find_comfy_python(comfy_dir)
        if python is None:
            print("Couldn't find the Python it runs with, so HoloForge can't start it for you. "
                  "Rerun with --comfy-python PATH, or start ComfyUI yourself.", flush=True)  # fmt: skip
    else:
        if not confirm(f"Install ComfyUI (with CUDA PyTorch, about 6 GB) at {comfy_dir}?"):
            print("Skipped. Rerun with --comfy-dir PATH to use an existing ComfyUI.")
            return
        python = install_comfyui(comfy_dir)
    install_gguf_nodes(comfy_dir, python, managed=managed)
    save_config(
        comfy_dir=str(comfy_dir),
        comfy_python=str(python) if python else None,
        comfy_url=args.comfy_url,
        comfy_managed=managed,
    )

    if not args.skip_models:
        total = sum(m.size_gb for m in MODELS)
        print(f"\nHoloForge needs four MiniMax H3 model files ({total:.0f} GB).")
        print("The MiniMax H3 Community License excludes the EU, UK, South Korea and the USA "
              f"unless MiniMax grants a separate licence: {H3_LICENSE_URL}")  # fmt: skip
        # --yes never stands in for the licence: that needs the flag or a typed answer.
        if args.accept_h3_license or (
            not args.yes and ask("Does the licence cover you, and download now?", False)
        ):
            download_models(comfy_dir)
        else:
            print("Skipped the model download; rerun holoforge-setup (or pass "
                  "--accept-h3-license) when you're ready.")  # fmt: skip
    print(f"\nSetup complete. Settings saved to {config_path()}")


if __name__ == "__main__":
    main()
