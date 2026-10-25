#!/usr/bin/env sh
set -eu

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ and rerun this script." >&2
  exit 1
fi

uv python install 3.12
uv sync --python 3.12
echo
echo "HoloForge is installed. It renders with MiniMax H3 in ComfyUI, which needs a one-time"
echo "setup: use your existing ComfyUI or clone a new one, plus about 34 GB of model files."
echo "ffmpeg must be on PATH (macOS: brew install ffmpeg; Linux: sudo apt install ffmpeg)."
printf "Set up ComfyUI and the H3 models now? [Y/n] "
read -r answer || answer=""
case "$answer" in
  ""|[Yy]*)
    printf "Path to an existing ComfyUI folder (leave blank to clone a new one): "
    read -r comfy || comfy=""
    if [ -n "$comfy" ]; then uv run holoforge-setup --comfy-dir "$comfy"; else uv run holoforge-setup; fi
    ;;
  *) echo "Later: uv run holoforge-setup" ;;
esac
echo "Installation complete. Run ./run.sh to open HoloForge."
