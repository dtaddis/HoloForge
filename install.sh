#!/usr/bin/env sh
set -eu

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ and rerun this script." >&2
  exit 1
fi

uv python install 3.12
uv sync --python 3.12
echo "Installation complete. Run ./run.sh to open HoloForge."
