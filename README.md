# HoloForge

HoloForge is a private, local converter that turns one conventional photograph into a
multi-view Looking Glass hologram. It asks the MiniMax H3 video model, running in your own
ComfyUI, to fly a camera round the frozen scene, then builds the quilt from real views of
the people from either side, rather than warping a flat photo over a depth map. With a
Looking Glass Portrait plugged in, results go straight onto its standalone playlist.

This is an early working prototype, not an official Looking Glass Factory product.

![HoloForge parallax reconstruction of Migrant Mother](assets/demo/migrant_mother_holoforge.gif)

Demo: Dorothea Lange, *Migrant mother and children – California* (1936), cropped
from the [Library of Congress scan](https://www.loc.gov/pictures/item/93506674/).
The FSA/OWI collection lists no known publication restrictions. The animated image
is a HoloForge reconstruction: newly revealed pixels are generated, not historical evidence.

## Orbit method (default): MiniMax H3

The default method asks a video model to fly a camera around the photograph instead of
warping pixels:

1. The photo is given to MiniMax H3's first-and-last-frame mode (FL2VA) as **both** the first
   and the last frame, with a prompt for a camera orbit around subjects frozen like statues.
   The clip therefore starts and ends on the photographed viewpoint, and the frames just after
   the start and just before the end show the scene from either side of it.
2. By default Pablo Dawson's
   [360° orbit LoRA](https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA)
   (`minimax_h3_flf2v_lora_v1.safetensors` in ComfyUI's `models/loras`) is applied with its
   training caption as the prompt. Without it, the base model often turns the subjects on a
   turntable in front of a frozen background, which plays back with a flat background. With
   `--no-lora`, a local Ollama vision model (`qwen2.5vl:7b`) describes who is in the photo
   and its film look for a descriptive prompt instead. Either way, a clip whose background
   stays still while the view turns is re-rendered once with a new seed.
3. 48 views are placed at evenly spaced viewing angles over `--sweep` degrees (default 30)
   either side of the photograph; where no frame lands close enough, the two neighbouring
   frames are blended along their optical flow. LoRA clips orbit at constant speed, so frame
   number gives the angle. Base-model clips hold, ease in and out unevenly, and sometimes turn
   the subjects instead of the camera, so their progress is measured from accumulated
   frame-to-frame motion. In both cases, an essential matrix fitted to the moving features
   decides which side the clip leaves towards, so the quilt runs from the leftmost view.
4. Views are framed into the display aspect, packed into a quilt, and, when a Portrait is
   connected, interlaced and copied onto it (see below).

### ComfyUI setup

H3 runs locally through [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 0.30 or newer
with the [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) custom nodes. Put these files
in ComfyUI's `models` folders (these are the defaults; `--unet`, `--clip` and `--vae` select
others, and non-GGUF files such as Comfy-Org's int8/nvfp4 repacks load with the core loaders):

| folder | file | source |
|---|---|---|
| `diffusion_models` | `MiniMax-H3-FL2VA-Pruned-Q5_K_M.gguf` | [Abiray/MiniMax-H3-Pruned-GGUF](https://huggingface.co/Abiray/MiniMax-H3-Pruned-GGUF) |
| `text_encoders` | `qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf` | [realrebelai/MiniMax-H3_GGUFs](https://huggingface.co/realrebelai/MiniMax-H3_GGUFs) |
| `vae` | `minimax_h3_video_vae_fp16.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) (`vae/`) |
| `loras` | `minimax_h3_flf2v_lora_v1.safetensors` | [pablodawson/MiniMax-H3-360-Orbit-LoRA](https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA) |

On an RTX 4090 a 141-frame 768p clip (the longest that stays inside 24 GB) takes about 15
minutes at the default 28 steps; `--pdd` uses the 8-step PDD distillation
([alibaba-pai/MiniMax-H3-Acc-LoRAs](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs))
for about 2.5x speed at some cost in fidelity.

### Web app

```powershell
.un.ps1          # or: uv run holoforge-ui
```

Choose a photo, keep **H3 orbit** selected and press **Forge hologram**. The page shows the
generated orbit video, a parallax preview, the measured viewing angles, the quilt and (with a
Portrait connected) the Portrait recording, which it also adds to the device's playlist.

### Command line

```powershell
# ComfyUI must be running on http://127.0.0.1:8188
uv run holoforge "photo.jpg" --push
uv run holoforge "D:\Photos\Birthday" --push          # a whole folder, captioned and queued
uv run holoforge "photo.jpg" --motion sweep             # swing side to side instead of orbiting
uv run holoforge "photo.jpg" --no-lora --subject "a bride holding roses" --style "1980s colour print"
```

Each run keeps the H3 frames under `outputs/<photo>/orbit/`, an MP4 of every generated clip
under `outputs/<photo>/videos/`, and reuses the frames when settings are unchanged, so trying a stronger or gentler 3D effect (`--sweep 20`, `--sweep 36`) takes about a
minute per photo instead of another H3 render. `parallax.png` plots the measured viewing angle
per frame against the chosen views.

HoloForge shares ComfyUI politely: before each H3 job it waits until the server's queue has
been empty for `--idle-seconds` (default 180). Pass `--idle-seconds 0` when you have the GPU to
yourself.

**Licence:** the open H3 weights are under the MiniMax H3 Community License, which excludes the
EU, UK, South Korea and USA unless MiniMax grants a separate licence. Check it applies to you.

## Looking Glass Portrait standalone playback

The Portrait does not play quilts directly in standalone mode. Looking Glass Studio renders each
quilt through its lenticular shader into a 3072×2048 HEVC clip (`<quilt>__lkgrec.mp4`) and lists
it in `playlist.json`/`playlist.m3u` on the device's USB drive. HoloForge does the same from the
device's own `LKG_calibration/visual.json`, reproducing Studio 1.7's `quilt_to_lent` shader and
its logged uniforms, and encodes with the ffmpeg bundled in Studio (libkvazaar), so Studio is
installed but never has to be opened.

```powershell
uv run holoforge-portrait "outputs\photo\photo_quilt_qs8x6a0.75.jpg"   # any existing quilt
```

New items go to the front of the playlist and nothing existing is removed; the original
playlists are backed up once as `*.holoforge-backup`. A later "sync" from Looking Glass Studio
rewrites the playlist from Studio's own library and may drop these items.

## Layered method: what is different

A basic RGB-D converter estimates one depth value per visible pixel, shifts those pixels,
and has nothing to show when a foreground object moves aside. HoloForge instead builds a
small layered scene:

1. Depth Anything V2 Large estimates fine relative depth.
   RGB-guided boundary snapping removes the intermediate-depth fringe that otherwise
   detaches from hair, clothing, and other silhouettes at endpoint views.
2. A bounded facial-feature prior adds subtle cheek and nose relief only inside detected
   face pixels. It never guesses a spherical cranium or paints a head edge over scenery,
   so profile and three-quarter portraits retain their real silhouette.
3. Two depth thresholds identify surfaces that will be exposed by parallax.
4. LaMa removes masked objects into a coherent coarse backplate; a restrained SDXL pass
   refines its texture while depth inpainting reconstructs hidden geometry. Each generated
   plate is generated with a wide cleanup margin but revealed only in the narrow bands
   that parallax can expose. This keeps blend edges and hallucinated subject remnants out
   of extreme views.
5. A CUDA forward-splat renderer projects the shared scene through 60 viewpoints with a
   real z-buffer.
6. Views are packed bottom-left to top-right using Looking Glass quilt ordering.

The centre view remains the original photograph. Generated content only becomes visible
at disocclusions in off-axis views, which greatly reduces identity drift and view-to-view
flicker.

## Hardware

The high-quality defaults target an NVIDIA RTX 4090 with 24 GB VRAM and roughly 60–120
seconds per image after model warm-up. CUDA 13 wheels are installed on Windows and Linux.
Apple Silicon installs CPU PyTorch and will work, but the generative path is not expected
to meet that timing target.

Model downloads on first use are several gigabytes and do not count toward conversion
time. After download, photographs and inference remain local.

## Install

### Windows

Open PowerShell in the cloned repository:

```powershell
.\install.ps1
.\run.ps1
```

The installer uses the `uv` Python manager. If it is not already present, Windows `winget`
installs it. The application opens a local browser UI.

### Linux / macOS

Install [`uv`](https://docs.astral.sh/uv/), then:

```bash
chmod +x install.sh run.sh
./install.sh
./run.sh
```

## Layered method command line

```powershell
uv run holoforge "photo.jpg" --method layered --preset musubi --inpaint generative
```

Useful alternatives:

```powershell
# Quick diagnostic without diffusion
uv run holoforge "photo.jpg" --method layered --preset preview --inpaint fast

# Preserve the full image over a blurred ambient field
uv run holoforge "photo.jpg" --method layered --layout ambient

# Override a future official display grid without changing code
uv run holoforge "photo.jpg" --method layered --columns 10 --rows 6 --quilt-size 8192 --aspect 0.5625
```

Each run creates a folder under `outputs/` containing:

- the named quilt;
- a left-to-right wiggle GIF;
- the centre view;
- explicit left/right endpoint images and a side-by-side endpoint audit;
- depth, face, generation-mask, parallax-reveal, and generated-backplate diagnostics;
- `conversion.json` with every setting required to reproduce the run.

## Musubi status

Looking Glass publicly specifies musubi as a portrait 9:16 HLD device and says its own
converter uses local AI-powered Gaussian splatting. As of this prototype, Looking Glass
has not published a final developer quilt grid for musubi. The `musubi` preset therefore
uses the 10×6 grid seen in current Blocks exports with the published 9:16 content aspect;
all four values are CLI-overridable once an official specification appears.

The standard presets use Looking Glass' published layouts:

- Portrait: 8×6 at 3360×3360, aspect 0.75
- Looking Glass 16: 5×9 at 4096×4096, aspect 1.777…

## Honest limits

One photograph does not contain ground-truth pixels from behind a person, nor the true
profile of a face. HoloForge generates a plausible and view-consistent 2.5D reconstruction;
it cannot match a calibrated multi-camera light-field capture in geometric truth. Keep the
view cone conservative. Large rotations expose invented structure and eventually reveal
that the representation is layered rather than a complete 3D scan.

For irreplaceable family photographs, retain the source and inspect `wiggle.gif` before
sharing the quilt. Generated hidden detail is an interpretation, not historical evidence.

## Model licences

- HoloForge source: Apache-2.0.
- MiniMax H3 weights and their quantised repacks, and the 360° orbit LoRA trained on them:
  MiniMax H3 Community License, which excludes the EU, UK, South Korea and USA unless
  MiniMax grants a separate licence.
- Depth Anything V2 **Large** weights: CC-BY-NC-4.0 (non-commercial). Small is Apache-2.0
  and can be selected by changing `depth_model` to
  `depth-anything/Depth-Anything-V2-Small-hf`.
- SDXL inpainting weights: use is governed by the model card and OpenRAIL-family licence
  downloaded with the model.
- LaMa object-removal weights and the small TorchScript wrapper source are Apache-2.0.

The weights are not redistributed by this repository; Hugging Face downloads them into
the user's local cache on first use.

The public demo photograph and generated preview are not included in the HoloForge
software licence; their source and rights information are given above.

## Development

```powershell
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```
