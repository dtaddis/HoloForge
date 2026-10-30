import json
from pathlib import Path

import pytest
from PIL import Image
from test_orbit import synthetic_orbit

from holoforge import orbit_pipeline
from holoforge.config import OrbitConfig
from holoforge.orbit import OrbitSettings
from holoforge.orbit_pipeline import OrbitPipeline


@pytest.fixture
def stub_h3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the pipeline on a synthetic clip, without ComfyUI, a GPU claim or ffmpeg."""
    monkeypatch.setenv("HOLOFORGE_HOME", str(tmp_path / "home"))
    frames = synthetic_orbit(61, 24)

    def generate_orbit(source, directory, settings, **_):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "orbit.json").write_text("{}", encoding="utf-8")
        return frames, {"signature": {"lora": settings.lora}}

    def encode(_, path, *args):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"mp4")
        return Path(path)

    monkeypatch.setattr(orbit_pipeline, "generate_orbit", generate_orbit)
    monkeypatch.setattr(orbit_pipeline, "save_frames_video", encode)
    monkeypatch.setattr(orbit_pipeline, "save_sweep_video", encode)
    photo = tmp_path / "photo.png"
    frames[0].save(photo)
    return photo


def convert(photo: Path, looking_glass: bool):
    cfg = OrbitConfig(
        output_dir=photo.parent / "out",
        flat_retry=False,
        portrait_recording=False,
        looking_glass=looking_glass,
    ).with_preset_overrides(columns=3, rows=3, quilt_size=300)
    return OrbitPipeline(cfg, OrbitSettings()).convert(photo)


def test_orbit_only_saves_views_and_videos_without_a_quilt(stub_h3: Path) -> None:
    result = convert(stub_h3, looking_glass=False)
    assert result.quilt_path is None and result.recording_path is None
    assert not list(result.metadata_path.parent.glob("*_quilt_*"))
    assert [p.name for p in result.view_paths] == [f"photo_view_{i:02d}.jpg" for i in range(9)]
    # Views keep the clip's own framing rather than being fitted to a display.
    assert Image.open(result.view_paths[0]).size == (160, 120)
    assert result.orbit_video_path.exists() and result.sweep_video_path.exists()
    meta = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert meta["quilt"] is None and len(meta["view_images"]) == 9


def test_looking_glass_output_is_unchanged(stub_h3: Path) -> None:
    result = convert(stub_h3, looking_glass=True)
    assert result.quilt_path.name == "photo_quilt_qs3x3a0.75.jpg"
    assert result.view_paths == [] and result.sweep_video_path is None
