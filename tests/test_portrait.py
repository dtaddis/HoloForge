import json
from pathlib import Path

import numpy as np
from PIL import Image

from holoforge.portrait import (
    Calibration,
    push_recordings,
    recording_name,
    render_lenticular,
    replace_playlist,
)

# The Portrait (LKG-PORT-03650) whose uniforms Looking Glass Studio logged while syncing.
VISUAL = {
    "configVersion": "1.0",
    "serial": "LKG-PORT-03650",
    "pitch": {"value": 52.56363856600123},
    "slope": {"value": -7.086395566007788},
    "center": {"value": 0.6487457606842117},
    "invView": {"value": 1},
    "DPI": {"value": 324.0},
    "screenW": {"value": 1536.0},
    "screenH": {"value": 2048.0},
}


def calibration(tmp_path: Path) -> Calibration:
    path = tmp_path / "visual.json"
    path.write_text(json.dumps(VISUAL), encoding="utf-8")
    return Calibration.from_file(path)


def test_uniforms_match_looking_glass_studio(tmp_path: Path) -> None:
    cal = calibration(tmp_path)
    assert abs(cal.shader_pitch - 246.7459) < 1e-3
    assert abs(cal.shader_slope - -0.18815394) < 1e-6
    assert abs(cal.subpixel_size - 0.00021701389) < 1e-10
    assert cal.aspect == 0.75


def test_uniform_quilt_renders_uniformly(tmp_path: Path) -> None:
    quilt = Image.new("RGB", (64, 48), (200, 100, 50))
    out = render_lenticular(quilt, 8, 6, 0.75, calibration(tmp_path), width=96, height=64)
    array = np.asarray(out).astype(int)
    assert out.size == (96, 64)
    assert np.abs(array - [200, 100, 50]).max() <= 1


def test_each_subpixel_comes_from_one_view(tmp_path: Path) -> None:
    # Give every view a distinct grey level; away from view boundaries each output
    # subpixel must reproduce one of them.
    columns, rows = 8, 6
    quilt = np.zeros((rows * 8, columns * 8, 3), np.uint8)
    for view in range(columns * rows):
        col, row_from_bottom = view % columns, view // columns
        top = (rows - 1 - row_from_bottom) * 8
        quilt[top : top + 8, col * 8 : col * 8 + 8] = 5 * view
    out = render_lenticular(
        Image.fromarray(quilt), columns, rows, 0.75, calibration(tmp_path), width=60, height=40
    )
    values = np.asarray(out).astype(int)
    levels = 5 * np.arange(columns * rows)
    nearest = np.abs(values[..., None] - levels).min(-1)
    # Blends between neighbouring views stay within one step of a level.
    assert nearest.max() <= 5


def test_push_recordings_prepends_and_dedupes(tmp_path: Path) -> None:
    device = tmp_path / "device"
    device.mkdir()
    (device / "playlist.json").write_text(
        json.dumps([{"filename": "a.mp4", "last_updated": "never", "needs_sync": False}]),
        encoding="utf-8",
    )
    (device / "playlist.m3u").write_text("a.mp4__lkgrec.mp4\n", encoding="utf-8")
    recording = tmp_path / "x.mp4"
    recording.write_bytes(b"video")

    push_recordings(device, [("photo_qs8x6a0.75.jpg", recording)])
    push_recordings(device, [("photo_qs8x6a0.75.jpg", recording)])

    playlist = json.loads((device / "playlist.json").read_text(encoding="utf-8"))
    assert [item["filename"] for item in playlist] == ["photo_qs8x6a0.75.jpg", "a.mp4"]
    lines = (device / "playlist.m3u").read_text(encoding="utf-8").split()
    assert lines == [recording_name("photo_qs8x6a0.75.jpg"), "a.mp4__lkgrec.mp4"]
    assert (device / recording_name("photo_qs8x6a0.75.jpg")).read_bytes() == b"video"
    assert (device / "playlist.m3u.holoforge-backup").read_text(encoding="utf-8").strip() == (
        "a.mp4__lkgrec.mp4"
    )


def test_replace_playlist_keeps_only_the_given_recordings(tmp_path: Path) -> None:
    device = tmp_path / "device"
    (device / "LKG_calibration").mkdir(parents=True)
    (device / "LKG_calibration" / "visual.json").write_text("{}", encoding="utf-8")
    (device / "old.png__lkgrec.mp4").write_bytes(b"old")
    (device / "playlist.json").write_text(
        json.dumps([{"filename": "old.png", "last_updated": "never", "needs_sync": False}]),
        encoding="utf-8",
    )
    (device / "playlist.m3u").write_text("old.png__lkgrec.mp4\n", encoding="utf-8")
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"a")
    b.write_bytes(b"b")

    removed = replace_playlist(device, [("b_qs8x6a0.75.jpg", b), ("a_qs8x6a0.75.jpg", a)])

    assert removed == ["old.png__lkgrec.mp4"]
    playlist = json.loads((device / "playlist.json").read_text(encoding="utf-8"))
    assert [item["filename"] for item in playlist] == ["b_qs8x6a0.75.jpg", "a_qs8x6a0.75.jpg"]
    lines = (device / "playlist.m3u").read_text(encoding="utf-8").split()
    assert lines == [recording_name("b_qs8x6a0.75.jpg"), recording_name("a_qs8x6a0.75.jpg")]
    assert sorted(p.name for p in device.glob("*__lkgrec.mp4")) == sorted(lines)
    assert (device / "LKG_calibration" / "visual.json").exists()
