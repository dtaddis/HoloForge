import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from holoforge import portrait
from holoforge.cli import portrait_main
from holoforge.portrait import (
    Calibration,
    find_holograms,
    playlist_names,
    prepare_holograms,
    push_recordings,
    recording_name,
    remove_from_playlist,
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


def make_device(tmp_path: Path, names: tuple[str, ...] = ()) -> Path:
    device = tmp_path / "device"
    (device / "LKG_calibration").mkdir(parents=True)
    (device / "LKG_calibration" / "visual.json").write_text(json.dumps(VISUAL), encoding="utf-8")
    for name in names:
        (device / recording_name(name)).write_bytes(name.encode())
    (device / "playlist.json").write_text(
        json.dumps([{"filename": n, "last_updated": "never", "needs_sync": False} for n in names]),
        encoding="utf-8",
    )
    (device / "playlist.m3u").write_text(
        "".join(recording_name(n) + "\n" for n in names), encoding="utf-8"
    )
    return device


@pytest.fixture
def fake_render(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stand in for interlacing and encoding; records which quilts were rendered."""
    rendered: list[str] = []

    def prepare_recording(quilt, columns, rows, aspect, calibration, output_dir):
        rendered.append(Path(quilt).name)
        recording = Path(output_dir) / recording_name(Path(quilt).name)
        recording.write_bytes(b"rendered " + Path(quilt).name.encode())
        return recording, None

    monkeypatch.setattr(portrait, "prepare_recording", prepare_recording)
    return rendered


def test_find_holograms_pairs_quilts_with_recordings(tmp_path: Path) -> None:
    run = tmp_path / "results" / "b"
    run.mkdir(parents=True)
    (run / "b_quilt_qs8x6a0.75.jpg").write_bytes(b"q")
    (run / "b_quilt_qs8x6a0.75_lenticular.png").write_bytes(b"preview")
    (run / recording_name("b_quilt_qs8x6a0.75.jpg")).write_bytes(b"r")
    (run / "b_wiggle.gif").write_bytes(b"gif")
    loose = tmp_path / recording_name("a_quilt_qs8x6a0.75.jpg")
    loose.write_bytes(b"r")

    items = find_holograms([tmp_path / "results", loose])

    assert [item.name for item in items] == ["b_quilt_qs8x6a0.75.jpg", "a_quilt_qs8x6a0.75.jpg"]
    assert items[0].quilt == run / "b_quilt_qs8x6a0.75.jpg"
    assert items[0].recording == run / recording_name("b_quilt_qs8x6a0.75.jpg")
    assert items[1].quilt is None and items[1].recording == loose
    with pytest.raises(ValueError):
        find_holograms([run / "b_wiggle.gif"])


def test_ready_recordings_are_not_rendered_again(tmp_path: Path, fake_render: list[str]) -> None:
    quilt = tmp_path / "a_quilt_qs8x6a0.75.jpg"
    quilt.write_bytes(b"q")
    ready = tmp_path / "b_quilt_qs8x6a0.75.jpg"
    ready.write_bytes(b"q")
    (tmp_path / recording_name(ready.name)).write_bytes(b"ready")
    items = find_holograms([quilt, ready])

    entries = prepare_holograms(items, calibration(tmp_path), report=lambda _: None)

    assert fake_render == [quilt.name]
    assert [path.read_bytes() for _, path in entries] == [
        b"rendered " + quilt.name.encode(),
        b"ready",
    ]
    prepare_holograms(items, calibration(tmp_path), rerender=True, report=lambda _: None)
    assert fake_render == [quilt.name, quilt.name, ready.name]


def test_recordings_for_another_portrait_are_rendered_again(
    tmp_path: Path, fake_render: list[str]
) -> None:
    quilt = tmp_path / "a_quilt_qs8x6a0.75.jpg"
    quilt.write_bytes(b"q")
    recording = tmp_path / recording_name(quilt.name)
    recording.write_bytes(b"old")
    meta = {"recording": str(recording.resolve()), "calibration": {"serial": "LKG-PORT-00001"}}
    (tmp_path / "conversion.json").write_text(json.dumps(meta), encoding="utf-8")

    prepare_holograms(find_holograms([quilt]), calibration(tmp_path), report=lambda _: None)

    assert fake_render == [quilt.name]
    meta = json.loads((tmp_path / "conversion.json").read_text(encoding="utf-8"))
    assert meta["calibration"]["serial"] == VISUAL["serial"]
    prepare_holograms(find_holograms([quilt]), calibration(tmp_path), report=lambda _: None)
    assert fake_render == [quilt.name]


def test_push_skips_identical_recordings_and_leaves_no_partial_files(tmp_path: Path) -> None:
    device = make_device(tmp_path, ("a.jpg",))
    same = tmp_path / "same.mp4"
    same.write_bytes(b"a.jpg")
    new = tmp_path / "new.mp4"
    new.write_bytes(b"new")
    before = (device / recording_name("a.jpg")).stat().st_mtime_ns

    push_recordings(device, [("a.jpg", same), ("b.jpg", new)])

    assert (device / recording_name("a.jpg")).stat().st_mtime_ns == before
    assert (device / recording_name("b.jpg")).read_bytes() == b"new"
    assert not list(device.glob("*.partial"))
    assert playlist_names(device) == ["a.jpg", "b.jpg"]


def test_remove_from_playlist(tmp_path: Path) -> None:
    device = make_device(tmp_path, ("a.jpg", "b.jpg", "c.jpg"))

    removed = remove_from_playlist(device, ["b.jpg", recording_name("c.jpg"), "zzz.jpg"])

    assert removed == ["b.jpg", "c.jpg"]
    assert playlist_names(device) == ["a.jpg"]
    lines = (device / "playlist.m3u").read_text(encoding="utf-8").split()
    assert lines == [recording_name("a.jpg")]
    assert sorted(p.name for p in device.glob("*__lkgrec.mp4")) == [recording_name("a.jpg")]


def test_portrait_command_copies_a_results_folder(
    tmp_path: Path, fake_render: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    device = make_device(tmp_path, ("old.jpg",))
    run = tmp_path / "outputs" / "photo"
    run.mkdir(parents=True)
    (run / "photo_quilt_qs8x6a0.75.jpg").write_bytes(b"q")

    portrait_main(["--device", str(device), str(tmp_path / "outputs")])

    assert fake_render == ["photo_quilt_qs8x6a0.75.jpg"]
    assert playlist_names(device) == ["photo_quilt_qs8x6a0.75.jpg", "old.jpg"]
    assert "2 item(s)" in capsys.readouterr().out
    portrait_main(["--device", str(device), "--remove", "old.jpg"])
    assert playlist_names(device) == ["photo_quilt_qs8x6a0.75.jpg"]
