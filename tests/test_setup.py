import os
import subprocess
from pathlib import Path

import pytest

from holoforge import portrait, setup


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOLOFORGE_HOME", str(tmp_path / "home"))
    return tmp_path


def test_config_round_trip_keeps_earlier_values() -> None:
    setup.save_config(comfy_dir="C:/ComfyUI", comfy_managed=False)
    setup.save_config(ffmpeg="C:/tools/ffmpeg.exe", comfy_python=None)
    assert setup.load_config() == {
        "comfy_dir": "C:/ComfyUI",
        "comfy_managed": False,
        "ffmpeg": "C:/tools/ffmpeg.exe",
    }


def test_configured_ffmpeg_wins(tmp_path: Path) -> None:
    ffmpeg = tmp_path / "my-ffmpeg.exe"
    ffmpeg.write_bytes(b"")
    setup.save_config(ffmpeg=str(ffmpeg))
    assert setup.ensure_ffmpeg(download=False) == ffmpeg


def test_missing_ffmpeg_explains_how_to_get_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup, "managed_ffmpeg", lambda: Path("missing/ffmpeg.exe"))
    monkeypatch.setattr(setup.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="holoforge-setup"):
        setup.ensure_ffmpeg(download=False)


def test_finds_comfy_python_in_venv_or_portable_build(tmp_path: Path) -> None:
    exe = "python.exe" if os.name == "nt" else "python"
    sub = "Scripts" if os.name == "nt" else "bin"
    comfy = tmp_path / "ComfyUI_windows_portable" / "ComfyUI"
    comfy.mkdir(parents=True)
    assert setup.find_comfy_python(comfy) is None
    (comfy.parent / "python_embeded").mkdir()
    (comfy.parent / "python_embeded" / exe).write_bytes(b"")
    assert setup.find_comfy_python(comfy) == comfy.parent / "python_embeded" / exe
    (comfy / ".venv" / sub).mkdir(parents=True)
    (comfy / ".venv" / sub / exe).write_bytes(b"")
    assert setup.find_comfy_python(comfy) == comfy / ".venv" / sub / exe


def test_start_on_demand_needs_a_configured_comfyui(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup, "comfy_is_running", lambda url: False)
    started = []
    monkeypatch.setattr(setup.subprocess, "Popen", lambda *a, **k: started.append(a))
    assert setup.start_comfyui_if_needed("http://127.0.0.1:8188") is False
    assert started == []
    assert "holoforge-setup" in setup.comfy_status("http://127.0.0.1:8188")


def test_start_on_demand_is_a_no_op_when_running(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup, "comfy_is_running", lambda url: True)
    assert setup.start_comfyui_if_needed("http://127.0.0.1:8188") is True
    assert "running" in setup.comfy_status("http://127.0.0.1:8188")


def test_start_on_demand_launches_main_py(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    (comfy / "main.py").write_text("", encoding="utf-8")
    python = tmp_path / "python.exe"
    python.write_bytes(b"")
    setup.save_config(comfy_dir=str(comfy), comfy_python=str(python))
    answers = iter([False, True])
    monkeypatch.setattr(setup, "comfy_is_running", lambda url: next(answers))
    launched = {}

    class FakeProcess:
        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        launched.update(command=command, cwd=kwargs["cwd"])
        return FakeProcess()

    monkeypatch.setattr(setup.subprocess, "Popen", fake_popen)
    assert setup.start_comfyui_if_needed("http://127.0.0.1:8190", timeout=5) is True
    assert launched["command"] == [
        str(python),
        "main.py",
        "--listen",
        "127.0.0.1",
        "--port",
        "8190",
    ]
    assert launched["cwd"] == str(comfy)


def test_hevc_encoder_prefers_free_x265(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = " V....D libx265  libx265 H.265 / HEVC\n V....D libkvazaar  kvazaar\n"
    monkeypatch.setattr(
        portrait.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=listing, stderr=""),
    )
    args = portrait.encoder_args("ffmpeg")
    assert args[:2] == ["-c:v", "libx265"] and "hev1" in args and "18" in args


def test_gpu_claim_waits_for_go_and_releases_after_outer_session(tmp_path: Path) -> None:
    request, go = tmp_path / "request.flag", tmp_path / "go.flag"
    setup.save_config(gpu_request_flag=str(request), gpu_go_flag=str(go))
    go.touch()  # the other pipeline has already paused
    with setup.gpu_session():
        with setup.gpu_session():
            setup.claim_gpu(timeout=5)
            assert request.exists()
        assert request.exists()  # still held: the outer session (a batch) is open
    assert not request.exists()


def test_gpu_claim_is_skipped_without_configured_flags(tmp_path: Path) -> None:
    with setup.gpu_session():
        setup.claim_gpu(timeout=1)
    assert list(tmp_path.glob("*.flag")) == []
