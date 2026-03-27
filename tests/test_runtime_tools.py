import os
import stat
from pathlib import Path

import pytesseract

from graphai.core.common.runtime_tools import configure_runtime_external_tools


def _make_executable(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_configure_runtime_external_tools_resolves_all_required_tools_from_env(tmp_path, monkeypatch):
    tools_dir = tmp_path / "tools"
    ffmpeg = tools_dir / "ffmpeg"
    ffprobe = tools_dir / "ffprobe"
    tesseract = tools_dir / "tesseract"
    pdftoppm = tools_dir / "pdftoppm"
    pdfinfo = tools_dir / "pdfinfo"
    magick = tools_dir / "magick"

    for tool in (ffmpeg, ffprobe, tesseract, pdftoppm, pdfinfo, magick):
        _make_executable(tool)

    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("FFMPEG_PATH", str(ffmpeg))
    monkeypatch.setenv("FFPROBE_PATH", str(ffprobe))
    monkeypatch.setenv("TESSERACT_PATH", str(tesseract))
    monkeypatch.setenv("PDFTOPPM_PATH", str(pdftoppm))
    monkeypatch.setenv("PDFINFO_PATH", str(pdfinfo))
    monkeypatch.setenv("MAGICK_PATH", str(magick))

    resolved = configure_runtime_external_tools()

    assert resolved["ffmpeg"] == str(ffmpeg)
    assert resolved["ffprobe"] == str(ffprobe)
    assert resolved["tesseract"] == str(tesseract)
    assert resolved["pdftoppm"] == str(pdftoppm)
    assert resolved["pdfinfo"] == str(pdfinfo)
    assert resolved["magick"] == str(magick)
    assert resolved["poppler_path"] == str(tools_dir)
    assert pytesseract.pytesseract.tesseract_cmd == str(tesseract)


def test_configure_runtime_external_tools_finds_user_local_bins(tmp_path, monkeypatch):
    home_dir = tmp_path / "home"
    local_bin = home_dir / ".local" / "bin"
    mamba_bin = home_dir / ".local" / "share" / "mamba" / "bin"

    # Split across both candidate paths to validate fallback behavior.
    _make_executable(local_bin / "ffmpeg")
    _make_executable(local_bin / "ffprobe")
    _make_executable(mamba_bin / "tesseract")
    _make_executable(mamba_bin / "pdftoppm")
    _make_executable(mamba_bin / "pdfinfo")
    _make_executable(mamba_bin / "magick")

    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("PATH", "/usr/bin")
    for env_name in (
        "FFMPEG_PATH",
        "FFPROBE_PATH",
        "TESSERACT_PATH",
        "PDFTOPPM_PATH",
        "PDFINFO_PATH",
        "MAGICK_PATH",
    ):
        monkeypatch.delenv(env_name, raising=False)

    resolved = configure_runtime_external_tools()

    assert resolved["ffmpeg"] == str(local_bin / "ffmpeg")
    assert resolved["ffprobe"] == str(local_bin / "ffprobe")
    assert resolved["tesseract"] == str(mamba_bin / "tesseract")
    assert resolved["pdftoppm"] == str(mamba_bin / "pdftoppm")
    assert resolved["pdfinfo"] == str(mamba_bin / "pdfinfo")
    assert resolved["magick"] == str(mamba_bin / "magick")
    assert resolved["poppler_path"] == str(mamba_bin)

    current_path = os.environ.get("PATH", "")
    assert str(local_bin) in current_path
    assert str(mamba_bin) in current_path
