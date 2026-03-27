import os
import shutil
from pathlib import Path

import pytesseract


def _candidate_bin_dirs():
    home = Path.home()
    return [
        home / ".local" / "bin",
        home / ".local" / "share" / "mamba" / "bin",
    ]


def _prepend_path_dirs(dirs):
    existing = [str(d) for d in dirs if d.is_dir()]
    if not existing:
        return
    current = os.environ.get("PATH", "")
    path_parts = current.split(":") if current else []
    new_parts = []
    for d in existing:
        if d not in path_parts:
            new_parts.append(d)
    if new_parts:
        os.environ["PATH"] = ":".join(new_parts + path_parts)


def resolve_binary(binary_name, env_var_name=None):
    explicit = os.environ.get(env_var_name) if env_var_name else None
    if explicit and os.path.isfile(explicit) and os.access(explicit, os.X_OK):
        return explicit

    resolved = shutil.which(binary_name)
    if resolved:
        return resolved

    for bin_dir in _candidate_bin_dirs():
        candidate = bin_dir / binary_name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def configure_runtime_external_tools():
    # Make common user-level bin dirs visible even in non-interactive runtimes.
    _prepend_path_dirs(_candidate_bin_dirs())

    resolved = {
        "ffmpeg": resolve_binary("ffmpeg", "FFMPEG_PATH"),
        "ffprobe": resolve_binary("ffprobe", "FFPROBE_PATH"),
        "tesseract": resolve_binary("tesseract", "TESSERACT_PATH"),
        "pdftoppm": resolve_binary("pdftoppm", "PDFTOPPM_PATH"),
        "pdfinfo": resolve_binary("pdfinfo", "PDFINFO_PATH"),
        "magick": resolve_binary("magick", "MAGICK_PATH"),
    }

    # Back-fill env vars so downstream libs/subprocesses inherit deterministic paths.
    for key, env in (
        ("ffmpeg", "FFMPEG_PATH"),
        ("ffprobe", "FFPROBE_PATH"),
        ("tesseract", "TESSERACT_PATH"),
        ("pdftoppm", "PDFTOPPM_PATH"),
        ("pdfinfo", "PDFINFO_PATH"),
        ("magick", "MAGICK_PATH"),
    ):
        if resolved[key] and not os.environ.get(env):
            os.environ[env] = resolved[key]

    # pytesseract supports explicit command override.
    if resolved["tesseract"]:
        pytesseract.pytesseract.tesseract_cmd = resolved["tesseract"]

    pdftoppm = resolved["pdftoppm"]
    pdfinfo = resolved["pdfinfo"]
    if pdftoppm and pdfinfo:
        p1 = str(Path(pdftoppm).parent)
        p2 = str(Path(pdfinfo).parent)
        if p1 == p2:
            resolved["poppler_path"] = p1
        else:
            resolved["poppler_path"] = None
    else:
        resolved["poppler_path"] = None

    return resolved
