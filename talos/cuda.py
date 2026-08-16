"""Make the CUDA libraries shipped in the venv findable by CTranslate2.

Without this, `faster-whisper` fails at the first inference call with

    RuntimeError: Library cublas64_12.dll is not found or cannot be loaded

which is a confusing place for it to appear, because everything up to that
point looks healthy: `nvidia-smi` reports the card, and
`ctranslate2.get_cuda_device_count()` returns 1. That call only asks the driver
whether a device exists. Actually running an encoder needs cuBLAS and cuDNN,
and those are separate libraries that Windows will not find on its own.

We install them as pip packages (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`)
rather than requiring the full CUDA Toolkit, which keeps everything inside the
virtual environment and off the system. The cost is that their DLLs land in
site-packages, somewhere Windows does not search — hence this module.

Import order matters: call `register_cuda_dll_directories()` *before* importing
`faster_whisper` or `ctranslate2`. The search path is read when the extension
module loads, so registering afterwards has no effect.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Relative to site-packages. Both packages follow the same layout.
_DLL_SUBPATHS = (
    Path("nvidia") / "cublas" / "bin",
    Path("nvidia") / "cudnn" / "bin",
)

_registered = False


def _site_packages() -> Path:
    return Path(sys.prefix) / "Lib" / "site-packages"


def register_cuda_dll_directories() -> list[Path]:
    """Add the bundled CUDA library directories to the DLL search path.

    Returns the directories that were registered. An empty list means the
    packages are not installed, in which case CTranslate2 falls back to CPU —
    slower, but not broken.

    Safe to call more than once; only the first call does anything.
    """
    global _registered
    if _registered or sys.platform != "win32":
        return []

    registered: list[Path] = []
    for subpath in _DLL_SUBPATHS:
        directory = _site_packages() / subpath
        if not directory.is_dir():
            continue
        try:
            os.add_dll_directory(str(directory))
        except OSError:
            # Directory disappeared between the check and the call, or the
            # path is malformed. Not worth failing over — the caller finds out
            # soon enough by falling back to CPU.
            continue
        registered.append(directory)

    # add_dll_directory alone is not enough. It only extends the search used by
    # LoadLibraryEx when called with the LOAD_LIBRARY_SEARCH_* flags, and
    # CTranslate2 resolves cuBLAS with a plain load that ignores it — so the
    # first inference still dies with "cublas64_12.dll is not found" even
    # though the directory was registered successfully. PATH is consulted by
    # both, so setting it as well is what actually fixes it.
    if registered:
        os.environ["PATH"] = os.pathsep.join(
            [*(str(d) for d in registered), os.environ.get("PATH", "")]
        )

    _registered = True
    return registered


def describe() -> str:
    """One line on what was found, for diagnostics."""
    missing = [
        str(subpath) for subpath in _DLL_SUBPATHS if not (_site_packages() / subpath).is_dir()
    ]
    if not missing:
        return "bundled CUDA libraries present"
    return (
        f"missing: {', '.join(missing)} — "
        "pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
    )
