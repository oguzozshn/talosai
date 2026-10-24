"""Whether Talos starts at all.

Every other test imports the module it is about, and none of them imports
main.py. So a syntax error there — a stray line break inside an f-string,
which is how this file came to exist — passed all 176 tests while the
assistant could not be launched.

Two checks, because they catch different things. Compiling every file finds a
syntax error anywhere, including in modules main.py only imports inside a
function (wake mode, the device list) and which --help never loads. Running
--help finds what compiling cannot: a top-level import that is broken.
"""

from __future__ import annotations

import os
import py_compile
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "main.py", *sorted((ROOT / "talos").rglob("*.py"))]


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_source_file_compiles(path):
    py_compile.compile(str(path), doraise=True)


def test_main_starts_and_prints_its_help():
    result = subprocess.run(
        [sys.executable, str(ROOT / "main.py"), "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=ROOT,
        timeout=60,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode == 0, result.stderr
    assert "--wake" in result.stdout
