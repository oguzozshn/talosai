"""The one directory skills are allowed to touch.

Every skill that reads a path goes through `resolve` here. Not because the
model is malicious, but because it is small and will occasionally produce
`../../..` or an absolute path from somewhere else in a filename, and a file
search that happily walks out of the workspace is a file search that can read
anything on the machine.

The check is done after resolving symlinks, so a link pointing outside the
workspace does not slip through.
"""

from __future__ import annotations

from pathlib import Path

from talos.config import load

_config = load()
WORKSPACE = Path(_config.skills.workspace).expanduser().resolve()


class OutsideWorkspace(ValueError):
    pass


def resolve(relative: str = "") -> Path:
    """Resolve a path inside the workspace, or refuse.

    Accepts paths relative to the workspace. Absolute paths are allowed only
    if they land inside it, so a model that helpfully expands a filename to a
    full path is not punished for it.
    """
    candidate = Path(relative.strip()) if relative else Path()
    base = candidate if candidate.is_absolute() else WORKSPACE / candidate

    try:
        resolved = base.resolve()
    except OSError as exc:
        raise OutsideWorkspace(f"Could not resolve '{relative}': {exc}") from exc

    if resolved != WORKSPACE and WORKSPACE not in resolved.parents:
        raise OutsideWorkspace(
            f"'{relative}' is outside the workspace. Skills can only reach "
            f"inside {WORKSPACE}."
        )
    return resolved


def describe(path: Path) -> str:
    """Path as the user would recognise it — relative to the workspace."""
    try:
        return str(path.relative_to(WORKSPACE))
    except ValueError:
        return str(path)
