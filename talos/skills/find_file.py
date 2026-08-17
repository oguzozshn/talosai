"""Finding files and reading what is in them."""

from __future__ import annotations

from talos.skills._workspace import (
    WORKSPACES,
    OutsideWorkspace,
    SecretFile,
    describe,
    resolve,
)
from talos.skills.base import Permission, skill

# Directories that hold thousands of files nobody is ever looking for. Walking
# them makes every search slow and buries the real results.
IGNORED = {
    ".git", "node_modules", "bin", "obj", "__pycache__", ".vs", ".venv",
    "venv", "dist", "build", "packages", ".idea",
}

MAX_READ_CHARS = 4000


@skill(Permission.OPEN)
def find_file(pattern: str, limit: int = 10) -> str:
    """Search the projects folder for files whose name matches a pattern.

    Args:
        pattern: Name or glob to look for, such as README* or *.csproj
        limit: How many results to return at most
    """
    if "*" not in pattern and "?" not in pattern:
        pattern = f"*{pattern}*"

    matches = []
    for root in WORKSPACES:
        for path in root.rglob(pattern):
            if any(part in IGNORED for part in path.parts):
                continue
            if path.is_file():
                matches.append(path)
                if len(matches) >= limit:
                    break
        if len(matches) >= limit:
            break

    if not matches:
        return f"No file matching '{pattern}' anywhere I can look."

    lines = [f"{len(matches)} match(es) for '{pattern}':"]
    lines += [f"  {describe(p)}" for p in matches]
    return "\n".join(lines)


@skill(Permission.OPEN)
def read_file(path: str) -> str:
    """Read a text file from the projects folder and return its contents.

    Args:
        path: Path to the file, relative to the projects folder
    """
    try:
        target = resolve(path)
    except (OutsideWorkspace, SecretFile) as exc:
        return str(exc)

    if not target.is_file():
        return f"'{path}' is not a file. Try find_file to locate it first."

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"Could not read '{path}': {exc}"

    if len(text) > MAX_READ_CHARS:
        # The reply is spoken aloud, and the context window is 4k. Neither
        # survives a whole source file, so truncate here rather than letting
        # it crowd out the conversation.
        text = text[:MAX_READ_CHARS] + f"\n\n[truncated, {len(text)} characters total]"
    return f"{describe(target)}:\n\n{text}"
