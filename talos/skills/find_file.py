"""Finding files and reading what is in them."""

from __future__ import annotations

import fnmatch
import re
import subprocess
import time
from pathlib import Path

from talos.skills import _index, _workspace
from talos.skills._workspace import (
    OutsideWorkspace,
    SecretFile,
    describe,
    is_secret,
    reach,
    resolve,
    search_roots,
)
from talos.skills.base import Permission, skill

# Directories that hold thousands of files nobody is ever looking for. Walking
# them makes every search slow and buries the real results.
IGNORED = {
    ".git", "node_modules", "bin", "obj", "__pycache__", ".vs", ".venv",
    "venv", "dist", "build", "packages", ".idea", "PackageCache",
}

MAX_READ_CHARS = 4000


# Results read back. Not the model's choice: given a limit, it asked for one
# result and read the README of whichever project happened to come first.
LIMIT = 8

# Walking the directories without the index is slow enough on the desktop
# (15 to 42 seconds) that it has to stop somewhere and say so.
WALK_SECONDS = 8


def _walk(pattern: str, roots: tuple[Path, ...]) -> tuple[list[Path], bool]:
    """Matches found by walking, and whether the walk finished in time."""
    deadline = time.monotonic() + WALK_SECONDS
    found: list[Path] = []
    for root in roots:
        for path in root.rglob(pattern):
            if time.monotonic() > deadline:
                return found, False
            found.append(path)
    return found, True


def _rank(path: Path, wanted: str) -> tuple:
    # An exact name beats a prefix beats a match anywhere; then folders, which
    # are what "where is" usually means; then the shallower path.
    name = path.stem.casefold() if path.is_file() else path.name.casefold()
    if name == wanted:
        closeness = 0
    elif name.startswith(wanted):
        closeness = 1
    elif re.search(rf"(?<![^\W_]){re.escape(wanted)}", name):
        # The start of a word inside the name: "notes RAG" for "rag", but
        # not "drag" or "storage".
        closeness = 2
    else:
        closeness = 3
    return closeness, not path.is_dir(), len(path.parts)


def _near_the_top() -> list[Path]:
    """Everything directly inside each root, and inside each project.

    Looked at directly, whatever the index says. It is a few hundred names,
    and they are the folders and project files people ask about most — the
    index had skipped the Talos project, so "the README in Talos" read some
    other project's README instead.
    """
    near: list[Path] = []
    for root in reach():
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        near += children
        if root in _workspace.WORKSPACES:
            for project in children:
                if project.is_dir() and project.name not in IGNORED:
                    try:
                        near += list(project.iterdir())
                    except OSError:
                        continue
    return near


@skill(Permission.OPEN)
def find_file(pattern: str) -> str:
    """Find files or folders by name, in the projects and anywhere on the desktop. Use for "where is".

    Args:
        pattern: A name, a few words from it, or a glob, such as Dersler, README.md or *.csproj; put a folder first to look only inside it, as Talos/README.md
    """
    # "Talos/README.md": the last part is the name, the rest are folders the
    # match must be inside. Without this, the README of whichever of 36
    # projects ranked first was read for "the README in Talos".
    *inside, pattern = [part for part in re.split(r"[\\/]", pattern.strip()) if part]
    inside = [part.casefold() for part in inside]
    words = re.findall(r"\w+", pattern.casefold())
    # Several words are matched in any order, so "motivation letter" finds
    # "Letter of motivation". The index is asked for the longest of them.
    several = "*" not in pattern and "?" not in pattern and len(words) > 1
    if several:
        glob = f"*{max(words, key=len)}*"
    elif "*" not in pattern and "?" not in pattern:
        glob = f"*{pattern}*"
    else:
        glob = pattern
    wanted = pattern.strip("*").casefold()

    near = _near_the_top()
    roots = search_roots()
    if inside:
        # Searched from the named folder itself when it is one of those near
        # the top. From the desktop, the index's first few hundred PDFs did
        # not include a single one in Dersler.
        named = tuple(p for p in near if p.is_dir() and p.name.casefold() == inside[0])
        roots = named or roots
    found = _index.find(glob, roots)
    complete = True
    if found is None:
        found, complete = _walk(glob, roots)
    for path in near:
        if fnmatch.fnmatch(path.name.casefold(), glob.casefold()) and path not in found:
            found.append(path)

    matches = [
        p for p in found
        if not any(part in IGNORED for part in p.parts) and not is_secret(p)
        and (not several or all(w in p.name.casefold() for w in words))
        and all(folder in {q.casefold() for q in p.parent.parts} for folder in inside)
    ]
    if not matches:
        if not complete:
            return f"Nothing matching '{pattern}' turned up before I had to stop looking."
        # Most names here are Turkish. Asked for "motivation letter", the
        # model searched in English and gave up; told this, it can try again.
        return (
            f"No file or folder matching '{pattern}' anywhere I can look. Many names "
            f"here are Turkish — try the Turkish word, or one word on its own."
        )

    matches.sort(key=lambda p: _rank(p, wanted))
    shown = matches[:LIMIT]
    lines = [f"{len(matches)} match(es) for '{pattern}':"]
    lines += [f"  {describe(p)}{' (folder)' if p.is_dir() else ''}" for p in shown]
    if not complete:
        lines.append("The search was cut short, so there may be more.")
    return "\n".join(lines)


@skill(Permission.OPEN)
def read_file(path: str) -> str:
    """Read a text file from the projects or the desktop and return its contents.

    Args:
        path: Path to the file, as find_file gave it
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


# OPEN: it changes nothing and reads nothing aloud. A file is shown selected in
# its folder, never opened — opening runs whatever program the file belongs
# to, and for a .cmd or .exe that program is the file itself.
@skill(Permission.OPEN)
def show_in_explorer(path: str) -> str:
    """Open a folder in Explorer, or show a file selected in its folder, so the owner can see it.

    Args:
        path: The folder or file, as find_file gave it
    """
    try:
        # Showing where a file is reads nothing from it, so credentials are
        # not refused here; reading them still is.
        target = resolve(path, allow_secret=True)
    except OutsideWorkspace as exc:
        return str(exc)

    if not target.exists():
        return f"'{path}' doesn't exist. Try find_file to locate it first."

    if target.is_dir():
        subprocess.Popen(["explorer", str(target)])
        return f"Opened {describe(target)} in Explorer."
    subprocess.Popen(["explorer", f"/select,{target}"])
    return f"Showing {describe(target)} in its folder."
