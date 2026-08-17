"""Changing files, reversibly.

An 8B model editing source code by voice will eventually get something wrong,
and a silent wrong edit is the worst kind — you find it days later and cannot
tell when it arrived. So the design here is not "be careful", it is "make every
edit undoable and make the user see it first".

Three guards, in order of how much they matter:

**Git, or nothing.** A file is only edited if it is inside a git repository and
tracked by it. That makes `git diff` the record of everything Talos changed and
`git checkout` the undo. A file git does not know about cannot be recovered,
so it is not touched.

**Exact match, exactly once.** Replacement is on a literal string that must
appear precisely once. No regular expressions, no line numbers, no "the second
one". If the model is vague about what it is changing, the edit is refused
rather than applied to the wrong place.

**Shown, then confirmed.** These are CONFIRM tier, and the confirmation carries
the arguments, so what is approved is the change and not the verb.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from talos.skills._workspace import OutsideWorkspace, describe, resolve
from talos.skills.base import Permission, skill

log = logging.getLogger(__name__)

# Bigger than this and it is not a file anyone dictates an edit to.
MAX_FILE_BYTES = 400_000

# Editing these by voice is a way to lose a repository, not a feature.
FORBIDDEN_PARTS = {".git", "node_modules", ".venv", "venv", "__pycache__"}


def _git(repo: Path, *args: str) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=15,
            encoding="utf-8", errors="replace",
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return proc.returncode, (proc.stdout or "").strip()


def _repo_of(path: Path) -> Path | None:
    code, out = _git(path.parent, "rev-parse", "--show-toplevel")
    return Path(out) if code == 0 and out else None


def _tracked(path: Path) -> bool:
    repo = _repo_of(path)
    if repo is None:
        return False
    code, _ = _git(repo, "ls-files", "--error-unmatch", str(path))
    return code == 0


def _check(path: str, *, must_exist: bool) -> tuple[Path | None, str | None]:
    """Resolve and vet a path. Returns (path, refusal)."""
    try:
        target = resolve(path)
    except OutsideWorkspace as exc:
        return None, str(exc)

    if any(part in FORBIDDEN_PARTS for part in target.parts):
        return None, f"I won't touch anything inside {', '.join(sorted(FORBIDDEN_PARTS))}."

    if must_exist:
        if not target.is_file():
            return None, f"'{path}' is not a file I can find. Try find_file first."
        if target.stat().st_size > MAX_FILE_BYTES:
            return None, f"'{path}' is too large for me to edit safely."
        if not _tracked(target):
            return None, (
                f"'{path}' isn't tracked by git, so I couldn't undo a mistake. "
                f"Commit it first and I'll edit it."
            )
    return target, None


@skill(Permission.CONFIRM)
def edit_file(path: str, find: str, replace: str) -> str:
    """Replace an exact piece of text in a file with something else.

    Args:
        path: File to change, relative to the projects folder
        find: The exact text to replace, which must appear once
        replace: What to put in its place
    """
    target, refusal = _check(path, must_exist=True)
    if refusal:
        return refusal

    try:
        original = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return f"Could not read '{path}': {exc}"

    occurrences = original.count(find)
    if occurrences == 0:
        return f"I couldn't find that text in {describe(target)}, so I changed nothing."
    if occurrences > 1:
        # Guessing which one is how the wrong line gets edited.
        return (
            f"That text appears {occurrences} times in {describe(target)}. "
            f"Give me more of the surrounding line so I know which one."
        )

    try:
        target.write_text(original.replace(find, replace, 1), encoding="utf-8")
    except OSError as exc:
        return f"Could not write '{path}': {exc}"

    line = original[: original.index(find)].count("\n") + 1
    return (
        f"Changed line {line} of {describe(target)}. "
        f"Run git diff to see it, or ask me to undo it."
    )


@skill(Permission.CONFIRM)
def create_file(path: str, content: str) -> str:
    """Create a new file with the given contents.

    Args:
        path: Where to create it, relative to the projects folder
        content: What to put in it
    """
    target, refusal = _check(path, must_exist=False)
    if refusal:
        return refusal

    if target.exists():
        return f"'{path}' already exists. I won't overwrite it — edit it instead."

    if _repo_of(target) is None:
        return f"'{path}' isn't inside a git repository, so I couldn't undo this."

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"Could not create '{path}': {exc}"

    return f"Created {describe(target)}, {len(content)} characters."


@skill(Permission.OWNER)
def undo_changes(path: str) -> str:
    """Discard uncommitted changes to a file, restoring the committed version.

    Args:
        path: File to restore, relative to the projects folder
    """
    target, refusal = _check(path, must_exist=True)
    if refusal:
        return refusal

    repo = _repo_of(target)
    if repo is None:
        return f"'{path}' isn't in a git repository."

    code, status = _git(repo, "status", "--porcelain", str(target))
    if code != 0:
        return f"Could not read git status for '{path}'."
    if not status:
        return f"{describe(target)} has no uncommitted changes to undo."

    # Highest tier, and worth being plain about why: this throws away work,
    # including anything you changed by hand since the last commit — not only
    # what Talos did.
    code, out = _git(repo, "checkout", "--", str(target))
    if code != 0:
        return f"Could not restore '{path}': {out}"
    return f"Restored {describe(target)} to its last committed state."
