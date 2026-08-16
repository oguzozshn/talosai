"""What state a project is in, according to git."""

from __future__ import annotations

import subprocess
from pathlib import Path

from talos.skills._workspace import OutsideWorkspace, WORKSPACE, describe, resolve
from talos.skills.base import Permission, skill


def _git(repo: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


@skill(Permission.OPEN)
def list_projects(limit: int = 25) -> str:
    """List the projects in the projects folder.

    Args:
        limit: How many to list at most
    """
    try:
        root = resolve()
    except OutsideWorkspace as exc:
        return str(exc)

    names = sorted(
        p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if not names:
        return "The projects folder is empty."

    shown = names[:limit]
    text = f"{len(names)} projects: " + ", ".join(shown)
    if len(names) > limit:
        text += f", and {len(names) - limit} more"
    return text


@skill(Permission.OPEN)
def project_status(project: str) -> str:
    """Report the git branch and uncommitted changes of a project.

    Args:
        project: Folder name of the project, for example RatchetSync
    """
    try:
        repo = resolve(project)
    except OutsideWorkspace as exc:
        return str(exc)

    if not repo.is_dir():
        return f"There is no project folder called '{project}'. Try list_projects."

    if not (repo / ".git").exists():
        return f"{describe(repo)} is not a git repository."

    branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    if branch is None:
        return f"Could not read git status for {describe(repo)}. Is git installed?"

    status = _git(repo, "status", "--porcelain") or ""
    changes = [line for line in status.splitlines() if line.strip()]
    last = _git(repo, "log", "-1", "--format=%s (%cr)") or "no commits yet"

    parts = [f"{describe(repo)} is on branch {branch}."]
    if changes:
        # Split rather than slicing a fixed width. Porcelain status codes are
        # two columns plus a space, but _git strips the whole output, which
        # eats the leading space of the first line and shifts it by one.
        names = [line.split(maxsplit=1)[-1] for line in changes[:3]]
        # Spoken aloud, so a count and a couple of examples beat a file list.
        parts.append(f"{len(changes)} uncommitted change(s),")
        parts.append("for example " + ", ".join(names) + ".")
    else:
        parts.append("Working tree is clean.")
    parts.append(f"Last commit: {last}")
    return " ".join(parts)


@skill(Permission.OPEN)
def recent_work(days: int = 7, limit: int = 10) -> str:
    """Report which projects have been committed to recently.

    Args:
        days: How many days back to look
        limit: How many projects to report at most
    """
    root = WORKSPACE
    found: list[tuple[str, str]] = []

    for path in sorted(root.iterdir()):
        if not path.is_dir() or not (path / ".git").exists():
            continue
        last = _git(path, "log", "-1", f"--since={days}.days.ago", "--format=%s (%cr)")
        if last:
            found.append((path.name, last))
        if len(found) >= limit:
            break

    if not found:
        return f"No commits in any project in the last {days} days."
    return f"Worked on {len(found)} project(s) in the last {days} days: " + "; ".join(
        f"{name} — {last}" for name, last in found
    )
