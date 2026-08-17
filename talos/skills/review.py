"""Looking at uncommitted changes.

This does not review anything — it fetches the diff and hands it to the model,
which does the reviewing. Worth being clear about what that is worth: an 8B
model catches typos, a forgotten null check, a name that does not match its
neighbours. It does not catch a race condition, a leak, or a design that will
hurt in six months.

Useful for "is there anything obviously wrong here". Not a reason to skip
reading it yourself.

Diffs are truncated, because the reply is spoken and the context window is
4k. A review of the first two hundred lines, honestly labelled, beats a
context window full of diff and no room left to think in.
"""

from __future__ import annotations

from pathlib import Path

from talos.skills._run import git
from talos.skills._workspace import OutsideWorkspace, describe, resolve
from talos.skills.base import Permission, skill

MAX_DIFF_LINES = 200
MAX_DIFF_CHARS = 6000


_git = git


def _project(name: str) -> tuple[Path | None, str | None]:
    name = name.strip()
    if not name:
        return None, "Which project?"
    try:
        path = resolve(name)
    except OutsideWorkspace as exc:
        return None, str(exc)
    if not path.is_dir():
        return None, f"There's no project called '{name}'. Try list_projects."
    if not (path / ".git").exists():
        return None, f"{describe(path)} isn't a git repository, so there's no diff to read."
    return path, None


def _trim(diff: str) -> tuple[str, bool]:
    lines = diff.splitlines()
    truncated = False
    if len(lines) > MAX_DIFF_LINES:
        lines = lines[:MAX_DIFF_LINES]
        truncated = True
    text = "\n".join(lines)
    if len(text) > MAX_DIFF_CHARS:
        text = text[:MAX_DIFF_CHARS]
        truncated = True
    return text, truncated


@skill(Permission.OPEN)
def review_changes(project: str, staged: bool = False) -> str:
    """Fetch the uncommitted changes in a project so they can be looked over.

    Args:
        project: Folder name of the project
        staged: Look at staged changes instead of unstaged ones
    """
    repo, refusal = _project(project)
    if refusal:
        return refusal

    args = ["diff", "--staged"] if staged else ["diff"]
    code, diff = _git(repo, *args)
    if code != 0:
        return f"Could not read the diff for {project}."

    if not diff:
        which = "staged" if staged else "unstaged"
        return f"{describe(repo)} has no {which} changes."

    text, truncated = _trim(diff)
    note = (
        "\n\n[Truncated — this is the start of a larger diff.]" if truncated else ""
    )
    return (
        f"Uncommitted changes in {describe(repo)}:\n\n{text}{note}\n\n"
        f"Look these over and say what stands out. Be specific about lines. "
        f"If nothing stands out, say so rather than inventing something."
    )


@skill(Permission.OPEN)
def review_last_commit(project: str) -> str:
    """Fetch the most recent commit in a project so it can be looked over.

    Args:
        project: Folder name of the project
    """
    repo, refusal = _project(project)
    if refusal:
        return refusal

    code, subject = _git(repo, "log", "-1", "--format=%s%n%n%b")
    if code != 0:
        return f"Could not read the last commit in {project}."

    code, diff = _git(repo, "show", "--format=", "HEAD")
    if code != 0:
        return f"Could not read the last commit's changes in {project}."

    text, truncated = _trim(diff)
    note = "\n\n[Truncated.]" if truncated else ""
    return (
        f"Last commit in {describe(repo)}:\n\n{subject}\n\n{text}{note}\n\n"
        f"Say whether the message matches what the code actually does."
    )


# There was a skill here that checked every project for uncommitted work. It
# is gone on purpose.
#
# It multiplied a per-repository cost by thirty-seven, in a folder where a
# single git status can block for minutes on a OneDrive placeholder. Bounding
# the timeout did not save it: the interesting repositories are exactly the
# slow ones, so the honest answer became "most of them were too slow to
# check", which is worse than not asking.
#
# `project_status` answers the same question for one project, quickly, and
# that is the question anyone actually has.
