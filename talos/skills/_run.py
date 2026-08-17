"""Running an external command without the risk of never getting it back.

`subprocess.run(timeout=...)` is not enough on Windows. It kills the process
it started, but not that process's children, and then blocks in
`communicate()` waiting for a pipe that a surviving grandchild still holds
open. The timeout expires and the call hangs anyway.

That is not theoretical here. These projects live in a OneDrive folder with
files-on-demand, so their contents are cloud placeholders rather than real
files. A `git status` touches them, OneDrive starts fetching, and git blocks
— for minutes, sometimes indefinitely. Four git processes were left running
for a quarter of an hour before this was written, each one holding a pipe that
a Python process was politely waiting on.

So: kill the whole tree, and never wait on a pipe after deciding to give up.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15.0


class Timeout(RuntimeError):
    pass


def _kill_tree(process: subprocess.Popen) -> None:
    if sys.platform != "win32":
        process.kill()
        return
    # /T takes the children with it, which is the whole point; /F because a
    # process blocked on a cloud file will not respond to anything gentler.
    subprocess.run(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        capture_output=True,
        timeout=10,
        check=False,
    )


def run(
    *command: str,
    cwd: Path | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> tuple[int, str]:
    """Run a command, returning (exit code, stdout). Raises Timeout if it hangs."""
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (FileNotFoundError, OSError) as exc:
        return 1, str(exc)

    try:
        out, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(process)
        # Deliberately not calling communicate() again. That is exactly the
        # call that hangs — the pipe stays open as long as anything in the
        # tree holds it, and by now we have stopped caring what it would say.
        log.warning("gave up on %s after %.0fs", command[0], timeout)
        raise Timeout(f"{command[0]} took longer than {timeout:.0f} seconds") from None

    return process.returncode, (out or "").strip()


def git(repo: Path, *args: str, timeout: float = DEFAULT_TIMEOUT) -> tuple[int, str]:
    """Run git in a repository. Returns (1, message) rather than raising on timeout."""
    try:
        return run("git", "-C", str(repo), *args, timeout=timeout)
    except Timeout as exc:
        return 1, str(exc)
