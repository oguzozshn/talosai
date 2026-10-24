"""Todoist.

The one thing here that leaves the machine. Everything else in Talos runs
locally by design; this talks to Todoist's servers, which means task text goes
over the network and the assistant stops working when the internet does. That
was a deliberate exception, not an oversight — see the README.

The token is never asked for, stored, or written by Talos. It goes in
`config/local.toml`, which is git-ignored, or in the TODOIST_API_TOKEN
environment variable. Both are places you put it and Talos only reads.

Reading is OPEN; anything that creates or completes a task is CONFIRM. The
permission tiers are about what you cannot trivially undo, and a task
completed in a shared project is seen by other people before you can take it
back.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime

from talos.config import load
from talos.skills._net import ipv4_only
from talos.skills.base import Permission, skill

log = logging.getLogger(__name__)

# The unified v1 API. The older /rest/v2 endpoints now answer 410 Gone —
# found by probing, because with an invalid token a retired endpoint and a
# rejected one look the same from the error message alone: v1 answers 401 as
# it should, v2 answers 410 whatever you send it.
API = "https://api.todoist.com/api/v1"
TIMEOUT = 15

_config = load()


class NoToken(RuntimeError):
    pass


def _token() -> str:
    token = (_config.todoist.token or "").strip() or os.environ.get("TODOIST_API_TOKEN", "").strip()
    if not token:
        raise NoToken(
            "I don't have a Todoist token. Put it in config/local.toml under "
            "[todoist] as token, or in the TODOIST_API_TOKEN environment variable."
        )
    return token


def _call(path: str, method: str = "GET", body: dict | None = None) -> object:
    request = urllib.request.Request(
        f"{API}/{path}",
        method=method,
        data=json.dumps(body).encode("utf-8") if body else None,
        headers={
            "Authorization": f"Bearer {_token()}",
            "Content-Type": "application/json",
        },
    )
    with ipv4_only():
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            raw = response.read()
    return json.loads(raw) if raw else None


def _tasks(payload: object) -> list[dict]:
    """Pull the task list out of a response.

    v1 paginates, returning {"results": [...], "next_cursor": ...} where v2
    returned a bare array. Accepting both means the skill survives whichever
    shape arrives, and one page is more than enough for something read aloud.
    """
    if isinstance(payload, dict):
        payload = payload.get("results", [])
    return [t for t in (payload or []) if isinstance(t, dict)]


def _friendly(exc: Exception) -> str:
    """Turn a network failure into something worth saying out loud."""
    if isinstance(exc, NoToken):
        return str(exc)
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code in (401, 403):
            return "Todoist rejected the token. It may be wrong or expired."
        if exc.code == 429:
            return "Todoist is rate limiting me. Try again in a minute."
        return f"Todoist returned an error ({exc.code})."
    if isinstance(exc, (urllib.error.URLError, TimeoutError)):
        return "I can't reach Todoist. Check the internet connection."
    return f"Something went wrong talking to Todoist: {exc}"


def _due_text(task: dict) -> str:
    due = task.get("due") or {}
    return due.get("string") or due.get("date") or ""


def _describe(tasks: list[dict], limit: int) -> str:
    shown = tasks[:limit]
    lines = []
    for task in shown:
        due = _due_text(task)
        lines.append(f"  {task.get('content', '(untitled)')}" + (f" — {due}" if due else ""))
    more = f" and {len(tasks) - len(shown)} more" if len(tasks) > len(shown) else ""
    return f"{len(tasks)} task(s){more}:\n" + "\n".join(lines)


def due_today(limit: int = 5) -> str:
    """The Todoist tasks due today or overdue. Asked by `day.my_day`."""
    # "today | overdue" is what "what's due" means to a person — yesterday's
    # unfinished work is still due.
    return due_on("today | overdue", limit, empty="Nothing is due today.")


def due_on(query: str, limit: int = 5, empty: str = "") -> str:
    """Tasks matching one of Todoist's own filters, such as "tomorrow"."""
    try:
        # The dedicated filter endpoint in v1.
        encoded = urllib.parse.urlencode({"query": query})
        tasks = _tasks(_call(f"tasks/filter?{encoded}"))
    except Exception as exc:  # noqa: BLE001
        return _friendly(exc)

    if not tasks:
        return empty or f"Nothing is due {query}."
    return _describe(tasks, limit)


# Words Todoist's own filters take as a date.
DUE_WORDS = {
    "today", "tomorrow", "overdue", "yesterday",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}


@skill(Permission.OPEN)
def todoist_search(about: str, limit: int = 5) -> str:
    """Find Todoist tasks by words in their text, or by when they are due: today, tomorrow, overdue, or a day name.

    Args:
        about: Word or phrase to look for
        limit: How many matches to read out
    """
    needle = about.strip().casefold()
    if not needle:
        return "I need something to look for."

    # Asked "what's due in Todoist tomorrow", the model comes here with
    # "tomorrow" every time, however the descriptions are worded, and a text
    # search for the word finds nothing. Todoist's filter language already
    # understands it, so it is answered as the question it is.
    if needle in DUE_WORDS:
        return due_on(needle, limit)

    try:
        tasks = _tasks(_call("tasks"))
    except Exception as exc:  # noqa: BLE001
        return _friendly(exc)

    matches = [t for t in tasks if needle in str(t.get("content", "")).casefold()]
    if not matches:
        return f"No Todoist task mentions '{about}'."
    return _describe(matches, limit)


@skill(Permission.CONFIRM)
def todoist_add(content: str, due: str = "") -> str:
    """Add a task to Todoist.

    Args:
        content: What the task says
        due: When it is due, in plain words such as "tomorrow" or "next Friday"
    """
    content = content.strip()
    if not content:
        return "I need something to put in the task."

    body: dict = {"content": content}
    if due.strip():
        # Todoist parses this itself, and does it better than I would.
        body["due_string"] = due.strip()

    try:
        created = _call("tasks", method="POST", body=body)
    except Exception as exc:  # noqa: BLE001
        return _friendly(exc)

    when = _due_text(created or {})
    return f"Added '{content}' to Todoist" + (f", due {when}." if when else ".")


@skill(Permission.CONFIRM)
def todoist_complete(about: str) -> str:
    """Mark a Todoist task as done.

    Args:
        about: Enough of the task's text to identify it
    """
    needle = about.strip().casefold()
    if not needle:
        return "I need to know which task."

    try:
        tasks = _tasks(_call("tasks"))
    except Exception as exc:  # noqa: BLE001
        return _friendly(exc)

    matches = [t for t in tasks if needle in str(t.get("content", "")).casefold()]
    if not matches:
        return f"No Todoist task mentions '{about}'."
    if len(matches) > 1:
        # Completing the wrong task is quiet and annoying to discover.
        names = ", ".join(str(t.get("content")) for t in matches[:4])
        return f"{len(matches)} tasks match '{about}': {names}. Which one?"

    task = matches[0]
    try:
        _call(f"tasks/{task['id']}/close", method="POST")
    except Exception as exc:  # noqa: BLE001
        return _friendly(exc)
    return f"Completed '{task.get('content')}'."
