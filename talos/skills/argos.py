"""Reading what Argos, the planner, knows: tasks, study plans, goals.

Argos keeps everything in plain JSON under the personal assistant project's
`data` directory — the same folder `planner` reads the weekly plan from. This
reads three more parts of it. Nothing here writes: Argos checks each file's
modification time before saving, so a write from here would surface there as
a conflict the owner never caused.

Argos saves by writing a temporary file and renaming it over the real one, so
a half-written file is never seen. On Windows, opening a file at the instant
of that rename can fail, which is why a failed read is tried once more.

Todoist is not read here even though Argos mirrors it: the `todoist` skills
read the real thing, and two answers to "what's due" that can disagree are
worse than one.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path

from talos.skills._workspace import WORKSPACES
from talos.skills.base import Permission, skill

log = logging.getLogger(__name__)

# The file that marks a directory as Argos's, rather than any other `data`.
MARKER = "calisma-planlari.json"


def _data_dir() -> Path | None:
    """ARGOS_DATA if set, as Argos itself honours it; otherwise found."""
    override = os.environ.get("ARGOS_DATA")
    if override and (Path(override) / MARKER).is_file():
        return Path(override)
    for root in WORKSPACES:
        if (root / "data" / MARKER).is_file():
            return root / "data"
    return None


def _load(path: Path) -> dict | None:
    for attempt in range(2):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            if attempt == 0:
                time.sleep(0.1)
                continue
            log.warning("could not read %s: %s", path.name, exc)
        except json.JSONDecodeError as exc:
            log.warning("could not parse %s: %s", path.name, exc)
            return None
    return None


def _duration(minutes: int) -> str:
    hours, rest = divmod(max(minutes, 0), 60)
    if hours and rest:
        return f"{hours} h {rest} min"
    return f"{hours} h" if hours else f"{rest} min"


def _target_day(day: str) -> date | None:
    wanted = day.strip().lower()
    if wanted in {"", "today", "bugun", "bugün"}:
        return date.today()
    if wanted in {"tomorrow", "yarin", "yarın"}:
        return date.today() + timedelta(days=1)
    from talos.skills.planner import DAY_NAMES

    if wanted in DAY_NAMES:
        # The next one, counting today: "Friday" asked on a Friday is today.
        ahead = (DAY_NAMES[wanted] - date.today().isoweekday()) % 7
        return date.today() + timedelta(days=ahead)
    try:
        return date.fromisoformat(wanted)
    except ValueError:
        return None


def open_tasks(data: Path, on: date) -> str:
    tasks = (_load(data / "tasks" / "tasks.json") or {}).get("tasks", [])
    lists = {
        item.get("id"): item.get("name", "")
        for item in (_load(data / "tasks" / "lists.json") or {}).get("lists", [])
    }
    open_ = [t for t in tasks if not t.get("completedAt")]
    due = [t for t in open_ if t.get("due") and str(t["due"])[:10] <= on.isoformat()]
    undated = len([t for t in open_ if not t.get("due")])

    if not due:
        return f"No Argos tasks are due by {on.isoformat()}. {undated} open task(s) have no date."

    due.sort(key=lambda t: (str(t["due"]), t.get("priority", 4)))
    lines = [f"{len(due)} Argos task(s) due by {on.isoformat()}:"]
    for task in due:
        when = str(task["due"])
        late = " (overdue)" if when[:10] < on.isoformat() else ""
        clock = f" at {when[11:16]}" if len(when) > 10 else ""
        flag = ", important" if task.get("important") else ""
        where = lists.get(task.get("listId"), "")
        lines.append(f"  {task.get('title', '')}{clock}{late} — {where}{flag}")
    return "\n".join(lines)


def _done_minutes(calendar: dict, plan_id: str, alt_id: str | None = None, own: bool = False) -> int:
    total = 0
    for sessions in calendar.values():
        for s in sessions:
            if s.get("planId") != plan_id or not s.get("yapildi"):
                continue
            if own and s.get("altId"):
                continue
            if alt_id is not None and s.get("altId") != alt_id:
                continue
            total += int(s.get("dk") or 0)
    return total


# Sub-plans are usually exams, named in Turkish. Left bare, "Vize" was read
# out as "quiz" and its hours were answered from the whole course instead.
EXAM_WORDS = {"vize": "midterm", "final": "final exam", "bütünleme": "resit", "butunleme": "resit"}


def _part_name(alt: dict) -> str:
    name = str(alt.get("ad", ""))
    meaning = EXAM_WORDS.get(name.strip().lower())
    return f"{name} ({meaning})" if meaning else name


def _days_left(deadline: str | None, today: date) -> str:
    if not deadline:
        return "no deadline"
    try:
        days = (date.fromisoformat(deadline[:10]) - today).days
    except ValueError:
        return "no deadline"
    if days < 0:
        return f"deadline passed {-days} day(s) ago"
    return f"{days} day(s) to {deadline[:10]}"


def plan_progress(data: Path, name: str, today: date) -> str:
    plans_file = _load(data / MARKER) or {}
    plans = plans_file.get("planlar", [])
    calendar = plans_file.get("takvim", {})
    if not plans:
        return "Argos has no study plans."

    wanted = name.strip().lower()
    chosen = [p for p in plans if wanted in str(p.get("ad", "")).lower()] if wanted else plans
    if not chosen:
        names = ", ".join(str(p.get("ad", "")) for p in plans)
        return f"No plan matches '{name}'. The plans are: {names}."

    lines: list[str] = []
    for plan in chosen:
        target = int(plan.get("hedefDk") or 0)
        done = _done_minutes(calendar, plan["id"])
        lines.append(
            f"{plan.get('ad')}: {_duration(done)} done of {_duration(target)}, "
            f"{_duration(target - done)} left, {_days_left(plan.get('sonTarih'), today)}."
        )
        for alt in plan.get("altlar", []):
            alt_target = int(alt.get("hedefDk") or 0)
            alt_done = _done_minutes(calendar, plan["id"], alt.get("id"))
            lines.append(
                f"  Part {_part_name(alt)}: {_duration(alt_target - alt_done)} left of "
                f"{_duration(alt_target)}, {_days_left(alt.get('sonTarih'), today)}."
            )

    todays = sessions_on(plans_file, today, {p["id"] for p in chosen})
    if todays:
        lines.append("Today's sessions:")
        lines += [f"  {line}" for line in todays]
    return "\n".join(lines)


def sessions_on(plans_file: dict, day: date, plan_ids: set[str] | None = None) -> list[str]:
    """The study sessions planned for a day, as one line each."""
    by_id = {p["id"]: p for p in plans_file.get("planlar", [])}
    lines = []
    for s in plans_file.get("takvim", {}).get(day.isoformat(), []):
        # Missed sessions stay on the calendar marked yapilmadi; they are
        # history, not plan.
        if s.get("yapilmadi") or (plan_ids is not None and s.get("planId") not in plan_ids):
            continue
        plan = by_id.get(s.get("planId"), {})
        alt = next((a for a in plan.get("altlar", []) if a.get("id") == s.get("altId")), None)
        label = f"{plan.get('ad', '')}{' — ' + _part_name(alt) if alt else ''}"
        state = "done" if s.get("yapildi") else "to do"
        lines.append(f"{label}, {_duration(int(s.get('dk') or 0))} ({state})")
    return lines


def goals_and_reminders(data: Path) -> str:
    goals = (_load(data / "hedefler.json") or {}).get("hedefler", [])
    notes = [m for m in (_load(data / "hatirla.json") or {}).get("maddeler", []) if not m.get("arsiv")]
    if not goals and not notes:
        return "Argos has no goals or things to remember."

    lines: list[str] = []
    if goals:
        lines.append("Goals:")
        lines += [f"  {g.get('ad')} — by {g.get('hedefTarih', 'no date')}" for g in goals]
    if notes:
        lines.append("To remember:")
        lines += [f"  {m.get('metin')}" for m in notes]
    return "\n".join(lines)


def day_overview(data: Path, on: date) -> str:
    """Everything local that says what a day holds, in one answer.

    "What's on my plate today" spans the weekly plan, study sessions and
    tasks. Asked it, the model picked one of the three, got nothing back
    because there was no plan file for the week, and offered to look further
    instead of looking. One skill that already covers the whole question
    leaves it nothing to choose wrongly.
    """
    from talos.skills import planner

    parts: list[str] = []
    week = planner._active_week()
    if week is not None:
        iso = on.isoformat()
        day = next((d for d in week.get("gunler", []) if str(d.get("isoTarih")) == iso), None)
        if day is not None:
            slots = sorted(
                (s for s in week.get("slotlar", []) if s.get("gun") == day.get("no")),
                key=lambda s: str(s.get("saat", "")),
            )
            if slots:
                parts.append("Weekly plan:\n" + "\n".join(f"  {planner._slot_line(s)}" for s in slots))

    sessions = sessions_on(_load(data / MARKER) or {}, on)
    if sessions:
        parts.append("Study sessions:\n" + "\n".join(f"  {line}" for line in sessions))

    parts.append(open_tasks(data, on))
    return "\n".join(parts)


NOT_FOUND = "I can't find the Argos data folder."


@skill(Permission.OPEN)
def my_day(day: str = "today") -> str:
    """Say what a day holds: weekly plan slots, study sessions and Argos tasks due. Use for "what's on my plate".

    Args:
        day: today, tomorrow, a day name such as Friday, or YYYY-MM-DD
    """
    data = _data_dir()
    if data is None:
        return NOT_FOUND
    on = _target_day(day)
    if on is None:
        return f"'{day}' isn't a day I understand. Use today, tomorrow, a day name or YYYY-MM-DD."
    return day_overview(data, on)


@skill(Permission.OPEN)
def argos_tasks(day: str = "today") -> str:
    """List open tasks from the Argos planner that are due by a day, overdue ones included.

    Args:
        day: today, tomorrow, a day name such as Friday, or YYYY-MM-DD
    """
    data = _data_dir()
    if data is None:
        return NOT_FOUND
    on = _target_day(day)
    if on is None:
        return f"'{day}' isn't a day I understand. Use today, tomorrow, a day name or YYYY-MM-DD."
    return open_tasks(data, on)


@skill(Permission.OPEN)
def study_progress(plan: str = "") -> str:
    """Report study plans from Argos: hours done and left, deadlines, and today's sessions.

    Args:
        plan: Part of a plan's name, such as a course; empty for all plans
    """
    data = _data_dir()
    return NOT_FOUND if data is None else plan_progress(data, plan, date.today())


@skill(Permission.OPEN)
def goals() -> str:
    """List the owner's long-term goals and the things they asked Argos to remember."""
    data = _data_dir()
    return NOT_FOUND if data is None else goals_and_reminders(data)
