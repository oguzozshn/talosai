"""The day, from every place it is written down.

What a day holds is spread over three places: the weekly plan, Argos (study
sessions and its own tasks) and Todoist. Asked "what's on my plate", the model
used to pick one of the skills for one of them, get nothing back because that
one place was empty, and offer to look further instead of looking. Seven
skills covered the question between them, and every one was a chance to
choose wrong.

So there is one. It asks every source and says what each one has. A source
that is missing or unreachable is named rather than skipped, because "nothing
due" and "I could not check" sound the same otherwise.
"""

from __future__ import annotations

from datetime import date, timedelta

from talos.skills import argos, planner, todoist
from talos.skills.base import Permission, skill


def _plan_slots(on: date) -> list[str]:
    week = planner._active_week()
    if week is None:
        return []
    day = next((d for d in week.get("gunler", []) if str(d.get("isoTarih")) == on.isoformat()), None)
    if day is None:
        return []
    slots = sorted(
        (s for s in week.get("slotlar", []) if s.get("gun") == day.get("no")),
        key=lambda s: str(s.get("saat", "")),
    )
    return [planner._slot_line(s) for s in slots]


def overview(on: date, today: date) -> str:
    parts: list[str] = []

    if on == today:
        upcoming = planner.next_up()
        if upcoming:
            parts.append(f"Next: {upcoming}")

    slots = _plan_slots(on)
    if slots:
        parts.append("Weekly plan:\n" + "\n".join(f"  {line}" for line in slots))

    data = argos._data_dir()
    if data is None:
        parts.append(argos.NOT_FOUND)
    else:
        sessions = argos.sessions_on(argos._load(data / argos.MARKER) or {}, on)
        if sessions:
            parts.append("Study sessions:\n" + "\n".join(f"  {line}" for line in sessions))
        parts.append(argos.open_tasks(data, on))

    # Todoist's filter language has words for these two days, and they are the
    # two anyone asks about. Any other day is said to be unchecked.
    if on == today:
        parts.append("Todoist: " + todoist.due_today())
    elif on == today + timedelta(days=1):
        parts.append("Todoist: " + todoist.due_on("tomorrow"))
    else:
        parts.append("Todoist was not checked for that day.")

    return "\n".join(parts)


@skill(Permission.OPEN)
def my_day(day: str = "today") -> str:
    """Say what a day holds and what is next: weekly plan, study sessions, Argos and Todoist tasks. Use for "what's on my plate", "what's next", "what's due", "how does my week look".

    Args:
        day: today, tomorrow, a day name such as Friday, a date as YYYY-MM-DD, or week
    """
    if day.strip().casefold() in {"week", "this week", "hafta", "bu hafta"}:
        # The planner falls back to the latest week it has, which answers
        # "how does my week look" with a week that is over. Only a plan that
        # contains today is this week's.
        week = planner._active_week()
        today = date.today().isoformat()
        if week is None or today not in {str(d.get("isoTarih")) for d in week.get("gunler", [])}:
            return "There is no weekly plan for this week."
        return planner.week_summary()

    on = argos._target_day(day)
    if on is None:
        return f"'{day}' isn't a day I understand. Use today, tomorrow, a day name, a date or week."
    return overview(on, date.today())
