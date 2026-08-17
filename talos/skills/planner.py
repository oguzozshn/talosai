"""Reading the weekly plan kept by the personal assistant project.

That project already writes down what the week is meant to contain — slots
with a day, a time, an area and a topic, in `data/W<week>.json`. This reads
it. Nothing here plans anything or writes anything; asking "what am I meant to
be doing" should not be able to change the answer.

The file is found rather than configured: any workspace root holding a `data`
directory with `W*.json` in it will do, so moving the project only means
updating `workspaces` in settings.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

from talos.skills._workspace import WORKSPACES
from talos.skills.base import Permission, skill

log = logging.getLogger(__name__)

# Slot statuses as the plan file writes them, and what to call them out loud.
STATUS_WORDS = {
    "planli": "planned",
    "tamam": "done",
    "tamamlandi": "done",
    "kacti": "missed",
    "kaçtı": "missed",
    "iptal": "cancelled",
    "devam": "in progress",
}

DAY_NAMES = {
    "monday": 1, "tuesday": 2, "wednesday": 3, "thursday": 4,
    "friday": 5, "saturday": 6, "sunday": 7,
    "pazartesi": 1, "sali": 2, "salı": 2, "carsamba": 3, "çarşamba": 3,
    "persembe": 4, "perşembe": 4, "cuma": 5, "cumartesi": 6, "pazar": 7,
}


def _plan_files() -> list[Path]:
    found: list[Path] = []
    for root in WORKSPACES:
        data = root / "data"
        if data.is_dir():
            found += sorted(data.glob("W*.json"))
    # Backups sit alongside the real files and would otherwise be read as
    # duplicates of the same week.
    return [p for p in found if p.suffix == ".json" and ".bak" not in p.name]


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read %s: %s", path.name, exc)
        return None


def _active_week() -> dict | None:
    """The week marked active, or failing that the one containing today."""
    weeks = [w for w in (_load(p) for p in _plan_files()) if w]
    if not weeks:
        return None

    for week in weeks:
        if str(week.get("durum", "")).lower() == "aktif":
            return week

    today = date.today().isoformat()
    for week in weeks:
        days = [str(d.get("isoTarih", "")) for d in week.get("gunler", [])]
        if today in days:
            return week

    # Highest week number, as the best guess at "most recent".
    return max(weeks, key=lambda w: (w.get("yil", 0), w.get("hafta", 0)))


def _day_number(week: dict, when: str) -> tuple[int | None, str]:
    """Resolve 'today', 'tomorrow' or a day name to a day number in this week."""
    wanted = when.strip().lower()
    days = week.get("gunler", [])

    if wanted in {"", "today", "bugun", "bugün"}:
        target = date.today()
    elif wanted in {"tomorrow", "yarin", "yarın"}:
        target = date.today() + timedelta(days=1)
    elif wanted in DAY_NAMES:
        number = DAY_NAMES[wanted]
        match = next((d for d in days if d.get("no") == number), None)
        return number, str(match.get("ad", wanted)) if match else wanted
    else:
        return None, wanted

    iso = target.isoformat()
    for day in days:
        if str(day.get("isoTarih")) == iso:
            return int(day.get("no", 0)), str(day.get("ad", iso))
    return None, target.strftime("%A")


def _slot_line(slot: dict) -> str:
    status = STATUS_WORDS.get(str(slot.get("durum", "")).lower(), str(slot.get("durum", "")))
    minutes = slot.get("sureDk")
    length = f", {minutes} minutes" if minutes else ""
    return f"{slot.get('saat', '')} {slot.get('alan', '')} — {slot.get('konu', '')}{length} ({status})"


@skill(Permission.OPEN)
def todays_plan(day: str = "today") -> str:
    """Report what the weekly plan says is scheduled, for today or another day.

    Args:
        day: Which day — today, tomorrow, or a day name such as Friday
    """
    week = _active_week()
    if week is None:
        return "I can't find a weekly plan file."

    number, label = _day_number(week, day)
    if number is None:
        return f"'{day}' isn't a day in {week.get('baslik', 'this week')}."

    slots = [s for s in week.get("slotlar", []) if s.get("gun") == number]
    if not slots:
        return f"Nothing is scheduled for {label}."

    slots.sort(key=lambda s: str(s.get("saat", "")))
    lines = [f"{label} has {len(slots)} slot(s):"]
    lines += [f"  {_slot_line(s)}" for s in slots]
    return "\n".join(lines)


@skill(Permission.OPEN)
def weekly_plan() -> str:
    """Summarise the current week's plan: how many slots, and how many are done."""
    week = _active_week()
    if week is None:
        return "I can't find a weekly plan file."

    slots = week.get("slotlar", [])
    if not slots:
        return f"{week.get('baslik', 'This week')} has no slots planned."

    done = sum(1 for s in slots if STATUS_WORDS.get(str(s.get("durum", "")).lower()) == "done")
    minutes = sum(int(s.get("sureDk") or 0) for s in slots)

    areas: dict[str, int] = {}
    for slot in slots:
        area = str(slot.get("alan", "unspecified"))
        areas[area] = areas.get(area, 0) + 1
    spread = ", ".join(f"{name} {count}" for name, count in sorted(areas.items(), key=lambda kv: -kv[1]))

    return (
        f"{week.get('baslik', 'This week')}: {len(slots)} slots, "
        f"{minutes // 60} hours {minutes % 60} minutes in total, {done} done so far. "
        f"By area — {spread}."
    )


@skill(Permission.OPEN)
def whats_next() -> str:
    """Report the next thing scheduled from now, whether today or later this week."""
    week = _active_week()
    if week is None:
        return "I can't find a weekly plan file."

    by_number = {int(d.get("no", 0)): d for d in week.get("gunler", [])}
    now = datetime.now()
    upcoming: list[tuple[datetime, dict, str]] = []

    for slot in week.get("slotlar", []):
        day = by_number.get(int(slot.get("gun", 0)))
        if not day:
            continue
        if STATUS_WORDS.get(str(slot.get("durum", "")).lower()) in {"done", "cancelled"}:
            continue
        try:
            when = datetime.fromisoformat(f"{day.get('isoTarih')}T{slot.get('saat', '00:00')}")
        except ValueError:
            continue
        if when >= now:
            upcoming.append((when, slot, str(day.get("ad", ""))))

    if not upcoming:
        return "Nothing left on the plan for this week."

    when, slot, day_name = min(upcoming, key=lambda item: item[0])
    same_day = when.date() == now.date()
    lead = "Later today" if same_day else day_name
    return f"{lead} at {slot.get('saat')}: {slot.get('alan')}, {slot.get('konu')}."
