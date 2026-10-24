"""Speaking without being asked: the morning brief, and reminders before what is planned.

Until this, nothing happened unprompted — no timers, no reminders. This is a
thread that wakes every half minute, decides whether anything is due, and
says it: out loud through `Assistant.announce`, which waits for any turn in
progress and keeps the microphone from hearing it, and on Telegram, for when
nobody is in the room.

What it says:

- **The morning brief**, once a day: the first time Talos is running after
  `brief_time`, unless it is already past `brief_until` — a "good morning"
  at four in the afternoon is not a brief, it is noise. The day comes from
  the same place `my_day` reads it, and the model turns it into a few spoken
  sentences.
- **A reminder** `remind_minutes` before anything with a clock time: weekly
  plan slots and Argos tasks due at an hour. Argos study sessions have a day
  but no hour, so they are in the brief instead.

Never between `quiet_from` and `quiet_until`. What was said is kept in
`data/proactive.json`, so a restart does not repeat the brief or a reminder.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

from talos.config import ProactiveConfig

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "proactive.json"
TICK_SECONDS = 30


@dataclass(frozen=True)
class Event:
    at: datetime
    what: str
    # Stable across restarts, so a reminder is given once.
    key: str


def _clock(text: str) -> time:
    hours, minutes = text.split(":")
    return time(int(hours), int(minutes))


def quiet(now: datetime, config: ProactiveConfig) -> bool:
    start, end = _clock(config.quiet_from), _clock(config.quiet_until)
    current = now.time()
    if start <= end:
        return start <= current < end
    return current >= start or current < end  # Overnight, the usual case.


def timed_events(day: date) -> list[Event]:
    """Everything with a clock time today: plan slots, and Argos tasks due at an hour."""
    from talos.skills import argos, planner

    events: list[Event] = []

    week = planner._active_week()
    if week is not None:
        number = next(
            (d.get("no") for d in week.get("gunler", []) if str(d.get("isoTarih")) == day.isoformat()),
            None,
        )
        for slot in week.get("slotlar", []):
            if slot.get("gun") != number or not slot.get("saat"):
                continue
            if planner.STATUS_WORDS.get(str(slot.get("durum", "")).lower()) in {"done", "cancelled"}:
                continue
            try:
                at = datetime.combine(day, _clock(str(slot["saat"])))
            except ValueError:
                continue
            minutes = slot.get("sureDk")
            length = f", {minutes} minutes" if minutes else ""
            events.append(
                Event(at, f"{slot.get('alan', '')}: {slot.get('konu', '')}{length}", f"slot:{slot.get('id', at)}")
            )

    data = argos._data_dir()
    if data is not None:
        for task in (argos._load(data / "tasks" / "tasks.json") or {}).get("tasks", []):
            due = str(task.get("due") or "")
            if task.get("completedAt") or len(due) < 16 or due[:10] != day.isoformat():
                continue
            try:
                at = datetime.fromisoformat(due[:16])
            except ValueError:
                continue
            events.append(Event(at, str(task.get("title", "")), f"task:{task.get('id', due)}"))

    return sorted(events, key=lambda e: e.at)


def compose_brief(config, overview: str) -> str:
    """The day, as a few sentences worth hearing at breakfast."""
    import ollama

    try:
        response = ollama.chat(
            model=config.llm.model,
            messages=[
                {"role": "system", "content": config.character},
                {
                    "role": "user",
                    "content": (
                        "Turn this into a short spoken morning brief for the owner: "
                        "greet them in a few words, then what matters today, at most "
                        "five sentences, no lists. Say nothing that is not below.\n\n"
                        + overview
                    ),
                },
            ],
            think=config.llm.think,
            keep_alive=config.llm.keep_alive,
            options={"num_ctx": config.llm.context, "num_predict": config.llm.max_tokens},
        )
        return response["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001 — without the model, read the day plainly
        log.warning("could not compose the brief: %s", exc)
        return "Good morning. Here is today. " + " ".join(overview.split())


class Scheduler:
    def __init__(self, config, say, *, state: Path = STATE, events=timed_events, brief=None) -> None:
        self._config = config
        self._settings: ProactiveConfig = config.proactive
        self._say = say
        self._state_path = state
        self._events = events
        self._brief = brief or self._default_brief
        self._state = self._read()
        self._stop = threading.Event()

    def _read(self) -> dict:
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(json.dumps(self._state), encoding="utf-8")

    def _default_brief(self, today: date) -> str:
        from talos.skills import day

        return compose_brief(self._config, day.overview(today, today))

    def tick(self, now: datetime) -> list[str]:
        """Say whatever is due at `now`. Returns what was said."""
        said: list[str] = []
        today = now.date().isoformat()
        if quiet(now, self._settings):
            return said

        if (
            self._settings.brief
            and self._state.get("brief") != today
            and _clock(self._settings.brief_time) <= now.time() < _clock(self._settings.brief_until)
        ):
            text = self._brief(now.date())
            self._say(text)
            said.append(text)
            self._state["brief"] = today
            self._write()

        reminded = set(self._state.get("reminded", {}).get(today, []))
        ahead = timedelta(minutes=self._settings.remind_minutes)
        for event in self._events(now.date()):
            if event.key in reminded or not (now <= event.at <= now + ahead):
                continue
            minutes = max(1, round((event.at - now).total_seconds() / 60))
            text = f"In {minutes} minute{'s' if minutes != 1 else ''}, at {event.at:%H:%M}: {event.what}."
            self._say(text)
            said.append(text)
            reminded.add(event.key)
            # Only today is kept: yesterday's keys can never match again.
            self._state["reminded"] = {today: sorted(reminded)}
            self._write()
        return said

    def run(self) -> None:
        while not self._stop.wait(TICK_SECONDS):
            try:
                self.tick(datetime.now())
            except Exception:  # noqa: BLE001 — a bad tick must not end the thread
                log.exception("proactive tick failed")

    def start(self) -> None:
        # The first tick straight away, so a brief due at start-up is not half
        # a minute late.
        def first_then_loop() -> None:
            try:
                self.tick(datetime.now())
            except Exception:  # noqa: BLE001
                log.exception("proactive tick failed")
            self.run()

        threading.Thread(target=first_then_loop, name="proactive", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()


def deliver(assistant, text: str, speak: bool = True) -> None:
    """Out loud, and on Telegram if it is set up — whoever is not in the room still hears."""
    from talos import telegram

    settings = assistant.config.proactive
    if settings.speak and speak:
        assistant.announce(text)
    else:
        print(f"  {text}")
    bot = telegram.active()
    if settings.telegram and bot is not None:
        chat = bot.owner_chat()
        if chat is not None:
            try:
                bot.send_text(chat, text)
            except Exception as exc:  # noqa: BLE001
                log.warning("could not send to Telegram: %s", exc)


def start(assistant, speak: bool = True) -> Scheduler | None:
    """Start the scheduler. `speak` is off when nobody is meant to be in the room."""
    if not assistant.config.proactive.enabled:
        return None
    scheduler = Scheduler(assistant.config, lambda text: deliver(assistant, text, speak))
    scheduler.start()
    return scheduler
