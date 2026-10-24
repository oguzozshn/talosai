"""The morning brief and reminders, with the clock and the day's events supplied."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pytest

from talos import proactive
from talos.config import ProactiveConfig

DAY = date(2026, 10, 26)


def at(clock: str, day: date = DAY) -> datetime:
    return datetime.combine(day, proactive._clock(clock))


@dataclass
class FakeConfig:
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)


def scheduler(tmp_path, events=(), **settings):
    said: list[str] = []
    made = proactive.Scheduler(
        FakeConfig(ProactiveConfig(**settings)),
        said.append,
        state=tmp_path / "proactive.json",
        events=lambda day: list(events),
        brief=lambda day: f"Brief for {day}.",
    )
    return made, said


def event(clock: str, what: str, key: str) -> proactive.Event:
    return proactive.Event(at(clock), what, key)


class TestBrief:
    def test_given_once_in_the_morning(self, tmp_path):
        made, said = scheduler(tmp_path)
        made.tick(at("08:40"))
        made.tick(at("09:10"))
        assert said == ["Brief for 2026-10-26."]

    def test_not_before_its_time(self, tmp_path):
        made, said = scheduler(tmp_path, quiet_until="06:00")
        made.tick(at("07:00"))
        assert said == []

    def test_not_in_the_afternoon(self, tmp_path):
        made, said = scheduler(tmp_path)
        made.tick(at("16:00"))
        assert said == []

    def test_not_repeated_after_a_restart(self, tmp_path):
        made, _ = scheduler(tmp_path)
        made.tick(at("08:40"))
        again, said = scheduler(tmp_path)
        again.tick(at("09:00"))
        assert said == []

    def test_given_again_the_next_day(self, tmp_path):
        made, said = scheduler(tmp_path)
        made.tick(at("08:40"))
        made.tick(at("08:40", date(2026, 10, 27)))
        assert len(said) == 2


class TestReminders:
    def test_said_shortly_before(self, tmp_path):
        made, said = scheduler(tmp_path, brief=False, events=[event("14:00", "Requirements: chapter 7", "a")])
        made.tick(at("13:45"))
        made.tick(at("13:52"))
        assert said == ["In 8 minutes, at 14:00: Requirements: chapter 7."]

    def test_said_once(self, tmp_path):
        made, said = scheduler(tmp_path, brief=False, events=[event("14:00", "Gym", "a")])
        made.tick(at("13:52"))
        made.tick(at("13:55"))
        assert len(said) == 1

    def test_not_for_what_has_started(self, tmp_path):
        made, said = scheduler(tmp_path, brief=False, events=[event("14:00", "Gym", "a")])
        made.tick(at("14:05"))
        assert said == []

    def test_nothing_in_the_quiet_hours(self, tmp_path):
        made, said = scheduler(tmp_path, brief=False, events=[event("23:30", "Late", "a")])
        made.tick(at("23:25"))
        assert said == []


@pytest.mark.parametrize(
    ("clock", "expected"),
    [("23:30", True), ("02:00", True), ("07:59", True), ("08:00", False), ("15:00", False)],
)
def test_quiet_hours_cross_midnight(clock, expected):
    assert proactive.quiet(at(clock), ProactiveConfig()) is expected


class FakeVoice:
    def __init__(self, log):
        self.log = log

    def say(self, text):
        self.log.append(("say", text))


def test_announcing_holds_the_microphone_while_speaking():
    import contextlib
    import threading

    from talos.assistant import Assistant

    log: list = []

    @contextlib.contextmanager
    def hold():
        log.append("hold")
        yield
        log.append("release")

    assistant = Assistant.__new__(Assistant)
    assistant._turn = threading.Lock()
    assistant.voice = FakeVoice(log)
    assistant.hold_microphone = hold
    assistant.announce("Gym in ten minutes.")
    assert log == ["hold", ("say", "Gym in ten minutes."), "release"]
