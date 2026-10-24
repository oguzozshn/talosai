"""Reading Argos's files, against fixtures shaped like the real ones."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from talos.skills import argos

TODAY = date(2026, 10, 24)


def write(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def data(tmp_path: Path) -> Path:
    write(tmp_path / "tasks" / "lists.json", {"lists": [{"id": "l1", "name": "Argos"}]})
    write(
        tmp_path / "tasks" / "tasks.json",
        {
            "tasks": [
                {"id": "a", "listId": "l1", "title": "Late one", "due": "2026-10-20", "completedAt": None},
                {"id": "b", "listId": "l1", "title": "Today at ten", "due": "2026-10-24T10:00",
                 "important": True, "completedAt": None},
                {"id": "c", "listId": "l1", "title": "Next week", "due": "2026-10-31", "completedAt": None},
                {"id": "d", "listId": "l1", "title": "Finished", "due": "2026-10-24",
                 "completedAt": "2026-10-24T08:00:00Z"},
                {"id": "e", "listId": "l1", "title": "Someday", "due": None, "completedAt": None},
            ]
        },
    )
    write(
        tmp_path / argos.MARKER,
        {
            "planlar": [
                {"id": "p1", "ad": "Software Requirements Engineering", "hedefDk": 1800,
                 "sonTarih": "2026-12-18",
                 "altlar": [{"id": "v", "ad": "Vize", "hedefDk": 900, "sonTarih": "2026-11-03"}]},
                {"id": "p2", "ad": "Game Programming", "hedefDk": 600, "sonTarih": "2026-12-18", "altlar": []},
            ],
            "takvim": {
                "2026-10-19": [
                    {"planId": "p1", "altId": "v", "dk": 60, "yapildi": True},
                    {"planId": "p1", "dk": 30, "yapildi": True},
                    {"planId": "p2", "dk": 120, "yapilmadi": True},
                ],
                "2026-10-24": [
                    {"planId": "p1", "altId": "v", "dk": 60},
                    {"planId": "p2", "dk": 120, "yapilmadi": True},
                ],
            },
        },
    )
    write(tmp_path / "hedefler.json", {"hedefler": [{"ad": "Masters abroad", "hedefTarih": "2028-09"}]})
    write(
        tmp_path / "hatirla.json",
        {"maddeler": [{"metin": "Book IELTS", "arsiv": False}, {"metin": "Old", "arsiv": True}]},
    )
    return tmp_path


class TestTasks:
    def test_due_and_overdue_are_listed_but_not_future_or_done(self, data):
        text = argos.open_tasks(data, TODAY)
        assert "Late one (overdue)" in text
        assert "Today at ten at 10:00" in text and "important" in text
        assert "Next week" not in text and "Finished" not in text

    def test_undated_tasks_are_counted_when_nothing_is_due(self, data):
        assert "1 open task(s) have no date" in argos.open_tasks(data, date(2026, 10, 1))


class TestPlans:
    def test_time_left_counts_only_done_sessions(self, data):
        text = argos.plan_progress(data, "requirements", TODAY)
        # 60 on Vize + 30 on the plan itself are done; today's 60 is not.
        assert "1 h 30 min done of 30 h" in text
        assert "Part Vize (midterm): 14 h left of 15 h, 10 day(s) to 2026-11-03" in text

    def test_missed_sessions_are_not_today_plan(self, data):
        text = argos.plan_progress(data, "", TODAY)
        assert "1 h of study: Software Requirements Engineering, for the Vize (midterm) (to do)" in text
        assert "Game Programming, 2 h" not in text

    def test_an_unknown_name_lists_the_real_ones(self, data):
        assert "Game Programming" in argos.plan_progress(data, "physics", TODAY)


def test_archived_reminders_are_left_out(data):
    text = argos.goals_and_reminders(data)
    assert "Masters abroad" in text and "Book IELTS" in text and "Old" not in text


def test_a_missing_folder_says_so(tmp_path, monkeypatch):
    monkeypatch.delenv("ARGOS_DATA", raising=False)
    monkeypatch.setattr(argos, "WORKSPACES", (tmp_path,))
    assert argos.study_progress() == argos.NOT_FOUND


def test_argos_data_overrides_the_search(data, monkeypatch):
    monkeypatch.setenv("ARGOS_DATA", str(data))
    monkeypatch.setattr(argos, "WORKSPACES", ())
    assert argos._data_dir() == data


class TestDay:
    """my_day asks every source; these stub the two that are not files here."""

    @pytest.fixture(autouse=True)
    def sources(self, data, monkeypatch):
        from talos.skills import planner, todoist

        monkeypatch.setattr(planner, "_active_week", lambda: None)
        monkeypatch.setattr(argos, "_data_dir", lambda: data)
        monkeypatch.setattr(todoist, "due_today", lambda limit=5: "Nothing is due today.")
        monkeypatch.setattr(todoist, "due_on", lambda query, limit=5, empty="": f"checked {query}")

    def test_today_has_sessions_tasks_and_todoist(self):
        from talos.skills import day

        text = day.overview(TODAY, TODAY)
        assert "Study sessions:" in text and "1 h of study: Software Requirements Engineering, for the Vize (midterm) (to do)" in text
        assert "Today at ten" in text
        assert "Todoist: Nothing is due today." in text

    def test_tomorrow_asks_todoist_for_tomorrow(self):
        from datetime import timedelta

        from talos.skills import day

        assert "Todoist: checked tomorrow" in day.overview(TODAY + timedelta(days=1), TODAY)

    def test_a_later_day_says_todoist_was_not_checked(self):
        from datetime import timedelta

        from talos.skills import day

        assert "Todoist was not checked" in day.overview(TODAY + timedelta(days=3), TODAY)

    def test_a_week_that_is_over_is_not_this_week(self, monkeypatch):
        from talos.skills import day, planner

        old = {"gunler": [{"no": 1, "isoTarih": "2026-01-05"}], "slotlar": []}
        monkeypatch.setattr(planner, "_active_week", lambda: old)
        assert day.my_day("week") == "There is no weekly plan for this week."


def test_a_date_word_searches_todoist_by_due_date(monkeypatch):
    from talos.skills import todoist

    monkeypatch.setattr(todoist, "due_on", lambda query, limit=5, empty="": f"due {query}")
    assert todoist.todoist_search("Tomorrow") == "due tomorrow"
