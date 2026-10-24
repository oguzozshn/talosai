"""Answering questions about which applications got used.

The data comes from `talos/usage.py`, which samples the focused window while
Talos is running. That last part is the whole reason these answers say what
period they actually cover: a total for "today" that began when Talos started
at two in the afternoon is not a day, and reporting it as one would be a lie
of omission.
"""

from __future__ import annotations

from datetime import date, timedelta

from talos.skills.base import Permission, skill
from talos.usage import UsageLog, friendly, spoken_duration

_log = UsageLog()

# Appended to every answer. The model reads it as instruction rather than as
# something to repeat.
#
# It is here rather than in the character definition because it is specific to
# this data. Usage figures invite commentary in a way that a git branch or a
# file listing does not, and an assistant that volunteers an opinion about how
# someone spends their time is one they stop asking.
MANNER = (
    "Report, never judge. Three hours in Chrome is a fact. Three hours in "
    "Chrome, maybe cut back a little, is an opinion nobody asked for. Never "
    "comment on whether the usage looks high, low, healthy or wasted, and do "
    "not offer to help with focus or habits unless asked outright. "
    "State the period plainly. If he asked for something outside what is "
    "tracked, say what there is instead of estimating."
)


def _period(period: str) -> tuple[date | None, date, str]:
    """Resolve a period to a date range and how to say it out loud."""
    wanted = period.strip().lower()
    today = date.today()
    if wanted in {"", "today"}:
        return today, today, "today"
    if wanted == "yesterday":
        # A closed range, or "yesterday" silently includes today too.
        yesterday = today - timedelta(days=1)
        return yesterday, yesterday, "yesterday"
    if wanted in {"week", "this week", "7 days", "last week"}:
        return today - timedelta(days=6), today, "the last seven days"
    # Named so the refusal is specific. Without these the model tends to
    # substitute "week" for "month" on its way in, and the honest answer —
    # that a month is not tracked — never gets the chance to be given.
    if any(word in wanted for word in ("month", "year", "all time", "ever")):
        return None, today, wanted
    return None, today, wanted


@skill(Permission.OPEN)
def app_usage(app: str = "", period: str = "today", limit: int = 5) -> str:
    """Report time spent in applications: the most used ones, or one in particular.

    Args:
        app: One application to ask about, such as Chrome; empty for the most used
        period: "today" for since midnight, "week" for the last 7 days
        limit: how many applications to include, most-used first

    Examples:
        how long was I in Chrome today
        which apps did I use most this week
        bugün bilgisayarda ne kadar vakit geçirdim
    """
    if app.strip():
        return _time_in_app(app, period)
    return _most_used(period, limit)


def _most_used(period: str, limit: int) -> str:
    since, until, spoken = _period(period)
    if since is None:
        return (
            f"I only track today, yesterday and the last seven days, not "
            f"{spoken}. Want one of those instead? {MANNER}"
        )

    first = _log.first_day()
    if first is None:
        return (
            "I have not recorded any application use yet. I only count while I "
            "am running, so there is nothing from before I started. " + MANNER
        )

    totals = _log.totals(since, limit=limit, until=until)
    if not totals:
        return f"Nothing recorded for {spoken}. " + MANNER

    tracked = _log.recorded_seconds(since, until)

    # One flowing sentence rather than a list on separate lines. A list invites
    # the model to complete the pattern, and it did — inventing a row reading
    # "[Other apps] — [Time]". It also reads better aloud, which is the point.
    parts = [f"{friendly(app)} {spoken_duration(seconds)}" for app, seconds in totals]
    if len(parts) > 1:
        listed = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    else:
        listed = parts[0]

    # The caveat that matters, stated as fact rather than apology: the figures
    # are of recorded time, and recording only happens while Talos is up.
    if first > since:
        coverage = (
            f"That is from {first:%d %B} onwards, since I was not recording "
            f"before then."
        )
    else:
        coverage = f"That is {spoken_duration(tracked)} of recorded time in total."

    return f"Most used {spoken}: {listed}. {coverage} {MANNER}"


def _time_in_app(app: str, period: str) -> str:
    wanted = app.strip().casefold()
    if not wanted:
        return "Which application? " + MANNER

    since, until, spoken = _period(period)
    if since is None:
        return f"I only track today, yesterday and the last seven days. {MANNER}"

    # Matched on both the executable and the spoken name, so "Chrome",
    # "chrome" and "chrome.exe" all find the same thing.
    for name, seconds in _log.totals(since, limit=100, until=until):
        if wanted in name.casefold() or wanted in friendly(name).casefold():
            return f"{friendly(name)}, {spoken_duration(seconds)} {spoken}. {MANNER}"

    return (
        f"I have no recorded time for {app} {spoken}. Either it was not used "
        f"while I was running, or it goes by another name. {MANNER}"
    )
