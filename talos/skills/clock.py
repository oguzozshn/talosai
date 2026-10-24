"""The time somewhere else, without asking the internet.

Asked "what time is it in New York", web search returned the titles of three
live-clock pages and no time — the clock on those pages is drawn by
JavaScript, and a search snippet is text. The answer was on this machine all
along: the time zone database knows every zone's rules, daylight saving
included.

Places are matched against the zone names themselves ("America/New_York"
holds New York) and a short list of places that are not zone names — capitals
that share a zone with a bigger city, and countries with a single zone.
Anything else is said to be unknown rather than guessed.
"""

from __future__ import annotations

import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo, available_timezones

from talos.skills.base import Permission, skill

# Places people ask about that are not themselves in a zone name.
ALIASES = {
    "ankara": "Europe/Istanbul", "izmir": "Europe/Istanbul", "turkey": "Europe/Istanbul",
    "turkiye": "Europe/Istanbul", "bursa": "Europe/Istanbul", "antalya": "Europe/Istanbul",
    "san francisco": "America/Los_Angeles", "seattle": "America/Los_Angeles",
    "silicon valley": "America/Los_Angeles", "california": "America/Los_Angeles",
    "boston": "America/New_York", "washington": "America/New_York", "miami": "America/New_York",
    "texas": "America/Chicago", "dallas": "America/Chicago", "houston": "America/Chicago",
    "munich": "Europe/Berlin", "munchen": "Europe/Berlin", "hamburg": "Europe/Berlin",
    "frankfurt": "Europe/Berlin", "stuttgart": "Europe/Berlin", "cologne": "Europe/Berlin",
    "germany": "Europe/Berlin", "almanya": "Europe/Berlin",
    "uk": "Europe/London", "england": "Europe/London", "manchester": "Europe/London",
    "edinburgh": "Europe/London", "ingiltere": "Europe/London",
    "netherlands": "Europe/Amsterdam", "hollanda": "Europe/Amsterdam",
    "france": "Europe/Paris", "fransa": "Europe/Paris", "italy": "Europe/Rome",
    "spain": "Europe/Madrid", "sweden": "Europe/Stockholm", "switzerland": "Europe/Zurich",
    "zurich": "Europe/Zurich", "geneva": "Europe/Zurich", "austria": "Europe/Vienna",
    "japan": "Asia/Tokyo", "japonya": "Asia/Tokyo", "osaka": "Asia/Tokyo",
    "china": "Asia/Shanghai", "beijing": "Asia/Shanghai", "korea": "Asia/Seoul",
    "india": "Asia/Kolkata", "delhi": "Asia/Kolkata", "new delhi": "Asia/Kolkata",
    "mumbai": "Asia/Kolkata", "bangalore": "Asia/Kolkata",
    "dubai": "Asia/Dubai", "uae": "Asia/Dubai", "abu dhabi": "Asia/Dubai",
    "sydney": "Australia/Sydney", "canberra": "Australia/Sydney",
    "utc": "UTC", "gmt": "UTC",
}


def _key(text: str) -> str:
    """Lowercase, accents dropped, underscores as spaces: "São_Paulo" -> "sao paulo"."""
    text = text.replace("ı", "i").replace("İ", "i")
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return " ".join(plain.replace("_", " ").casefold().split())


def _zones() -> dict[str, str]:
    """The last part of every zone name, as a place: "new york" -> America/New_York."""
    zones: dict[str, str] = {}
    for name in sorted(available_timezones()):
        if "/" in name and not name.startswith(("Etc/", "SystemV/", "posix/", "right/")):
            zones.setdefault(_key(name.rsplit("/", 1)[1]), name)
    return zones


ZONES = _zones()


def zone_for(place: str) -> str | None:
    wanted = _key(place)
    return ALIASES.get(wanted) or ZONES.get(wanted)


@skill(Permission.OPEN)
def time_in(place: str = "") -> str:
    """Tell the current time in a city or country, worked out here without the internet; empty for the time here.

    Args:
        place: A city or country, such as New York, Tokyo or Germany

    Examples:
        what time is it in New York
        what's the time in Tokyo right now
        is it night in Sydney
        Almanya'da saat kaç
    """
    if not place.strip():
        return f"It's {datetime.now().strftime('%H:%M on %A')} here."

    zone = zone_for(place)
    if zone is None:
        return (
            f"I don't know which time zone {place} is in. Name the nearest big "
            f"city, or I can look it up online."
        )

    there = datetime.now(ZoneInfo(zone))
    here = datetime.now().astimezone()
    hours = (there.utcoffset() - here.utcoffset()).total_seconds() / 3600
    if hours == 0:
        difference = "the same as here"
    else:
        amount = f"{abs(hours):g} hour{'s' if abs(hours) != 1 else ''}"
        difference = f"{amount} {'ahead of' if hours > 0 else 'behind'} here"
    day = "" if there.date() == here.date() else f", {there.strftime('%A')}"
    return f"In {place.strip()} it's {there.strftime('%H:%M')}{day} — {difference}."
