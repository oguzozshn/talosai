"""The weather, from a weather service rather than from search results.

Asked about the weather, Talos called `web_search` and then said it did not
have the current conditions — which was true. Search engine snippets for
"weather in Istanbul" describe a page rather than carrying a temperature, so
the skill fired correctly and still had nothing to report.

This uses Open-Meteo: free, no account, no key, and it returns the number.
That makes three skills that reach the internet, listed in the README with the
other two.

Nothing is sent but a place name and a coordinate.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request

from talos.skills._net import ipv4_only
from talos.skills.base import Permission, skill

log = logging.getLogger(__name__)

GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 15

# Where it is, unless asked otherwise.
DEFAULT_PLACE = "Istanbul"

# WMO weather codes, as a person would say them. The API returns a number and
# nothing else, so this table is the difference between "code 61" and "light
# rain".
CONDITIONS = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "freezing rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light showers", 81: "showers", 82: "heavy showers",
    85: "snow showers", 86: "heavy snow showers",
    95: "thunderstorms", 96: "thunderstorms with hail",
    99: "thunderstorms with hail",
}


# Coordinates do not move. Looking the same town up twice is a network call
# for an answer we already had.
_PLACES: dict[str, tuple[float, float, str]] = {}


def _get(url: str, params: dict) -> object:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(full, headers={"User-Agent": "talos"})
    with ipv4_only():
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read())


def _find_place(name: str) -> tuple[float, float, str] | None:
    cached = _PLACES.get(name.casefold())
    if cached:
        return cached

    data = _get(GEOCODE, {"name": name, "count": 1, "language": "en"})
    results = (data or {}).get("results") or []
    if not results:
        return None
    place = results[0]
    label = place.get("name", name)
    country = place.get("country")
    found = (
        place["latitude"],
        place["longitude"],
        f"{label}, {country}" if country else label,
    )
    _PLACES[name.casefold()] = found
    return found


def _describe(code: object) -> str:
    try:
        return CONDITIONS.get(int(code), "unclear")
    except (TypeError, ValueError):
        return "unclear"


@skill(Permission.OPEN)
def weather(place: str = DEFAULT_PLACE) -> str:
    """Report the current weather somewhere.

    Args:
        place: Town or city. Defaults to Istanbul.
    """
    place = place.strip() or DEFAULT_PLACE
    try:
        found = _find_place(place)
        if found is None:
            return f"I couldn't find anywhere called {place}."
        latitude, longitude, label = found

        data = _get(FORECAST, {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,apparent_temperature,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
            "forecast_days": 1,
        })
    except (urllib.error.URLError, TimeoutError):
        return "I can't reach the weather service. Check the internet connection."
    except Exception as exc:  # noqa: BLE001
        log.warning("weather lookup failed: %s", exc)
        return "The weather lookup didn't work."

    current = (data or {}).get("current") or {}
    daily = (data or {}).get("daily") or {}

    now = current.get("temperature_2m")
    feels = current.get("apparent_temperature")
    condition = _describe(current.get("weather_code"))
    if now is None:
        return f"I got a reply about {label} but no temperature in it."

    sentence = f"{label} is {round(now)} degrees and {condition}"

    # Only worth saying when it disagrees with the thermometer.
    if feels is not None and abs(feels - now) >= 3:
        sentence += f", feeling more like {round(feels)}"

    highs = (daily.get("temperature_2m_max") or [None])[0]
    lows = (daily.get("temperature_2m_min") or [None])[0]
    if highs is not None and lows is not None:
        sentence += f". Today runs between {round(lows)} and {round(highs)}"

    return sentence + "."
