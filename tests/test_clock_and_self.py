"""Telling the time elsewhere, and telling the owner how to ask for things."""

from __future__ import annotations

import pytest

from talos.skills import clock
from talos.skills.about_self import about_me


@pytest.mark.parametrize(
    ("place", "zone"),
    [
        ("New York", "America/New_York"),
        ("new york", "America/New_York"),
        ("São Paulo", "America/Sao_Paulo"),
        ("Sao Paulo", "America/Sao_Paulo"),
        ("Ankara", "Europe/Istanbul"),
        ("Almanya", "Europe/Berlin"),
        ("İzmir", "Europe/Istanbul"),
    ],
)
def test_places_find_their_zone(place, zone):
    assert clock.zone_for(place) == zone


def test_an_unknown_place_is_said_to_be_unknown():
    assert "don't know" in clock.time_in("Atlantis")


def test_no_place_means_here():
    assert "here" in clock.time_in("")


class TestHowToAsk:
    def test_an_area_answer_gives_things_to_say(self):
        text = about_me("files")
        assert "You could say" in text and '"open the Dersler folder"' in text

    def test_an_area_is_found_from_a_loose_description(self):
        assert "sending files" in about_me("sending a file")
        assert "your day" in about_me("my day")

    def test_nothing_matching_says_so(self):
        assert "nothing for astrology" in about_me("astrology")
