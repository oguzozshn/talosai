"""The router, with a fake embedding: one dimension per word it knows."""

from __future__ import annotations

from talos.router import ALWAYS, Router
from talos.skills import Registry
from talos.skills.base import Permission, skill

WORDS = ["weather", "file", "todoist", "screen", "note"]


def fake_embed(texts):
    return [[text.lower().count(word) for word in WORDS] for text in texts]


def make(name: str, about: str):
    def function() -> str:
        return name

    function.__doc__ = about
    return skill(Permission.OPEN, name=name)(function)._skill


def registry() -> Registry:
    made = Registry()
    for name, about in [
        ("weather", "Report the weather."),
        ("find_file", "Find a file by name."),
        ("todoist_add", "Add a task to Todoist."),
        ("read_screen", "Read the screen."),
        *[(n, "Always there.") for n in ALWAYS],
    ]:
        made._skills[name] = make(name, about)
    return made


def test_the_closest_skill_is_offered_with_the_ones_always_there():
    router = Router(registry(), top=1, embed=fake_embed)
    assert router.prepare()
    offered = router.choose("what's the weather")
    assert "weather" in offered and "find_file" not in offered
    assert set(ALWAYS) <= set(offered)


def test_a_skill_just_used_is_offered_for_the_follow_up():
    router = Router(registry(), top=1, embed=fake_embed)
    router.prepare()
    router.used(["find_file"])
    assert "find_file" in router.choose("what's the weather")


def test_the_same_request_gives_the_same_order():
    router = Router(registry(), top=2, embed=fake_embed)
    router.prepare()
    first = router.choose("weather and a file")
    assert first == sorted(first) == router.choose("weather and a file")


def test_without_embeddings_everything_is_offered():
    router = Router(registry(), embed=lambda texts: None)
    assert router.prepare() is False
    assert router.choose("anything") is None
