"""Proposing facts about the owner, and keeping them out of the prompt.

The profile goes into every reply unchanged and unsearched, which is what
makes it valuable and what makes automating it dangerous. Two things have to
hold. A fact the model invented must never reach the file — this week Whisper
turned clean Turkish into a fluent, confident, half-wrong English sentence, and
a fact extractor fed that would have written it somewhere nobody looks. And
nothing a person has not approved may reach a prompt at all.
"""

from __future__ import annotations

import pytest

from talos import profile as profile_module
from talos.memory import Exchange, Memory, load_profile
from talos.profile import Candidates, supported

PREAMBLE = """# Profile

Written for a person. Nothing may touch this part.

---

You are talking to Oğuz.
"""


@pytest.fixture
def store(tmp_path) -> Candidates:
    return Candidates(Memory(tmp_path / "memory.db"))


@pytest.fixture
def profile(tmp_path, monkeypatch):
    """A profile file of our own, so no test edits the real one."""
    path = tmp_path / "profile.md"
    path.write_text(PREAMBLE, encoding="utf-8")
    monkeypatch.setattr(profile_module, "PROFILE", path)
    monkeypatch.setattr("talos.memory.PROFILE", path)
    return path


def exchange(asked: str, answered: str = "Noted.") -> Exchange:
    from datetime import datetime

    return Exchange(when=datetime.now(), asked=asked, answered=answered)


class TestTheHonestyGuard:
    """A fact has to be quotable from the conversation it claims to come from."""

    def test_a_real_quote_is_supported(self):
        source = exchange("I have moved the BACnet driver over to Rust")
        assert supported("moved the BACnet driver over to Rust", source)

    def test_punctuation_and_case_do_not_matter(self):
        source = exchange("I write mostly C# and .NET.")
        assert supported("I WRITE MOSTLY C#, AND .NET", source)

    def test_an_invented_quote_is_not(self):
        source = exchange("What is the weather like today?")
        assert not supported("I prefer very detailed answers", source)

    def test_a_quote_too_short_to_be_evidence_is_not(self):
        # "he said yes" appears in half the log and supports anything.
        assert not supported("yes", exchange("yes"))

    def test_the_answer_counts_as_source_too(self):
        source = exchange("Remind me", "You said your standup is at nine.")
        assert supported("your standup is at nine", source)


class TestCandidates:
    def test_a_proposal_is_pending(self, store):
        assert store.propose("He works in C#", "I write mostly C#", 1)
        assert [c.fact for c in store.pending()] == ["He works in C#"]

    def test_the_same_fact_is_never_proposed_twice(self, store):
        assert store.propose("He works in C#", "I write mostly C#", 1)
        assert not store.propose("he works in c#.", "I write mostly C#", 2)
        assert len(store.pending()) == 1

    def test_a_rejected_fact_is_never_proposed_again(self, store):
        store.propose("He hates tabs", "tabs are fine actually", 1)
        store.reject(store.pending()[0].id)
        assert store.pending() == []
        assert not store.propose("He hates tabs", "tabs are fine actually", 9)
        assert store.pending() == []

    def test_an_accepted_fact_leaves_the_queue(self, store):
        store.propose("He works in C#", "I write mostly C#", 1)
        store.accept(store.pending()[0].id)
        assert store.pending() == []
        assert store.counts()["accepted"] == 1

    def test_the_watermark_survives_a_reopen(self, tmp_path):
        path = tmp_path / "memory.db"
        Candidates(Memory(path)).scanned_to = 42
        assert Candidates(Memory(path)).scanned_to == 42

    def test_nothing_scanned_starts_at_zero(self, store):
        assert store.scanned_to == 0


class TestPromotion:
    def test_a_promoted_fact_reaches_the_body(self, profile):
        assert profile_module.promote("He works in C#", max_chars=1400)
        assert "He works in C#." in load_profile()

    def test_the_preamble_is_untouched(self, profile):
        profile_module.promote("He works in C#", max_chars=1400)
        text = profile.read_text(encoding="utf-8")
        assert text.startswith("# Profile")
        assert "Written for a person" in text
        # And it stays out of the prompt, which is the reason it may exist.
        assert "Written for a person" not in load_profile()

    def test_promoted_facts_are_readable_back(self, profile):
        profile_module.promote("He works in C#", max_chars=1400)
        profile_module.promote("He prefers small commits", max_chars=1400)
        assert profile_module.learned() == ["He works in C#.", "He prefers small commits."]

    def test_the_hand_written_body_survives(self, profile):
        profile_module.promote("He works in C#", max_chars=1400)
        assert "You are talking to Oğuz." in load_profile()

    def test_it_refuses_rather_than_truncating(self, profile):
        before = load_profile()
        assert not profile_module.promote("He works in C#", max_chars=10)
        assert load_profile() == before

    def test_a_dropped_fact_makes_room(self, profile):
        profile_module.promote("He works in C#", max_chars=1400)
        assert profile_module.drop("He works in C#.")
        assert profile_module.learned() == []
        assert "He works in C#" not in load_profile()

    def test_dropping_something_absent_changes_nothing(self, profile):
        profile_module.promote("He works in C#", max_chars=1400)
        assert not profile_module.drop("He works in Rust")
        assert profile_module.learned() == ["He works in C#."]

    def test_size_is_the_body_only(self, profile):
        assert profile_module.size() == len(load_profile())


class TestTheSplit:
    """The whole point: a candidate is not in any prompt until it is promoted."""

    def test_a_pending_candidate_is_not_in_the_profile(self, store, profile):
        store.propose("He secretly loves tabs", "tabs are fine actually", 1)
        assert "tabs" not in load_profile()
        assert profile_module.size() == len(load_profile())

    def test_only_promotion_puts_it_there(self, store, profile):
        store.propose("He works in C#", "I write mostly C#", 1)
        assert "C#" not in load_profile()
        profile_module.promote(store.pending()[0].fact, max_chars=1400)
        assert "C#" in load_profile()


class TestParsing:
    """A bounded answer sometimes stops in the middle of an object."""

    def test_ordinary_json(self):
        content = '{"facts": [{"fact": "a", "quote": "b", "exchange": 1}]}'
        assert profile_module._parse(content) == [{"fact": "a", "quote": "b", "exchange": 1}]

    def test_no_facts_is_not_an_error(self):
        assert profile_module._parse('{"facts": []}') == []

    def test_a_truncated_reply_keeps_the_objects_that_finished(self):
        # num_predict cut this off mid-object. Losing the whole batch would
        # throw away the model call that found the first two.
        content = (
            '{"facts": [{"fact": "a", "quote": "aa", "exchange": 1}, '
            '{"fact": "b", "quote": "bb", "exchange": 2}, '
            '{"fact": "c", "quote": "cc'
        )
        assert [f["fact"] for f in profile_module._parse(content)] == ["a", "b"]

    def test_nothing_usable_is_empty_not_an_exception(self):
        assert profile_module._parse("I'm afraid I can't do that") == []


class TestAlreadyKnown:
    """The model returns the profile's own wording back as a finding."""

    def test_a_fact_already_in_the_profile_is_not_proposed(self, store):
        known = "He is a software engineer, primarily C# and .NET."
        outcome = profile_module._consider(
            {"fact": "He is a software engineer, primarily C# and .NET", "quote": "x", "exchange": 1},
            {1: exchange("something else entirely")},
            known,
            store,
        )
        assert outcome == "known"
        assert store.pending() == []

    def test_a_new_fact_still_gets_through(self, store):
        source = exchange("I have started using Rust for the driver")
        outcome = profile_module._consider(
            {"fact": "He uses Rust", "quote": "started using Rust for the driver", "exchange": 1},
            {1: source},
            "He is a software engineer.",
            store,
        )
        assert outcome == "kept"
        assert [c.fact for c in store.pending()] == ["He uses Rust"]

    def test_a_wrong_exchange_id_still_finds_the_real_source(self, store):
        source = exchange("I have started using Rust for the driver")
        outcome = profile_module._consider(
            {"fact": "He uses Rust", "quote": "started using Rust for the driver", "exchange": 99},
            {7: source},
            "",
            store,
        )
        assert outcome == "kept"
        assert store.pending()[0].exchange_id == 7

    def test_an_unquotable_fact_is_refused(self, store):
        outcome = profile_module._consider(
            {"fact": "He prefers detailed answers", "quote": "I love detail", "exchange": 1},
            {1: exchange("what is the weather")},
            "",
            store,
        )
        assert outcome == "invented"
        assert store.pending() == []


class TestRoundTrip:
    """Promoting and dropping has to leave the profile exactly as it was.

    It did not. The last fact went, the empty "## Learned" heading stayed, and
    a dozen characters of nothing sat in the prompt of every reply from then
    on — a leak the profile could never lose again.
    """

    def test_promote_then_drop_leaves_no_trace(self, profile):
        before = load_profile()
        profile_module.promote("He works in C#", max_chars=1400)
        profile_module.drop("He works in C#.")
        assert load_profile() == before
        assert profile_module.size() == len(before)

    def test_the_heading_survives_while_a_fact_remains(self, profile):
        profile_module.promote("He works in C#", max_chars=1400)
        profile_module.promote("He prefers small commits", max_chars=1400)
        profile_module.drop("He works in C#.")
        assert profile_module.learned() == ["He prefers small commits."]
        assert profile_module.LEARNED_HEADING in load_profile()

    def test_room_is_actually_freed(self, profile):
        # The swap the reviewer offers when the budget is full has to work.
        budget = len(load_profile()) + 40
        assert profile_module.promote("He works in C#", budget)
        assert not profile_module.promote("He prefers small commits", budget)
        profile_module.drop("He works in C#.")
        assert profile_module.promote("He prefers small commits", budget)
