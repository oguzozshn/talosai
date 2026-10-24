"""Deciding that "remember this" was a request to write something down.

Every case here is one the assistant has to get right by itself, because the
model does not. Told "remember my girlfriend's name is Dilara" it answered
"I'll remember that" and called nothing — with the skill's own description
rewritten to say this is what it is for, and after three attempts at saying so
in the character definition. The promise is the worst kind of failure available
here: it sounds exactly like the thing working, so nobody checks, and the fact
is gone by morning.

Both directions matter. A missed request loses something the owner said out
loud and believed was kept. A false one files the weather forecast as a note.
"""

from __future__ import annotations

import pytest

from talos.skills.notes import note_request


class TestRecords:
    """Said to Talos, these mean write it down."""

    @pytest.mark.parametrize(
        "said,recorded",
        [
            ("Remember my girlfriend's name is Dilara", "my girlfriend's name is Dilara"),
            ("remember that the standup moved to ten", "the standup moved to ten"),
            ("Note that the BACnet bug is in the signed path", "the BACnet bug is in the signed path"),
            ("Make a note that I promised Ali a review", "I promised Ali a review"),
            ("write down the licence expires in March", "the licence expires in March"),
            ("Don't forget the meeting is on Friday", "the meeting is on Friday"),
            ("Keep in mind that I prefer small commits", "I prefer small commits"),
        ],
    )
    def test_english(self, said, recorded):
        assert note_request(said) == recorded

    @pytest.mark.parametrize(
        "said,recorded",
        [
            ("Hatırla: Cuma günü BACnet hatasını düzelteceğim", "Cuma günü BACnet hatasını düzelteceğim"),
            ("Kız arkadaşımın adı Dilara, bunu hatırla", "Kız arkadaşımın adı Dilara"),
            ("Toplantı Cuma günü, unutma", "Toplantı Cuma günü"),
            ("Not al: parola yöneticisini güncellemem lazım", "parola yöneticisini güncellemem lazım"),
        ],
    )
    def test_turkish_leads_or_trails(self, said, recorded):
        # Turkish puts the imperative last as readily as first, and the wake
        # listener already taught this project that word order is not optional.
        assert note_request(said) == recorded

    def test_filler_before_the_request_is_ignored(self):
        assert note_request("Talos, remember I parked on level three") == "I parked on level three"

    def test_please_is_ignored(self):
        assert note_request("please remember the code is 4417") == "the code is 4417"


class TestDoesNotRecord:
    """Said to Talos, these mean something else entirely."""

    @pytest.mark.parametrize(
        "said",
        [
            "Do you remember my girlfriend's name?",
            "do you remember what I said yesterday",
            "Can you remember the BACnet bug",
            "Hatırlıyor musun dün ne dedim",
        ],
    )
    def test_asking_is_a_search_not_a_note(self, said):
        # Answering "do you remember X?" by filing a new note would be absurd,
        # and recall already exists for it.
        assert note_request(said) is None

    @pytest.mark.parametrize(
        "said",
        [
            "What is the weather like today",
            "Read my notes",
            "How long was I in Visual Studio",
            "Bugün ne yaptım",
            "",
            "   ",
        ],
    )
    def test_ordinary_speech(self, said):
        assert note_request(said) is None

    def test_a_question_is_never_a_note(self):
        assert note_request("Should I remember to call him?") is None

    @pytest.mark.parametrize("said", ["remember", "Remember.", "hatırla", "unutma"])
    def test_the_word_on_its_own_is_half_a_sentence(self, said):
        assert note_request(said) is None


class TestTheTurnRecordsIt:
    """The matcher is only useful if the turn actually runs the skill."""

    @pytest.fixture
    def brain(self):
        from talos.llm import Brain

        class FakeRegistry:
            def __init__(self):
                self.ran = []

            def get(self, name):
                return object() if name == "take_note" else None

            def execute(self, name, arguments, confirm=None):
                self.ran.append((name, arguments))
                return f"Written down in your notes: {arguments['text']}"

        brain = object.__new__(Brain)
        brain._registry = FakeRegistry()
        brain._confirm = None
        brain.last_tool_calls = []
        brain.last_tool_results = []
        return brain

    def test_the_skill_runs(self, brain):
        messages = [{"role": "user", "content": "x"}]
        brain._record_if_asked("Remember the code is 4417", messages)
        assert brain._registry.ran == [("take_note", {"text": "the code is 4417"})]

    def test_the_model_is_told_what_was_written(self, brain):
        # So the sentence it speaks is built on what actually happened rather
        # than on what it would have promised.
        messages = [{"role": "user", "content": "x"}]
        brain._record_if_asked("Remember the code is 4417", messages)
        assert messages[-1]["role"] == "tool"
        assert "4417" in messages[-1]["content"]

    def test_it_shows_up_in_diagnostic(self, brain):
        brain._record_if_asked("Remember the code is 4417", [])
        assert brain.last_tool_calls == [("take_note", {"text": "the code is 4417"})]
        assert len(brain.last_tool_results) == 1

    def test_ordinary_speech_runs_nothing(self, brain):
        messages = [{"role": "user", "content": "x"}]
        brain._record_if_asked("What is the weather like today", messages)
        assert brain._registry.ran == []
        assert len(messages) == 1

    def test_a_deleted_skill_takes_the_behaviour_with_it(self, brain):
        # Skills are files. Deleting take_note.py has to remove this too.
        brain._registry.get = lambda name: None
        brain._record_if_asked("Remember the code is 4417", [])
        assert brain._registry.ran == []
