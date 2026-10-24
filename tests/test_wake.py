"""Deciding whether Talos was spoken to.

Every case here came from a real session. Whisper does not reliably hear
"Talos" — it produced Kalos, Salos, Taroj, Tarot, titles and "Tell us" across
nine utterances on the first day — so the matching is fuzzy, and fuzzy
matching earns its keep only if it stays narrow enough to ignore ordinary
speech.

Both directions matter and they pull against each other. A missed wake means
saying the name again. A false wake interrupts you, and during development,
when the project is discussed constantly, an assistant that answers to every
mention of its own name is unusable.
"""

from __future__ import annotations

import pytest

from talos.wake import strip_wake_word


class TestWakes:
    """Transcriptions that must be treated as addressed to Talos."""

    @pytest.mark.parametrize(
        "heard,command",
        [
            ("Talos, what did I work on today?", "what did I work on today?"),
            ("talos what did I work on today", "what did I work on today"),
            ("Hey Talos, take a note.", "take a note."),
            ("Um, Talos, read my notes", "read my notes"),
            ("Okay, Talos.", ""),
            ("Talos", ""),
        ],
    )
    def test_the_name_is_stripped_and_the_rest_survives(self, heard, command):
        assert strip_wake_word(heard) == command

    @pytest.mark.parametrize(
        "heard",
        [
            "Kalos.",                                   # every one of these is
            "Salos, what did I work on today?",         # something Whisper
            "Taroj, what did I work on today?",         # actually returned for
            "Tarot.",                                   # the word "Talos"
            "Tallos, list my projects.",
            "Thalos what time is it",
        ],
    )
    def test_misheard_spellings_still_wake_it(self, heard):
        assert strip_wake_word(heard) is not None

    def test_capitalisation_and_punctuation_are_kept(self):
        # Matching is done on a normalised copy, but what reaches the model
        # should be a real sentence: a flattened one reads worse and loses the
        # question mark that tells it a question was asked.
        assert strip_wake_word("Talos, what did I work on today?") == (
            "what did I work on today?"
        )

    def test_the_name_split_in_two_is_still_the_name(self):
        # Whisper sometimes breaks it across a space.
        assert strip_wake_word("Tal os, list my projects") == "list my projects"


class TestStaysQuiet:
    """Transcriptions that must be ignored."""

    @pytest.mark.parametrize(
        "heard",
        [
            "What did I work on today?",
            "Hello, how are you doing?",
            "",
            "   ",
        ],
    )
    def test_speech_without_the_name_is_not_a_command(self, heard):
        assert strip_wake_word(heard) is None

    @pytest.mark.parametrize(
        "heard",
        [
            "I was telling Ahmet about Talos yesterday.",
            "The project is called Talos and it runs locally.",
            "Can you ask Talos about it later",
        ],
    )
    def test_the_name_late_in_a_sentence_is_discussion_not_address(self, heard):
        # This is why the name is only looked for in the first two words.
        # Without it, talking *about* the project summons it, which during
        # development happens constantly.
        assert strip_wake_word(heard) is None

    @pytest.mark.parametrize(
        "heard",
        [
            "Tables are ready.",     # two edits from "talos"
            "Talks start at nine.",  # one edit
            "Tales of the city.",
            "Titles are wrong.",
        ],
    )
    def test_ordinary_words_close_to_the_name_do_not_wake_it(self, heard):
        # Found by testing rather than by guessing: fuzzy matching at distance
        # two accepts "tables", so the words that collide are excluded by name.
        # That list is finishable, unlike the list of misspellings.
        assert strip_wake_word(heard) is None


class TestTheListenerItself:
    """Built for real, not stubbed. Adding hold() once deleted the line that
    made `vad` a property; every test here passed, and wake mode died on the
    first sound it heard."""

    @pytest.fixture
    def listener(self):
        from talos.config import load
        from talos.wake import WakeListener

        config = load()
        return WakeListener(config.audio, config.wake, lambda audio: "")

    def test_voice_activity_detection_loads(self, listener):
        import numpy as np
        import torch

        silence = torch.from_numpy(np.zeros(512, dtype=np.float32))
        assert 0.0 <= listener.vad(silence, 16000).item() <= 1.0

    def test_holding_ignores_the_microphone_and_drains_after(self, listener):
        with listener.hold():
            assert listener._holding.is_set()
        assert not listener._holding.is_set()
        assert listener._spoke


class TestCuttingIn:
    SAYING = "The weather in Istanbul is nineteen degrees and mostly clear today."

    @pytest.mark.parametrize(
        ("heard", "meaning"),
        [
            ("Talos, stop.", ""),
            ("Stop.", ""),
            ("Dur!", ""),
            ("Talos.", ""),
            ("Talos, what time is it in Tokyo?", "what time is it in Tokyo?"),
        ],
    )
    def test_the_owner_cutting_in(self, heard, meaning):
        from talos.wake import interruption

        assert interruption(heard, self.SAYING) == meaning

    @pytest.mark.parametrize(
        "heard",
        [
            "The weather in Istanbul is nineteen degrees",  # its own voice
            "mostly clear today",
            "We should stop the build",  # "stop" inside a sentence
        ],
    )
    def test_what_does_not_cut_in(self, heard):
        from talos.wake import interruption

        assert interruption(heard, self.SAYING) is None

    def test_its_own_name_in_its_own_voice_does_not_stop_it(self):
        from talos.wake import interruption

        assert interruption("Talos here, your assistant", "Talos here, your assistant.") is None

    def test_the_watcher_stops_the_reply(self):
        import queue
        import types

        import numpy as np

        from talos.config import load
        from talos.wake import WakeListener

        config = load()
        listener = WakeListener(config.audio, config.wake, lambda audio, hint=None: "Talos, stop.")
        # Loud blocks are speech, quiet ones silence: enough to be one utterance.
        listener._vad = lambda block, rate: types.SimpleNamespace(item=lambda: float(block.abs().max() > 0.5))
        listener._chunks = queue.Queue()
        for _ in range(20):
            listener._chunks.put(np.ones(512, dtype=np.float32))
        for _ in range(20):
            listener._chunks.put(np.zeros(512, dtype=np.float32))

        stopped = []
        with listener.interruptible(lambda: stopped.append(True), lambda: self.SAYING):
            for _ in range(50):
                if stopped:
                    break
                __import__("time").sleep(0.05)
        assert stopped and listener.interruption == ""
