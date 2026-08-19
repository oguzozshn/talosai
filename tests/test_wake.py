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
