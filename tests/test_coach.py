"""Deciding whether a correction is really a correction.

The grader is a separate model call that sees one sentence and returns one
clause. Its most common failure is handing the sentence straight back, or
returning it with different punctuation, which is not a correction anyone can
act on and is worse than saying nothing.

Filtering that in code rather than by asking the prompt more firmly cost one
round to get right: the first version tested the wrong direction and threw
away five good corrections out of six. A real correction quotes the original
*and* the fix, so it always contains every word of the sentence. What makes it
an echo is introducing nothing new.
"""

from __future__ import annotations

import pytest

from talos.coach import Word, pronunciation_note, shaky_words
from talos.coach import _same_words


class TestEchoDetection:
    @pytest.mark.parametrize(
        "verdict",
        [
            "I went to the cinema yesterday.",              # identical
            "I went to the cinema yesterday",               # punctuation only
            "i went to the cinema yesterday.",              # case only
            "I went to the cinema, yesterday.",             # a comma
        ],
    )
    def test_the_sentence_returned_unchanged_is_not_a_correction(self, verdict):
        assert _same_words(verdict, "I went to the cinema yesterday.")

    @pytest.mark.parametrize(
        "verdict,sentence",
        [
            ('"for two years", not "since two years".',
             "I have been working here since two years."),
            ('"I want to improve", not "I want improve".',
             "I want improve my English."),
            ('"I agree with you", not "I am agree with you".',
             "I am agree with you."),
        ],
    )
    def test_a_real_correction_survives(self, verdict, sentence):
        # These contain every word of the original plus the fix. Testing
        # containment the other way round rejected all of them.
        assert not _same_words(verdict, sentence)

    def test_an_empty_sentence_is_not_echoed_by_anything(self):
        assert not _same_words("anything at all", "")


class TestShakyWords:
    def test_filler_is_never_reported(self):
        # Short unstressed words are the ones the recogniser is least sure of
        # and the ones carrying no pronunciation worth commenting on.
        words = [Word("to", 0.2), Word("a", 0.1), Word("and", 0.3)]
        assert shaky_words(words) == []

    def test_confident_speech_produces_no_note(self):
        words = [Word("hello", 0.99), Word("there", 0.97)]
        assert pronunciation_note(words) == ""

    def test_the_least_certain_content_words_come_first(self):
        words = [
            Word("vocabulary", 0.52),
            Word("pronunciation", 0.28),
            Word("improve", 0.31),
        ]
        assert shaky_words(words)[0] == "pronunciation"

    def test_a_word_is_only_reported_once(self):
        words = [Word("pronunciation", 0.2), Word("pronunciation", 0.3)]
        assert shaky_words(words) == ["pronunciation"]

    def test_the_note_is_phrased_as_a_hint_not_a_verdict(self):
        # Noise, an unusual name and a mispronunciation are indistinguishable
        # from a confidence score, so the wording must not claim otherwise.
        note = pronunciation_note([Word("pronunciation", 0.2)])
        assert "unsure" in note
        assert "not as a mistake" in note
