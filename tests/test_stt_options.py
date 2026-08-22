"""Which language the decoder is told to expect.

Whisper does not fail loudly when the pin is wrong. Forced to English, a clean
Turkish recording of "Benim adım Oğuz" came back as 'My name is Oğuz Talos.' —
fluent, confident, and missing half the sentence. The assistant then answers
the paraphrase, which is the kind of fault you debug for an hour because
nothing anywhere reports an error.

So an empty language has to reach faster-whisper as None, the value that means
detect it, rather than as "", which is not a language code at all.
"""

from __future__ import annotations

import pytest

from talos.config import SttConfig
from talos.stt import SpeechToText


def options_for(language: str, hotwords: str | None = None) -> dict:
    """The decoder options, without loading a model to get at them."""
    stt = object.__new__(SpeechToText)
    stt._config = SttConfig(language=language)
    stt._vocabulary = ""
    return stt._options(hotwords)


class TestLanguage:
    @pytest.mark.parametrize("language", ["en", "tr", "de"])
    def test_a_code_is_passed_through(self, language):
        assert options_for(language)["language"] == language

    @pytest.mark.parametrize("empty", ["", None])
    def test_empty_means_detect_it(self, empty):
        # None is what faster-whisper reads as "work it out"; "" is not.
        assert options_for(empty)["language"] is None


class TestHotwords:
    def test_the_callers_hint_comes_first(self):
        assert options_for("en", "Talos")["hotwords"].startswith("Talos")

    def test_no_hint_and_no_vocabulary_means_no_hotwords(self):
        assert "hotwords" not in options_for("en")


class TestWarmUp:
    """Warm-up takes its own path to the model and has to translate too.

    It did not, and an empty language reached faster-whisper as "" —
    "'' is not a valid language code". Warm-up catches everything and logs a
    warning, so the only symptom was one line at startup and a first sentence
    that paid the kernel-compilation cost the warm-up existed to prevent.
    """

    def test_it_does_not_pass_an_empty_language(self):
        calls: dict = {}

        class FakeModel:
            def transcribe(self, audio, **options):
                calls.update(options)
                return iter([]), None

        stt = object.__new__(SpeechToText)
        stt._config = SttConfig(language="")
        stt._sample_rate = 16_000
        stt._model = FakeModel()
        stt.warm_up()

        assert calls["language"] is None

    def test_a_pinned_language_still_reaches_the_model(self):
        calls: dict = {}

        class FakeModel:
            def transcribe(self, audio, **options):
                calls.update(options)
                return iter([]), None

        stt = object.__new__(SpeechToText)
        stt._config = SttConfig(language="tr")
        stt._sample_rate = 16_000
        stt._model = FakeModel()
        stt.warm_up()

        assert calls["language"] == "tr"
