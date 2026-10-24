"""Answering in the language spoken to, and choosing the voice to say it in."""

from __future__ import annotations

import pytest

from talos import language
from talos.skills.day import day_request


@pytest.mark.parametrize(
    "text",
    ["Merhaba, nasılsın?", "Bugün ne yapmam lazım?", "RAG ne demek?", "Almanya'da saat kaç?", "bugun ne var"],
)
def test_turkish_is_recognised(text):
    assert language.detect(text) == language.TURKISH


@pytest.mark.parametrize(
    "text",
    ["What is on my plate today?", "Open the Dersler folder.", "What did Oğuz say about Gülşah?", "One moment."],
)
def test_english_with_turkish_names_stays_english(text):
    assert language.detect(text) == language.ENGLISH


def test_what_whisper_heard_wins_over_the_text():
    assert language.resolve("tr", "Okay") == language.TURKISH
    assert language.resolve("de", "Bugün ne var?") == language.TURKISH


@pytest.mark.parametrize(
    ("text", "day"),
    [
        ("Bugün ne yapmam lazım?", "today"),
        ("Bugün ne yapmalıyım?", "today"),
        ("Yarın ne var?", "tomorrow"),
        ("Yarın programım ne?", "tomorrow"),
        ("What do I need to do today?", "today"),
        ("Bugün hava nasıl?", None),
        ("Bugün ne giysem?", None),
    ],
)
def test_the_days_question_is_recognised_before_the_model(text, day):
    assert day_request(text) == day


def test_a_turkish_sentence_gets_the_turkish_voice(tmp_path, monkeypatch):
    from talos import tts
    from talos.config import TtsConfig

    for name in ("en_US-lessac-medium", "tr_TR-dfki-medium"):
        (tmp_path / f"{name}.onnx").write_bytes(b"")
        (tmp_path / f"{name}.onnx.json").write_text('{"audio": {"sample_rate": 22050}}', encoding="utf-8")
    (tmp_path / "piper.exe").write_bytes(b"")
    monkeypatch.setattr(tts, "VOICE_DIR", tmp_path)
    monkeypatch.setattr(tts, "PIPER_DIR", tmp_path)

    voice = tts.Voice(TtsConfig(voice="lessac", turkish_voice="tr_TR"), mute=True)
    assert voice._voice_for("Bugün iki saat çalışman var.")[0].stem == "tr_TR-dfki-medium"
    assert voice._voice_for("You have two hours of study.")[0].stem == "en_US-lessac-medium"
