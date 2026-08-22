"""What --diagnostic prints, and about the turns that print nothing else.

Every case here is a failure that actually happened. A headset was handing
over audio that no decoder could read; the microphone the Windows default
pointed at was capturing a room nobody was in; the decoder was pinned to
English while the speaking was in Turkish. All three printed the same line —
"(nothing heard)" or "(nothing recognised)" — and none of them printed a
single number to tell them apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from talos.assistant import ToolCall, Turn, Timings
from talos.audio import Recorder
from talos.config import AudioConfig


@pytest.fixture
def recorder() -> Recorder:
    # An empty input_device resolves to the system default without asking
    # PortAudio anything, so this needs no sound card.
    return Recorder(AudioConfig(sample_rate=16_000, min_seconds=0.4, silence_threshold=0.01))


def tone(seconds: float, amplitude: float) -> np.ndarray:
    samples = int(seconds * 16_000)
    return (amplitude * np.sin(np.linspace(0, 220 * seconds, samples))).astype(np.float32)


class TestDescribe:
    def test_a_quiet_room_says_so(self, recorder):
        described = recorder.describe(tone(2.0, 0.0001))
        assert "peak under silence_threshold" in described
        assert "peak 0.0001" in described

    def test_a_misfired_key_says_so(self, recorder):
        assert "shorter than min_seconds" in recorder.describe(tone(0.1, 0.5))

    def test_audio_that_should_have_worked_says_so(self, recorder):
        # The case that matters: past the gate, so whatever went wrong next
        # was not the microphone level.
        assert "loud enough and long enough" in recorder.describe(tone(2.0, 0.5))

    def test_it_reports_the_numbers_the_gate_used(self, recorder):
        described = recorder.describe(tone(2.0, 0.5))
        assert "2.0s" in described
        assert "peak 0.50" in described
        assert "system default" in described

    def test_empty_audio_does_not_divide_by_zero(self, recorder):
        assert "0.0s" in recorder.describe(np.zeros(0, dtype=np.float32))


class TestSave:
    def test_it_writes_something_that_can_be_read_back(self, recorder, tmp_path):
        import soundfile as sf

        audio = tone(1.0, 0.5)
        written = recorder.save(audio, tmp_path / "unrecognised.wav")
        assert written is not None

        read_back, rate = sf.read(written, dtype="float32")
        assert rate == 16_000
        assert np.allclose(read_back, audio, atol=1e-4)

    def test_it_makes_the_directory(self, recorder, tmp_path):
        assert recorder.save(tone(1.0, 0.5), tmp_path / "missing" / "a.wav") is not None

    def test_nothing_captured_writes_nothing(self, recorder, tmp_path):
        target = tmp_path / "empty.wav"
        assert recorder.save(np.zeros(0, dtype=np.float32), target) is None
        assert not target.exists()


class TestTurnDiagnostic:
    def test_it_shows_what_was_heard(self):
        turn = Turn(heard="what did I work on today?", timings=Timings())
        assert '"what did I work on today?"' in turn.diagnostic()

    def test_a_plain_answer_says_no_skills_ran(self):
        assert "skills     none" in Turn(heard="hello", timings=Timings()).diagnostic()

    def test_a_skill_shows_its_arguments_and_what_it_returned(self):
        turn = Turn(
            heard="what's the weather",
            timings=Timings(),
            tool_calls=[ToolCall("weather", "open", {"place": "Ankara"}, "18 degrees")],
        )
        diagnostic = turn.diagnostic()
        assert "weather(place='Ankara')  [open]" in diagnostic
        assert "-> 18 degrees" in diagnostic

    def test_a_long_result_is_cut_rather_than_flooding_the_terminal(self):
        turn = Turn(
            heard="read the file",
            timings=Timings(),
            tool_calls=[ToolCall("read_file", "read", {}, "x" * 500)],
        )
        line = [n for n in turn.diagnostic().splitlines() if "->" in n][0]
        assert line.endswith("...")
        assert len(line) < 250

    def test_the_practice_note_is_shown_when_there_is_one(self):
        turn = Turn(heard="thought", timings=Timings(), note="'th' was unclear")
        assert "'th' was unclear" in turn.diagnostic()


class TestTimings:
    def test_rounds_are_hidden_for_an_ordinary_answer(self):
        assert "rounds" not in Timings(rounds=1).summary()

    def test_rounds_are_shown_when_a_skill_ran(self):
        assert "rounds 2" in Timings(rounds=2).summary()
