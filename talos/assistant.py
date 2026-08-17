"""One turn of conversation, and the timing of every step in it.

Phase 0 measured each component alone. The number that actually matters is the
assembled one, and when it is worse than the sum of the parts, the difference
is in the wiring — so every step is timed here rather than inside the
components.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from talos.config import Config
from talos.devices import resolve
from talos.llm import Brain
from talos.memory import Memory
from talos.skills import ConfirmFn, Registry
from talos.skills import load as load_skills
from talos.stt import SpeechToText
from talos.tts import Voice, VoiceUnavailable

log = logging.getLogger(__name__)


@dataclass
class Timings:
    """Seconds spent in each stage of one turn."""

    transcribe: float = 0.0
    first_audio: float | None = None
    total: float = 0.0

    def summary(self) -> str:
        first = f"{self.first_audio:.2f}s" if self.first_audio is not None else "—"
        return (
            f"transcribe {self.transcribe:.2f}s  "
            f"first audio {first}  "
            f"total {self.total:.2f}s"
        )


@dataclass
class Turn:
    heard: str
    timings: Timings


class Assistant:
    def __init__(
        self,
        config: Config,
        *,
        mute: bool = False,
        confirm: ConfirmFn | None = None,
    ) -> None:
        self.config = config
        self.mute = mute

        # No confirm callback means skills above OPEN refuse rather than run.
        # That is the right default: an assistant with no way to ask should not
        # decide on the user's behalf.
        self.skills: Registry | None = load_skills() if config.skills.enabled else None

        self.memory = Memory() if config.assistant.remember else None

        self.ears = SpeechToText(
            config.stt,
            config.audio.sample_rate,
            workspace=Path(config.skills.workspace).expanduser(),
        )
        self.brain = Brain(
            config.llm,
            config.assistant,
            config.character,
            registry=self.skills,
            confirm=confirm,
            max_rounds=config.skills.max_rounds,
            memory=self.memory,
        )

        self.voice: Voice | None = None
        try:
            self.voice = Voice(
                config.tts,
                mute=mute,
                device=resolve(config.audio.output_device, kind="output"),
            )
        except VoiceUnavailable as exc:
            # Not fatal. Everything except the speaking works, and the replies
            # are still printed — which is enough to develop against.
            log.warning("%s", exc)

    def set_confirm(self, confirm: ConfirmFn) -> None:
        """Change how permission is asked for.

        The wake-word loop swaps in a spoken confirmation once its listener
        exists, since that cannot be built before the assistant it belongs to.
        """
        self.brain.set_confirm(confirm)

    def warm_up(self) -> None:
        """Pay every cold-start cost before the first utterance.

        Whisper compiles kernels on its first call, Ollama loads several GB
        into VRAM, and Piper starts a process. Together that is over a minute,
        and all of it would otherwise land on the user's first sentence.
        """
        self.ears.warm_up()
        self.brain.warm_up()
        if self.voice is not None:
            self.voice.warm_up()

    def respond_to_text(self, text: str) -> Turn:
        """Run a turn from text, skipping transcription. Used for testing."""
        started = time.perf_counter()
        timings = Timings()
        self._speak(self.brain.reply(text), timings, started)
        timings.total = time.perf_counter() - started
        return Turn(heard=text, timings=timings)

    def respond_to_audio(self, audio: np.ndarray) -> Turn | None:
        """Run a full turn. Returns None when nothing intelligible was said."""
        started = time.perf_counter()
        timings = Timings()

        transcribe_started = time.perf_counter()
        heard = self.ears.transcribe(audio)
        timings.transcribe = time.perf_counter() - transcribe_started

        if not heard:
            return None

        self._speak(self.brain.reply(heard), timings, started)
        timings.total = time.perf_counter() - started
        return Turn(heard=heard, timings=timings)

    def _speak(self, sentences, timings: Timings, started: float) -> None:
        """Consume the reply stream, printing it and speaking it."""
        first_seen: float | None = None

        def echo(stream):
            nonlocal first_seen
            for sentence in stream:
                if first_seen is None:
                    first_seen = time.perf_counter() - started
                print(f"  {sentence}")
                yield sentence

        if self.voice is None:
            for _ in echo(sentences):
                pass
            timings.first_audio = first_seen
            return

        # say_stream measures from when it is called, but the interesting
        # number is measured from the start of the turn.
        speak_started = time.perf_counter()
        offset = self.voice.say_stream(echo(sentences))
        if offset is not None:
            timings.first_audio = (speak_started - started) + offset
