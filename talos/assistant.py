"""One turn of conversation, and the timing of every step in it.

Phase 0 measured each component alone. The number that actually matters is the
assembled one, and when it is worse than the sum of the parts, the difference
is in the wiring — so every step is timed here rather than inside the
components.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from talos.coach import pronunciation_note
from talos.config import Config
from talos.devices import resolve
from talos.llm import Brain
from talos.memory import Memory
from talos.session import current as session
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
    # 1 for a plain answer; more means at least one skill call happened. Only
    # shown when it says something — see Brain.last_rounds for why it costs
    # what it costs.
    rounds: int = 1

    def summary(self) -> str:
        first = f"{self.first_audio:.2f}s" if self.first_audio is not None else "—"
        extra = f"  rounds {self.rounds}" if self.rounds > 1 else ""
        return (
            f"transcribe {self.transcribe:.2f}s  "
            f"first audio {first}  "
            f"total {self.total:.2f}s{extra}"
        )


@dataclass
class ToolCall:
    """One skill invocation, and what it returned."""

    name: str
    permission: str
    arguments: dict
    result: str


@dataclass
class Turn:
    heard: str
    timings: Timings
    # Empty unless this turn was in practice mode and something was hard to
    # make out. See talos/coach.py.
    note: str = ""
    # Empty whenever `think` is off in settings.toml, which is the default.
    thinking: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # What was said back, in full. Spoken turns print and speak it as it
    # streams; a Telegram turn sends this.
    reply: str = ""

    def diagnostic(self) -> str:
        """Everything a plain run does not show: heard, thought, called, got back.

        Kept separate from timings.summary() rather than folded into it —
        most turns should stay quiet on the terminal, and this is for when
        something needs explaining, not for every turn. --diagnostic decides
        whether it gets printed at all.
        """
        lines = [f'  heard      "{self.heard}"']
        if self.note:
            lines.append(f"  note       {self.note}")
        if self.thinking:
            lines.append(f"  thinking   {self.thinking.strip()}")
        if not self.tool_calls:
            lines.append("  skills     none")
        else:
            for call in self.tool_calls:
                args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
                result = call.result if len(call.result) <= 200 else call.result[:197] + "..."
                lines.append(f"  skill      {call.name}({args})  [{call.permission}]")
                lines.append(f"             -> {result}")
        return "\n".join(lines)


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
        # One turn at a time. Telegram answers on its own thread while the
        # microphone loop runs on another, and the brain keeps one
        # conversation: two turns interleaved would mix their skill calls and
        # their history.
        self._turn = threading.Lock()

        # No confirm callback means skills above OPEN refuse rather than run.
        # That is the right default: an assistant with no way to ask should not
        # decide on the user's behalf.
        self.skills: Registry | None = load_skills() if config.skills.enabled else None

        self.memory = Memory() if config.assistant.remember else None

        self.ears = SpeechToText(
            config.stt,
            config.audio.sample_rate,
            workspaces=[Path(w).expanduser() for w in config.skills.workspaces],
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

    def _tool_calls(self) -> list[ToolCall]:
        """The last turn's skill calls, paired with what they returned.

        Read from the brain right after a reply is fully drained — that is
        the point at which Brain.last_tool_calls / last_tool_results are
        settled for the turn that just ran.
        """
        calls = []
        for (name, arguments), result in zip(
            self.brain.last_tool_calls, self.brain.last_tool_results
        ):
            skill = self.skills.get(name) if self.skills else None
            permission = skill.permission.value if skill else "?"
            calls.append(ToolCall(name, permission, arguments, result))
        return calls

    def respond_to_text(self, text: str) -> Turn:
        """Run a turn from text, skipping transcription, and say the answer."""
        with self._turn:
            started = time.perf_counter()
            timings = Timings()
            reply = self._speak(self.brain.reply(text), timings, started)
            timings.total = time.perf_counter() - started
            timings.rounds = self.brain.last_rounds or 1
            return Turn(
                heard=text,
                timings=timings,
                thinking=self.brain.last_thinking,
                tool_calls=self._tool_calls(),
                reply=reply,
            )

    def respond_in_writing(self, text: str, confirm: ConfirmFn) -> Turn:
        """Run a turn whose answer is written back rather than spoken.

        For Telegram: the person is not in the room, so the answer is not
        read out to it, and permission is asked where they are — through
        `confirm` — for this turn only.
        """
        with self._turn:
            started = time.perf_counter()
            timings = Timings()
            previous = self.brain.confirm
            self.brain.set_confirm(confirm)
            try:
                sentences = list(self._guarded(self.brain.reply(text)))
            finally:
                self.brain.set_confirm(previous)
            timings.total = time.perf_counter() - started
            timings.rounds = self.brain.last_rounds or 1
            return Turn(
                heard=text,
                timings=timings,
                thinking=self.brain.last_thinking,
                tool_calls=self._tool_calls(),
                reply=" ".join(sentences),
            )

    def hear(self, audio: np.ndarray) -> str:
        """Transcribe audio that did not come through the microphone, such as a voice message."""
        with self._turn:
            return self.ears.transcribe(audio)

    def respond_to_audio(self, audio: np.ndarray) -> Turn | None:
        """Run a full turn. Returns None when nothing intelligible was said."""
        with self._turn:
            return self._respond_to_audio(audio)

    def _respond_to_audio(self, audio: np.ndarray) -> Turn | None:
        started = time.perf_counter()
        timings = Timings()

        transcribe_started = time.perf_counter()
        note = ""
        if session.coaching:
            # Only while practising: word timestamps cost extra alignment work
            # for a signal nothing else uses.
            heard, words = self.ears.transcribe_words(audio)
            note = pronunciation_note(words)
        else:
            heard = self.ears.transcribe(audio)
        timings.transcribe = time.perf_counter() - transcribe_started

        if not heard:
            return None

        reply = self._speak(self.brain.reply(heard, note=note), timings, started)
        timings.total = time.perf_counter() - started
        timings.rounds = self.brain.last_rounds or 1
        return Turn(
            heard=heard,
            timings=timings,
            note=note,
            thinking=self.brain.last_thinking,
            tool_calls=self._tool_calls(),
            reply=reply,
        )

    def _guarded(self, sentences):
        """Yield a reply, turning a failure into something sayable.

        The model runs in a separate process that can stop — the machine
        sleeps, the tray app exits, someone runs ollama stop. Before this, that
        surfaced as a stack trace and a dead program in the middle of a
        conversation. An assistant should be able to say that it cannot think
        right now.
        """
        try:
            yield from sentences
        except Exception as exc:  # noqa: BLE001 — one bad turn is not a crash
            # Not `except ConnectionError`. The client raises httpx.ConnectError,
            # which descends from Exception and not from the builtin, so the
            # first version of this caught nothing and printed a full traceback
            # for the most ordinary failure there is.
            if "Connect" in type(exc).__name__ or isinstance(exc, (ConnectionError, OSError)):
                log.warning("model unreachable: %s", type(exc).__name__)
                yield "I can't reach the model right now. Is Ollama running?"
            else:
                log.exception("turn failed")
                yield f"Something went wrong while I was thinking: {type(exc).__name__}."

    def _speak(self, sentences, timings: Timings, started: float) -> str:
        """Consume the reply stream, printing it and speaking it. Returns what was said."""
        sentences = self._guarded(sentences)
        first_seen: float | None = None
        said: list[str] = []

        def echo(stream):
            nonlocal first_seen
            for sentence in stream:
                if first_seen is None:
                    first_seen = time.perf_counter() - started
                print(f"  {sentence}")
                said.append(sentence)
                yield sentence

        if self.voice is None:
            for _ in echo(sentences):
                pass
            timings.first_audio = first_seen
            return " ".join(said)

        # say_stream measures from when it is called, but the interesting
        # number is measured from the start of the turn.
        speak_started = time.perf_counter()
        offset = self.voice.say_stream(echo(sentences))
        if offset is not None:
            timings.first_audio = (speak_started - started) + offset
        return " ".join(said)
