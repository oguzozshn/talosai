"""Listening for the wake word, without training a wake word model.

The usual approach trains a small classifier on thousands of synthetic samples
of one word. That buys very low idle cost — a percent of one CPU core — at the
price of hours of setup and days of threshold tuning, and the accuracy still
depends on getting that threshold right.

This does it differently, because this machine can afford to. Voice activity
detection finds speech, Whisper transcribes it, and the transcript is checked
for the wake word. Transcription costs 0.22 s on this GPU, so the expensive
step is cheap enough to run on every utterance.

What it buys:

- No training, no recordings, no synthetic data pipeline.
- The wake word is recognised as a *word*, by a model that already knows it,
  rather than as an acoustic pattern matched against a threshold.
- The command can follow in the same breath. "Talos, what did I work on" is
  one utterance, with no beep to wait for.

What it costs: every utterance in the room is transcribed, not just the ones
addressed to Talos. In a quiet room that is nothing. With a television on it
is a transcription every few seconds, which is power rather than a malfunction.

Whether that trade is right is a measurement, not an opinion — run with
--verbose and watch how often it wakes when it should not.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
import unicodedata
from collections.abc import Iterator

import numpy as np

from talos.config import AudioConfig, WakeConfig
from talos.devices import InputSelector, refresh

log = logging.getLogger(__name__)

# Whisper does not reliably hear "Talos". Measured on real utterances it
# produced Kalos, Salos, Taroj, Tarot, titles and "Tell us" — a proper noun it
# has no reason to expect, said by a Turkish speaker in English mode.
#
# Two defences. The first is telling Whisper the word to expect, via hotwords,
# which is what actually fixes this. The second is this: matching by edit
# distance rather than against a list, because enumerating the misspellings of
# a name is a race that cannot be won. The list below is only the spellings
# common enough to accept outright.
_VARIANTS = {
    "talos", "talus", "tallos", "thalos", "talo", "talls",
    "talas", "tholos", "tallus", "taos", "kalos", "salos", "taroj",
}

# Edit distance at or below this counts as the name. Two allows Kalos, Salos,
# Taroj and Tarot; three would start accepting ordinary words like "tables"
# and "colours", and a false wake mid-conversation is worse than a missed one
# the user can simply repeat.
_MAX_DISTANCE = 2

# Below this length, an edit distance of two is most of the word.
_MIN_FUZZY_LENGTH = 4

# Ordinary English words that land within two edits of "talos". Unlike the list
# of misspellings, this one is bounded and can be finished: the set of words
# that close to a five-letter string is small. Without it, "Tables are ready"
# wakes the assistant, which was found by testing rather than by guessing.
_NOT_THE_NAME = {
    "tables", "table", "tales", "talks", "tacos", "tools", "toes", "ties",
    "halos", "solos", "sales", "males", "males", "salad", "tall", "tabs",
    "titles", "tiles", "tanks", "tapes", "tastes", "days", "class",
}

# How often to try the pinned microphone again while the default stands in
# for it. The stream is only reopened when it dies, and the default does not
# die when the headset comes back, so without this the headset would not be
# used again until a restart. Each try closes the stream for a moment, which
# is why it only happens between utterances and not more often than this.
_RETRY_PINNED_SECONDS = 30.0


def _distance(a: str, b: str) -> int:
    """Levenshtein distance. Small inputs, so the simple version is fine."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def _is_name(token: str) -> bool:
    if token in _VARIANTS:
        return True
    # Checked after the variants, so a spelling Whisper genuinely produces
    # still wins if it ever collides with an ordinary word.
    if token in _NOT_THE_NAME or len(token) < _MIN_FUZZY_LENGTH:
        return False
    return _distance(token, "talos") <= _MAX_DISTANCE

# Leading filler Whisper likes to add, stripped before looking for the name.
_LEADING_NOISE = re.compile(r"^(?:um|uh|ah|oh|hey|hi|hello|okay|ok|so|well)[\s,]+", re.I)
_PUNCTUATION = re.compile(r"[^\w\s]")


def _normalise(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(_PUNCTUATION.sub(" ", text).lower().split())


def strip_wake_word(text: str) -> str | None:
    """Return the command with the wake word removed, or None if absent.

    The name must be at or near the start. Talos answering to its name buried
    in the middle of a sentence would mean answering whenever it is discussed,
    which during development is constantly.

    Normalisation is used for *matching* only. What comes back is a slice of
    the original, keeping capitalisation and the question mark — the model
    reads a real sentence rather than a flattened one.
    """
    original = _LEADING_NOISE.sub("", text.strip())
    if not original:
        return None

    tokens = original.split()

    # Whisper sometimes splits the name in two — "Tal os", "Ta los". Check the
    # joined pair before the singles, so the longer match wins.
    if len(tokens) >= 2:
        joined = _normalise(tokens[0]) + _normalise(tokens[1])
        if _is_name(joined):
            return " ".join(tokens[2:]).lstrip(" ,.:;-—").strip()

    # Only the first two positions, so "tell Talos about it" does not wake it
    # but "okay Talos, what next" does.
    for index in range(min(2, len(tokens))):
        if _is_name(_normalise(tokens[index])):
            remainder = " ".join(tokens[index + 1 :])
            # The name is usually followed by a comma, which now begins the
            # sentence.
            remainder = remainder.lstrip(" ,.:;-—")
            # Bare "Talos" with nothing after it is still a wake — the user is
            # getting its attention and will say the rest next.
            return remainder.strip()
    return None


class WakeListener:
    """Streams microphone audio, yielding the command after each wake word."""

    def __init__(self, audio: AudioConfig, wake: WakeConfig, transcribe, announce=None) -> None:
        self._audio = audio
        self._wake = wake
        self._transcribe = transcribe
        # announce(message) tells the person about the microphone: that the
        # pinned one dropped out or came back, or that none works at all.
        self._announce = announce
        self._microphone = InputSelector(audio.input_device, self._say)
        # Set when something was said while the stream was open, so the
        # listening loop knows to discard its own voice coming back.
        self._spoke = False
        # True from the first failed open until one succeeds, so "no
        # microphone" is said once rather than every two seconds.
        self._no_microphone = False
        self._vad = None
        self._stop = threading.Event()
        # Set while listen() is running, so a confirmation can be collected
        # from the same stream. Opening a second input stream on the same
        # device is how you find out the driver only allows one.
        self._chunks: queue.Queue | None = None

    @property
    def vad(self):
        if self._vad is None:
            from silero_vad import load_silero_vad

            # ONNX rather than torch: it is the same model, runs on the CPU in
            # about a millisecond per chunk, and keeps torch off the hot path.
            self._vad = load_silero_vad(onnx=True)
        return self._vad

    def stop(self) -> None:
        self._stop.set()

    def _say(self, message: str) -> None:
        if self._announce is not None:
            self._announce(message)
            self._spoke = True

    def _drain(self, chunks: queue.Queue) -> None:
        """Discard what the microphone picked up while Talos was speaking.

        Draining once is not enough: the last of the reply is still
        travelling from the speaker to the microphone when the queue first
        reads as empty, and arrives a moment later as a short utterance that
        transcribes to nothing. So it waits a fixed span, not "until the
        queue stays empty": the callback delivers a block every 32 ms, so a
        loop that waits for emptiness never ends.
        """
        settle_until = time.monotonic() + self._wake.settle_seconds
        while time.monotonic() < settle_until:
            try:
                chunks.get(timeout=0.05)
            except queue.Empty:
                pass
        while not chunks.empty():
            try:
                chunks.get_nowait()
            except queue.Empty:
                break

    def ask(self, timeout: float = 8.0, hint: str = "yes no") -> str:
        """Capture and transcribe one short reply, for a confirmation.

        Reads from the stream listen() already has open rather than opening
        another. It is called from inside a turn — the generator is paused at
        its yield, the callback is still filling the queue — so the audio is
        there for the taking.

        Returns "" if nothing was said before the timeout, which the caller
        must treat as "no": silence is not consent.
        """
        chunks = self._chunks
        if chunks is None:
            return ""

        import torch

        rate = self._audio.sample_rate
        block_size = 512
        max_silence = int(0.6 * rate / block_size)

        # Drop whatever the question itself put into the microphone.
        settle_until = time.monotonic() + self._wake.settle_seconds
        while time.monotonic() < settle_until:
            try:
                chunks.get(timeout=0.05)
            except queue.Empty:
                pass

        speech: list[np.ndarray] = []
        silence = 0
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                block = chunks.get(timeout=0.3)
            except queue.Empty:
                continue

            with torch.no_grad():
                probability = self.vad(torch.from_numpy(block), rate).item()

            if probability >= self._wake.speech_threshold:
                speech.append(block)
                silence = 0
            elif speech:
                speech.append(block)
                silence += 1
                if silence >= max_silence:
                    break

        if not speech:
            return ""
        return self._transcribe(np.concatenate(speech), hint)

    def listen(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield (command text, audio) once per utterance addressed to Talos.

        An empty command means the name was said on its own. That opens a
        short window during which the next utterance is taken as the command
        without the name — because people say "Talos", pause, and then ask.
        Requiring one breath is how the assistant feels, not how anyone talks.
        """
        rate = self._audio.sample_rate
        # Silero v5 wants exactly 512 samples at 16 kHz.
        chunk = 512
        chunks: queue.Queue = queue.Queue()
        self._chunks = chunks

        def callback(indata, _frames, _time, status):
            if status:
                log.debug("input status: %s", status)
            chunks.put(indata.copy().flatten())

        speech: list[np.ndarray] = []
        silence_chunks = 0
        # While this is in the future, an utterance without the name is still
        # taken as a command.
        awake_until = 0.0
        # Consecutive half-seconds with no audio delivered at all.
        starved = 0
        # When to next try the pinned microphone, if the default is in use.
        retry_pinned_at = time.monotonic() + _RETRY_PINNED_SECONDS
        # When the current utterance began. Zero means none is in progress.
        speech_started = 0.0
        # Silence long enough to mean the sentence ended rather than a pause
        # for breath.
        max_silence = int(self._wake.silence_seconds * rate / chunk)
        max_speech = int(self._wake.max_seconds * rate / chunk)

        import torch

        # Through _reopen, so a first open that fails is handled like a
        # microphone lost later: said once and retried, instead of a crash
        # before the assistant has heard a word.
        stream = self._reopen(None, rate, chunk, callback)
        try:
            log.info("listening for '%s'", self._wake.word)
            while not self._stop.is_set():
                if self._spoke:
                    # A notice about the microphone was just spoken into it.
                    self._drain(chunks)
                    self._spoke = False
                    speech, silence_chunks, speech_started = [], 0, 0.0

                try:
                    block = chunks.get(timeout=0.5)
                except queue.Empty:
                    # No audio at all, not even silence. The callback should
                    # deliver a block every 32 ms, so this means the stream has
                    # stopped — almost always because the microphone it was
                    # opened on was unplugged — or that none could be opened.
                    starved += 1
                    if starved >= 4:
                        if stream is not None:
                            log.warning("no audio for 2s, reopening the microphone")
                        stream = self._reopen(stream, rate, chunk, callback)
                        starved = 0
                    continue
                starved = 0

                now = time.monotonic()
                if (
                    self._microphone.fallen_back
                    and not speech
                    and now >= awake_until
                    and now >= retry_pinned_at
                ):
                    retry_pinned_at = now + _RETRY_PINNED_SECONDS
                    stream = self._reopen(stream, rate, chunk, callback)
                    continue

                with torch.no_grad():
                    speech_probability = self.vad(torch.from_numpy(block), rate).item()

                if speech_probability >= self._wake.speech_threshold:
                    if not speech:
                        speech_started = time.monotonic()
                    speech.append(block)
                    silence_chunks = 0
                elif speech:
                    silence_chunks += 1
                    # Keep the trailing silence: cutting at the exact moment
                    # speech stops clips the final consonant, and Whisper
                    # guesses badly at a truncated last word.
                    speech.append(block)

                ended = speech and silence_chunks >= max_silence
                too_long = len(speech) >= max_speech
                if not (ended or too_long):
                    continue

                utterance = np.concatenate(speech)
                speech, silence_chunks = [], 0

                seconds = len(utterance) / rate
                if seconds < self._wake.min_seconds:
                    continue

                # Tell Whisper the name to expect. Without this it produces a
                # different spelling almost every time, and no amount of
                # matching afterwards recovers reliably from that.
                # Measured from when speaking *started*, not from now. Checking
                # at the end of the utterance means a long sentence expires the
                # window it began inside — an eight second window rejects an
                # eight second reply that started immediately, which is exactly
                # the sentence someone gives when asked to remember something.
                awake = speech_started < awake_until
                text = self._transcribe(utterance, self._wake.word)
                if not text:
                    # Whisper heard speech and produced nothing. Worth seeing
                    # while the window is open, because from the outside it is
                    # indistinguishable from not having listened at all.
                    log.info("heard %.1fs but transcribed nothing (awake=%s)", seconds, awake)
                    continue

                command = strip_wake_word(text)
                if command is None and awake:
                    # Inside the follow-up window, the whole utterance is the
                    # command. The name was already said.
                    command = text.strip()
                    log.info("follow-up: %s", text)
                elif command is None:
                    # Ignoring speech while the window is open means something
                    # is wrong, not that the room was noisy — say so loudly.
                    level = log.info if awake else log.debug
                    level("ignored (%.1fs, awake=%s): %s", seconds, awake, text)
                    continue
                else:
                    log.info("woke on: %s", text)

                # An empty command is the name on its own; the caller
                # acknowledges it. Either way, control does not come back here
                # until the answer has finished being spoken.
                yield command, utterance

                # Whatever was captured while the caller was answering is the
                # assistant's own voice coming back through the microphone, or
                # the user waiting. Either way it is not the next command.
                self._drain(chunks)
                self._spoke = False

                speech, silence_chunks, speech_started = [], 0, 0.0

                # Start the window now rather than before yielding: the reply
                # takes seconds to speak, and a window opened beforehand is
                # mostly gone by the time the person can talk.
                window = (
                    self._wake.follow_up_seconds
                    if not command
                    else self._wake.continue_seconds
                )
                awake_until = time.monotonic() + window if window > 0 else 0.0
        finally:
            self._close(stream)


    def _open_stream(self, rate: int, chunk: int, callback):
        """Open the microphone, resolving the device fresh each time.

        Resolved here rather than once in __init__: a device index that was
        right at startup points at nothing after the headset it named is
        unplugged. The pinned device is tried first every time, and the
        system default stands in when it cannot be opened.
        """
        return self._microphone.open(
            samplerate=rate, channels=1, dtype="float32",
            callback=callback, blocksize=chunk,
        )

    @staticmethod
    def _close(stream) -> None:
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as exc:  # noqa: BLE001 — it may already be gone
            log.debug("could not close the input stream: %s", exc)

    def _reopen(self, stream, rate: int, chunk: int, callback):
        """Close, re-read the device list, and open again.

        The refresh is the part that matters. PortAudio caches its device list
        at initialisation, so after a headset is unplugged it keeps offering
        the device that left and keeps resolving "the default" to it — opening
        a new stream without refreshing gets the same dead microphone back.
        """
        self._close(stream)
        refresh()
        try:
            reopened = self._open_stream(rate, chunk, callback)
        except Exception as exc:  # noqa: BLE001
            # Nothing usable right now, not even the default. Returning None
            # keeps the loop alive so it can try again when something is
            # plugged back in.
            if not self._no_microphone:
                self._no_microphone = True
                log.warning("no microphone available: %s", exc)
                self._say("I can't find a working microphone. I'll keep trying.")
            return None
        log.debug("microphone opened")
        if self._no_microphone:
            self._no_microphone = False
            log.warning("a microphone is available again")
            self._say("I can hear you again.")
        return reopened
