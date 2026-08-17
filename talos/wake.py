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
import unicodedata
from collections.abc import Iterator

import numpy as np
import sounddevice as sd

from talos.config import AudioConfig, WakeConfig
from talos.devices import resolve

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

    def __init__(self, audio: AudioConfig, wake: WakeConfig, transcribe) -> None:
        self._audio = audio
        self._wake = wake
        self._transcribe = transcribe
        self._device = resolve(audio.input_device, kind="input")
        self._vad = None
        self._stop = threading.Event()

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

    def listen(self) -> Iterator[tuple[str, np.ndarray]]:
        """Yield (command text, audio) once per utterance that names Talos."""
        rate = self._audio.sample_rate
        # Silero v5 wants exactly 512 samples at 16 kHz.
        chunk = 512
        chunks: queue.Queue = queue.Queue()

        def callback(indata, _frames, _time, status):
            if status:
                log.debug("input status: %s", status)
            chunks.put(indata.copy().flatten())

        speech: list[np.ndarray] = []
        silence_chunks = 0
        # Silence long enough to mean the sentence ended rather than a pause
        # for breath.
        max_silence = int(self._wake.silence_seconds * rate / chunk)
        max_speech = int(self._wake.max_seconds * rate / chunk)

        import torch

        with sd.InputStream(
            samplerate=rate, channels=1, dtype="float32",
            callback=callback, blocksize=chunk, device=self._device,
        ):
            log.info("listening for '%s'", self._wake.word)
            while not self._stop.is_set():
                try:
                    block = chunks.get(timeout=0.5)
                except queue.Empty:
                    continue

                with torch.no_grad():
                    speech_probability = self.vad(torch.from_numpy(block), rate).item()

                if speech_probability >= self._wake.speech_threshold:
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
                text = self._transcribe(utterance, self._wake.word)
                if not text:
                    continue

                command = strip_wake_word(text)
                if command is None:
                    log.debug("ignored (%.1fs): %s", seconds, text)
                    continue

                log.info("woke on: %s", text)
                yield command, utterance
