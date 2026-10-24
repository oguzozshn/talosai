"""Text to speech, via the Piper binary.

Two things here exist to cut perceived latency rather than actual latency.

Sentences are synthesised and played on separate threads, so sentence two is
being generated while sentence one is still being spoken. And Piper is asked
for raw audio on stdout instead of a wav file, which skips a write and a read
per sentence.

The result is that the assistant starts talking after its first clause rather
than after its last, which is most of the difference between feeling
conversational and feeling like a form submission.
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import sounddevice as sd

from talos import language
from talos.config import TtsConfig

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PIPER_DIR = ROOT / "vendor" / "piper"
VOICE_DIR = ROOT / "models" / "piper"

_STOP = object()


class VoiceUnavailable(RuntimeError):
    pass


class Voice:
    """Speaks text aloud, one sentence at a time, in order."""

    def __init__(
        self, config: TtsConfig, *, mute: bool = False, device: int | None = None
    ) -> None:
        self._config = config
        self._device = device
        # Muting belongs to the output device, not to the layer: synthesis
        # still runs and is still timed, so a muted run exercises everything
        # except the speaker.
        self._mute = mute
        self._binary, self._voice = self._locate()
        self._sample_rate = self._read_sample_rate(self._voice)
        # The voice for Turkish sentences. The English voice reading Turkish
        # pronounces every word as if it were English; this one is Piper's
        # only Turkish voice. None if it is not there, and then the English
        # voice reads everything, as before.
        self._turkish = self._find(config.turkish_voice) if config.turkish_voice else None
        self._turkish_rate = self._read_sample_rate(self._turkish) if self._turkish else None

    def _play(self, audio: np.ndarray, rate: int | None = None) -> None:
        if self._mute:
            return
        sd.play(audio, rate or self._sample_rate, device=self._device)
        sd.wait()

    def _find(self, fragment: str) -> Path | None:
        voices = sorted(VOICE_DIR.glob("*.onnx")) if VOICE_DIR.is_dir() else []
        return next((v for v in voices if fragment.casefold() in v.stem.casefold()), None)

    def _voice_for(self, text: str) -> tuple[Path, int]:
        if self._turkish is not None and language.detect(text) == language.TURKISH:
            return self._turkish, self._turkish_rate
        return self._voice, self._sample_rate

    def _locate(self) -> tuple[Path, Path]:
        binary = PIPER_DIR / "piper.exe"
        if not binary.exists():
            found = shutil.which("piper")
            if not found:
                raise VoiceUnavailable(
                    f"Piper not found at {binary}. Run: run scripts\\fetch_piper.py"
                )
            binary = Path(found)

        voices = sorted(VOICE_DIR.glob("*.onnx")) if VOICE_DIR.is_dir() else []
        if not voices:
            raise VoiceUnavailable(
                f"No voice in {VOICE_DIR}. Run: run scripts\\fetch_piper.py"
            )

        wanted = (self._config.voice or "").casefold()
        if wanted:
            matches = [v for v in voices if wanted in v.stem.casefold()]
            if matches:
                return binary, matches[0]
            # Falling back rather than failing: a missing voice should sound
            # wrong, not stop the assistant from starting.
            log.warning(
                "No voice matching '%s' in %s, using %s",
                self._config.voice, VOICE_DIR, voices[0].stem,
            )
        return binary, voices[0]

    @staticmethod
    def _read_sample_rate(voice: Path) -> int:
        """Piper emits raw PCM with no header, so the rate has to come from the config."""
        config_path = voice.with_suffix(".onnx.json")
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            return int(data["audio"]["sample_rate"])
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            log.warning("Could not read sample rate from %s, assuming 22050", config_path.name)
            return 22_050

    @property
    def voice_name(self) -> str:
        return self._voice.stem

    def _synthesise(self, text: str) -> tuple[np.ndarray, int] | None:
        """Run one sentence through Piper, in the voice for its language.

        Returns float32 mono samples and their rate, which differs between
        voices.
        """
        voice, rate = self._voice_for(text)
        try:
            proc = subprocess.run(
                [
                    str(self._binary),
                    "--model", str(voice),
                    "--output_raw",
                    "--length_scale", str(self._config.length_scale),
                    "--noise_w", str(self._config.noise_w),
                    "--noise_scale", str(self._config.noise_scale),
                    "--sentence_silence", str(self._config.sentence_silence),
                    "--quiet",
                ],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=60,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            log.error("Piper failed: %s", exc.stderr.decode("utf-8", "replace").strip())
            return None
        except subprocess.TimeoutExpired:
            log.error("Piper timed out on: %.40s", text)
            return None

        if not proc.stdout:
            return None
        # --output_raw is signed 16-bit little-endian mono.
        pcm = np.frombuffer(proc.stdout, dtype="<i2")
        return pcm.astype(np.float32) / 32768.0, rate

    def say(self, text: str) -> None:
        """Speak a single string, blocking until it finishes."""
        spoken = self._synthesise(text)
        if spoken is not None and spoken[0].size:
            self._play(*spoken)

    def say_stream(self, sentences: Iterable[str]) -> float | None:
        """Speak sentences as they arrive. Returns seconds until the first audio.

        The caller's iterator is consumed on this thread, so a slow producer
        (the model still generating) does not stall playback of what has
        already been synthesised.
        """
        started = time.perf_counter()
        first_audio: float | None = None

        pending: queue.Queue = queue.Queue()
        ready: queue.Queue = queue.Queue(maxsize=4)

        def synth_worker() -> None:
            while True:
                item = pending.get()
                if item is _STOP:
                    ready.put(_STOP)
                    return
                spoken = self._synthesise(item)
                if spoken is not None and spoken[0].size:
                    ready.put(spoken)

        def play_worker() -> None:
            nonlocal first_audio
            while True:
                item = ready.get()
                if item is _STOP:
                    return
                if first_audio is None:
                    first_audio = time.perf_counter() - started
                self._play(*item)

        synth = threading.Thread(target=synth_worker, daemon=True)
        play = threading.Thread(target=play_worker, daemon=True)
        synth.start()
        play.start()

        try:
            for sentence in sentences:
                pending.put(sentence)
        finally:
            pending.put(_STOP)

        synth.join()
        play.join()
        return first_audio

    def warm_up(self) -> None:
        """Start the binary once so the first real sentence does not pay for it."""
        try:
            self._synthesise("ready")
            if self._turkish is not None:
                self._synthesise("hazırım")
        except Exception as exc:  # noqa: BLE001 — warm-up is best effort
            log.warning("Voice warm-up failed: %s", exc)

    @staticmethod
    def stop() -> None:
        sd.stop()
