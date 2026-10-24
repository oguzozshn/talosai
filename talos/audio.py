"""Microphone capture.

Records at 16 kHz because that is what Whisper wants; recording there means no
resampling step per utterance. Mono float32 throughout, which is also what
faster-whisper accepts directly, so audio never touches the disk.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
import sounddevice as sd

from talos.config import AudioConfig
from talos.devices import InputSelector, refresh, resolve_or_default

log = logging.getLogger(__name__)


class Recorder:
    """Captures audio between start() and stop().

    Frames accumulate on the audio callback thread, so start() returns
    immediately and the caller stays responsive while the user is talking.
    """

    def __init__(self, config: AudioConfig, announce=None) -> None:
        self._config = config
        # announce(message) is told when the pinned microphone drops out or
        # comes back.
        self._microphone = InputSelector(config.input_device, announce)
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self.device = resolve_or_default(config.input_device, kind="input")

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def _callback(self, indata, _frames, _time, status) -> None:
        if status:
            log.debug("input stream status: %s", status)
        with self._lock:
            self._frames.append(indata.copy())

    def start(self) -> None:
        if self._stream is not None:
            return
        # Re-resolved on every recording rather than once at startup: the
        # headset the default pointed at may have been unplugged since, and
        # PortAudio has to be asked again before it will admit that.
        refresh()
        with self._lock:
            self._frames.clear()
        self._stream = self._microphone.open(
            samplerate=self._config.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=int(self._config.sample_rate * 0.05),
        )
        self.device = self._microphone.device

    def stop(self) -> np.ndarray:
        """Stop and return what was captured, as mono float32."""
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)

        self._stream.stop()
        self._stream.close()
        self._stream = None

        with self._lock:
            frames = list(self._frames)
            self._frames.clear()

        if not frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames, axis=0).flatten()

    def describe(self, audio: np.ndarray) -> str:
        """Why is_speech() decided what it decided, in one line.

        The gate is the quietest place a turn can die: it prints "nothing
        heard" and throws the audio away, so the two numbers that would
        explain it — how loud, how long — are gone before anyone can look. A
        microphone that is muted, and a microphone that is capturing a room
        nobody is speaking in, produce the same message and very different
        numbers.
        """
        seconds = audio.size / self._config.sample_rate
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0

        if seconds < self._config.min_seconds:
            verdict = f"shorter than min_seconds ({self._config.min_seconds}s)"
        elif peak < self._config.silence_threshold:
            verdict = f"peak under silence_threshold ({self._config.silence_threshold})"
        else:
            verdict = "loud enough and long enough"

        where = "system default" if self.device is None else f"device {self.device}"
        return f"{seconds:.1f}s  peak {peak:.4f}  rms {rms:.4f}  {where}  — {verdict}"

    def save(self, audio: np.ndarray, path) -> object | None:
        """Write the audio a turn could not use, so it can be listened to.

        Numbers narrow it down; the recording settles it. Deciding between a
        bad microphone and a bad decode took re-running the same six seconds
        through Whisper three ways, which is only possible if those six
        seconds still exist. One file, overwritten each time — a diagnostic
        that fills the disk is one you turn off.

        Returns the path written, or None if it could not be.
        """
        if not audio.size:
            return None
        try:
            import soundfile as sf

            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, audio, self._config.sample_rate)
            return path
        except Exception as exc:  # noqa: BLE001 — a failed diagnostic is not a failed turn
            log.debug("could not save the unrecognised audio: %s", exc)
            return None

    def is_speech(self, audio: np.ndarray) -> bool:
        """Cheap gate before the expensive part of the chain.

        A key pressed and released by accident, or a recording of a quiet room,
        should not cost a transcription and a model round trip.
        """
        if audio.size < self._config.sample_rate * self._config.min_seconds:
            return False
        return float(np.max(np.abs(audio))) >= self._config.silence_threshold
