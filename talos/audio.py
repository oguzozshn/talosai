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

log = logging.getLogger(__name__)


class Recorder:
    """Captures audio between start() and stop().

    Frames accumulate on the audio callback thread, so start() returns
    immediately and the caller stays responsive while the user is talking.
    """

    def __init__(self, config: AudioConfig) -> None:
        self._config = config
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream: sd.InputStream | None = None

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
        with self._lock:
            self._frames.clear()
        self._stream = sd.InputStream(
            samplerate=self._config.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=int(self._config.sample_rate * 0.05),
        )
        self._stream.start()

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

    def is_speech(self, audio: np.ndarray) -> bool:
        """Cheap gate before the expensive part of the chain.

        A key pressed and released by accident, or a recording of a quiet room,
        should not cost a transcription and a model round trip.
        """
        if audio.size < self._config.sample_rate * self._config.min_seconds:
            return False
        return float(np.max(np.abs(audio))) >= self._config.silence_threshold
