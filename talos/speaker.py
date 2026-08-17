"""Recognising whose voice it is.

An ECAPA-TDNN model turns a few seconds of speech into a 192-dimension
embedding. Two recordings of the same person land close together; two people
land far apart. Enrolment averages several recordings into one profile, and
verification is a cosine similarity against it.

Runs on the CPU deliberately. The model is small enough that it costs about
50 ms there, and the VRAM it would occupy is worth far more to Whisper and the
language model.

**This is a filter, not authentication.** A recording of the enrolled voice
will pass it. So will a good impression, sometimes. A cold, a whisper, or a
noisy room will fail it. Use it to ignore the television and other people in
the room — never to guard anything that actually matters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROFILE = ROOT / "data" / "voice_profile.npz"
MODEL_DIR = ROOT / "models" / "speechbrain"
MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

# Cosine similarity above this counts as a match. Not a universal constant —
# it depends on the microphone, the room and the speaker, which is why
# enrol_voice.py measures the actual distribution and suggests one.
DEFAULT_THRESHOLD = 0.65

# Below this there is not enough signal for a stable embedding, and the
# similarity score becomes noise.
MIN_SECONDS = 1.0


@dataclass
class Verdict:
    matched: bool
    similarity: float
    reason: str = ""


class SpeakerVerifier:
    def __init__(self, sample_rate: int = 16_000, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._encoder = None
        self._centroid: np.ndarray | None = None
        self._load_profile()

    # -- model ---------------------------------------------------------------

    @property
    def encoder(self):
        """Loaded on first use — importing speechbrain pulls in torch, which is
        slow enough that doing it at startup would be felt."""
        if self._encoder is None:
            from speechbrain.inference.speaker import EncoderClassifier

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            self._encoder = EncoderClassifier.from_hparams(
                source=MODEL_SOURCE,
                savedir=str(MODEL_DIR),
                run_opts={"device": "cpu"},
            )
        return self._encoder

    def embed(self, audio: np.ndarray) -> np.ndarray | None:
        """Turn mono float32 audio into a normalised embedding."""
        if audio.size < self.sample_rate * MIN_SECONDS:
            return None

        import torch

        with torch.no_grad():
            tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
            embedding = self.encoder.encode_batch(tensor.unsqueeze(0))
        vector = embedding.squeeze().cpu().numpy().astype(np.float32)

        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return None
        # Normalising here means comparison is a plain dot product later.
        return vector / norm

    # -- profile -------------------------------------------------------------

    @property
    def enrolled(self) -> bool:
        return self._centroid is not None

    def _load_profile(self) -> None:
        if not PROFILE.exists():
            return
        try:
            data = np.load(PROFILE)
            self._centroid = data["centroid"]
            if "threshold" in data:
                self.threshold = float(data["threshold"])
        except (OSError, KeyError, ValueError) as exc:
            log.warning("Could not read voice profile: %s", exc)

    def enrol(self, samples: list[np.ndarray], threshold: float | None = None) -> int:
        """Build a profile from several recordings. Returns how many were used.

        Several short recordings rather than one long one, on purpose: they
        cover more phonetic ground, and averaging means one bad take — a cough,
        a door — moves the centroid a little instead of defining it.
        """
        vectors = [v for v in (self.embed(s) for s in samples) if v is not None]
        if not vectors:
            return 0

        centroid = np.mean(vectors, axis=0)
        centroid /= np.linalg.norm(centroid)
        self._centroid = centroid.astype(np.float32)
        if threshold is not None:
            self.threshold = threshold

        PROFILE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            PROFILE,
            centroid=self._centroid,
            threshold=np.float32(self.threshold),
            samples=np.int32(len(vectors)),
        )
        return len(vectors)

    def similarity(self, audio: np.ndarray) -> float | None:
        """Cosine similarity against the profile, or None if it cannot be computed."""
        if self._centroid is None:
            return None
        vector = self.embed(audio)
        if vector is None:
            return None
        return float(np.dot(vector, self._centroid))

    def verify(self, audio: np.ndarray) -> Verdict:
        if self._centroid is None:
            # No profile means no opinion. Refusing everything would make the
            # assistant useless before enrolment; claiming a match would be a
            # lie. Pass, and say why.
            return Verdict(True, 0.0, "no voice profile enrolled")

        score = self.similarity(audio)
        if score is None:
            return Verdict(False, 0.0, "too short to identify")
        if score >= self.threshold:
            return Verdict(True, score)
        return Verdict(False, score, "voice does not match the enrolled profile")
