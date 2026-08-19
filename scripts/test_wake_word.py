"""Measure a wake word model against real speech it has never heard.

Held-out accuracy on synthetic clips is close to meaningless: the test clips
come from the same two voices saying the same phrases as the training ones, so
a model can score ninety-nine percent and still fire at the television.

The number that matters is false wakes per hour on real audio. This runs the
model across ten hours of ordinary speech and recordings — none of which
contain the wake word — and counts how often it thinks it heard it. Then it
does the same on the synthetic positives to show what detection rate each
threshold buys.

There is no single right threshold. There is a curve, and where you sit on it
is a preference: one false wake a day and the occasional missed call, or
instant response and being interrupted while watching something.

    python scripts/test_wake_word.py
    python scripts/test_wake_word.py --model models/wakeword/talos.onnx
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "wakeword" / "talos.onnx"
VALIDATION = ROOT / "models" / "wakeword" / "validation" / "negatives.npy"
POSITIVES = ROOT / "models" / "wakeword" / "positive"

FRAMES, EMBEDDING = 16, 96
# Each embedding frame covers 80 ms of audio.
SECONDS_PER_FRAME = 0.08
CLIP_SECONDS = 2.0
SAMPLE_RATE = 16000

THRESHOLDS = (0.3, 0.5, 0.7, 0.9, 0.95, 0.99)


def windows(features: np.ndarray, stride: int = 1) -> np.ndarray:
    """Sliding windows of sixteen frames, the way the model sees them live."""
    count = (len(features) - FRAMES) // stride + 1
    shape = (count, FRAMES, EMBEDDING)
    strides = (features.strides[0] * stride, features.strides[0], features.strides[1])
    return np.lib.stride_tricks.as_strided(features, shape=shape, strides=strides)


def score(session: ort.InferenceSession, batch: np.ndarray) -> np.ndarray:
    name = session.get_inputs()[0].name
    out = []
    # Chunked because ten hours of windows is several gigabytes if done at once.
    for start in range(0, len(batch), 4096):
        chunk = np.ascontiguousarray(batch[start : start + 4096], dtype=np.float32)
        out.append(session.run(None, {name: chunk})[0].ravel())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def load_positive_clips(extractor) -> np.ndarray:
    clips = []
    wanted = int(CLIP_SECONDS * SAMPLE_RATE)
    for path in sorted(POSITIVES.glob("*.wav")):
        try:
            with wave.open(str(path), "rb") as source:
                samples = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16)
        except Exception:  # noqa: BLE001
            continue
        if len(samples) < wanted:
            samples = np.pad(samples, (wanted - len(samples), 0))
        clips.append(samples[-wanted:])
    if not clips:
        return np.zeros((0, FRAMES, EMBEDDING), dtype=np.float32)
    return np.asarray(extractor.embed_clips(np.stack(clips), batch_size=32), dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"No model at {model_path}\nRun: run scripts\\train_wake_word.py")
        return 1
    if not VALIDATION.exists():
        print(f"No validation audio at {VALIDATION}")
        return 1

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    negatives = np.load(VALIDATION, mmap_mode="r")
    hours = len(negatives) * SECONDS_PER_FRAME / 3600
    print(f"Real speech, none of it the wake word: {hours:.1f} hours")

    # Stride of four rather than one: the model sees every 320 ms instead of
    # every 80 ms, which is four times faster and cannot hide a detection,
    # since an activation persists across neighbouring windows.
    print("Scoring...", flush=True)
    negative_scores = score(session, windows(np.asarray(negatives), stride=4))

    print("Scoring the wake word clips...", flush=True)
    import warnings

    warnings.filterwarnings("ignore")
    import scipy.special as sp

    if not hasattr(sp, "sph_harm"):
        sp.sph_harm = lambda m, n, theta, phi: sp.sph_harm_y(n, m, phi, theta)
    from openwakeword.utils import AudioFeatures

    positive = load_positive_clips(AudioFeatures())
    positive_scores = score(session, positive) if len(positive) else np.zeros(0)

    print()
    print(f"{'threshold':>10}{'false wakes/hour':>20}{'wake word caught':>20}")
    print("-" * 50)
    for threshold in THRESHOLDS:
        # Consecutive windows over one utterance count once, the way the
        # runtime debounces them.
        firing = negative_scores > threshold
        events = int(np.sum(firing[1:] & ~firing[:-1])) + int(firing[0] if len(firing) else 0)
        per_hour = events / hours
        caught = float((positive_scores > threshold).mean()) if len(positive_scores) else 0.0
        print(f"{threshold:>10.2f}{per_hour:>20.1f}{caught:>19.0%}")

    print()
    print("A model worth using sits under about one false wake per hour while")
    print("still catching most of the word. If nothing on this table does both,")
    print("the training data was too small — which is what the 16 GB negative")
    print("set exists to fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
