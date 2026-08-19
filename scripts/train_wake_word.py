"""Train a wake word model for "Talos", at the small end of the scale.

openWakeWord's own pipeline wants roughly twenty gigabytes: two thousand hours
of precomputed negative features, background audio, room impulse responses,
and a multi-speaker sample generator. That produces a model robust to voices
and rooms it has never heard.

This trains the same shape of model on what is already here — clips
synthesised by the Piper we installed for speech output. It is a deliberately
cheap first attempt, made to be measured rather than trusted: if the false
wake rate is bad, that is the argument for the twenty gigabytes, and it will
have been earned rather than assumed.

The architecture is fixed by what openWakeWord loads: sixteen frames of
ninety-six dimensional embeddings in, one probability out. The embeddings come
from the same shared model every openWakeWord voice uses, so only the small
head on top is trained here.

    python scripts/train_wake_word.py
    python scripts/train_wake_word.py --epochs 60
"""

from __future__ import annotations

import argparse
import sys
import warnings
import wave
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# acoustics, pulled in by openwakeword, imports scipy.special.sph_harm, which
# SciPy removed in 1.15. It is used for microphone directivity, which nothing
# here touches — but the import fails before that is discoverable. The
# replacement takes its arguments in a different order.
import scipy.special as _special  # noqa: E402

if not hasattr(_special, "sph_harm"):
    def _sph_harm(m, n, theta, phi):
        return _special.sph_harm_y(n, m, phi, theta)

    _special.sph_harm = _sph_harm

import torch  # noqa: E402
from torch import nn  # noqa: E402

from openwakeword.utils import AudioFeatures  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "models" / "wakeword"
OUTPUT = ROOT / "models" / "wakeword" / "talos.onnx"

# Fixed by the runtime: openWakeWord feeds exactly this and expects a scalar.
FRAMES, EMBEDDING = 16, 96

# Two seconds produces exactly the sixteen frames the model expects. Measured
# rather than derived: the embedding model has its own window and stride, and
# 1.44 seconds — which looked right — gives nine.
CLIP_SECONDS = 2.0
SAMPLE_RATE = 16000


def load_clip(path: Path) -> np.ndarray | None:
    """One clip as int16, padded or trimmed to the window the model sees."""
    try:
        with wave.open(str(path), "rb") as source:
            if source.getframerate() != SAMPLE_RATE or source.getnchannels() != 1:
                return None
            samples = np.frombuffer(source.readframes(source.getnframes()), dtype=np.int16)
    except Exception:  # noqa: BLE001
        # Deliberately broad. A half-written file raises EOFError, which is
        # neither OSError nor wave.Error, and one unreadable clip out of six
        # hundred should not end a training run.
        return None

    wanted = int(CLIP_SECONDS * SAMPLE_RATE)
    if len(samples) < wanted:
        # Pad at the front. The word then sits at the end of the window, which
        # is where it lands in real use — detection happens as you finish
        # saying it, not before.
        return np.pad(samples, (wanted - len(samples), 0))
    return samples[-wanted:]


def features_for(folder: Path, extractor: AudioFeatures) -> np.ndarray:
    clips = [c for c in (load_clip(p) for p in sorted(folder.glob("*.wav"))) if c is not None]
    if not clips:
        return np.zeros((0, FRAMES, EMBEDDING), dtype=np.float32)

    print(f"  {folder.name}: {len(clips)} clips", flush=True)
    embeddings = extractor.embed_clips(np.stack(clips), batch_size=32)
    return np.asarray(embeddings, dtype=np.float32)


class WakeWord(nn.Module):
    """The head that sits on top of the shared embedding model.

    Small on purpose. There are a few hundred training clips here, and a
    larger network would memorise them rather than learn the word.
    """

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FRAMES * EMBEDDING, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args()

    positive_dir, negative_dir = SAMPLES / "positive", SAMPLES / "negative"
    if not positive_dir.is_dir() or not negative_dir.is_dir():
        print("No samples yet. Run: run scripts\\make_wake_samples.py")
        return 1

    print("Computing embeddings")
    extractor = AudioFeatures()
    positive = features_for(positive_dir, extractor)
    negative = features_for(negative_dir, extractor)

    if len(positive) < 20 or len(negative) < 20:
        print(f"\nToo few clips ({len(positive)} positive, {len(negative)} negative).")
        print("Generate more: run scripts\\make_wake_samples.py --count 400")
        return 1

    x = torch.from_numpy(np.concatenate([positive, negative]))
    y = torch.cat([torch.ones(len(positive)), torch.zeros(len(negative))]).unsqueeze(1)

    # Hold out a fifth, shuffled, so the score means something.
    generator = torch.Generator().manual_seed(20260818)
    order = torch.randperm(len(x), generator=generator)
    x, y = x[order], y[order]
    split = int(len(x) * 0.8)
    train_x, train_y, test_x, test_y = x[:split], y[:split], x[split:], y[split:]

    model = WakeWord(args.hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    # A false wake interrupts you; a missed wake means saying the name again.
    # The first is worse, so negatives are weighted more heavily.
    weight = torch.where(train_y > 0.5, 1.0, 2.0)
    criterion = nn.BCELoss(reduction="none")

    print(f"\nTraining on {len(train_x)} clips, holding back {len(test_x)}")
    for epoch in range(args.epochs):
        model.train()
        optimiser.zero_grad()
        loss = (criterion(model(train_x), train_y) * weight).mean()
        loss.backward()
        optimiser.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                predicted = model(test_x) > 0.5
                accuracy = (predicted == (test_y > 0.5)).float().mean().item()
            print(f"  epoch {epoch + 1:3d}   loss {loss.item():.4f}   held-out {accuracy:.0%}")

    model.eval()
    with torch.no_grad():
        scores = model(test_x)
        actually = test_y > 0.5
        said_yes = scores > 0.5
        false_wakes = int((said_yes & ~actually).sum())
        missed = int((~said_yes & actually).sum())

    print(f"\nOn the held-out clips: {false_wakes} false wakes, {missed} missed")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, FRAMES, EMBEDDING),
        str(OUTPUT),
        input_names=["x.1"],
        output_names=["output"],
        dynamic_axes={"x.1": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )
    print(f"Saved {OUTPUT}")
    print("\nThis was trained on synthetic speech only. The number that matters")
    print("is how often it fires when nobody said it — measure that before")
    print("trusting it:  run scripts\\test_wake_word.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
