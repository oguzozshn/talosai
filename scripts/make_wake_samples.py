"""Generate the spoken samples a wake word model is trained on.

openWakeWord learns one word from thousands of examples of it being said, and
thousands more of it not being said. Recording those by hand is not a weekend,
it is a month, so the positives are synthesised — which is how the published
models were made too.

The official pipeline uses piper-sample-generator, a fork of Piper with a
multi-speaker model that gives hundreds of distinct voices. We already have
Piper and two voices, so the variety has to come from somewhere else:
different speaking rates, different phrasings around the word, and pitch
shifts applied afterwards. That is thinner variety than hundreds of speakers,
and the model will be correspondingly less robust to voices unlike the ones it
heard. It is trained on you saying it, so that may be fine.

Negatives matter as much as positives and are easier to get wrong. A model
trained on "Talos" against silence learns to fire at any speech. The negatives
here are deliberately adversarial: words that share its sounds — Carlos, tacos,
talk, tallest — because those are what it will actually mistake it for.

    python scripts/make_wake_samples.py            everything
    python scripts/make_wake_samples.py --positive  just the wake word
    python scripts/make_wake_samples.py --count 500
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPER = ROOT / "vendor" / "piper" / "piper.exe"
VOICE_DIR = ROOT / "models" / "piper"
OUTPUT = ROOT / "models" / "wakeword"

WAKE = "Talos"

# The word alone, and the word as it is actually said — leading filler, a
# comma, a question after it. A model trained only on the isolated word fires
# poorly when it arrives mid-sentence, which is how people speak.
POSITIVE_TEMPLATES = [
    "{w}",
    "{w}.",
    "{w}?",
    "Hey {w}.",
    "Okay {w}.",
    "{w}, are you there?",
    "{w}, what time is it?",
    "{w}, take a note.",
    "Um, {w}.",
    "So {w}, listen.",
    "{w}, can you help me?",
    "Hey {w}, quick question.",
]

# What it must not fire on. Words sharing the vowels and consonants of the
# wake word are the ones that matter — silence teaches nothing.
NEGATIVE_PHRASES = [
    "Carlos", "tacos", "talk to me", "the tallest one", "colours",
    "tables are ready", "tell us about it", "chaos", "tell me a story",
    "close the door", "hello there", "what time is it",
    "take a note", "are you there", "thank you very much",
    "call us later", "tallow", "Dallas", "gallows", "salads",
    "it is called Talos", "the Talos project runs locally",
    "I was telling Ahmet about it", "solos", "toss it over",
    "how are you doing", "open the file", "what did I work on",
    "let us practise English", "read my notes back",
]

# Applied per clip so the model hears the word at different speeds and pitches
# rather than one voice saying it identically a thousand times.
LENGTH_SCALES = (0.8, 0.9, 1.0, 1.1, 1.25)
NOISE_SCALES = (0.5, 0.667, 0.8)
NOISE_WIDTHS = (0.6, 0.8, 1.0)


def voices() -> list[Path]:
    if not VOICE_DIR.is_dir():
        return []
    # English voices only. The wake word is said in an English sentence, and
    # the Turkish voice pronounces it with entirely different vowels.
    return sorted(v for v in VOICE_DIR.glob("*.onnx") if v.name.startswith("en"))


def synthesise(voice: Path, text: str, target: Path, rng: random.Random) -> bool:
    """One clip, with randomised delivery. Returns whether it worked."""
    command = [
        str(PIPER),
        "--model", str(voice),
        "--output_file", str(target),
        "--length_scale", str(rng.choice(LENGTH_SCALES)),
        "--noise_scale", str(rng.choice(NOISE_SCALES)),
        "--noise_w", str(rng.choice(NOISE_WIDTHS)),
        "--quiet",
    ]
    try:
        subprocess.run(
            command, input=text.encode("utf-8"),
            capture_output=True, timeout=30, check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return target.exists() and target.stat().st_size > 1000


def resample_to_16k(path: Path) -> None:
    """openWakeWord expects 16 kHz mono. Piper produces 22.05 kHz."""
    import numpy as np

    with wave.open(str(path), "rb") as source:
        rate = source.getframerate()
        frames = source.readframes(source.getnframes())
        channels = source.getnchannels()

    if rate == 16000 and channels == 1:
        return

    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    # Linear interpolation. Crude next to a proper filter, but these clips are
    # about to be augmented with noise and reverb anyway.
    count = int(len(samples) * 16000 / rate)
    resampled = np.interp(
        np.linspace(0, len(samples) - 1, count),
        np.arange(len(samples)),
        samples,
    ).astype(np.int16)

    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(resampled.tobytes())


def generate(kind: str, phrases: list[str], count: int, rng: random.Random) -> int:
    available = voices()
    if not available:
        print("No English Piper voice found. Run: run scripts\\fetch_piper.py")
        return 0

    folder = OUTPUT / kind
    folder.mkdir(parents=True, exist_ok=True)
    made = 0

    for index in range(count):
        voice = available[index % len(available)]
        text = phrases[index % len(phrases)]
        if "{w}" in text:
            text = text.format(w=WAKE)

        target = folder / f"{kind}_{index:05d}.wav"
        if synthesise(voice, text, target, rng):
            try:
                resample_to_16k(target)
                made += 1
            except (OSError, wave.Error):
                target.unlink(missing_ok=True)

        if (index + 1) % 50 == 0:
            print(f"  {kind}: {made}/{index + 1}", flush=True)

    return made


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=400, help="clips per kind")
    parser.add_argument("--positive", action="store_true", help="only the wake word")
    parser.add_argument("--negative", action="store_true", help="only the rest")
    args = parser.parse_args()

    if not PIPER.exists():
        print(f"Piper not found at {PIPER}\nRun: run scripts\\fetch_piper.py")
        return 1

    rng = random.Random(20260818)
    print(f"Voices: {', '.join(v.stem for v in voices()) or 'none'}")
    print(f"Writing to {OUTPUT}\n")

    total = 0
    if not args.negative:
        total += generate("positive", POSITIVE_TEMPLATES, args.count, rng)
    if not args.positive:
        # More negatives than positives on purpose. False wakes are the
        # failure people actually notice, and they come from the negative set
        # being too small or too easy.
        total += generate("negative", NEGATIVE_PHRASES, int(args.count * 1.5), rng)

    print(f"\n{total} clips written to {OUTPUT}")
    print("These are synthetic. Recording yourself saying it a few dozen times")
    print("and adding those will matter more than another thousand of these.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
