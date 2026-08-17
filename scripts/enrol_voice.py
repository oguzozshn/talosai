"""Record a voice profile, then measure how well it recognises you.

    python scripts/enrol_voice.py            record and measure
    python scripts/enrol_voice.py --test     measure an existing profile
    python scripts/enrol_voice.py --threshold 0.6   set it by hand

Reads six short sentences rather than one long one. They cover more phonetic
ground, and averaging means a cough or a door slam moves the profile slightly
instead of defining it.

Afterwards it takes three more recordings and reports how similar they are to
the profile just built. That distribution is what a sensible threshold is
chosen from — a number picked without measuring is a number that either locks
you out or lets the television in.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from talos.speaker import PROFILE, SpeakerVerifier  # noqa: E402

SAMPLE_RATE = 16_000

# Varied vowels, consonant clusters, and the Turkish-specific letters, so the
# profile is not built entirely out of one narrow set of sounds.
PROMPTS = [
    "Merhaba Talos, bugün nasılsın?",
    "Yağmurlu havalarda çalışmak daha keyifli oluyor.",
    "Şu üç proje için öğleden sonra bir plan çıkaralım.",
    "Bilgisayarımdaki dosyaları düzenlemem gerekiyor.",
    "Kahvaltıdan sonra ığdır'a doğru yola çıktık.",
    "Bir, iki, üç, dört, beş, altı, yedi, sekiz, dokuz, on.",
]

VERIFY_PROMPTS = [
    "Talos, bugün hangi projede kaldım?",
    "Bana yarınki toplantıları hatırlat.",
    "Teşekkürler, şimdilik bu kadar.",
]


def record(seconds: float) -> np.ndarray:
    """Record with a countdown and a level meter."""
    frames: list[np.ndarray] = []

    def callback(indata, _frames, _time, _status):
        frames.append(indata.copy())

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        callback=callback, blocksize=int(SAMPLE_RATE * 0.1),
    ):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            sd.sleep(100)
            if frames:
                rms = float(np.sqrt(np.mean(np.square(frames[-1]))))
                bar = "#" * min(30, int(rms * 360))
                print(f"\r    [{bar:<30}] {deadline - time.monotonic():3.1f}s", end="")

    print("\r" + " " * 60 + "\r", end="")
    return np.concatenate(frames).flatten() if frames else np.zeros(0, dtype=np.float32)


def read_aloud(prompts: list[str], seconds: float) -> list[np.ndarray]:
    samples: list[np.ndarray] = []
    for index, prompt in enumerate(prompts, start=1):
        print(f"\n  {index}/{len(prompts)}  \"{prompt}\"")
        try:
            input("        Enter to record, Ctrl+C to stop... ")
        except (EOFError, KeyboardInterrupt):
            print("\nStopped.")
            break

        audio = record(seconds)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak < 0.01:
            print("        Too quiet — skipped. Move closer to the microphone.")
            continue
        print(f"        ok ({audio.size / SAMPLE_RATE:.1f}s, peak {peak:.2f})")
        samples.append(audio)
    return samples


def measure(verifier: SpeakerVerifier) -> None:
    print("\n\nNow a few more, to see how well it recognises you.")
    samples = read_aloud(VERIFY_PROMPTS, seconds=4.0)
    if not samples:
        print("\nNothing recorded, so nothing to measure.")
        return

    scores = [s for s in (verifier.similarity(a) for a in samples) if s is not None]
    if not scores:
        print("\nCould not score any of the recordings.")
        return

    print("\n\nSimilarity to your profile")
    print("--------------------------")
    for index, score in enumerate(scores, start=1):
        verdict = "match" if score >= verifier.threshold else "REJECTED"
        print(f"  sample {index}   {score:.3f}   {verdict}")

    lowest = min(scores)
    print(f"\n  lowest      {lowest:.3f}")
    print(f"  threshold   {verifier.threshold:.3f}")

    # Leave room below the worst honest attempt, or a bad-throat day locks you
    # out. Not so much room that another voice slips through.
    suggested = max(0.45, round(lowest - 0.08, 2))
    if lowest < verifier.threshold:
        print(
            f"\n  One of your own recordings was rejected. The threshold is too\n"
            f"  high for this microphone. Try:  --threshold {suggested}"
        )
    elif lowest - verifier.threshold > 0.20:
        print(
            f"\n  Comfortable margin. It could be tightened to {suggested} to make\n"
            f"  it harder for another voice to pass."
        )
    else:
        print("\n  Sensible margin. Leave it here.")

    print(
        "\n  Remember what this is: a filter, not a lock. A recording of your\n"
        "  voice will pass it. Do not put anything behind it that would hurt\n"
        "  to lose."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="measure the existing profile")
    parser.add_argument("--threshold", type=float, help="set the match threshold")
    args = parser.parse_args()

    verifier = SpeakerVerifier(SAMPLE_RATE)
    if args.threshold is not None:
        verifier.threshold = args.threshold

    if args.test:
        if not verifier.enrolled:
            print("No profile yet. Run without --test first.")
            return 1
        print(f"Testing the profile in {PROFILE.name}")
        measure(verifier)
        return 0

    print("Voice enrolment")
    print("===============\n")
    print("Read each sentence in your normal speaking voice — not slower, not")
    print("clearer than usual. The profile should match how you actually talk.")
    print("\nLoading the speaker model (first run downloads about 80 MB)...")
    verifier.encoder  # noqa: B018 — force the load now, with an explanation on screen
    print("ready.")

    samples = read_aloud(PROMPTS, seconds=5.0)
    if len(samples) < 3:
        print(f"\nOnly {len(samples)} usable recording(s). Need at least three.")
        return 1

    used = verifier.enrol(samples, threshold=args.threshold)
    print(f"\n\nProfile built from {used} recording(s) → {PROFILE}")

    measure(verifier)

    if args.threshold is None:
        print("\nTo change the threshold later:")
        print("  run scripts\\enrol_voice.py --test --threshold 0.60")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
