"""Record from the microphone and play it back.

Proves the most basic thing there is: audio comes in, audio goes out, at the
sample rate everything downstream expects. Whisper resamples to 16 kHz anyway,
so recording there avoids a conversion later.

    python scripts/test_mic.py [seconds]

Writes data/sample.wav, which test_stt.py then transcribes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SAMPLE_RATE = 16_000  # what Whisper wants; recording here avoids resampling
CHANNELS = 1
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "sample.wav"


def level_meter(block: np.ndarray, width: int = 40) -> str:
    """A crude VU meter, so you can see the microphone is actually picking you up."""
    rms = float(np.sqrt(np.mean(np.square(block))))
    filled = min(width, int(rms * width * 12))
    return "#" * filled + "." * (width - filled)


def record(seconds: float) -> np.ndarray:
    frames: list[np.ndarray] = []

    def callback(indata, _frames, _time, status):
        if status:
            print(f"\n  (stream warning: {status})", file=sys.stderr)
        frames.append(indata.copy())

    print(f"\nRecording {seconds:.0f} seconds — say something.\n")
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        callback=callback,
        blocksize=int(SAMPLE_RATE * 0.1),
    ):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            sd.sleep(100)
            if frames:
                remaining = deadline - time.monotonic()
                print(f"\r  [{level_meter(frames[-1])}] {remaining:4.1f}s ", end="")

    print("\r" + " " * 70 + "\r", end="")
    return np.concatenate(frames, axis=0) if frames else np.zeros((0, CHANNELS), "float32")


def main() -> int:
    seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0

    device_in, device_out = sd.default.device
    devices = sd.query_devices()
    print(f"Input : {devices[device_in]['name']}")
    print(f"Output: {devices[device_out]['name']}")

    audio = record(seconds)
    if audio.size == 0:
        print("Nothing was captured. The microphone may be muted or in use.")
        return 1

    peak = float(np.max(np.abs(audio)))
    print(f"Captured {len(audio) / SAMPLE_RATE:.1f}s, peak amplitude {peak:.3f}")
    if peak < 0.01:
        print(
            "\nThat is effectively silence. Check that the right input device is\n"
            "selected in Windows sound settings and that it is not muted."
        )
        return 1
    if peak > 0.99:
        print("\nClipping. Lower the input gain or move back from the microphone.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sf.write(OUTPUT, audio, SAMPLE_RATE)
    print(f"Saved to {OUTPUT}")

    print("\nPlaying it back...")
    sd.play(audio, SAMPLE_RATE)
    sd.wait()

    print("\nIf you heard yourself clearly, audio I/O works. Next: test_stt.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
