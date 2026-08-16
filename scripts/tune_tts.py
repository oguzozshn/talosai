"""Play the same sentence through Piper at several settings, so you can pick.

    python scripts/tune_tts.py ["sentence"]

Piper exposes three knobs that matter and one that does not exist:

  length_scale   phoneme duration. Below 1.0 is faster, above is slower. This
                 is the one that most affects whether speech sounds fluent or
                 laboured.
  noise_w        variation in phoneme duration. Low is metronomic, high is
                 loose. Flatness usually reads as "robotic" more than timbre
                 does.
  noise_scale    generator noise. Mostly affects texture; large changes make
                 it sound unstable rather than natural.

  pitch          not adjustable. Piper synthesises the voice it was trained
                 on. If the timbre is wrong, no setting here fixes it — the
                 voice has to change.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPER = ROOT / "vendor" / "piper" / "piper.exe"
VOICE_DIR = ROOT / "models" / "piper"

DEFAULT_TEXT = (
    "Merhaba, ben Talos. Bugün dört toplantın var, ilki saat onda. "
    "Başka bir şey ister misin?"
)

# (label, length_scale, noise_w, noise_scale)
PRESETS = [
    ("default          ", 1.00, 0.80, 0.667),
    ("faster           ", 0.90, 0.80, 0.667),
    ("faster + looser  ", 0.90, 1.00, 0.667),
    ("slower + looser  ", 1.10, 1.00, 0.667),
    ("flat / metronomic", 1.00, 0.40, 0.667),
    ("smoother texture ", 0.95, 0.90, 0.500),
]


def synthesise(voice: Path, text: str, length: float, noise_w: float, noise: float) -> tuple[Path, float] | None:
    out = Path(tempfile.gettempdir()) / f"talos_tune_{length}_{noise_w}_{noise}.wav"
    started = time.perf_counter()
    try:
        subprocess.run(
            [
                str(PIPER),
                "--model", str(voice),
                "--output_file", str(out),
                "--length_scale", str(length),
                "--noise_w", str(noise_w),
                "--noise_scale", str(noise),
                "--quiet",
            ],
            input=text,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=60,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"  failed: {exc.stderr.strip() if exc.stderr else exc}")
        return None
    return out, time.perf_counter() - started


def main() -> int:
    if not PIPER.exists():
        print(f"Piper not found at {PIPER}\nRun:  run scripts\\fetch_piper.py")
        return 1

    voices = sorted(VOICE_DIR.glob("*.onnx"))
    if not voices:
        print(f"No voice in {VOICE_DIR}\nRun:  run scripts\\fetch_piper.py")
        return 1
    voice = voices[0]

    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
    print(f"Voice: {voice.stem}")
    print(f'Text:  "{text}"\n')

    import sounddevice as sd
    import soundfile as sf

    for label, length, noise_w, noise in PRESETS:
        print(f"  {label}  length={length}  noise_w={noise_w}  noise={noise}", end="", flush=True)
        result = synthesise(voice, text, length, noise_w, noise)
        if result is None:
            continue
        path, elapsed = result
        audio, rate = sf.read(path, dtype="float32")
        print(f"   [{elapsed:.2f}s synth, {len(audio) / rate:.1f}s audio]")
        sd.play(audio, rate)
        sd.wait()
        path.unlink(missing_ok=True)

    print(
        "\nIf none of these are good enough, the limit is the voice, not the\n"
        "settings. tr_TR-dfki-medium is the only Turkish voice Piper publishes,\n"
        "and it is trained on a small dataset. See MEASUREMENTS.md for what\n"
        "replacing the engine would cost."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
