"""Speak a sentence and report how long it took to produce audio.

Prefers Piper. Falls back to the voice built into Windows so that this script
always proves *something* — if the fallback speaks, the output path works and
only Piper is missing, which is a much easier problem to chase.

    python scripts/test_tts.py ["sentence to speak"]

Piper is not pip-installed on Windows; it is a binary plus a voice file. See
scripts/README.md for where to put them.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPER_DIR = ROOT / "vendor" / "piper"
VOICE_DIR = ROOT / "models" / "piper"

DEFAULT_TEXT = "Merhaba. Ben Talos. Çağırdığında buradayım."

SETUP_HINT = f"""
Piper is not installed yet. It is two downloads:

  1. The binary — piper_windows_amd64.zip from
     https://github.com/rhasspy/piper/releases
     Unzip so that this path exists:
       {PIPER_DIR / "piper.exe"}

  2. A Turkish voice — both the .onnx and the .onnx.json from
     https://huggingface.co/rhasspy/piper-voices/tree/main/tr/tr_TR/fettah/medium
     Put them in:
       {VOICE_DIR}

Falling back to the built-in Windows voice for now. It will mispronounce
Turkish badly — that is expected and does not mean anything is broken.
"""


def find_piper() -> tuple[Path, Path] | None:
    """Locate the Piper binary and a voice, or None if either is missing."""
    binary = PIPER_DIR / "piper.exe"
    if not binary.exists():
        found = shutil.which("piper")
        if not found:
            return None
        binary = Path(found)

    voices = sorted(VOICE_DIR.glob("*.onnx")) if VOICE_DIR.exists() else []
    if not voices:
        return None

    # Prefer a Turkish voice if several are present.
    turkish = [v for v in voices if v.name.startswith("tr")]
    return binary, (turkish or voices)[0]


def speak_with_piper(binary: Path, voice: Path, text: str) -> float | None:
    """Synthesise to a temp wav and play it. Returns seconds to produce audio."""
    out = Path(tempfile.gettempdir()) / "talos_tts_test.wav"
    started = time.perf_counter()
    try:
        subprocess.run(
            [str(binary), "--model", str(voice), "--output_file", str(out)],
            input=text,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=60,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Piper failed: {exc.stderr.strip() if exc.stderr else exc}")
        return None
    except subprocess.TimeoutExpired:
        print("Piper timed out.")
        return None
    elapsed = time.perf_counter() - started

    try:
        import sounddevice as sd
        import soundfile as sf

        audio, rate = sf.read(out, dtype="float32")
        sd.play(audio, rate)
        sd.wait()
    except Exception as exc:  # noqa: BLE001 — playback is a nicety, not the test
        print(f"(synthesised fine, but playback failed: {exc})")

    out.unlink(missing_ok=True)
    return elapsed


def speak_with_windows(text: str) -> float | None:
    """Last resort: the voice that ships with Windows, driven through .NET."""
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Speak([Console]::In.ReadToEnd())"
    )
    started = time.perf_counter()
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            input=text,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=60,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"Windows speech failed too: {exc}")
        return None
    return time.perf_counter() - started


def main() -> int:
    text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TEXT
    print(f'Speaking: "{text}"\n')

    piper = find_piper()
    if piper:
        binary, voice = piper
        print(f"Engine: Piper ({voice.stem})")
        elapsed = speak_with_piper(binary, voice, text)
        if elapsed is not None:
            print(f"\n  synthesis time  {elapsed:.2f}s")
            # Piper is fast enough that the real system streams sentence by
            # sentence; a whole paragraph taking a second is fine because the
            # first sentence is already playing.
            if elapsed > 1.0:
                print("  Slower than expected — try the 'low' quality voice.")
            return 0
        print("\nFalling back to the built-in Windows voice.\n")
    else:
        print(SETUP_HINT)

    print("Engine: Windows built-in")
    elapsed = speak_with_windows(text)
    if elapsed is None:
        return 1
    print(f"\n  synthesis time  {elapsed:.2f}s")
    print("\nAudio output works. Install Piper for a voice worth listening to.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
