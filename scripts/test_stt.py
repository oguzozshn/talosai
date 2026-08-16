"""Transcribe a wav file and report how long each stage took.

Two things this is really checking:

1. Whether the GPU is actually being used. CTranslate2 falls back to CPU
   without complaining, so the only way to know is to try CUDA explicitly and
   see whether it raises.
2. The real-time factor. Anything above ~0.3 means transcription alone will
   eat most of the latency budget, and the model needs to get smaller.

    python scripts/test_stt.py [wav-path] [--cpu] [--model NAME]

Defaults to data/sample.wav, which test_mic.py writes.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WAV = ROOT / "data" / "sample.wav"

# large-v3-turbo is 809M parameters — roughly a quarter the cost of large-v3
# for a small accuracy loss, which is the right trade when latency is the
# whole point.
DEFAULT_MODEL = "large-v3-turbo"
LANGUAGE = "tr"


def load(model_name: str, force_cpu: bool) -> tuple[WhisperModel, str]:
    """Load on GPU if we can, CPU if we must. Returns the model and which one."""
    if not force_cpu:
        try:
            started = time.perf_counter()
            model = WhisperModel(model_name, device="cuda", compute_type="int8_float16")
            print(f"  loaded on GPU in {time.perf_counter() - started:.1f}s")
            return model, "cuda"
        except Exception as exc:  # noqa: BLE001 — any failure means "no usable GPU"
            print(f"  GPU unavailable ({type(exc).__name__}), falling back to CPU")
            print(f"    {exc}")

    started = time.perf_counter()
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    print(f"  loaded on CPU in {time.perf_counter() - started:.1f}s")
    return model, "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", nargs="?", default=str(DEFAULT_WAV))
    parser.add_argument("--cpu", action="store_true", help="skip the GPU attempt")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    wav = Path(args.wav)
    if not wav.exists():
        print(f"No such file: {wav}")
        print("Record one first:  python scripts/test_mic.py")
        return 1

    info = sf.info(wav)
    duration = info.frames / info.samplerate
    print(f"Input: {wav.name}  ({duration:.1f}s, {info.samplerate} Hz)\n")

    print(f"Loading {args.model} — first run downloads ~1.5 GB, be patient.")
    model, device = load(args.model, args.cpu)

    print("\nTranscribing...")
    started = time.perf_counter()
    segments, result = model.transcribe(
        str(wav),
        language=LANGUAGE,
        beam_size=1,  # greedy: measurably faster, and the accuracy cost is small
        vad_filter=True,  # drop leading and trailing silence
    )
    # faster-whisper is lazy — the work happens as segments are consumed.
    text = " ".join(segment.text.strip() for segment in segments)
    elapsed = time.perf_counter() - started

    print(f"\n  \"{text}\"\n")
    print(f"  device            {device}")
    print(f"  detected language {result.language} ({result.language_probability:.0%})")
    print(f"  transcribe time   {elapsed:.2f}s")
    print(f"  real-time factor  {elapsed / duration:.2f}x")

    if elapsed / duration > 0.3:
        print(
            "\nSlower than the latency budget allows. Options, in order of\n"
            "preference: get CUDA working, or drop to the 'small' model."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
