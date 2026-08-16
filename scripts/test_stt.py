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
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Must run before faster_whisper is imported: the DLL search path is read when
# the CTranslate2 extension loads, so registering afterwards has no effect.
from talos.cuda import register_cuda_dll_directories  # noqa: E402

register_cuda_dll_directories()

import soundfile as sf  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

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

    def transcribe() -> tuple[str, object, float]:
        started = time.perf_counter()
        segments, result = model.transcribe(
            str(wav),
            language=LANGUAGE,
            beam_size=1,  # greedy: measurably faster, and the accuracy cost is small
            vad_filter=True,  # drop leading and trailing silence
        )
        # faster-whisper is lazy — the work happens as segments are consumed.
        text = " ".join(segment.text.strip() for segment in segments)
        return text, result, time.perf_counter() - started

    # The first call on a GPU pays for kernel compilation and cuDNN autotuning,
    # which on this model is several seconds — enough to make a warm system look
    # 25x slower than it is. Both numbers matter, so report both: the cold one
    # is what the first utterance after startup costs, the warm one is what the
    # latency budget is actually built on. Phase 1 hides the cold cost by
    # warming the model at startup rather than on the user's first sentence.
    print("\nTranscribing (cold)...")
    _, _, cold = transcribe()

    print("Transcribing (warm)...")
    text, result, warm = transcribe()

    print(f'\n  "{text}"\n')
    print(f"  device            {device}")
    print(f"  detected language {result.language} ({result.language_probability:.0%})")
    print(f"  first call        {cold:.2f}s   ({cold / duration:.2f}x real time)")
    print(f"  steady state      {warm:.2f}s   ({warm / duration:.2f}x real time)")

    rtf = warm / duration
    if rtf > 0.3:
        if device == "cpu":
            print(
                "\nToo slow for the latency budget. Getting CUDA working is the\n"
                "fix — see talos/cuda.py for what that needs."
            )
        else:
            print(
                "\nToo slow for the latency budget even on the GPU. Drop to the\n"
                "'small' model:  run scripts\\test_stt.py --model small"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
