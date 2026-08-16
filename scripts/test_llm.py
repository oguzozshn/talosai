"""Send one prompt to the local model and measure how fast it answers.

Time to first token is the number that matters. Because responses are streamed
into text-to-speech sentence by sentence, Talos starts speaking after the first
sentence rather than the last — so the user perceives time-to-first-token, not
total generation time.

    python scripts/test_llm.py [--model NAME] [--prompt TEXT]

With no model given, uses whichever one Ollama lists first.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import ollama

DEFAULT_PROMPT = "Kendini bir cümleyle tanıt."

# The character definition, kept in sync with config/character.md by hand for
# now. Phase 1 loads it from that file instead.
SYSTEM_PROMPT = """You are Talos, a local voice assistant.

You speak Turkish unless addressed in another language. Your answers are spoken
aloud, so keep them short — two or three sentences unless asked for more. No
markdown, no bullet points, no emoji; none of it survives text-to-speech.

You point the way, you do not take the wheel. When you are unsure, say so
plainly instead of guessing."""


def first_available_model() -> str | None:
    try:
        proc = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=30
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    lines = [ln for ln in proc.stdout.strip().splitlines()[1:] if ln.strip()]
    return lines[0].split()[0] if lines else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    model = args.model or first_available_model()
    if not model:
        print("No models found. Install Ollama, then pull one:")
        print("  ollama pull qwen3:8b")
        return 1

    print(f"Model:  {model}")
    print(f"Prompt: {args.prompt}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.prompt},
    ]

    started = time.perf_counter()
    first_token_at: float | None = None
    tokens = 0

    try:
        stream = ollama.chat(model=model, messages=messages, stream=True)
        for chunk in stream:
            piece = chunk.get("message", {}).get("content", "")
            if not piece:
                continue
            if first_token_at is None:
                first_token_at = time.perf_counter() - started
            tokens += 1
            print(piece, end="", flush=True)
    except ollama.ResponseError as exc:
        print(f"\nOllama refused the request: {exc}")
        print(f"Is the model pulled?  ollama pull {model}")
        return 1
    except ConnectionError:
        print("\nCannot reach Ollama. Is the background service running?")
        return 1

    total = time.perf_counter() - started
    print("\n")

    if first_token_at is None:
        print("The model returned nothing.")
        return 1

    print(f"  time to first token  {first_token_at:.2f}s")
    print(f"  total                {total:.2f}s")
    print(f"  throughput           {tokens / max(total, 1e-6):.1f} chunks/s")

    if first_token_at > 0.6:
        print(
            "\nSlow to start. Usually means the model was cold — run it again.\n"
            "If it stays slow, the model is spilling out of VRAM into system RAM;\n"
            "drop to a smaller quantisation or a smaller model."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
