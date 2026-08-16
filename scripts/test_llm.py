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
    """Ask the Ollama service what it has, over HTTP.

    Deliberately not `subprocess.run(["ollama", "list"])`: the CLI is only on
    PATH in shells started after the installer ran, so shelling out reports
    "no models" on a machine that has them. The service is what actually
    matters here, and the client talks to it directly.
    """
    try:
        listing = ollama.list()
    except Exception:  # noqa: BLE001 — unreachable service is handled by the caller
        return None

    # The client returned plain dicts before 0.4 and typed objects after.
    models = getattr(listing, "models", None) or listing.get("models", [])
    names = [getattr(m, "model", None) or m.get("model") or m.get("name") for m in models]
    available = [n for n in names if n]
    return available[0] if available else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--think",
        action="store_true",
        help="let a reasoning model think first (measurably slower; see below)",
    )
    args = parser.parse_args()

    model = args.model or first_available_model()
    if not model:
        print("No models available. Either the Ollama service is not running,")
        print("or nothing has been pulled yet:")
        print("  ollama serve")
        print("  ollama pull qwen3:8b")
        return 1

    print(f"Model:  {model}")
    print(f"Prompt: {args.prompt}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.prompt},
    ]

    print(f"Thinking: {'on' if args.think else 'off'}\n")

    started = time.perf_counter()
    first_token_at: float | None = None
    tokens = 0
    thinking_chars = 0

    try:
        # keep_alive matters as much as the model choice here. Ollama unloads
        # after five idle minutes by default, and reloading 5 GB costs about a
        # minute — which a voice assistant would pay in the middle of a
        # conversation. Phase 1 keeps the model resident.
        stream = ollama.chat(
            model=model,
            messages=messages,
            stream=True,
            think=args.think,
            keep_alive="30m",
        )
        for chunk in stream:
            message = chunk.get("message", {})
            thinking_chars += len(message.get("thinking") or "")
            piece = message.get("content", "")
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
    if thinking_chars:
        print(f"  thinking             {thinking_chars} characters before answering")

    if first_token_at > 0.6:
        if thinking_chars:
            print(
                "\nThat is the reasoning trace, not the machine being slow. The\n"
                "model writes its thinking before the answer, and none of it is\n"
                "spoken. Drop --think to see the difference."
            )
        else:
            print(
                "\nSlow to start. If this was the first call, the model was being\n"
                "loaded into VRAM — run it again. If it stays slow, check\n"
                "'ollama ps': anything less than 100% GPU means it is spilling\n"
                "into system RAM, and a smaller model is the fix."
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
