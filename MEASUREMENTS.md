# Baseline measurements

Phase 0, on the target machine: RTX 5070 Laptop (8 GB), i7-13650HX, 24 GB RAM,
Windows 11, driver 610.47, CUDA compute 12.0.

Recorded so that phase 1 has something to compare against. When the assembled
chain is slower than the sum of its parts, the difference is in the wiring, and
these are what tell you that.

## Speech to text

`faster-whisper large-v3-turbo`, int8_float16, CUDA, greedy decoding, VAD
filter on. Input: 5.0 s of Turkish speech, 16 kHz mono.

| | |
|---|---|
| Model load | 3.6 s |
| First call | 0.97 s (0.19x real time) |
| Steady state | **0.22 s (0.04x real time)** |
| Language detection | `tr`, 100% |

The first call is not representative — it pays for kernel compilation and cuDNN
autotuning. Before that cache existed at all, the very first call on this
machine took 11.4 s. Warm the model at startup and none of this is ever seen.

## Reasoning

Ollama 0.32.14, 4096 context, thinking off, `keep_alive` pinned. Prompt:
*"Kendini bir cümleyle tanıt."* with the character definition as system prompt.

| Model | Size | First token | Total | VRAM |
|---|---|---|---|---|
| `qwen3:8b` | 5.2 GB | **0.35 s** | 0.63 s | 5.6 GB, 100% GPU |
| `gemma3:4b` | 3.3 GB | 0.55 s | 0.79 s | 2.9 GB, 100% GPU |

Cold load — model on disk, not in VRAM — costs 62 s for Qwen and 25 s for
Gemma. That is the cost `keep_alive` exists to avoid.

### Only one of them can be resident

5.6 + 2.9 GB does not fit in 8 GB. Loading Gemma evicted Qwen, and switching
back would pay the full cold load again. So the phase 3 comparison has to run
the models one after another rather than side by side, and the finished system
commits to one.

### Thinking is not optional-cost, it is the whole cost

Qwen3 is a reasoning model, and its reasoning goes to a separate field that is
never spoken:

| | First token | Reasoning emitted |
|---|---|---|
| Thinking on | 12.58 s | 847 characters |
| Thinking off | 0.45 s | — |

A 28x latency difference for output the user never hears.

### Turkish quality, one sample

Not a benchmark — one prompt, one run each — but the gap was wide enough to
note. Same system prompt, which asks the assistant to point the way rather than
take the wheel:

- **qwen3:8b** — *"Ben Talos, sana yardımcı olmamı ister misin?"*
- **gemma3:4b** — *"Ben Talos, yerel sesli asistanım ve Türkçe konuşurum. Sana
  yön gösterebilirim."*

Gemma's Turkish is more natural and it picked up the character; Qwen ended on
exactly the kind of filler question the character definition tells it not to
ask. Against that, Qwen is expected to be the better tool caller, which is what
phase 3 depends on. Decide it there, by measuring, not here.

## Not yet measured

- End-of-speech detection (phase 2)
- Text-to-speech — Piper not fetched yet
- The assembled chain, which is the only number that actually matters
