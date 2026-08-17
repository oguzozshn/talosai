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

### Memory costs 323 MB and one cold load

`nomic-embed-text` sits alongside the language model rather than displacing it:
5.6 GB and 323 MB, both reported at 100% GPU. Its first use cost 18 s while it
loaded; the turn after was 3.15 s.

### The model choice, decided

`scripts/compare_models.py`, twelve cases, one run each. Seven expect a
particular skill; five expect none, because a model that reaches for a tool on
every question is not being helpful — each call costs a full extra round trip.

| Model | Correct | Wrong | Missed | Unwanted | Bad args | Median |
|---|---|---|---|---|---|---|
| `qwen3:8b` | **12/12** | 0 | 0 | 0 | 0 | 2.14 s |
| `llama3.1:8b` | 9/12 | 0 | 0 | 3 | 0 | 2.26 s |
| `gemma3:4b` | — | — | — | — | — | — |

Llama's recall is perfect — seven of seven on the cases that want a skill. It
loses entirely on restraint: it tried to take a note when told "thanks, that's
all for now", took another when asked how it was doing, and went to read a file
to explain what a race condition is. Three unnecessary round trips in twelve
questions, which in conversation is an assistant that pauses for nine seconds
to look something up before telling you it is fine, thanks.

That is the half of tool use that goes unmeasured, and it is why five of the
twelve cases expect no call at all.

**Gemma 3 cannot call tools at all.** Every request returns
`does not support tools (status code: 400)`. It was pulled as the tool-calling
comparison and was never a candidate — which should have been checked before
half the VRAM planning was built on the possibility of switching to it.

That also closes the idea of freeing 2.3 GB by dropping to a 4B and spending it
on neural text-to-speech. There is no 4B to drop to.

Qwen3 is kept. Worth being honest about what the score means: twelve
unambiguous cases, one run each. It says Qwen3 is not obviously broken at tool
choice, not that it is reliable in conversation.

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

## The assembled chain

Text in, muted, `qwen3:8b` and `tr_TR-dfki-medium`. "First audio" is from the
start of the turn to the first sample leaving the synthesiser.

| Turn | First audio | Total |
|---|---|---|
| Plain answer | **0.93 s** | 2.06 s |
| Answer requiring a skill call | **9.52 s** | 10.01 s |

Add ~0.22 s of transcription for a spoken turn. That puts a plain exchange
around 1.2 s against a 1.4 s budget.

### Attaching tools is free until one is used

The worry was that seven tool schemas in every request would tax every answer.
It does not:

| | First audio |
|---|---|
| Skills enabled, no skill needed | 0.93 s |
| Skills disabled entirely | 0.97 s |

Within noise. Prefill of the schemas is cheap next to generation, so there is
no reason to gate tools behind a classifier or trim the toolset for latency.

### A skill call costs a full extra round trip

0.93 s to 9.52 s is the real price, and it is structural rather than
incidental: the model generates the call, the skill runs, and then the whole
conversation is prefilled again for the answer. Ten seconds is too long to sit
in silence, so phase 3 should say something before the tool runs — an
acknowledgement covers the gap that no amount of optimisation will close.

### Spoken turns, measured for real

Push-to-talk, English, `en_US-lessac-medium`, audio actually playing. Four
consecutive turns:

| | Transcribe | First audio | Total |
|---|---|---|---|
| First turn of the session | 0.34 s | 4.06 s | 7.00 s |
| Second | 0.31 s | 1.69 s | 12.92 s |
| Third | 0.33 s | 1.88 s | 13.47 s |

Transcription lands at 0.31–0.34 s against a text-mode 0.22 s — the difference
is real audio rather than a clean file. First audio settles at 1.7–1.9 s
against a 1.4 s budget: close, and the gap is generation, not plumbing.

The first turn costs an extra 2 s despite warm-up, so something is still cold
on the first real utterance. Not chased yet.

Totals of 13 s are long answers being spoken in full, not slow machinery.
Reply length is the thing to tune, not the pipeline.

### The first turn, explained

The first turn of a session was slower than the ones after it, and warming the
model up had not fixed it. Measured properly it was much worse than it looked
from the spoken timings:

| | First audio |
|---|---|
| Turn 1, warm-up sending `"hi"` | **9.77 s** |
| Turn 1, warm-up sending the real prompt and tools | **1.61 s** |
| Turns 2-4, either way | 1.2-1.3 s |

The model was already resident both times — `ollama ps` confirmed it. Loading
weights was never the problem. The character definition, the profile and nine
tool schemas come to well over a thousand tokens that are identical on every
request, and prefilling them is what the first turn was paying for.

A warm-up that does not resemble the request it is warming up for warms
nothing.

## Not yet measured

- End-of-speech detection (phase 2)
- How often the wake word fires when it should not, in ordinary use
