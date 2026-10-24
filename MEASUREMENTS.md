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

### A 4B, measured again: `qwen3:4b-instruct`

2026-09-26, Ollama 0.33.3, 37 skills loaded. The question was whether a model
small enough to fine-tune locally (QLoRA on this card) could replace the 8B.
The tag is `qwen3:4b-instruct`; `qwen3:4b-instruct-2507` does not exist in the
registry.

**The full prompt no longer fits in 4096 on this model.** Character, profile
and 37 tool schemas measure:

| | Prompt tokens, before the question has been answered |
|---|---|
| `qwen3:8b` | 3444 |
| `qwen3:4b-instruct` | 4083 |

Same tokenizer; the 4B's chat template renders tool schemas more verbosely.
At 4096 every turn that calls a skill fails outright — `request (4110 tokens)
exceeds the available context size` — because the tool result pushes it over.
On Talos's own settings it scored 18/21, with 15 cases lost to that error.
So the comparison below runs both models at 8192, which is what the 4B would
need in practice.

Worth noticing for the 8B too: 3444 of 4096 is 84% spent before anyone speaks.
One tool result or a couple of turns of history fills the rest.

`scripts/compare_models.py`, twelve cases, three runs each, `num_ctx` 8192:

| Model | Correct | Wrong | Missed | Unwanted | Bad args | Median |
|---|---|---|---|---|---|---|
| `qwen3:8b` | **36/36** | 0 | 0 | 0 | 0 | 1.17 s |
| `qwen3:4b-instruct` | 33/36 | 0 | 3 | 0 | 0 | 0.91 s |

The 4B's restraint is perfect. It missed one case three times out of three:
"What's the status of the Talos project?" — it answered from the profile, which
says he is building Talos, instead of looking. Consistent rather than noisy.

The same caveat as before applies more strongly now: 12 cases cover 7 of 37
skills.

Latency and memory, with Whisper `large-v3-turbo` resident. Warm model, median
of three after one discarded run, prompt *"Kendini bir cümleyle tanıt."* with
the real system prompt. First-token numbers are low because the identical
prefix is cached between runs; compare them with each other, not with the
table above.

| Model | Context | Tools | First token | Total | Model VRAM | Card total |
|---|---|---|---|---|---|---|
| `qwen3:8b` | 4096 | yes | 0.06 s | 0.83 s | 5.6 GB | 7748 MiB |
| `qwen3:4b-instruct` | 8192 | yes | 0.08 s | 0.54 s | 3.9 GB | 6198 MiB |
| `qwen3:4b-instruct` | 4096 | no | 0.04 s | 0.26 s | 3.2 GB | 5530 MiB |

Baseline with nothing loaded was 1636 MiB (desktop, browser); Whisper added
1.2 GB. Even at 8192 context the 4B leaves 1.5 GB more free than the 8B does
at 4096.

**Language.** `config/character.md` says Talos works in English in both
directions. Asked three things in Turkish, the 4B answered all three in English,
in character. The 8B answered all three in Turkish, against its instructions.
So on this measure the smaller model is the more obedient one. Its Turkish
quality is untested, because it would not produce any.

Verdict: a credible fine-tuning base. The context cost is real and has to be
paid in VRAM, but it fits. It is not a drop-in replacement: at Talos's current
4096 it breaks on every skill call. The model setting is unchanged.

### A 12B: `gemma4:12b`

2026-10-24, Ollama 0.40.6, on Talos's own settings (`num_ctx` 4096).
`scripts/compare_models.py`, twelve cases, three runs each, with `qwen3:8b`
run again the same day as the baseline:

| Model | Correct | Wrong | Missed | Unwanted | Bad args | Median |
|---|---|---|---|---|---|---|
| `qwen3:8b` | **36/36** | 0 | 0 | 0 | 0 | **1.21 s** |
| `gemma4:12b` | **36/36** | 0 | 0 | 0 | 0 | 3.28 s |

Unlike the 4B, it fits its prompt into 4096 and calls skills without error.
It also got the project-status case the 4B missed. But on these cases there is
nothing left to win: both are perfect, and the 12B takes 2.7 times as long.

**It does not fit beside Whisper.** The card is an 8 GB RTX 5070 Laptop.
With `gemma4:12b` loaded and nothing else, 7688 MiB of 8151 was in use;
`qwen3:8b` in the same state used 6443. Whisper's 1.2 GB fits beside the 8B
and not beside the 12B, so live it would spill to the CPU and get slower
still.

Verdict: no change. Worth trying again only on a larger card, or if a case
turns up that the 8B gets wrong.

### Twice the context, for no memory: `q8_0` KV cache

2026-10-24, Ollama 0.40.6, `qwen3:8b` with Whisper `large-v3-turbo` resident,
the real system prompt and skills. Warm model, median of three after one
discarded run. Each row is a separate Ollama server, so the cache type is the
only difference. The card read 757 MiB with nothing loaded.

| KV cache | Context | Total | Model VRAM | Card total |
|---|---|---|---|---|
| f16 (default) | 4096 | 0.54 s | 5.59 GB | 6744 MiB |
| f16 (default) | 8192 | 0.91 s | 6.19 GB | 7020 MiB |
| `q8_0` + flash attention | 4096 | 0.51 s | 5.30 GB | 6237 MiB |
| `q8_0` + flash attention | 8192 | **0.53 s** | 5.63 GB | **6547 MiB** |

At 8 bits, 8192 tokens cost less than the default 4096 did, and are no
slower. The default cache at 8192 also fits, at 300 MB more and 0.4 s more
per reply. `scripts/compare_models.py` on the `q8_0` server at 8192 scored
36/36 — the smaller cache did not cost accuracy on these cases.

`context` is now 8192. The cache type belongs to the Ollama service, not to a
request, so it is set in the environment Ollama starts in:
`OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0`.

### Fewer skills: 40 to 23

Same day. Removed code editing and review (five skills), two diagnostics, and
merged every group that answered one question: seven planning skills into
`my_day`, three about Talos into `about_me`, three for listening into one, two
each for English practice and application time.

| | Skills | Prompt tokens before the question |
|---|---|---|
| Before | 40 | 3584 |
| After | 23 | **2812** |

`compare_models.py` still 36/36. Fifteen new cases for the merged skills
scored 13/15 on the first try. Two misses were fixed by rewording a
description; the third — "what's due in Todoist tomorrow" sent as a text
search for "tomorrow" every time — by making `todoist_search` treat a date
word as the date filter it obviously is. "Are you listening all the time?"
still sometimes gets an answer from the model without calling `listening`.

### Finding a file on the desktop

2026-10-24. The whole desktop, OneDrive-backed, projects included.

| Search | Walking with `rglob` | Windows Search index |
|---|---|---|
| a folder by name (`Dersler`) | 42.3 s | 0.97 s |
| a word (`RAG`) | 19.0 s | 1.04 s |
| something that does not exist | 15.0 s | — |

Excluding folders such as `node_modules` inside the index query, rather than
afterwards, took every search to 19 s; it is filtered afterwards instead.

The index had skipped the Talos project entirely. `find_file` also looks
directly at the top of every root and every project, which is a few hundred
names and covers it.

One finding beyond speed: given `Projects/Talos\README.md` — forward slashes
from the root, backslashes after — `qwen3:8b` ended the turn without a word in
3 of 6 tries instead of reading the file. With forward slashes throughout,
0 of 10.

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

## What it costs to leave running

Idle in wake-word mode: models loaded, listening, nothing being asked of it.

| | |
|---|---|
| CPU | **3.7% of one core** — 0.2% of a 20-thread machine |
| RAM | 921 MB Python, plus 168 MB for the Ollama service |
| VRAM | **6716 MB of 8151** |
| Startup | 15 s — 5 s to build components, 10 s to warm them |

The VRAM splits about 1.1 GB for Whisper and 5.6 GB for the language model.
Whisper is the part that cannot be released while listening, because it *is*
the wake word.

The steady 3.7% is voice activity detection running continuously on the CPU:
every 32 ms, a chunk goes through Silero to decide whether it is speech. That
is the price of not having a trained wake-word model, and it is a good deal —
a fifth of a percent of the machine to avoid a training pipeline.

Idle costs nothing on the GPU. The card is *occupied* rather than *busy*: the
weights sit there and nothing computes until you speak, which matters because
occupied is what stops you playing something, and `release_gpu` is the answer.

Disk, separately, is about 21 GB: 13 GB of Ollama models, 3.1 GB of virtual
environment — 2 GB of that being the bundled CUDA libraries — 2.8 GB of Ollama
itself, and 1.6 GB of Whisper and speaker-recognition weights.

## Not yet measured

- End-of-speech detection (phase 2)
- How often the wake word fires when it should not, in ordinary use
