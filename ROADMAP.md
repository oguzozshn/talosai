# Roadmap

Each phase ends with something that demonstrably works. Projects like this die
when you install everything first and only then try to run it — by the time
something breaks, you have no idea which layer broke.

---

## Phase 0 — Environment and verification

**Deliverable:** four independent scripts, all passing. Nothing wired together.

- [x] Python 3.12 virtual environment, created outside OneDrive
- [x] Ollama installed, `qwen3:8b` pulled and answering
- [x] `check_env.py` — reports GPU, CUDA, and which components can use it
- [x] `test_mic.py` — record five seconds, play it back
- [x] `test_stt.py` — transcribe a wav file, report elapsed time
- [ ] `test_tts.py` — speak a sentence *(Piper not yet fetched)*
- [x] `test_llm.py` — one round trip through Ollama, report time to first token

**Blackwell turned out to be fine**, but not for free. `nvidia-smi` saw the card
and `ctranslate2.get_cuda_device_count()` returned 1, and both were misleading:
the first inference still died on a missing `cublas64_12.dll`. The driver-level
check says nothing about whether the compute libraries are present. See
`talos/cuda.py`.

**Two traps that make a healthy machine look broken**, both worth knowing
before trusting any measurement here:

- *The first GPU call is not representative.* Kernel compilation and cuDNN
  autotuning made transcription look 25x slower than it is. Measure a warm
  call; warm the models at startup so users never see the cold one.
- *Reasoning models think before they answer.* Qwen3 emits its reasoning into
  a separate field, so a naive time-to-first-token measurement is really
  time-to-finished-thinking — 12.6 s against 0.45 s with thinking off. None of
  it would be spoken aloud, so it is pure latency. Thinking stays off.

---

## Phase 1 — The chain, push-to-talk

**Deliverable:** an assistant you can actually talk to.

Hold a hotkey, speak, release. Transcribe, think, answer aloud. No memory, no
skills, no wake word — but working end to end.

Log the duration of every step. The budget, against what phase 0 actually
measured on the target machine:

| Step | Budget | Measured |
|---|---|---|
| End-of-speech detection | 0.5 s | — |
| Transcription (turbo, GPU) | 0.4 s | **0.22 s** |
| First token from the model | 0.3 s | **0.35 s** |
| First audio from TTS | 0.15 s | — |
| **Perceived latency** | **~1.4 s** | — |

Two components are still unmeasured, and both are the assembled system's
problem rather than a component's, so phase 1 is where the real number appears.

Note what these figures assume: models already resident. A cold Whisper call
costs 0.97 s, and a cold model load about 60 s for 5 GB. Ollama unloads after
five idle minutes by default, which would land that 60 s in the middle of a
conversation — so requests pin `keep_alive`, and startup warms both models.

Responses stream: the model writes, and TTS starts speaking sentence by sentence
rather than waiting for the full answer. Past roughly 2.5 seconds it stops
feeling like conversation and starts feeling like a form submission.

**Do not skip this phase.** It is tempting to jump straight to the wake word.
Resist — if the chain has never worked on its own, you will not be able to tell
which layer is at fault when the wake word version misbehaves.

---

## Phase 2 — Speaker verification and wake word

**Deliverable:** the hotkey goes away.

- [ ] Voice enrolment — 30-60 seconds of speech becomes an ECAPA embedding
- [ ] Wake word training for "Talos" — synthetic samples plus real recordings
- [ ] Voice activity detection, to know when a sentence has ended
- [ ] Background loop: listen, wake, verify the speaker, respond

Speaker verification is a **filter, not authentication**. A recording of your
voice will fool it, and a hoarse throat or a noisy room may lock you out. Use it
to ignore the television and other people in the room — not to guard anything
that actually matters.

By the end of this phase the assistant is genuinely usable. Everything after is
improvement.

**Risk:** false-trigger tuning is not finished when the code is finished. Expect
a day or two of adjusting the threshold during real use. This is the least
rewarding stretch of the project and the most common place to abandon it — plan
for it rather than being ambushed by it.

---

## Phase 3 — Skill infrastructure

**Deliverable:** adding a capability means adding a file.

The architecture is the product here; the first three skills exist mainly to
prove it works.

- [ ] `skills/` discovery, with JSON schemas generated from type hints
- [ ] Permission tiers: open / voice-verified / voice-verified plus spoken confirmation
- [ ] Tool-call loop with validation, error recovery, and retry
- [ ] First skills: file search, project status (git), take a note

Destructive actions need confirmation from the start. Retrofitting a permission
model after the assistant can already delete things is how you lose a folder.

**Risk:** this is the weakest link in the whole project. Small local models
sometimes call the wrong tool or malform arguments. Keep the toolset small and
strictly validated — five reliable skills beat twenty flaky ones. If Gemma
handles tool calls badly, Qwen3 is the fallback; pull both in phase 0 and
compare.

---

## Phase 4 — Memory

**Deliverable:** "what did we talk about last week?"

- [ ] Static profile — who the user is, injected into every prompt, never searched
- [ ] Conversation log in SQLite
- [ ] Retrieval over past conversations and notes via sqlite-vec
- [ ] Summarisation, so long histories do not crowd out the context window

Note the split: **files and code are reached through tools, conversation history
through retrieval.** Do not embed the codebase — the index goes stale on every
commit, and semantic similarity finds worse matches in source code than a plain
text search does.

---

## Phase 5 — Living with it

**Deliverable:** it stops being a project and becomes a habit.

- [ ] Tray application, starts with Windows
- [ ] Unload models from VRAM when idle
- [ ] More skills: reminders, application control, screen reading, calendar
- [ ] Optional: a C# WinUI shell over the Python core
