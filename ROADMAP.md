# Roadmap

Each phase ends with something that demonstrably works. Projects like this die
when you install everything first and only then try to run it — by the time
something breaks, you have no idea which layer broke.

---

## Phase 0 — Environment and verification

**Deliverable:** four independent scripts, all passing. Nothing wired together.

- [ ] Python 3.12 virtual environment, created outside OneDrive
- [ ] Ollama installed, a model pulled, working from the terminal
- [ ] `check_env.py` — reports GPU, CUDA, and which components can use it
- [ ] `test_mic.py` — record five seconds, play it back
- [ ] `test_stt.py` — transcribe a wav file, report elapsed time
- [ ] `test_tts.py` — speak a sentence
- [ ] `test_llm.py` — one round trip through Ollama, report time to first token

**Risk:** Blackwell (`sm_120`) is new enough that older CUDA builds will not
recognise the GPU and will quietly fall back to CPU. `check_env.py` exists to
make that failure loud instead of silent. CPU fallback still works — it is just
slower — so this is not a blocker, only something to know about.

---

## Phase 1 — The chain, push-to-talk

**Deliverable:** an assistant you can actually talk to.

Hold a hotkey, speak, release. Transcribe, think, answer aloud. No memory, no
skills, no wake word — but working end to end.

Log the duration of every step. The latency budget:

| Step | Target |
|---|---|
| End-of-speech detection | 0.5 s |
| Transcription (turbo, GPU) | 0.4 s |
| First token from the model | 0.3 s |
| First audio from TTS | 0.15 s |
| **Perceived latency** | **~1.4 s** |

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
