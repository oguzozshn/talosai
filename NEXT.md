# Prompt: what Talos needs next

Paste this into a fresh agent session started in the Talos repository. It is
written to stand alone — it assumes no memory of how the project got here.

---

You are working on **Talos**, a voice assistant that runs entirely on this
machine: Whisper for speech, Ollama with `qwen3:8b` for reasoning, Piper for
speech out, SQLite for memory, and 33 skills discovered from files in
`talos/skills/`. Read `README.md`, `HACKING.md` and `MEASUREMENTS.md` before
changing anything — between them they carry every decision and the number
behind it.

It works. A spoken turn takes about 1.2 seconds to first audio. What follows is
not rescue work; it is the difference between something that demos and
something that survives being used.

## The rule that matters most here

**Measure before you claim, and measure the thing you actually care about.**

This project has been wrong in the same way repeatedly, and always by
reasoning where it could have measured:

- A clip length was set to 1.44 s because that "should" be sixteen frames. It
  is nine. Two seconds is sixteen.
- A wake word model scored 99% on held-out clips and fired 750 times an hour
  on real speech, because the held-out clips came from the same two synthetic
  voices as the training ones.
- The first turn of a session took 9.8 s against 1.2 s for the rest. The model
  was already loaded; the warm-up simply did not resemble a real request.
- A Todoist skill was written against an API that had been retired, and the
  error looked like a bad token.

If you find yourself writing "this should be faster" or "this is probably
fine", stop and measure it instead. `MEASUREMENTS.md` is where the numbers go.

**And a rule enforced only by a prompt is not enforced.** The character file
has forbidden markdown since the beginning, and the synthesiser was still
being handed `**bold**` to read aloud until stripping was done in code. The
model imitates the shape of what it is shown far more readily than it follows
what it is told — it copied `He says:` from an example straight into its
output. When a rule matters, put it in the prompt *and* enforce it in code.

## Work, in the order it should be done

### 1. Tests, because there are none

Thirty-three skills, zero tests, one developer. Every bug found so far would
have been caught faster by a test, and nothing stops any of them returning.

Start with the pure logic that needs no microphone, no GPU and no model:

- `talos/wake.py` — `strip_wake_word` and the fuzzy name matching. There is a
  worked table of cases in the git history: "Kalos", "Salos", "Taroj" must
  wake it; "Tables are ready" and "I was telling Ahmet about Talos" must not.
- `talos/llm.py` — sentence splitting and markdown stripping. A bare `1.` on
  its own line must never reach the synthesiser.
- `talos/coach.py` — `_same_words`, which decides whether a correction is
  really an echo of the sentence. Getting its direction wrong silently threw
  away five good corrections out of six.
- `talos/skills/__init__.py` — argument coercion and the sandbox. `"5"` must
  become `5`; `../../..` and `gizli.json` must be refused.

Use `pytest`. Do not mock the sandbox — point it at a temporary directory and
let it really refuse things.

### 2. Grow the benchmark with the skill set

`scripts/compare_models.py` has 12 cases covering 7 skills. There are 33. The
model choice was decided on that benchmark, and 26 skills have never been
checked for whether the model picks them correctly.

Add cases until every skill is covered, and keep the ratio: roughly half the
cases must expect **no** skill call at all. Restraint is the half that usually
goes unmeasured, and a model that reaches for a tool on every question costs a
full extra round trip each time.

Then re-run it. If accuracy has dropped, that is a real finding — say so
rather than tuning the benchmark until it passes.

### 3. Delete what is not used

- `talos/speaker.py` and `scripts/enrol_voice.py` — speaker verification,
  built and then declined by the owner. Nothing calls it, and it is the only
  reason PyTorch (526 MB) is installed.
- The `OWNER` permission tier in `talos/skills/base.py`. It promises a
  verified voice, there is none, and it behaves exactly like `CONFIRM`. A tier
  that never differs from another tier is noise in the API. Move
  `undo_changes` to `CONFIRM` and remove the tier.
- Consider merging `todays_plan`, `whats_next` and `weekly_plan` into one
  skill with an argument. Three skills for one question is three chances to
  pick the wrong one.

Keep `scripts/make_wake_samples.py`, `train_wake_word.py` and
`test_wake_word.py` — they are a documented, measured, parked option and
HACKING.md explains what it would take to finish them.

### 4. Let it speak without being asked

Nothing currently happens unprompted. There are no timers, no reminders, no
"your meeting starts in ten minutes". Several things the owner has asked for
depend on this and none of the plumbing exists.

The hard part is not the timer, it is interrupting: the wake listener owns the
microphone and the voice, and something needs to speak through them without a
turn being in progress. Design that seam before writing any feature on top of
it, because reminders, timers and calendar alerts all sit on it.

### 5. Make it survive a day

Everything measured so far comes from sessions minutes long. Nobody knows:

- How often it wakes falsely over eight hours in a real room
- Whether `data/memory.db` grows without bound
- Whether anything leaks over a long session
- What happens when the machine sleeps and wakes

Run it for a working day with `--verbose` logging to a file, then read the log
and write what you find into `MEASUREMENTS.md`. This is the highest-value
measurement left and it needs no code to start.

## How to work here

- **Commit messages and all repo prose in English.** Small, frequent commits,
  one subject each. Explain *why*, not what the diff already shows.
- **Never commit `config/local.toml`.** It holds an API token. It is
  git-ignored; keep it that way.
- **Skills return a string that will be spoken aloud.** Not JSON, not a
  bulleted list. A list on separate lines once made the model invent an extra
  row to complete the pattern.
- **Never raise from a skill for an expected failure.** Return a sentence the
  model can act on.

## Traps specific to this machine

- **The projects live in OneDrive with files-on-demand.** Any command that
  reads file contents can block for minutes while OneDrive fetches them, and
  `subprocess.run(timeout=...)` does **not** save you on Windows — it kills
  the process but not its children, then hangs waiting on a pipe a survivor
  still holds. Use `talos/skills/_run.py`, which kills the process tree.
- **The console is cp1254.** `run.cmd` forces UTF-8; keep it. Without it, a
  library printing an em dash or an emoji crashes the program.
- **`openwakeword[full]` cannot be installed** — it pins TensorFlow 2.8.1,
  which has no Python 3.12 wheels. Only the tflite conversion needs it.
- **`acoustics` imports `scipy.special.sph_harm`,** removed in SciPy 1.15.
  There is a shim at the top of `scripts/train_wake_word.py`.
- **Ollama does not always survive a sleep.** If the model is unreachable,
  that is usually why.

## What not to do

Do not add skills. There are 33 and the benchmark covers 7. Do not tune
prompts to fix a behaviour that code should enforce. Do not replace the model
without running `scripts/compare_models.py` first — Gemma 3 was described as
the alternative for two days before anyone checked that it cannot call tools
at all.
