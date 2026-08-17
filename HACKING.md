# Extending Talos

How to change what Talos knows, how it behaves, and what it can do — without
reading the whole codebase first.

Three things you will want, in the order you will want them:

| Want to change | Edit | Restart needed |
|---|---|---|
| What it knows about you | `config/profile.md` | No |
| How it talks and behaves | `config/character.md` | No |
| Anything else — models, voice, reach | `config/settings.toml` | Yes |
| What it can *do* | Add a file to `talos/skills/` | Yes |

---

## The shape of it

One turn, end to end:

```
microphone → wake word → transcribe → think → speak
                            │           │
                            │           └── skills, memory, profile
                            └── Whisper, on the GPU
```

| File | Job |
|---|---|
| `main.py` | Entry point and the three modes: wake word, push-to-talk, text |
| `talos/wake.py` | Listens, decides whether it was addressed |
| `talos/stt.py` | Speech to text |
| `talos/llm.py` | The model, the tool-call loop, conversation history |
| `talos/tts.py` | Speech out, sentence by sentence |
| `talos/memory.py` | Profile, conversation log, recall |
| `talos/skills/` | Everything Talos can do |
| `talos/config.py` | Settings, with defaults matching `settings.toml` |

---

## Where its reach is decided

Two places, and they are the only two.

**`config/settings.toml`, `[skills] workspaces`** — the directories Talos may
read. This list is its entire reach. Adding a folder here is the whole of
granting access; nothing else needs changing.

```toml
[skills]
workspaces = [
    "C:/Users/oguzo/OneDrive/Desktop/Mhendislik/Projects",
    "C:/Users/oguzo/OneDrive/Desktop/personnal_assistant",
]
```

Paths that climb out with `..`, absolute paths from elsewhere, and symlinks
pointing outside are all refused — the check happens after resolving, so a
link cannot smuggle a path past it.

Do not point this at a home directory or a drive root. The point of the list
is that you can name what it sees.

**`talos/skills/_workspace.py`, `SECRET_PATTERNS`** — filenames refused even
inside those directories. `gizli.json`, `.env`, `*.key`, anything matching
`*token*`. Giving Talos a folder is not giving it the credentials that happen
to sit in that folder, and this assistant reads its answers out loud in a room
where other people might be.

Add patterns here when you put something sensitive somewhere it can reach.

---

## Changing how it behaves

**`config/profile.md`** — who it is talking to. Injected into every prompt,
unchanged, never searched. Read fresh on each reply, so edit it while Talos is
running and the next answer uses it.

Keep it short. Every line competes with the conversation for context. Facts
needed rarely belong in a skill; facts needed once belong in the conversation.

**`config/character.md`** — how it speaks and what it refuses. Same deal: read
fresh, no restart.

One thing to know before you write instructions here: **a rule enforced only
by a prompt is not enforced.** This has been true every single time in this
project. The character file has said "no markdown" since the beginning, and
the synthesiser was still handed `**bold**` and a bare `1.` to read aloud
until the stripping was done in code. If a rule matters, write it here *and*
enforce it in `talos/llm.py`.

**`config/local.toml`** — your overrides, git-ignored, merged over
`settings.toml` key by key. Secrets go here and nowhere else.

---

## Adding a skill

A skill is a function with type hints and a docstring. The schema the model
reads is generated from the signature, so there is no registration list and
nothing to keep in sync.

Create `talos/skills/weather.py`:

```python
"""Looking up the weather."""

from __future__ import annotations

from talos.skills.base import Permission, skill


@skill(Permission.OPEN)
def weather_today(city: str = "Istanbul") -> str:
    """Report today's weather.

    Args:
        city: Which city to look up
    """
    return f"It is probably raining in {city}."
```

Restart Talos. It appears in the skill list at startup and the model can call
it. That is the whole process.

### What the parts do

**The docstring is not documentation, it is the interface.** Its first line is
what the model reads to decide whether to call your skill, and the `Args:`
lines describe each parameter. A vague summary produces a skill that is called
at the wrong times, or never.

**Type hints become the schema.** `str`, `int`, `float`, `bool` map to JSON
types. Arguments are coerced before your function sees them — models send
`"5"` where the schema says integer, and that used to crash three skills — so
by the time you are called, `limit: int` really is an `int`.

**Return a string, phrased to be spoken.** Not JSON, not a table, not a
bulleted list. It goes to a synthesiser. Say "three projects: Talos, BACnet
and CalenDo", not `["Talos", "BACnet", "CalenDo"]`.

**Never raise for expected failures.** Return the problem as a sentence the
model can act on. `"No project called 'Talso'. Try list_projects."` is useful;
a traceback teaches it nothing. Unexpected exceptions are caught by the
registry, but a caught exception is a worse answer than a written one.

### Permission tiers

```python
@skill(Permission.OPEN)     # reads something. Runs without asking.
@skill(Permission.CONFIRM)  # changes something. Asks out loud first.
@skill(Permission.OWNER)    # destructive. Asks, and would check the voice.
```

The tiers are about **what cannot easily be undone**, not about reads versus
writes. `take_note` appends a timestamped line to one file and is OPEN;
`todoist_complete` is CONFIRM because other people see a completed task before
you can take it back.

`OWNER` currently behaves like `CONFIRM` — speaker verification exists in
`talos/speaker.py` but is not enrolled. It is not a lock.

Confirmation is spoken and carries your arguments, so the user hears *what*
will happen, not just which function. Silence and anything ambiguous count as
no.

### Reading files safely

Always go through the sandbox rather than touching paths directly:

```python
from talos.skills._workspace import OutsideWorkspace, SecretFile, describe, resolve

try:
    target = resolve(path)          # refuses escapes and credential files
except (OutsideWorkspace, SecretFile) as exc:
    return str(exc)                 # already phrased for the user

return f"{describe(target)} says..."  # path as the user would recognise it
```

### Testing without talking

```
run main.py --say "what's the weather"     one exchange, then exit
run main.py --text                          type instead of speaking
run main.py --mute                          run everything, stay silent
run main.py --verbose                       see which skills get called
```

Or call it directly, which is faster than going through the model:

```python
from talos.skills import load
registry = load()
print(registry.execute("weather_today", {"city": "Izmir"}))
```

And when you have added a few, check the model still picks correctly:

```
run scripts\compare_models.py --models qwen3:8b
```

Five of its twelve cases expect *no* skill call. A model that reaches for a
tool on every question is not being helpful — each call costs a full extra
round trip, about nine seconds.

---

## Things that will bite you

Each of these cost a debugging session here.

**A prompt is not an enforcement mechanism.** Said above, repeated because it
is the single most common failure in this codebase.

**Warm-ups must resemble the real request.** Warming the model with `"hi"`
left the first real turn paying to prefill the system prompt and every tool
schema: 9.8 s against 1.2 s. If you add anything that makes prompts longer,
make sure `Brain.warm_up` still sends it.

**Whisper mishears anything it does not expect.** "Talos" came back as Kalos,
Salos, Taroj and "Tell us"; "serialization" as "sterilization". The fix is to
tell it what to expect — `[stt] vocabulary` in settings. Add your jargon there
when a skill starts getting called with mangled arguments.

**Measure warm, not cold.** First GPU calls compile kernels and look 25x
slower than reality. First model loads take a minute. Neither is the number
you are optimising.

**The reply is heard once and cannot be re-read.** Two or three sentences.
This is why long procedures get written to a note instead of spoken.

---

## Worth building next

Ordered by usefulness per hour of work, for how this machine actually gets
used.

~~**Read the weekly plan.**~~ Built — `talos/skills/planner.py`, reading the
`data/W*.json` the personal assistant project already writes. Read-only on
purpose: asking what you are meant to be doing should not be able to change
the answer.

**Timers and reminders.** "Start a twenty-five minute timer." Needs a
background thread and a way to interrupt with speech — the plumbing for
speaking unprompted does not exist yet, and several later ideas depend on it.

**Application control.** "Open Visual Studio", "close Spotify". A short
allowlist of names to executables. Small, and immediately useful.

**Clipboard.** "What did I just copy?", "save that as a note". Two lines with
`pywin32`, and it bridges what you are looking at to what Talos knows.

**Terminal, with an allowlist.** Deliberately deferred. `git status`,
`dotnet build`, `npm test` run without asking; anything else is confirmed by
voice; no shell metacharacters, ever. Voice plus an 8B model plus an unguarded
shell is how a folder goes missing — Whisper heard "serialization" as
"sterilization" on the first day.

**Reading the screen.** Would need a vision model in the remaining VRAM, which
means dropping the language model to a smaller one. Expensive; judge it
against the others rather than assuming it is the impressive one.
