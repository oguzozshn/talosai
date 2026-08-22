# Talking to Talos

Thirty-three skills. You never name them — you say what you want and the model
picks. The phrases below are examples, not commands; anything that means the
same thing works.

## Starting it

```
run main.py --wake     say "Talos" to wake it, no key needed
run main.py            hold F9 while you talk
run main.py --text     type instead of speaking
```

In wake mode, say the name and wait for "Yes?", or say it all in one breath —
"Talos, what did I work on today?". After a reply it keeps listening for ten
seconds, so a follow-up needs no name.

You can switch without restarting: **"stop listening"** shuts the microphone
and leaves the hotkey, **"start listening"** goes back.

---

## What it can do

### Your projects

> "What have I worked on recently?"
> "What's the status of the BACnet project?"
> "List my projects."
> "Find the README in RatchetSync."
> "Read the config file in Talos."

### Looking over code

> "Show me the uncommitted changes in Talos."
> "What did the last commit in BACnet actually do?"

It fetches the diff and comments on it. Worth knowing what that is worth: an
8B model catches typos and forgotten null checks, not race conditions or a
design that will hurt in six months.

### Changing code

> "In the Talos README, change the licence line to say MIT."
> "Create a file called notes.md in the Talos project with a heading."
> "Undo my changes to that file."

Every edit asks first, out loud, and tells you what it is about to change.
Only files git already tracks can be edited, so `git diff` is always the
record and `git checkout` is always the undo. It refuses ambiguous edits
rather than guessing which of three matches you meant.

### Todoist

> "What's due today?"
> "Add a task to fix the serialization bug tomorrow."
> "Have I got anything about BACnet?"
> "Mark the parser task done."

Adding and completing ask first. Reading does not.

### Your week

Reads the plan the personal assistant project already writes.

> "What's on today?"
> "What's next?"
> "How does this week look?"

### Notes and memory

> "Remember that the deploy key expires in March."
> "What did I say about BACnet?"
> "What were we talking about earlier?"
> "Read my notes."

Notes are a file. Memory is every exchange, searchable, in `data/memory.db` —
it never leaves the machine, and `run main.py --forget` erases it.

### The screen

> "What's on my screen?"
> "Read the error I'm looking at."

Optical character recognition, not a vision model — it reads text, not
pictures. It cannot tell you a chart looks wrong.

### The web

> "Search for the faster-whisper hotwords parameter."

The one skill besides Todoist that leaves the machine.

### Time on the computer

> "What have I been using most today?"
> "How long was I in Visual Studio?"

Only counts while Talos is running, and drops time when you are away from the
keyboard. It reports and does not comment.

### The graphics card

> "How much graphics memory is in use?"
> "Let go of the GPU."

Releasing frees 5.6 GB for a game or anything else. The next question then
takes about a minute while the model loads again.

### English practice

> "Let's practise English."

It talks with you and corrects one thing per reply, conversation first. Say
**"stop practising"** to end it. Over twenty test sentences it corrected eight
of ten real mistakes and never invented one — an 8B model, so it misses
things.

### Time and date

> "What day is it?"
> "How many days until Friday?"

No skill involved; the date is in every prompt.

---

## What it learns about you

`config/profile.md` is the one thing Talos knows about you without going
looking. It is in the prompt of every single reply, which is what makes it
useful and what makes it expensive: every character there competes with the
conversation for room.

Talos never writes it. It reads the conversation log, proposes facts it thinks
are worth keeping, and stops there. You decide which ones are true.

```bash
run main.py --profile
```

It scans the exchanges it has not seen before, then walks what it found one at
a time, showing the fact and the exact words it came from:

```
  The owner has a girlfriend named Sila
    from [22 August] "I'll remember that you have a girlfriend named Sila."
    [a]ccept  [r]eject  [s]kip  [q]uit:
```

Accepting appends it under `## Learned` in the profile, below whatever you
wrote by hand — so what a person put there and what a model proposed stay
telling apart. Rejecting is permanent: that fact is never proposed again.
Skipping leaves it for next time.

**Read the quote, not the fact.** The first real run proposed "The owner's name
is Dilara", quoted from a misheard sentence about a friend. The fact reads
perfectly. Only the quote gives it away. That is the entire reason this is a
review step instead of something Talos does on its own — a wrong line in the
profile is in every future reply, and nothing about it looks broken.

When the profile is full, promoting a new fact means dropping an old one, and
it will ask you which. The budget is `[profile] max_chars`, and it is a real
constraint rather than a tidiness rule: this is the text that gets prefilled
before every answer.

Nothing is proposed during a conversation and nothing is written by voice. The
scan only runs when you run that command.

## What it will not do

- Edit a file git does not track, or one it cannot undo
- Open anything that looks like credentials — `.env`, `*.key`, `gizli.json`
- Reach outside the folders in `[skills] workspaces`
- Run a command that changes something without asking out loud first
- Treat silence or "maybe" as a yes

---

## When something goes wrong

**It answers but says nothing.** The voice failed; the reply is on screen.

**"I can't reach the model right now."** Ollama stopped. Launch it, or run
`ollama serve`.

**It ignores you in wake mode.** Run with `--verbose` and watch: it prints
what it heard and why it did not act. Usually the wake word was misheard, or
the follow-up window had closed.

**It says "(nothing heard)" or "(nothing recognised)".** Run with
`--diagnostic`. The first means the audio never passed the gate, the second
means it passed and the decoder returned nothing anyway — and the flag prints
the numbers that tell them apart: how long the recording was, how loud, which
device it came from, and which threshold it fell under.

Audio that produced no reply is written to `data/unrecognised.wav`, one file,
overwritten each time. Listen to it. Three separate faults print those same
two lines — a microphone capturing a room nobody is in, a Bluetooth headset
handing over audio no decoder can read, and `[stt] language` pinned to a
language you are not speaking — and the recording is what distinguishes them.
The last one is the quietest: forced to English, clean Turkish comes back as a
confident, fluent, half-wrong English sentence, and nothing reports an error.

**It answers something you did not say.** Same flag. `--diagnostic` prints
what it heard before it prints what it did, along with every skill it called,
the arguments it called them with, and what came back.

**It wakes when nobody spoke to it.** Raise `speech_threshold`, or set
`continue_seconds = 0` so the name is needed every time.

Both live in `config/local.toml`, which is not committed.
