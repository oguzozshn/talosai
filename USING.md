# Talking to Talos

Twenty-three skills. You never name them — you say what you want and the model
picks. The phrases below are examples, not commands; anything that means the
same thing works.

## Starting it

```
run main.py --wake     say "Talos" to wake it, no key needed
run main.py            hold F9 while you talk
run main.py --text     type instead of speaking
```

Or double-click **Talos** on the desktop, which starts wake mode. The shortcut
is made once with `run scripts\make_shortcut.py`, and again after moving the
project.

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
> "Read the config file in Talos."

### Your files

> "Where is my Dersler folder?"
> "Which PDFs are in Dersler?"
> "Find the README in RatchetSync."
> "Open the Dersler folder."

Everything on the desktop, not only the projects. Searches go through the
Windows Search index, so they take about a second; walking the desktop by hand
took up to 42. The index can miss things — on this machine it had skipped the
Talos project — so the top of every folder and every project is also looked at
directly. Many names here are Turkish, so if nothing turns up, say the Turkish
word.

"Open" shows a folder in Explorer, or a file selected in its folder. It never
runs a file: for a `.cmd` or `.exe` that would mean running a program.

### Code

Talos reads code, it does not change it. Asked to edit or review, it used to:
an 8B model with a few thousand words of context caught typos, not bugs, and
every edit was one more thing to check. Claude Code does both properly.

### Todoist

> "What's due today?"
> "Add a task to fix the serialization bug tomorrow."
> "Have I got anything about BACnet?"
> "Mark the parser task done."

Adding and completing ask first. Reading does not.

### Your week

Reads what Argos, the planner, already keeps: the weekly plan, study plans,
tasks, goals and the things you asked it to remember. Read only — Argos
checks its files before saving, and a write from here would look to it like a
conflict you caused.

> "What's on my plate today?"
> "What's on Friday?"
> "What's next?"
> "How does this week look?"
> "How much is left for the Requirements midterm?"
> "What are my goals?"

Todoist tasks come from Todoist itself, not from Argos's copy of them.

### Notes and memory

> "Remember that the deploy key expires in March."
> "What did I say about BACnet?"
> "What were we talking about earlier?"
> "Read my notes."

Notes are a file. Memory is every exchange, searchable, in `data/memory.db` —
it never leaves the machine, and `run main.py --forget` erases it.

**"Remember X" writes a note.** Not because the model decides to: the request
is matched before the model answers, and the note is taken either way. Asked to
remember something it used to say "I'll remember that" and record nothing,
which sounds exactly like the thing working — so nobody checks, and the fact is
gone by morning. Turkish works too, in either word order.

Asking whether something is remembered is still a search rather than a note:
"do you remember what I said" looks through the log.

Anything you tell it to remember also becomes a candidate for your profile on
the next `--profile` run, so a fact worth having in every reply only has to be
said once.

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

- Change a file. It reads them, and that is all
- Open anything that looks like credentials — `.env`, `*.key`, `gizli.json`
- Reach outside the folders in `[skills] workspaces` and `[skills] folders`
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
