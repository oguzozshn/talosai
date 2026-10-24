# Talking to Talos

Twenty-nine skills. You never name them — you say what you want and the model
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

## From your phone: Telegram

Write to Talos, or send it a voice message, and it answers in writing. It runs
the same skills as by voice, so "what's on my plate" and "send me the
Requirements PDF from Dersler" both work. The computer has to be on, and Talos
running: the bot is only a way in, the thinking still happens here.

Setting it up, once:

1. In Telegram, talk to **@BotFather**, send `/newbot`, and follow it. It
   gives you a token — treat it like a password.
2. Put it in `config/local.toml`, which is never committed:

   ```toml
   [telegram]
   token = "the token BotFather gave you"
   ```

3. Start Talos and write anything to your bot. Nothing comes back, and the
   Talos window prints the chat id the message came from. Add it:

   ```toml
   allowed_chat_ids = [123456789]
   ```

   and start Talos again. Only those chats are ever answered; anyone else who
   finds the bot gets silence.

Telegram starts with every mode on its own. To run it without the microphone
at all — on a machine nobody is sitting at:

```
run main.py --telegram
```

Anything that changes something or sends something away asks first, with Yes
and No buttons; no answer within two minutes is a no. Sending a file always
asks, because it leaves the machine. Files that look like credentials are
never sent, and the Bot API's limit is 50 MB.

A voice message is transcribed here, and what was heard is written back first,
so a mishearing shows before the answer to it.

## What it can do

### Your projects

> "What have I worked on recently?"
> "What's the status of the BACnet project?"
> "List my projects."
> "Read the config file in Talos."

### Studying

> "What does traceability mean?"
> "RAG ne demek?"
> "What do the slides say about elicitation techniques?"
> "Which terms have I asked about?"
> "Quiz me on my requirements terms."

The course folder (`[study] folder`, Dersler on the desktop) holds a folder
per course. Its slides, notes, books and syllabus are indexed in the
background after start-up — the first time takes a couple of minutes for a
textbook, after that only changed files are read. To index now and watch:

```
run main.py --index-courses
```

Answers from the material say which file and page or slide they came from.
Without a course named, the one in today's Argos study session is assumed.

Every "what is X" goes through the course material first, even for something
Talos knows — so the course's own sense of the word comes first, and the term
lands in your glossary with what Talos said about it. Asking to be quizzed
picks one and waits for your explanation before comparing.

### Your files

> "Where is my Dersler folder?"
> "Send me the Requirements PDF from Dersler."
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
> "What time is it in New York?"

The date is in every prompt. The time anywhere else is worked out here from
the time zone database, daylight saving included — web search could only find
pages with a clock drawn by JavaScript, which a search result cannot show.

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
