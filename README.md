# Talos

A fully local voice assistant for Windows. No cloud APIs, no subscriptions, no data leaving the machine.

Named after the bronze automaton of Greek myth that patrolled the shores of Crete — arguably the first artificial guardian anyone imagined. It waits quietly, answers when called, and gets out of the way.

## Design constraints

These are hard requirements, not preferences:

1. **Everything runs locally**, with one named exception. Speech, reasoning,
   memory and voice have no network path at all — no API keys, no accounts,
   nothing to leak. The exception is the Todoist skill, added deliberately and
   listed here so the claim stays true: it sends task text to Todoist's servers
   and stops working when the internet does. Delete `talos/skills/todoist.py`
   and the exception goes with it.
2. **Fits in 8 GB of VRAM.** Speech recognition and the language model must coexist on a single laptop GPU.
3. **Voice-gated.** The assistant recognises its owner's voice and refuses destructive actions from anyone else.
4. **Skills are files.** Adding a capability means dropping a Python file into `talos/skills/` — never editing the core.

## Stack

| Layer | Component | Runs on |
|---|---|---|
| Wake word | openWakeWord | CPU |
| Speech-to-text | faster-whisper (`large-v3-turbo`, int8) | GPU (~1.6 GB) |
| Speaker verification | SpeechBrain ECAPA-TDNN | CPU |
| Reasoning | Ollama (Qwen3 8B, Q4_K_M) | GPU (~5.2 GB) |
| Text-to-speech | Piper | CPU |
| Memory | SQLite + sqlite-vec | CPU |

### The VRAM budget is the whole design

| | |
|---|---|
| Total | 8151 MB |
| Reserved for the desktop | ~500 MB |
| Whisper `large-v3-turbo`, int8 | ~1500 MB |
| KV cache at 8k context | ~700 MB |
| **Left for the model** | **~5400 MB** |

Which rules out a 12B at Q4 — that is 8 GB on its own, and Ollama would quietly
spill the remainder into system RAM, where generation slows by roughly an order
of magnitude. Qwen3 8B fits with room to spare and is the better tool-caller
anyway.

If a larger model turns out to be worth it, the trade is to move Whisper to the
CPU. Twenty threads transcribe close enough to real time that it is a viable
swap, not a downgrade — measure both before deciding.

## Target hardware

Developed against an RTX 5070 Laptop (8 GB VRAM, Blackwell `sm_120`), 24 GB RAM, i7-13650HX.
Two consequences worth knowing before you install anything:

- **Blackwell needs recent CUDA builds.** Wheels compiled before CUDA 12.8 will not recognise `sm_120` and fall back to CPU silently.
- **Python 3.13+ is not usable yet.** Several audio and ML packages have no wheels for it. Use 3.12.

## Setup

Requires [Ollama](https://ollama.com/download) and Python 3.12.

```powershell
# Create the environment outside OneDrive — a synced venv is thousands of
# churning files and will fight you.
py -3.12 -m venv "$env:USERPROFILE\.venvs\talos"
& "$env:USERPROFILE\.venvs\talos\Scripts\Activate.ps1"
pip install -r requirements.txt
```

Then verify each component independently before wiring anything together:

```
run scripts\check_env.py
```

`run.cmd` calls the venv interpreter directly instead of activating it, so it
behaves the same in cmd and PowerShell and works from any directory. Activation
syntax differs between the two shells, and getting it wrong is reliably the
first thing that goes wrong when setting this up.

See [`scripts/README.md`](scripts/README.md) for what each verification script proves.

## Running it

```
run main.py                  hold F9, speak, release
run main.py --text           type instead of speaking
run main.py --say "merhaba"  one exchange, then exit
run main.py --mute           run the whole chain but stay silent
```

## Project status

Phases 0, 1 and 3 are built. A spoken turn works end to end, and skills are
discovered from files and called by the model.

| Phase | State |
|---|---|
| 0 — environment | Done. Numbers in [MEASUREMENTS.md](MEASUREMENTS.md) |
| 1 — the chain | Done, push-to-talk. 0.93 s to first audio |
| 2 — wake word | Done. No trained model — see below |
| 3 — skills | Done: 9 skills, sandboxed, permission-tiered |
| 4 — memory | Profile and recall done; summarisation open |
| 5 — living with it | Not started |

Two departures from the plan, both deliberate:

**No wake word model.** Training one costs hours of setup and days of threshold
tuning. Transcription costs 0.22 s on this GPU, so speech is simply transcribed
and the name matched as a word by a model that already knows it. The cost is
that every utterance in the room is transcribed, not only the ones addressed
here.

**No speaker verification.** Dropped at the owner's request. The consequence is
that nothing distinguishes your voice from the television's, which matters most
in the seconds after a reply, when Talos is still listening.

### Known rough edges

- The Turkish Piper voice (`tr_TR-dfki-medium`) is the only one published, and
  it is deep and stilted. Replacing the engine costs VRAM and would force a
  smaller model — see MEASUREMENTS.md for the arithmetic.
- A turn that calls a skill takes about ten seconds. The model should say
  something before the tool runs rather than leave silence.
- Ollama's tray app does not start the server on this machine; `ollama serve`
  has to be run by hand.

## Changing it

[HACKING.md](HACKING.md) covers where its reach is configured, how to change
how it behaves, and how to add a skill — which is one file with one function
in it. It also lists the traps this project already fell into, which is the
part worth reading before writing anything.

## Layout

```
talos/
├── skills/        Capabilities. One file per skill, auto-discovered.
├── scripts/       Standalone verification scripts (phase 0)
├── config/        Settings and the assistant's character definition
└── data/          Voice profile, conversation log, embeddings (git-ignored)
```

## Licence

Personal project. No licence granted yet.
