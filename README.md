# Talos

A fully local voice assistant for Windows. No cloud APIs, no subscriptions, no data leaving the machine.

Named after the bronze automaton of Greek myth that patrolled the shores of Crete — arguably the first artificial guardian anyone imagined. It waits quietly, answers when called, and gets out of the way.

## Design constraints

These are hard requirements, not preferences:

1. **Everything runs locally.** Every component has a free, offline implementation. No API keys.
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

```powershell
python scripts/check_env.py
```

See [`scripts/README.md`](scripts/README.md) for what each verification script proves.

## Project status

**Phase 0 — environment and verification.** Nothing is wired together yet; each
component is being proven in isolation first. See [ROADMAP.md](ROADMAP.md) for the
full plan and what each phase delivers.

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
