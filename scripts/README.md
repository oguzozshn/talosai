# Verification scripts (phase 0)

Each script proves one component works **on its own**. Nothing here imports
anything from `talos/` — that is the point. If all four pass and the assembled
system still misbehaves, the fault is in the wiring, not the parts.

Run them in this order:

| Script | What it proves | If it fails |
|---|---|---|
| `check_env.py` | Versions, GPU visibility, packages, devices | Read the notes it prints; it explains each failure |
| `test_mic.py` | Audio in and out, at 16 kHz | Wrong input device selected, or the mic is muted |
| `test_stt.py` | Speech to text, and whether it uses the GPU | Usually cuDNN missing from PATH — CPU still works |
| `test_tts.py` | Speech out | Falls back to the Windows voice, so a total failure means the audio path is broken |
| `test_llm.py` | The model answers, and how fast | Ollama service not running, or no model pulled |

## Typical first run

Use `run.cmd` in the project root — it finds the venv interpreter itself, so
there is no activation step to get wrong and it behaves identically in cmd and
PowerShell:

```
run scripts\check_env.py
run scripts\test_mic.py          REM writes data/sample.wav
run scripts\test_stt.py          REM reads it back
run scripts\test_tts.py
run scripts\test_llm.py
```

`test_stt.py` downloads roughly 1.5 GB of model on its first run. `test_llm.py`
needs a model pulled first:

```powershell
ollama pull qwen3:8b
ollama pull gemma3:4b
```

Both, deliberately. Qwen3 8B is the primary — it is the more reliable tool
caller, and at roughly 5.2 GB it leaves room for Whisper on the same card.
Gemma 3 4B is the comparison: noticeably better Turkish for its size, and small
enough that it stays resident no matter what else is loaded.

Which one wins is not decidable from documentation. Phase 3 picks by measuring;
having both already local makes that a five-minute comparison instead of an
afternoon.

Note what is *not* in that list. Gemma 3 12B is the obvious reach, but at Q4 it
is 8 GB by itself — Ollama would spill it into system RAM rather than refuse,
and generation slows by roughly an order of magnitude. If you want to try it,
move Whisper to the CPU first (`test_stt.py --cpu`) and compare the two configs
end to end rather than guessing.

## Installing Piper

Piper has no usable Windows wheel — it is a binary plus two voice files, from
two different sites. `fetch_piper.py` collects them:

```powershell
python scripts/fetch_piper.py
```

It downloads roughly 80 MB into `vendor/piper/` and `models/piper/`, both
git-ignored, and is safe to re-run — existing files are left alone unless you
pass `--force`.

If it cannot reach GitHub or Hugging Face, do it by hand:

1. `piper_windows_amd64.zip` from
   [the releases page](https://github.com/rhasspy/piper/releases) — unzip so
   that `vendor/piper/piper.exe` exists.
2. The Turkish voice, both `.onnx` and `.onnx.json`, from
   [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main/tr/tr_TR/fettah/medium)
   — put them in `models/piper/`.

Either way this can wait. `test_tts.py` falls back to the voice built into
Windows, which mangles Turkish but proves the audio output path works.

## Numbers worth writing down

Record what these scripts report on the first successful run. Phase 1 assembles
them into a chain, and when the assembled latency is worse than the sum of its
parts, these are what you compare against.

- `test_stt.py` — real-time factor, and whether it said `cuda` or `cpu`
- `test_llm.py` — time to first token
- `test_tts.py` — synthesis time
