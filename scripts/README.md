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

```powershell
& "$env:USERPROFILE\.venvs\talos\Scripts\Activate.ps1"

python scripts/check_env.py
python scripts/test_mic.py          # writes data/sample.wav
python scripts/test_stt.py          # reads it back
python scripts/test_tts.py
python scripts/test_llm.py
```

`test_stt.py` downloads roughly 1.5 GB of model on its first run. `test_llm.py`
needs a model pulled first:

```powershell
ollama pull gemma3:12b
ollama pull qwen3:8b
```

Both, deliberately. Gemma handles Turkish better; Qwen is more reliable at tool
calling. Which one wins is not decidable from documentation — phase 3 picks by
measuring, and having both already local makes that a five-minute comparison
instead of an afternoon.

## Installing Piper

Piper has no usable Windows wheel, so it is a binary and a voice file rather
than a pip install:

1. `piper_windows_amd64.zip` from
   [the releases page](https://github.com/rhasspy/piper/releases) — unzip so
   that `vendor/piper/piper.exe` exists.
2. The Turkish voice, both `.onnx` and `.onnx.json`, from
   [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices/tree/main/tr/tr_TR/fettah/medium)
   — put them in `models/piper/`.

Both directories are git-ignored. `test_tts.py` works without either, using the
voice built into Windows, so this can wait until the rest of the chain runs.

## Numbers worth writing down

Record what these scripts report on the first successful run. Phase 1 assembles
them into a chain, and when the assembled latency is worse than the sum of its
parts, these are what you compare against.

- `test_stt.py` — real-time factor, and whether it said `cuda` or `cpu`
- `test_llm.py` — time to first token
- `test_tts.py` — synthesis time
