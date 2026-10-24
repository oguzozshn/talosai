"""Giving the graphics card back.

Talos holds most of an 8 GB card while it is loaded: the language model is 5.6
GB of it, Whisper another 1.5, the embedder 0.3. That is fine until you want
to play something or render something, at which point there is nothing left.

Only the language model can be released. Whisper is the wake word — it is what
hears its own name — so unloading it would leave an assistant that cannot be
called. The embedder is small enough not to matter.

Releasing costs about a minute when you next speak, while the model loads from
disk. That is the whole trade: a minute later, against 5.6 GB now.
"""

from __future__ import annotations

import logging

from talos.config import load
from talos.skills.base import Permission, skill

log = logging.getLogger(__name__)

_config = load()


def _vram() -> tuple[int, int] | None:
    """Used and total VRAM in MiB, or None if it cannot be read."""
    import shutil
    import subprocess

    binary = shutil.which("nvidia-smi")
    if not binary:
        return None
    try:
        proc = subprocess.run(
            [binary, "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        used, total = (int(v.strip()) for v in proc.stdout.splitlines()[0].split(","))
    except (subprocess.SubprocessError, ValueError, IndexError):
        return None
    return used, total


# OPEN rather than CONFIRM: nothing is destroyed and nothing leaves the
# machine. The only cost is that the next question is slow, and that is the
# thing being asked for.
@skill(Permission.OPEN)
def release_gpu() -> str:
    """Unload the language model to free graphics memory for something else."""
    import ollama

    try:
        # keep_alive of zero tells Ollama to drop the model as soon as this
        # request finishes. num_predict of zero avoids generating anything on
        # the way out.
        ollama.chat(
            model=_config.llm.model,
            messages=[{"role": "user", "content": "bye"}],
            keep_alive=0,
            options={"num_predict": 1},
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("could not release the model: %s", exc)
        return "I couldn't release the model."

    reading = _vram()
    freed = f" {reading[0]} megabytes still in use." if reading else ""
    return (
        "Released the language model. Your next question will take about a "
        "minute while it loads again." + freed
    )
