"""Runtime state that outlives a single turn.

Skills are module-level functions with no handle on the assistant that called
them, which is deliberate — it keeps a skill a function rather than a class
wired into everything. But a few skills need to change how the *next* turn
behaves, not just answer this one. Practice mode is the first.

So: one small shared object. Everything that matters here is a mode the user
turned on out loud and can turn off the same way.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Session:
    # English practice: the assistant corrects rather than only answering, and
    # transcription is asked for per-word confidences it otherwise skips.
    coaching: bool = False

    # How Talos is being spoken to: "wake" listens continuously, "hotkey"
    # only records while a key is held. Changing this asks the running mode
    # to stop so the other can start — the loops check it between turns.
    mode: str = "hotkey"

    # Set when the mode should change. The loops own their own exits, so
    # they are asked rather than interrupted.
    switch_requested: bool = False

    def switch_to(self, mode: str) -> None:
        self.mode = mode
        self.switch_requested = True

    def reset(self) -> None:
        self.coaching = False
        self.switch_requested = False


# Deliberately a module-level singleton rather than something passed around.
# There is one conversation happening, and pretending otherwise would mean
# threading a parameter through every layer for the sake of a purity nobody
# benefits from here.
current = Session()
