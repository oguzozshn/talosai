"""Runtime state that outlives a single turn.

Skills are module-level functions with no handle on the assistant that called
them, which is deliberate — it keeps a skill a function rather than a class
wired into everything. But a few skills need to change how the *next* turn
behaves, not just answer this one. Practice mode is the first.

So: one small shared object. Everything that matters here is a mode the user
turned on out loud and can turn off the same way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


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

    # Called with the full answer once a reply is finished. A skill that
    # needs what was said — the glossary keeps Talos's explanation of a term —
    # registers here, and is called once.
    after_reply: list = field(default_factory=list)

    def reply_finished(self, answer: str) -> None:
        waiting, self.after_reply = self.after_reply, []
        for callback in waiting:
            try:
                callback(answer)
            except Exception:  # noqa: BLE001 — bookkeeping must not break a reply
                log.exception("after-reply callback failed")

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
