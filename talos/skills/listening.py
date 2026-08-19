"""Changing how Talos is spoken to, without restarting it.

Two ways in: the wake word, where the microphone is always open and every
utterance in the room is transcribed to see whether it was addressed here, and
the hotkey, where nothing is recorded until a key is held.

The second is not a worse version of the first. A meeting starts, someone else
is in the room, you are reading something aloud — those are all reasons to
want the microphone shut without shutting the assistant. Being able to say so
out loud matters more than it sounds, because the moment you want it is the
moment you do not want to go and find a terminal window.
"""

from __future__ import annotations

from talos.session import current
from talos.skills.base import Permission, skill


@skill(Permission.OPEN)
def stop_listening() -> str:
    """Stop listening continuously, and only record while the hotkey is held."""
    if current.mode == "hotkey":
        return "I'm already only listening when you hold the key."
    current.switch_to("hotkey")
    return (
        "Right, I'll stop listening. Hold F9 when you want me, and say "
        "start listening to go back."
    )


@skill(Permission.OPEN)
def start_listening() -> str:
    """Listen continuously again, waking on the name instead of a key."""
    if current.mode == "wake":
        return "I'm already listening."
    current.switch_to("wake")
    return "Listening again. Just say my name."


@skill(Permission.OPEN)
def listening_mode() -> str:
    """Report whether Talos is listening continuously or waiting for the hotkey."""
    if current.mode == "wake":
        return (
            "I'm listening continuously and waking on my name. Everything said "
            "in the room is transcribed to check whether it was meant for me."
        )
    return "I'm only recording while you hold the key. Nothing else is heard."
