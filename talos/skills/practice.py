"""Turning English practice on and off."""

from __future__ import annotations

from talos.session import current
from talos.skills.base import Permission, skill


@skill(Permission.OPEN)
def start_english_practice() -> str:
    """Begin practising spoken English, with corrections after each reply."""
    if current.coaching:
        return "We're already practising. Just keep talking."
    current.coaching = True
    return (
        "Right, let's practise. Talk to me normally and I'll correct the thing "
        "most worth correcting each time. Say stop practising when you've had enough."
    )


@skill(Permission.OPEN)
def stop_english_practice() -> str:
    """Stop practising English and go back to answering normally."""
    if not current.coaching:
        return "We weren't practising."
    current.coaching = False
    return "Back to normal. No more corrections."
