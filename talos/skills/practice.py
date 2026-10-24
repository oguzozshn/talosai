"""Turning English practice on and off."""

from __future__ import annotations

from talos.session import current
from talos.skills.base import Permission, skill


@skill(Permission.OPEN)
def english_practice(on: bool = True) -> str:
    """Start or stop practising spoken English, with a correction after each reply.

    Args:
        on: true to start practising, false to stop
    """
    if on:
        if current.coaching:
            return "We're already practising. Just keep talking."
        current.coaching = True
        return (
            "Right, let's practise. Talk to me normally and I'll correct the thing "
            "most worth correcting each time. Say stop practising when you've had enough."
        )
    if not current.coaching:
        return "We weren't practising."
    current.coaching = False
    return "Back to normal. No more corrections."
