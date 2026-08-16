"""Keeping and recalling notes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from talos.skills.base import Permission, skill

ROOT = Path(__file__).resolve().parent.parent.parent
NOTES = ROOT / "data" / "notes.md"


# OPEN rather than CONFIRM, and the distinction is worth being precise about:
# the tiers are not "reads versus writes", they are about what the user cannot
# undo. This appends a timestamped line to one file that belongs to the
# assistant. It cannot overwrite, cannot delete, and cannot touch anything
# else. Asking "are you sure?" after every spoken note would make the feature
# not worth having.
@skill(Permission.OPEN)
def take_note(text: str) -> str:
    """Write down a note for later.

    Args:
        text: What to remember
    """
    text = text.strip()
    if not text:
        return "There was nothing to write down."

    NOTES.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with NOTES.open("a", encoding="utf-8") as handle:
        handle.write(f"- [{stamp}] {text}\n")
    return f"Noted: {text}"


@skill(Permission.OPEN)
def read_notes(limit: int = 10, contains: str = "") -> str:
    """Read back recent notes, optionally filtered.

    Args:
        limit: How many notes to return, most recent first
        contains: Only return notes containing this word
    """
    if not NOTES.exists():
        return "There are no notes yet."

    lines = [ln.strip() for ln in NOTES.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if contains:
        needle = contains.casefold()
        lines = [ln for ln in lines if needle in ln.casefold()]

    if not lines:
        return f"No notes mentioning '{contains}'." if contains else "There are no notes yet."

    recent = lines[-limit:][::-1]
    return f"{len(lines)} note(s). Most recent:\n" + "\n".join(recent)
