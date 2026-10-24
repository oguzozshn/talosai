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
    """Write something down so it is not lost.

    Use this whenever the owner says to remember something, note something,
    write it down, or keep it in mind. It is the only thing that actually
    records what they say — without it, saying "I'll remember that" is a
    promise nothing keeps.

    Args:
        text: What to record, in the owner's own words

    Examples:
        remember that the deploy key expires in March
        take a note to call the bank
        not al
    """
    text = text.strip()
    if not text:
        return "There was nothing to write down."

    NOTES.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with NOTES.open("a", encoding="utf-8") as handle:
        handle.write(f"- [{stamp}] {text}\n")
    # Says where it went, not just that something happened. The model speaks
    # this result back, and "noted" is what it said before while nothing had
    # been written anywhere.
    return f"Written down in your notes: {text}"


@skill(Permission.OPEN)
def read_notes(limit: int = 10, contains: str = "") -> str:
    """Read back recent notes, optionally filtered.

    Args:
        limit: How many notes to return, most recent first
        contains: Only return notes containing this word

    Examples:
        read my notes
        what did I note about the parser
        notlarımı oku
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


# Told to remember something, the model answered "I'll remember that" and
# called nothing — every time, in three different phrasings of the character
# definition and after the skill's own description was rewritten to say this
# is what it is for. That is the case NEXT.md means by a behaviour code should
# enforce rather than a prompt should ask for, so the request is matched here
# and the skill is run whether the model would have thought of it or not.
_LEADING = (
    "remember that",
    "remember",
    "make a note that",
    "make a note",
    "take a note that",
    "take a note",
    "note that",
    "write down that",
    "write down",
    "jot down",
    "keep in mind that",
    "keep in mind",
    "don't forget that",
    "dont forget that",
    "don't forget",
    "dont forget",
    "do not forget",
    "hatırla",
    "hatirla",
    "not al",
    "unutma",
    "aklında tut",
    "aklinda tut",
)

# Turkish puts the imperative last: "kız arkadaşımın adı Dilara, bunu hatırla".
_TRAILING = (
    "bunu hatırla",
    "bunu hatirla",
    "bunu unutma",
    "hatırla",
    "hatirla",
    "unutma",
    "bunu not al",
    "not al",
    "aklında tut",
    "aklinda tut",
)

# Asking whether something is remembered is a search, not a note. These are
# the phrasings that begin with a trigger word and mean the opposite of it.
_ASKING = (
    "do you remember",
    "did you remember",
    "can you remember",
    "would you remember",
    "hatırlıyor musun",
    "hatirliyor musun",
)

_FILLER = ("please ", "hey ", "ok ", "okay ", "talos, ", "talos ", "um ", "uh ")


_PUNCTUATION = " 	:,-.!?;—"


def _meat(text: str) -> str | None:
    """What is left once the trigger and its punctuation are taken off."""
    return text.strip(_PUNCTUATION).strip() or None


def note_request(text: str) -> str | None:
    """What to write down, if this was a request to write something down.

    None for anything else, including questions about what was remembered —
    "do you remember what I said" is a search, and answering it with a new
    note would be absurd.
    """
    cleaned = text.strip()
    if not cleaned or cleaned.endswith("?"):
        return None

    lowered = cleaned.lower()
    for filler in _FILLER:
        while lowered.startswith(filler):
            cleaned, lowered = cleaned[len(filler):], lowered[len(filler):]

    if any(lowered.startswith(phrase) for phrase in _ASKING):
        return None

    for phrase in _LEADING:
        if lowered.startswith(phrase):
            # "remember" on its own is not a note, it is half a sentence.
            return _meat(cleaned[len(phrase):])

    for phrase in _TRAILING:
        if lowered.endswith(phrase):
            return _meat(cleaned[: len(cleaned) - len(phrase)])

    return None
