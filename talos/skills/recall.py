"""Looking things up in past conversations.

A skill rather than automatic retrieval. Running a search before every reply
would pay an embedding and a query on questions that have nothing to do with
the past, which is most of them — and worse, it would push whatever the search
happened to return into the context of a question it has nothing to do with.

Searching when someone is actually looking for something is both cheaper and
more accurate.
"""

from __future__ import annotations

from talos.memory import Memory, embed
from talos.skills.base import Permission, skill

# One connection for the process. SQLite handles this fine and it avoids
# reopening the file on every call.
_memory = Memory()


def _render(exchanges: list, header: str) -> str:
    if not exchanges:
        return "Nothing in our conversations matches that."
    lines = [header]
    lines += [f"  {e.summary()}" for e in exchanges]
    return "\n".join(lines)


@skill(Permission.OPEN)
def recall(about: str, limit: int = 3) -> str:
    """Search past conversations for something that was discussed before.

    Args:
        about: What to look for, in a few words
        limit: How many past exchanges to return

    Examples:
        what did I say about BACnet
        do you remember what I told you about the deadline
        geçen sefer ne konuşmuştuk
    """
    about = about.strip()
    if not about:
        return "I need something to look for."

    # Substring first. When someone names a specific thing — a project, an
    # error, a word — the exchange containing that word is what they want, and
    # similarity search will happily return something merely related instead.
    exact = _memory.search_text(about, limit)
    if exact:
        return _render(exact, f"Found {len(exact)} exchange(s) mentioning '{about}':")

    vector = embed(about)
    if vector is None:
        return f"Nothing mentioning '{about}', and semantic search is unavailable."

    similar = _memory.search_similar(vector, limit)
    return _render(similar, f"Nothing mentioned '{about}' exactly. The closest were:")


@skill(Permission.OPEN)
def last_conversations(limit: int = 3) -> str:
    """Report what was most recently discussed.

    Args:
        limit: How many recent exchanges to return

    Examples:
        what were we talking about earlier
        what did we discuss last time
    """
    exchanges = _memory.recent(limit)
    if not exchanges:
        return "We haven't spoken before, or the memory was cleared."
    return _render(exchanges, f"The last {len(exchanges)} exchange(s):")
