"""Remembering conversations, and being able to find them again.

Two different things share this file because they are easy to confuse and
should not be:

**The profile** is a static file of facts about the owner — name, work, how
they like to be answered. It goes into every prompt, unchanged, and is never
searched. Facts you always need are not a retrieval problem; putting them
behind a search means they sometimes arrive and sometimes do not, and an
assistant that intermittently forgets your name is worse than one that never
knew it.

**The log** is every exchange, stored with an embedding. It is searched only
when something is actually being looked for. This is the part that is properly
RAG, and it is deliberately a skill the model calls rather than a retrieval
that runs on every turn: most questions do not need last Tuesday, and paying
an embedding and a search for all of them buys nothing.

Storage is SQLite with sqlite-vec, so the whole memory is one file that can be
copied, inspected with any SQLite browser, or deleted.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATABASE = ROOT / "data" / "memory.db"
PROFILE = ROOT / "config" / "profile.md"

# nomic-embed-text. Small, fast, runs in Ollama alongside everything else, and
# the dimension has to match what the table was created with — changing the
# model means rebuilding the index.
EMBED_MODEL = "nomic-embed-text"
EMBED_DIMENSIONS = 768


@dataclass
class Exchange:
    when: datetime
    asked: str
    answered: str

    def as_text(self) -> str:
        """One string for embedding. Both halves, because a conversation is
        found as often by what was answered as by what was asked."""
        return f"{self.asked}\n{self.answered}"

    def summary(self) -> str:
        return f"[{self.when:%d %B, %H:%M}] You asked: {self.asked} — I said: {self.answered}"


def _pack(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


class Memory:
    def __init__(self, path: Path = DATABASE) -> None:
        self._path = path
        self._db: sqlite3.Connection | None = None
        self._vec_available = False

    # -- storage -------------------------------------------------------------

    @property
    def db(self) -> sqlite3.Connection:
        if self._db is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self._path, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._prepare(self._db)
        return self._db

    def _prepare(self, db: sqlite3.Connection) -> None:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS exchanges (
                id       INTEGER PRIMARY KEY,
                happened TEXT NOT NULL,
                asked    TEXT NOT NULL,
                answered TEXT NOT NULL
            )
            """
        )
        db.commit()

        try:
            import sqlite_vec

            db.enable_load_extension(True)
            sqlite_vec.load(db)
            db.enable_load_extension(False)
            db.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS exchange_vectors
                USING vec0(exchange_id INTEGER PRIMARY KEY,
                           embedding FLOAT[{EMBED_DIMENSIONS}])
                """
            )
            db.commit()
            self._vec_available = True
        except Exception as exc:  # noqa: BLE001
            # Without the extension the log still records everything and can
            # still be read back by date. Only similarity search is lost, and
            # that degradation is worth more than refusing to start.
            log.warning("sqlite-vec unavailable, semantic recall disabled: %s", exc)

    # -- writing -------------------------------------------------------------

    def record(self, asked: str, answered: str, embedding: list[float] | None = None) -> int:
        cursor = self.db.execute(
            "INSERT INTO exchanges (happened, asked, answered) VALUES (?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), asked.strip(), answered.strip()),
        )
        exchange_id = int(cursor.lastrowid)

        if embedding and self._vec_available:
            try:
                self.db.execute(
                    "INSERT INTO exchange_vectors (exchange_id, embedding) VALUES (?, ?)",
                    (exchange_id, _pack(embedding)),
                )
            except sqlite3.Error as exc:
                log.warning("Could not store embedding: %s", exc)

        self.db.commit()
        return exchange_id

    # -- reading -------------------------------------------------------------

    @staticmethod
    def _row_to_exchange(row: sqlite3.Row) -> Exchange:
        return Exchange(
            when=datetime.fromisoformat(row["happened"]),
            asked=row["asked"],
            answered=row["answered"],
        )

    def recent(self, limit: int = 5) -> list[Exchange]:
        rows = self.db.execute(
            "SELECT * FROM exchanges ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._row_to_exchange(r) for r in rows]

    def search_text(self, needle: str, limit: int = 5) -> list[Exchange]:
        """Plain substring search. Always available, and for a specific word
        it beats similarity — someone asking about "BACnet" wants the exchange
        containing that word, not the one that is thematically near it."""
        pattern = f"%{needle.strip()}%"
        rows = self.db.execute(
            """
            SELECT * FROM exchanges
            WHERE asked LIKE ? OR answered LIKE ?
            ORDER BY id DESC LIMIT ?
            """,
            (pattern, pattern, limit),
        ).fetchall()
        return [self._row_to_exchange(r) for r in rows]

    def search_similar(self, embedding: list[float], limit: int = 5) -> list[Exchange]:
        if not self._vec_available:
            return []
        try:
            rows = self.db.execute(
                """
                SELECT e.* FROM exchange_vectors v
                JOIN exchanges e ON e.id = v.exchange_id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (_pack(embedding), limit),
            ).fetchall()
        except sqlite3.Error as exc:
            log.warning("Similarity search failed: %s", exc)
            return []
        return [self._row_to_exchange(r) for r in rows]

    def count(self) -> int:
        return int(self.db.execute("SELECT COUNT(*) FROM exchanges").fetchone()[0])

    def forget_all(self) -> int:
        removed = self.count()
        self.db.execute("DELETE FROM exchanges")
        if self._vec_available:
            self.db.execute("DELETE FROM exchange_vectors")
        self.db.commit()
        return removed


def embed(text: str) -> list[float] | None:
    """Embed via Ollama. None on any failure — memory degrades, it does not break."""
    try:
        import ollama

        response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        vector = response.get("embedding") or []
        return list(vector) or None
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not embed (is %s pulled?): %s", EMBED_MODEL, exc)
        return None


def load_profile() -> str:
    """The static facts about the owner, for the system prompt.

    Read fresh each time rather than cached: this file gets edited while the
    assistant is running, and a stale profile is a confusing thing to debug.
    """
    if not PROFILE.exists():
        return ""
    text = PROFILE.read_text(encoding="utf-8")
    _, separator, body = text.partition("\n---\n")
    return (body if separator else text).strip()
