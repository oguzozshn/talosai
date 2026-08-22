"""Learning facts about the owner without writing them.

`config/profile.md` is the only thing Talos knows about its owner that is not a
search away: it goes into every prompt, unchanged and unsearched. That is what
makes it valuable and what makes it dangerous to automate. Every character of
it is in the context of every single reply, and a profile that grows on its own
degrades nothing visibly — there is no error, only worse answers.

Worse, a wrong fact here is invisible. Whisper, pinned to English and given
clean Turkish, once returned "My name is Oğuz Talos." — fluent, confident and
half missing. Fed straight into a profile, that sits in every future prompt
where nothing looks broken and nobody opens the file.

So noticing and committing are separate things, and Talos only does the first.
It proposes candidates into this module's tables, which no prompt ever reads.
Promoting one into the profile is a person's decision, made at the terminal
through `run main.py --profile`.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime

from talos.config import LlmConfig, ProfileConfig
from talos.memory import PROFILE, Exchange, Memory

log = logging.getLogger(__name__)

# The heading promoted facts live under, inside the profile body. Keeping them
# in their own section means what a person wrote and what a model proposed stay
# telling apart in the file, which matters the first time one of them is wrong.
LEARNED_HEADING = "## Learned"

_WATERMARK = "profile_scanned_to"


@dataclass
class Candidate:
    id: int
    fact: str
    quote: str
    exchange_id: int
    proposed: datetime


def _normalise(text: str) -> str:
    """Lowercased, punctuation-free, whitespace-collapsed, for comparing."""
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def supported(quote: str, exchange: Exchange) -> bool:
    """Does the exchange actually contain what the model says it does?

    A model asked what is important about someone will invent, confidently and
    in the owner's voice. Requiring a quote and then checking it is the
    cheapest guard there is, and it doubles as the thing that makes a candidate
    judgeable at review time: a fact drawn from a misheard sentence has to be
    visible as one.
    """
    quote = _normalise(quote)
    if len(quote) < 8:
        # Too short to be evidence of anything. "he said yes" appears in half
        # the log.
        return False
    return quote in _normalise(exchange.as_text())


class Candidates:
    """Proposed facts, and the memory of which ones were turned down.

    Its own schema on the memory's connection rather than a file of its own:
    rejections have to be deduplicated against, which is a query, and the
    database is already the one file that can be copied, browsed or deleted.
    Nothing here is ever read into a prompt.
    """

    def __init__(self, memory: Memory) -> None:
        self._memory = memory
        self._prepare(memory.db)

    @staticmethod
    def _prepare(db: sqlite3.Connection) -> None:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS profile_candidates (
                id          INTEGER PRIMARY KEY,
                fact        TEXT NOT NULL,
                normalised  TEXT NOT NULL,
                quote       TEXT NOT NULL,
                exchange_id INTEGER,
                proposed    TEXT NOT NULL,
                status      TEXT NOT NULL
            )
            """
        )
        # The index is what makes a rejection stick: the same fact cannot be
        # proposed a second time, whatever its status was.
        db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS profile_candidate_fact "
            "ON profile_candidates (normalised)"
        )
        db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        db.commit()

    # -- the watermark --------------------------------------------------------

    @property
    def scanned_to(self) -> int:
        row = self._memory.db.execute(
            "SELECT value FROM meta WHERE key = ?", (_WATERMARK,)
        ).fetchone()
        return int(row["value"]) if row else 0

    @scanned_to.setter
    def scanned_to(self, exchange_id: int) -> None:
        self._memory.db.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (_WATERMARK, str(exchange_id)),
        )
        self._memory.db.commit()

    # -- writing --------------------------------------------------------------

    def propose(self, fact: str, quote: str, exchange_id: int) -> bool:
        """Record a candidate. False if it has been seen before, in any state."""
        fact = fact.strip()
        if not fact:
            return False
        try:
            self._memory.db.execute(
                """
                INSERT INTO profile_candidates
                    (fact, normalised, quote, exchange_id, proposed, status)
                VALUES (?, ?, ?, ?, ?, 'pending')
                """,
                (
                    fact,
                    _normalise(fact),
                    quote.strip(),
                    exchange_id,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
        except sqlite3.IntegrityError:
            return False
        self._memory.db.commit()
        return True

    def _settle(self, candidate_id: int, status: str) -> None:
        self._memory.db.execute(
            "UPDATE profile_candidates SET status = ? WHERE id = ?", (status, candidate_id)
        )
        self._memory.db.commit()

    def accept(self, candidate_id: int) -> None:
        self._settle(candidate_id, "accepted")

    def reject(self, candidate_id: int) -> None:
        self._settle(candidate_id, "rejected")

    # -- reading --------------------------------------------------------------

    def pending(self) -> list[Candidate]:
        rows = self._memory.db.execute(
            "SELECT * FROM profile_candidates WHERE status = 'pending' ORDER BY id ASC"
        ).fetchall()
        return [
            Candidate(
                id=int(r["id"]),
                fact=r["fact"],
                quote=r["quote"],
                exchange_id=int(r["exchange_id"] or 0),
                proposed=datetime.fromisoformat(r["proposed"]),
            )
            for r in rows
        ]

    def counts(self) -> dict[str, int]:
        rows = self._memory.db.execute(
            "SELECT status, COUNT(*) AS n FROM profile_candidates GROUP BY status"
        ).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}


# -- the profile file ---------------------------------------------------------


def _split() -> tuple[str, str]:
    """The file above the rule, and the body below it.

    Same partition `load_profile()` uses. The preamble is written for a person
    — a title and an explanation of what the file costs — and nothing here may
    touch it.
    """
    text = PROFILE.read_text(encoding="utf-8") if PROFILE.exists() else ""
    head, separator, body = text.partition("\n---\n")
    if not separator:
        return "", text.strip()
    return head + separator, body.strip()


def learned() -> list[str]:
    """The facts that were promoted, in the order they went in."""
    _, body = _split()
    return _bullets_under(body) if LEARNED_HEADING in body else []


def size() -> int:
    """Characters in the body — what this costs on every single reply."""
    return len(_split()[1])


def _write(body: str) -> None:
    head, _ = _split()
    PROFILE.write_text(f"{head}\n{body.strip()}\n" if head else f"{body.strip()}\n",
                       encoding="utf-8")


def promote(fact: str, max_chars: int) -> bool:
    """Add a fact to the profile. False if it would not fit.

    Refusing rather than truncating: a budget that silently drops the end of
    the profile is worse than one that makes you choose, and choosing is the
    part that keeps the file worth reading.
    """
    fact = fact.strip().rstrip(".")
    if not fact:
        return False

    _, body = _split()
    if LEARNED_HEADING not in body:
        body = f"{body}\n\n{LEARNED_HEADING}\n"
    body = f"{body.rstrip()}\n- {fact}."

    if len(body) > max_chars:
        return False

    _write(body)
    return True


def drop(fact: str) -> bool:
    """Remove a promoted fact, so another one can take its place."""
    _, body = _split()
    wanted = _normalise(fact)
    kept, removed = [], False
    for line in body.splitlines():
        if not removed and line.strip().startswith("-") and _normalise(line.lstrip("- ")) == wanted:
            removed = True
            continue
        kept.append(line)
    if not removed:
        return False

    body = "\n".join(kept)
    # An empty heading is not free. It is a dozen characters in the prompt of
    # every reply, for a section with nothing in it, and the profile would
    # never lose them again.
    if LEARNED_HEADING in body and not _bullets_under(body):
        body = body.partition(LEARNED_HEADING)[0]

    _write(body)
    return True


def _bullets_under(body: str) -> list[str]:
    _, _, section = body.partition(LEARNED_HEADING)
    return [
        line.lstrip("- ").strip()
        for line in section.splitlines()
        if line.strip().startswith("-")
    ]


# -- extraction ---------------------------------------------------------------


_INSTRUCTION = (
    "You are reading a transcript of conversations between an assistant and its "
    "owner, looking for durable facts about the owner that would be worth "
    "knowing in every future conversation.\n\n"
    "Worth keeping: preferences, tools and languages they work in, projects, "
    "working habits, constraints, how they like to be answered.\n"
    "Not worth keeping: anything that happened once, anything about the "
    "assistant itself, and questions they asked.\n\n"
    "Every fact must be supported by an exact quote copied verbatim from the "
    "TRANSCRIPT. Not from these instructions. Do not paraphrase the quote. If "
    "you cannot quote it from the transcript, do not report it. Reporting "
    "nothing is a correct answer and a common one.\n\n"
    'Reply as JSON: {"facts": [{"fact": "...", "quote": "...", "exchange": 12}]}'
)


def _parse(content: str) -> list:
    """The facts in a reply, tolerating a reply that was cut off.

    num_predict bounds the answer, and a bounded answer sometimes stops in the
    middle of an object. Losing the whole batch because the last fact was
    truncated wastes the model call that found the first three.
    """
    try:
        return json.loads(content).get("facts") or []
    except (json.JSONDecodeError, AttributeError):
        pass

    salvaged = []
    for match in re.finditer(r"\{[^{}]*\}", content):
        try:
            salvaged.append(json.loads(match.group()))
        except json.JSONDecodeError:
            continue
    if not salvaged:
        log.warning("extraction returned nothing usable: %s", content[:200])
    return salvaged


def extract(
    memory: Memory,
    candidates: Candidates,
    llm: LlmConfig,
    settings: ProfileConfig,
    report=lambda message: None,
) -> int:
    """Walk the log forward, proposing facts. Returns how many were kept.

    Batched and on demand rather than once per turn: a pass over many
    conversations can notice that something came up repeatedly, which a single
    turn cannot, and it costs no latency in a conversation that is happening.
    """
    if not settings.learn:
        report("Learning is off ([profile] learn in settings.toml).")
        return 0

    import ollama

    from talos.memory import load_profile

    # In the system message, not beside the transcript. Beside it, the model
    # reads the profile as material and reports it straight back: on the first
    # run every batch returned facts quoted from the profile rather than from
    # the conversation, which the quote check caught but only after they had
    # eaten the whole answer.
    known = load_profile()
    instruction = _INSTRUCTION
    if known:
        instruction += (
            "\n\nThe following is ALREADY KNOWN. It is not part of the transcript, "
            "it is not evidence, and nothing in it may be reported back:\n" + known
        )

    kept = skipped = invented = 0

    while True:
        batch = memory.after(candidates.scanned_to, settings.batch)
        if not batch:
            break

        transcript = "\n\n".join(
            f"[exchange {i}]\nOwner: {e.asked}\nAssistant: {e.answered}" for i, e in batch
        )
        report(f"  exchanges {batch[0][0]}-{batch[-1][0]}...")

        try:
            response = ollama.chat(
                model=llm.model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": f"Transcript:\n{transcript}"},
                ],
                format="json",
                think=False,
                keep_alive=llm.keep_alive,
                options={"num_ctx": llm.context, "temperature": 0.1, "num_predict": 700},
            )
        except Exception as exc:  # noqa: BLE001 — a failed pass is not a failed database
            log.warning("could not extract from exchanges %s: %s", batch[0][0], exc)
            break

        content = (response.get("message", {}).get("content") or "").strip()
        for item in _parse(content):
            outcome = _consider(item, dict(batch), known, candidates)
            if outcome == "kept":
                kept += 1
            elif outcome == "known":
                skipped += 1
            else:
                invented += 1
                report(f"    unsupported: {item.get('fact', '')[:70]}")

        candidates.scanned_to = batch[-1][0]

    if skipped:
        report(f"  {skipped} already in the profile.")
    if invented:
        report(f"  {invented} could not be quoted from the transcript.")
    return kept


def _consider(item, sources: dict[int, Exchange], known: str, candidates: Candidates) -> str:
    """One reported fact: kept, already known, or unquotable."""
    if not isinstance(item, dict):
        return "invented"
    fact = str(item.get("fact") or "").strip()
    quote = str(item.get("quote") or "").strip()
    if not fact:
        return "invented"

    # Cheap containment rather than similarity: the model tends to return the
    # profile's own wording, which this catches, and a near-duplicate that gets
    # through is a thing a person can reject in one keystroke.
    if known and _normalise(fact) in _normalise(known):
        return "known"

    try:
        exchange_id = int(item.get("exchange") or 0)
    except (TypeError, ValueError):
        exchange_id = 0

    source = sources.get(exchange_id)
    if source is None or not supported(quote, source):
        # A made-up id is itself a reason to doubt the fact, but the quote may
        # still be real — so look for the exchange it actually came from.
        source = next((e for e in sources.values() if supported(quote, e)), None)
        exchange_id = next((i for i, e in sources.items() if e is source), 0)

    if source is None:
        return "invented"
    return "kept" if candidates.propose(fact, quote, exchange_id) else "known"
