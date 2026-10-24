"""Course material, searchable by meaning, and the terms asked about while studying.

The folder of courses (`[study] folder`, the desktop's Dersler) holds one
folder per course, and in each the slides, notes, books and syllabus. Every
file is cut into passages — a slide, a page, a few paragraphs — each embedded
once and stored in `data/study.db`. A question is embedded the same way and
answered from the closest passages, with the file and page they came from.

Indexing is incremental: a file is read again only when its size or time
changes. The first run is the slow one, mostly the textbook — reading its 673
pages took 42 seconds before any embedding — so it runs in the background
after start-up rather than in front of the first question.

The glossary is the other half. Asking what a term means while studying is
the moment it is worth keeping, so the term, the course and what Talos
answered are stored, to be listed or quizzed on before the exam.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
import zipfile
from dataclasses import dataclass
from datetime import datetime
from html import unescape
from pathlib import Path

import numpy as np

from talos.memory import EMBED_MODEL

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATABASE = ROOT / "data" / "study.db"

READABLE = {".pdf", ".pptx", ".docx", ".txt", ".md"}
# Characters per passage. A slide is usually well under this; a textbook page
# is about two, which keeps a passage to one idea and short enough to quote.
PASSAGE = 1000
OVERLAP = 150
# Passages sent to the embedding model per request.
BATCH = 64

DOCUMENT = "search_document: "
QUERY = "search_query: "


@dataclass
class Passage:
    course: str
    file: str
    where: str
    text: str
    score: float = 0.0

    def cite(self) -> str:
        return f"{self.course} — {self.file}, {self.where}"


# --- Reading files --------------------------------------------------------


def _pdf(path: Path) -> list[tuple[str, str]]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return [(f"page {n}", page.extract_text() or "") for n, page in enumerate(reader.pages, 1)]


def _xml_text(xml: str, tag: str) -> str:
    return " ".join(unescape(t) for t in re.findall(rf"<{tag}(?: [^>]*)?>([^<]*)</{tag}>", xml))


def _number(name: str) -> int:
    return int(re.search(r"\d+", name).group())


def _pptx(path: Path) -> list[tuple[str, str]]:
    # A .pptx is a zip of XML; the text of each slide is in its <a:t> runs.
    # Read directly rather than through python-pptx, which would be a
    # dependency for one regular expression.
    with zipfile.ZipFile(path) as archive:
        slides = [n for n in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        slides.sort(key=_number)
        return [
            (f"slide {_number(n)}", _xml_text(archive.read(n).decode("utf-8"), "a:t"))
            for n in slides
        ]


def _docx(path: Path) -> list[tuple[str, str]]:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    paragraphs = [_xml_text(p, "w:t") for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)]
    return [("text", "\n".join(p for p in paragraphs if p.strip()))]


def _plain(path: Path) -> list[tuple[str, str]]:
    return [("text", path.read_text(encoding="utf-8", errors="replace"))]


READERS = {".pdf": _pdf, ".pptx": _pptx, ".docx": _docx, ".txt": _plain, ".md": _plain}


def passages_of(path: Path) -> list[tuple[str, str]]:
    """(where, text) for every passage in a file."""
    out: list[tuple[str, str]] = []
    for where, text in READERS[path.suffix.lower()](path):
        text = " ".join(text.split())
        if len(text) < 40:
            continue  # A title slide or a blank page says nothing worth finding.
        start = 0
        while start < len(text):
            out.append((where, text[start:start + PASSAGE]))
            if start + PASSAGE >= len(text):
                break
            start += PASSAGE - OVERLAP
    return out


# --- The index ------------------------------------------------------------


def _embed(texts: list[str]) -> list[list[float]] | None:
    try:
        import ollama

        return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]
    except Exception as exc:  # noqa: BLE001
        log.warning("could not embed course material: %s", exc)
        return None


class Library:
    def __init__(self, folder: Path, database: Path = DATABASE, embed=_embed) -> None:
        self.folder = folder
        self._embed = embed
        database.parent.mkdir(parents=True, exist_ok=True)
        # Indexing runs on its own thread while turns search on another.
        self._db = sqlite3.connect(database, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY, size INTEGER, modified REAL);
            CREATE TABLE IF NOT EXISTS passages (
                id INTEGER PRIMARY KEY, path TEXT, course TEXT, file TEXT,
                place TEXT, text TEXT, vector BLOB);
            CREATE TABLE IF NOT EXISTS glossary (
                term TEXT, course TEXT, definition TEXT, asked TEXT, times INTEGER,
                PRIMARY KEY (term, course));
            """
        )
        self._matrix: np.ndarray | None = None
        self._rows: list[tuple] = []
        self.indexing = False

    # -- building --

    def course_of(self, path: Path) -> str:
        """The course a file belongs to: the folder under the courses folder."""
        relative = path.relative_to(self.folder)
        return relative.parts[0] if len(relative.parts) > 1 else "General"

    def update(self, progress=None) -> tuple[int, int]:
        """Index new and changed files, forget deleted ones. Returns (files read, passages)."""
        self.indexing = True
        try:
            return self._update(progress)
        finally:
            self.indexing = False
            self._matrix = None

    def _update(self, progress) -> tuple[int, int]:
        present = {
            str(p): p for p in self.folder.rglob("*")
            if p.is_file() and p.suffix.lower() in READABLE and not p.name.startswith("~$")
        }
        with self._lock:
            known = {row[0]: row[1:] for row in self._db.execute("SELECT path, size, modified FROM files")}
            for gone in set(known) - set(present):
                self._db.execute("DELETE FROM passages WHERE path = ?", (gone,))
                self._db.execute("DELETE FROM files WHERE path = ?", (gone,))
            self._db.commit()

        read = added = 0
        for key, path in sorted(present.items()):
            stat = path.stat()
            if known.get(key) == (stat.st_size, stat.st_mtime):
                continue
            try:
                pieces = passages_of(path)
            except Exception as exc:  # noqa: BLE001 — one unreadable file is not the end
                log.warning("could not read %s: %s", path.name, exc)
                continue
            vectors: list[list[float]] = []
            for start in range(0, len(pieces), BATCH):
                batch = self._embed([DOCUMENT + text for _, text in pieces[start:start + BATCH]])
                if batch is None:
                    return read, added  # Embedding is down; try again next time.
                vectors += batch
            course = self.course_of(path)
            with self._lock:
                self._db.execute("DELETE FROM passages WHERE path = ?", (key,))
                self._db.executemany(
                    "INSERT INTO passages (path, course, file, place, text, vector) VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        (key, course, path.name, where, text, np.asarray(v, dtype=np.float32).tobytes())
                        for (where, text), v in zip(pieces, vectors)
                    ],
                )
                self._db.execute(
                    "INSERT OR REPLACE INTO files (path, size, modified) VALUES (?, ?, ?)",
                    (key, stat.st_size, stat.st_mtime),
                )
                self._db.commit()
            read += 1
            added += len(pieces)
            if progress:
                progress(path.name, len(pieces))
        return read, added

    # -- searching --

    def courses(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._db.execute("SELECT DISTINCT course FROM passages ORDER BY course")]

    def _load(self) -> None:
        with self._lock:
            rows = list(self._db.execute("SELECT course, file, place, text, vector FROM passages"))
        self._rows = [r[:4] for r in rows]
        if rows:
            matrix = np.stack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
            self._matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        else:
            self._matrix = np.zeros((0, 1), dtype=np.float32)

    def match_course(self, name: str) -> str | None:
        """A course from a loose name: "requirements" finds SOFTWARE REQUIREMENTS ENGINEERING."""
        wanted = name.strip().casefold()
        if not wanted:
            return None
        for course in self.courses():
            if wanted in course.casefold() or course.casefold() in wanted:
                return course
        words = set(wanted.split())
        best = max(self.courses(), key=lambda c: len(words & set(c.casefold().split())), default=None)
        if best and words & set(best.casefold().split()):
            return best
        return None

    def search(self, question: str, course: str | None = None, limit: int = 4) -> list[Passage] | None:
        """The passages closest to a question, or None if embedding is unavailable."""
        if self._matrix is None:
            self._load()
        if not self._rows:
            return []
        query = self._embed([QUERY + question])
        if query is None:
            return None
        vector = np.asarray(query[0], dtype=np.float32)
        scores = self._matrix @ (vector / np.linalg.norm(vector))
        order = np.argsort(-scores)
        found: list[Passage] = []
        for i in order:
            row = self._rows[i]
            if course and row[0] != course:
                continue
            found.append(Passage(*row, score=float(scores[i])))
            if len(found) >= limit:
                break
        return found

    # -- the glossary --

    def remember_term(self, term: str, course: str, definition: str) -> None:
        with self._lock:
            self._db.execute(
                """
                INSERT INTO glossary (term, course, definition, asked, times) VALUES (?, ?, ?, ?, 1)
                ON CONFLICT (term, course) DO UPDATE SET
                    definition = excluded.definition, asked = excluded.asked, times = times + 1
                """,
                (term.strip(), course, definition.strip(), datetime.now().isoformat(timespec="seconds")),
            )
            self._db.commit()

    def terms(self, course: str | None = None) -> list[tuple[str, str, str, int]]:
        """(term, course, definition, times asked), most asked first."""
        query = "SELECT term, course, definition, times FROM glossary"
        params: tuple = ()
        if course:
            query += " WHERE course = ?"
            params = (course,)
        with self._lock:
            return list(self._db.execute(query + " ORDER BY times DESC, asked DESC", params))


_library: Library | None = None


def library() -> Library | None:
    return _library


def open_library(folder: str) -> Library | None:
    """Open the course index for the configured folder, if there is one."""
    global _library
    path = Path(folder).expanduser() if folder else None
    if path is None or not path.is_dir():
        return None
    _library = Library(path)
    return _library


def index_in_background(lib: Library) -> threading.Thread:
    def run() -> None:
        read, added = lib.update()
        if read:
            print(f"  study     indexed {read} new or changed file(s), {added} passages")

    thread = threading.Thread(target=run, name="study-index", daemon=True)
    thread.start()
    return thread
