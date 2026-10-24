"""Asking the Windows Search index for files by name.

Walking the whole desktop to find one folder took 15 to 42 seconds, measured
on this machine; the index answers the same question in under a second.
Windows keeps it up to date already, so using it costs nothing extra.

What it gives up: a file created in the last few seconds may not be indexed
yet, and a folder the indexer skips is invisible to it — on this machine that
included the Talos project itself. When the
index cannot be reached at all, `find` returns None and the caller falls back
to walking the directories itself.
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from urllib.parse import unquote

log = logging.getLogger(__name__)

CONNECTION = "Provider=Search.CollatorDSO;Extended Properties='Application=Windows';"
# Rows asked for before filtering. Generous, because ignored folders such as
# node_modules match common words ("drag" for "rag") by the hundred.
FETCH = 300


def _like(pattern: str) -> str:
    """A glob as a Windows Search LIKE pattern, quoted for the query."""
    out = []
    for ch in pattern:
        if ch == "*":
            out.append("%")
        elif ch == "?":
            out.append("_")
        elif ch in "%_[":
            out.append(f"[{ch}]")
        elif ch == "'":
            out.append("''")
        else:
            out.append(ch)
    return "".join(out)


def find(pattern: str, roots: tuple[Path, ...]) -> list[Path] | None:
    """Paths under the roots whose name matches the glob, or None if the index is unavailable."""
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        return None

    scopes = " OR ".join(f"SCOPE='file:{root.as_posix()}'" for root in roots)
    # A name that starts with the term is asked for first, so it cannot be
    # crowded out by names that merely contain it: searching for "Talos"
    # returned three hundred Unity cache files containing "dataloss" before
    # anything called Talos. Excluding such folders in the query instead was
    # tried, and made every search take 19 seconds.
    patterns = [pattern]
    if pattern.startswith("*") and len(pattern) > 1:
        patterns.insert(0, pattern[1:])

    found: list[Path] = []
    try:
        # Skills run on whichever thread asked, and COM has to be set up on
        # each one before use.
        pythoncom.CoInitialize()
        connection = win32com.client.Dispatch("ADODB.Connection")
        connection.Open(CONNECTION)
        try:
            for glob in patterns:
                query = (
                    f"SELECT TOP {FETCH} System.ItemUrl FROM SystemIndex "
                    f"WHERE ({scopes}) AND System.FileName LIKE '{_like(glob)}'"
                )
                records, _ = connection.Execute(query)
                while not records.EOF:
                    url = str(records.Fields.Item(0).Value)
                    # file:C:/Users/... — the real path, unlike
                    # ItemPathDisplay, which is translated ("Kullanıcılar",
                    # "Masaüstü").
                    path = Path(unquote(url.removeprefix("file:")))
                    if path not in found:
                        found.append(path)
                    records.MoveNext()
        finally:
            connection.Close()
    except Exception as exc:  # noqa: BLE001 — any failure means "use the fallback"
        log.warning("Windows Search index unavailable: %s", exc)
        return None

    # LIKE in the index is looser than a glob; check each name properly.
    wanted = pattern.casefold()
    return [p for p in found if fnmatch.fnmatch(p.name.casefold(), wanted)]
