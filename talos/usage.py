"""Recording which application is in front, so it can be asked about later.

Windows does not expose this. The Settings app shows screen time and the SRUM
database holds months of it, but neither is readable without administrator
rights and an ESE parser, so Talos samples it itself: every few seconds, ask
which window has focus and add the interval to that total.

Three things that make the numbers honest rather than impressive:

**Idle time is not usage.** If nothing has been typed or moved for a while,
the interval is dropped. A machine left on with a document open is not an hour
of writing, and counting it would make every total meaningless.

**The lock screen is not an application.** Neither is the shell furniture that
takes focus for a moment while you alt-tab.

**Nothing is counted while Talos is not running.** That is the honest
limitation, and the skill says so rather than presenting a partial day as a
whole one. Getting around it would mean a service that starts with Windows,
which is a different decision from this one.

Only the executable name is stored. Not window titles, which are documents,
messages and search queries, and are the business of nobody here.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import logging
import sqlite3
import sys
import threading
from datetime import date
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATABASE = ROOT / "data" / "usage.db"

# Often enough to catch a short visit to a window, rare enough to be free.
SAMPLE_SECONDS = 5.0

# No keyboard or mouse for this long and the interval is not counted.
IDLE_AFTER_SECONDS = 90.0

# Shell furniture rather than applications anyone means. Explorer is absent on
# purpose: time spent in File Explorer is real time spent.
IGNORED = {
    "lockapp.exe",
    "searchhost.exe",
    "shellexperiencehost.exe",
    "startmenuexperiencehost.exe",
    "applicationframehost.exe",
    "textinputhost.exe",
    "dwm.exe",
}

# How applications are usually spoken about, rather than by executable name.
FRIENDLY = {
    "chrome.exe": "Chrome",
    "msedge.exe": "Edge",
    "firefox.exe": "Firefox",
    "devenv.exe": "Visual Studio",
    "code.exe": "VS Code",
    "windowsterminal.exe": "Terminal",
    "cmd.exe": "Command Prompt",
    "powershell.exe": "PowerShell",
    "pwsh.exe": "PowerShell",
    "explorer.exe": "File Explorer",
    "spotify.exe": "Spotify",
    "discord.exe": "Discord",
    "slack.exe": "Slack",
    "teams.exe": "Teams",
    "outlook.exe": "Outlook",
    "olk.exe": "Outlook",
    "notepad.exe": "Notepad",
    "rider64.exe": "Rider",
    "idea64.exe": "IntelliJ",
    "pycharm64.exe": "PyCharm",
    "obsidian.exe": "Obsidian",
    "notion.exe": "Notion",
    "steam.exe": "Steam",
    "claude.exe": "Claude",
    "whatsapp.exe": "WhatsApp",
}


class _LastInput(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


def idle_seconds() -> float:
    """How long since the last keyboard or mouse input."""
    if sys.platform != "win32":
        return 0.0
    info = _LastInput()
    info.cbSize = ctypes.sizeof(_LastInput)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return 0.0
    ticks = ctypes.windll.kernel32.GetTickCount()
    return max(0.0, (ticks - info.dwTime) / 1000.0)


def foreground_app() -> str | None:
    """Executable name of the focused window, lowercased, or None."""
    if sys.platform != "win32":
        return None

    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    window = user32.GetForegroundWindow()
    if not window:
        return None

    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(window, ctypes.byref(pid))
    if not pid.value:
        return None

    # QUERY_LIMITED_INFORMATION is enough for the image name and, unlike the
    # fuller rights, is granted for processes this one does not own.
    handle = kernel32.OpenProcess(0x1000, False, pid.value)
    if not handle:
        return None
    try:
        size = wintypes.DWORD(260)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return None
        return Path(buffer.value).name.lower()
    finally:
        kernel32.CloseHandle(handle)


def friendly(app: str) -> str:
    known = FRIENDLY.get(app)
    if known:
        return known
    return app.removesuffix(".exe").replace("_", " ").title()


def spoken_duration(seconds: float) -> str:
    """A length of time as it would be said aloud."""
    minutes = int(round(seconds / 60))
    if minutes < 1:
        return "under a minute"
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours, rest = divmod(minutes, 60)
    if rest == 0:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    return f"{hours} hour{'s' if hours != 1 else ''} {rest} minute{'s' if rest != 1 else ''}"


class UsageLog:
    """Seconds per application per day, in its own SQLite file."""

    def __init__(self, path: Path = DATABASE) -> None:
        self._path = path
        self._local = threading.local()

    @property
    def db(self) -> sqlite3.Connection:
        # One connection per thread: the sampler writes from its own thread
        # while a skill reads from another, and SQLite objects do not cross.
        connection = getattr(self._local, "connection", None)
        if connection is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self._path)
            connection.row_factory = sqlite3.Row
            connection.execute(
                "CREATE TABLE IF NOT EXISTS usage ("
                " day TEXT NOT NULL,"
                " app TEXT NOT NULL,"
                " seconds REAL NOT NULL DEFAULT 0,"
                " PRIMARY KEY (day, app))"
            )
            connection.commit()
            self._local.connection = connection
        return connection

    def add(self, app: str, seconds: float, when: date | None = None) -> None:
        day = (when or date.today()).isoformat()
        self.db.execute(
            "INSERT INTO usage (day, app, seconds) VALUES (?, ?, ?)"
            " ON CONFLICT(day, app) DO UPDATE SET seconds = seconds + excluded.seconds",
            (day, app, seconds),
        )
        self.db.commit()

    def totals(
        self, since: date, limit: int = 5, until: date | None = None
    ) -> list[tuple[str, float]]:
        """Most-used applications between two dates, inclusive.

        `until` matters for a single day: without it, "yesterday" quietly
        means "yesterday and everything after it", which is today as well.
        """
        end = (until or date.today()).isoformat()
        rows = self.db.execute(
            "SELECT app, SUM(seconds) AS total FROM usage"
            " WHERE day >= ? AND day <= ? GROUP BY app ORDER BY total DESC LIMIT ?",
            (since.isoformat(), end, limit),
        ).fetchall()
        return [(row["app"], float(row["total"])) for row in rows]

    def first_day(self) -> date | None:
        row = self.db.execute("SELECT MIN(day) AS first FROM usage").fetchone()
        if not row or not row["first"]:
            return None
        return date.fromisoformat(row["first"])

    def recorded_seconds(self, since: date, until: date | None = None) -> float:
        end = (until or date.today()).isoformat()
        row = self.db.execute(
            "SELECT SUM(seconds) AS total FROM usage WHERE day >= ? AND day <= ?",
            (since.isoformat(), end),
        ).fetchone()
        return float(row["total"] or 0.0)


class Sampler:
    """Background thread writing to a UsageLog."""

    def __init__(self, store: UsageLog | None = None) -> None:
        self._store = store or UsageLog()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="usage")
        self._thread.start()
        log.info("recording application usage every %.0fs", SAMPLE_SECONDS)

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(SAMPLE_SECONDS):
            try:
                if idle_seconds() > IDLE_AFTER_SECONDS:
                    continue
                app = foreground_app()
                if not app or app in IGNORED:
                    continue
                self._store.add(app, SAMPLE_SECONDS)
            except Exception as exc:  # noqa: BLE001 — a sampler must not take the app down
                log.debug("usage sample failed: %s", exc)
