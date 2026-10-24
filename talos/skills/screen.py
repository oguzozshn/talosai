"""Reading what is on the screen, as text.

Optical character recognition rather than a vision model, and the reason is
arithmetic: the language model and Whisper already hold about 7.4 GB of an
8 GB card, so a vision model would mean unloading one of them for every
glance. Tesseract runs on the CPU and costs nothing that is in short supply.

The trade is real. This reads *text* — an error message, a stack trace, a
paragraph, a line of code. It cannot tell you a chart is wrong, a layout is
crooked, or which of two buttons is greyed out. For the questions people
actually ask a computer about its own screen, most are text.

Whatever is displayed gets read, including anything private that happens to be
visible. It runs only when asked, and nothing is written to disk.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from talos.skills.base import Permission, skill

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent

# Where the Windows installer puts it. Checked before PATH because a fresh
# install is not on PATH until the shell is restarted, which is exactly when
# someone first tries this.
KNOWN_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)

# A full screen of dense text runs to thousands of characters. This is spoken
# aloud and shares a 4k context window, so it gets cut early and says so.
MAX_CHARS = 2500

# Language data kept with the project rather than in Program Files, which
# needs administrator rights to write to. Both languages together because the
# machine is Turkish: menus, error dialogs and half the file names are, even
# when the code and the assistant are English. Reading with English alone
# turned "Kişisel" into "Kigisel" and "Dört" into a replacement character.
TESSDATA = ROOT / "models" / "tessdata"
DEFAULT_LANGS = "eng+tur"


def _tesseract() -> str | None:
    for path in KNOWN_PATHS:
        if Path(path).is_file():
            return path
    return shutil.which("tesseract")


def _capture(monitor: int):
    import mss

    with mss.mss() as sct:
        # sct.monitors[0] is every screen stitched together; 1 onwards are the
        # individual ones. Asking for "the screen" means the primary, not a
        # panorama with a laptop and a monitor side by side.
        screens = sct.monitors
        index = monitor if 0 < monitor < len(screens) else 1
        shot = sct.grab(screens[index])
        return shot, len(screens) - 1


@skill(Permission.OPEN)
def read_screen(monitor: int = 1, language: str = "") -> str:
    """Read the text currently on screen and report what it says.

    Args:
        monitor: Which screen to read, 1 for the main one
        language: Override the recognition language, such as eng or tur
    """
    binary = _tesseract()
    if binary is None:
        return (
            "Tesseract isn't installed, so I can't read the screen. "
            "Install it with: winget install UB-Mannheim.TesseractOCR"
        )

    try:
        import pytesseract
        from PIL import Image

        pytesseract.pytesseract.tesseract_cmd = binary
        shot, available = _capture(monitor)
        image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

        # TESSDATA_PREFIX rather than --tessdata-dir: the command-line form
        # keeps the quotes as part of the path and then looks for
        # "C:\...	essdata"/eng.traineddata, quotes and all.
        # Fall back to English alone when the extra language data is absent —
        # models/ is git-ignored, so a fresh clone has none of it and should
        # still be able to read an English screen.
        langs = language.strip() or DEFAULT_LANGS
        if TESSDATA.is_dir():
            os.environ["TESSDATA_PREFIX"] = str(TESSDATA)
            missing = [
                code for code in langs.split("+")
                if not (TESSDATA / f"{code}.traineddata").is_file()
            ]
            if missing:
                log.warning("no data for %s, reading in English only", ", ".join(missing))
                langs = "eng"
        elif not language.strip():
            langs = "eng"
        # Greyscale: colour tells OCR nothing and costs time on a 4K screen.
        text = pytesseract.image_to_string(
            image.convert("L"),
            lang=langs,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("screen read failed: %s", exc)
        return f"I couldn't read the screen: {exc}"

    # OCR of a desktop returns a lot of single stray characters from icons and
    # window furniture. Lines with nothing substantial on them are noise.
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln) > 2]
    cleaned = "\n".join(lines).strip()

    if not cleaned:
        return f"I looked at screen {monitor} and couldn't make out any text."

    truncated = len(cleaned) > MAX_CHARS
    if truncated:
        cleaned = cleaned[:MAX_CHARS].rsplit("\n", 1)[0]

    where = f"screen {monitor} of {available}" if available > 1 else "the screen"
    note = "\n\n[There was more than this.]" if truncated else ""
    return (
        f"Text on {where}:\n\n{cleaned}{note}\n\n"
        f"Answer what was asked about this in a sentence or two. Do not read "
        f"it back — he can see the screen, he wanted to know something about "
        f"it. This came from character recognition, so a mangled word is "
        f"probably a misreading: say you could not make it out rather than "
        f"guessing, and never translate a word that came out garbled."
    )
