"""Put a Talos shortcut on the desktop.

    run scripts\\make_shortcut.py

Double-clicking it starts Talos listening for its name, with the bronze
helmet as its icon. Run this again after moving the project: a shortcut holds
absolute paths, and an old one points at where the project used to be.

The desktop is asked of Windows rather than assumed, because with OneDrive
backing it up it is not the folder under the user's home that it looks like.
"""

from __future__ import annotations

import sys
from pathlib import Path

import win32com.client

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    shell = win32com.client.Dispatch("WScript.Shell")
    desktop = Path(shell.SpecialFolders("Desktop"))
    target = desktop / "Talos.lnk"

    shortcut = shell.CreateShortcut(str(target))
    shortcut.TargetPath = str(ROOT / "talos.cmd")
    shortcut.WorkingDirectory = str(ROOT)
    shortcut.IconLocation = str(ROOT / "assets" / "talos.ico")
    shortcut.Description = "Talos, the voice assistant"
    shortcut.Save()

    print(f"Shortcut created: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
