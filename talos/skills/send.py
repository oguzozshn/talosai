"""Sending a file to the owner's phone, over Telegram.

CONFIRM, every time. The file leaves the machine for Telegram's servers, and
the desktop holds documents nobody should send by accident — asked by voice
from across the room, a misheard name could pick the wrong one.
"""

from __future__ import annotations

from talos import telegram
from talos.skills._workspace import OutsideWorkspace, SecretFile, describe, resolve
from talos.skills.base import Permission, skill


@skill(Permission.CONFIRM)
def send_file(path: str) -> str:
    """Send a file to the owner's phone on Telegram, such as a PDF found with find_file.

    Args:
        path: The file, as find_file gave it

    Examples:
        send me the Requirements PDF from Dersler
        send my CV to my phone
        bu dosyayı telefonuma gönder
    """
    bot = telegram.active()
    if bot is None:
        return "Telegram isn't set up, so I can't send files. USING.md says how to set it up."

    try:
        target = resolve(path)
    except (OutsideWorkspace, SecretFile) as exc:
        return str(exc)
    if target.is_dir():
        return f"{describe(target)} is a folder. I can only send one file at a time."
    if not target.is_file():
        return f"'{path}' doesn't exist. Try find_file to locate it first."

    chat = bot.owner_chat()
    if chat is None:
        return "No Telegram chat is allowed yet, so there is nowhere to send it."
    try:
        bot.send_file(chat, target)
    except (telegram.TelegramError, OSError) as exc:
        return f"I couldn't send {target.name}: {exc}"
    return f"Sent {target.name} to Telegram."
