"""Talking to Talos from a phone, through a Telegram bot.

The third thing that leaves the machine, after Todoist and web search: what
you write to the bot, what it writes back, and any file you ask it to send go
through Telegram's servers. Nothing else does — the model, the transcription
and the skills still run here, which is also why the computer has to be on
for the bot to answer.

Only the chats in `[telegram] allowed_chat_ids` are answered. Anyone else who
finds the bot gets no reply at all, and their chat id is printed so the owner
can recognise their own. A bot answering strangers would hand them every
folder Talos can read.

Permission works as it does by voice, but in writing: a skill that changes
something or sends something away asks with Yes and No buttons, and no answer
in time is a no.

Plain HTTP against the Bot API rather than a library, the same as the Todoist
skill: three endpoints do not justify a dependency and its event loop.
"""

from __future__ import annotations

import io
import json
import logging
import mimetypes
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import deque
from pathlib import Path

from talos.config import TelegramConfig
from talos.skills._net import ipv4_only

log = logging.getLogger(__name__)

API = "https://api.telegram.org"
# Long polling: Telegram holds the request open until something arrives or
# this many seconds pass, so an idle bot makes one request every half minute.
POLL_SECONDS = 30
# A Telegram message holds at most 4096 characters.
MESSAGE_LIMIT = 4096


class TelegramError(RuntimeError):
    pass


class Bot:
    def __init__(self, token: str, config: TelegramConfig) -> None:
        self._token = token
        self._config = config
        self._allowed = {int(chat) for chat in config.allowed_chat_ids}
        self._offset = 0
        # Updates read while waiting for a button press, to be handled after.
        self._pending: deque[dict] = deque()
        self._strangers: set[int] = set()
        # The chat the turn in progress came from, so a skill sending a file
        # sends it back there. Per thread: voice turns run on another one.
        self._turn = threading.local()
        self.name = ""

    # --- The Bot API ------------------------------------------------------

    def call(self, method: str, params: dict | None = None, *, timeout: float = 20) -> object:
        body = json.dumps(params or {}).encode("utf-8")
        request = urllib.request.Request(
            f"{API}/bot{self._token}/{method}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        return self._send(request, timeout)

    def _send(self, request: urllib.request.Request, timeout: float) -> object:
        try:
            with ipv4_only():
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read())
            except (ValueError, OSError):
                raise TelegramError(f"Telegram returned {exc.code}") from exc
        if not payload.get("ok"):
            raise TelegramError(payload.get("description", "Telegram refused the request"))
        return payload.get("result")

    def _upload(self, method: str, fields: dict, field: str, path: Path) -> object:
        """A multipart request, which is the only way the Bot API takes a file."""
        boundary = uuid.uuid4().hex
        parts: list[bytes] = []
        for name, value in fields.items():
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
            )
        kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        # RFC 5987 form for the name, which keeps Turkish characters intact.
        quoted = urllib.parse.quote(path.name)
        parts.append(
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
                f"filename=\"{quoted}\"; filename*=UTF-8''{quoted}\r\n"
                f"Content-Type: {kind}\r\n\r\n"
            ).encode()
            + path.read_bytes()
            + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())
        request = urllib.request.Request(
            f"{API}/bot{self._token}/{method}",
            data=b"".join(parts),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        return self._send(request, timeout=300)

    def updates(self) -> list[dict]:
        if self._pending:
            pending = list(self._pending)
            self._pending.clear()
            return pending
        result = self.call(
            "getUpdates",
            {"offset": self._offset, "timeout": POLL_SECONDS, "allowed_updates": ["message", "callback_query"]},
            timeout=POLL_SECONDS + 10,
        )
        return self._advance(result or [])

    def _advance(self, updates: list[dict]) -> list[dict]:
        if updates:
            self._offset = max(u["update_id"] for u in updates) + 1
        return updates

    def send_text(self, chat_id: int, text: str) -> None:
        text = text.strip() or "…"
        for start in range(0, len(text), MESSAGE_LIMIT):
            self.call("sendMessage", {"chat_id": chat_id, "text": text[start:start + MESSAGE_LIMIT]})

    def typing(self, chat_id: int) -> None:
        try:
            self.call("sendChatAction", {"chat_id": chat_id, "action": "typing"})
        except (TelegramError, OSError):
            pass  # Cosmetic. Not worth failing a turn over.

    def send_file(self, chat_id: int, path: Path) -> None:
        size_mb = path.stat().st_size / 1_000_000
        if size_mb > self._config.max_file_mb:
            raise TelegramError(
                f"{path.name} is {size_mb:.0f} MB, and Telegram bots can send at most "
                f"{self._config.max_file_mb} MB."
            )
        self._upload("sendDocument", {"chat_id": chat_id}, "document", path)

    def download(self, file_id: str) -> bytes:
        info = self.call("getFile", {"file_id": file_id})
        request = urllib.request.Request(f"{API}/file/bot{self._token}/{info['file_path']}")
        with ipv4_only():
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()

    # --- Asking permission ------------------------------------------------

    def ask(self, chat_id: int, question: str) -> bool:
        """Ask with Yes and No buttons, and wait. Silence is not consent."""
        token = uuid.uuid4().hex[:8]
        sent = self.call(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": question,
                "reply_markup": {
                    "inline_keyboard": [[
                        {"text": "Yes", "callback_data": f"{token}:yes"},
                        {"text": "No", "callback_data": f"{token}:no"},
                    ]]
                },
            },
        )
        deadline = time.monotonic() + self._config.confirm_seconds
        answer: bool | None = None
        while answer is None and time.monotonic() < deadline:
            wait = max(1, min(POLL_SECONDS, int(deadline - time.monotonic())))
            try:
                result = self.call(
                    "getUpdates",
                    {"offset": self._offset, "timeout": wait, "allowed_updates": ["message", "callback_query"]},
                    timeout=wait + 10,
                )
            except (TelegramError, OSError) as exc:
                log.warning("waiting for an answer: %s", exc)
                time.sleep(2)
                continue
            for update in self._advance(result or []):
                query = update.get("callback_query")
                typed = self._typed_answer(update, chat_id)
                if query and str(query.get("data", "")).startswith(f"{token}:") and self._is_allowed(query):
                    answer = query["data"].endswith(":yes")
                    self.call("answerCallbackQuery", {"callback_query_id": query["id"]})
                elif typed is not None:
                    # "yes" typed instead of tapped counts too.
                    answer = typed
                else:
                    self._pending.append(update)

        outcome = "Allowed." if answer else ("Declined." if answer is False else "No answer, so no.")
        try:
            self.call(
                "editMessageText",
                {"chat_id": chat_id, "message_id": sent["message_id"], "text": f"{question}\n\n{outcome}"},
            )
        except (TelegramError, OSError):
            pass
        return bool(answer)

    def _typed_answer(self, update: dict, chat_id: int) -> bool | None:
        message = update.get("message") or {}
        if message.get("chat", {}).get("id") != chat_id:
            return None
        text = str(message.get("text", "")).strip().casefold()
        if text in {"yes", "y", "evet", "e", "ok", "tamam"}:
            return True
        if text in {"no", "n", "hayır", "hayir", "h", "iptal"}:
            return False
        return None

    # --- Who may talk to it -----------------------------------------------

    def _is_allowed(self, update_part: dict) -> bool:
        chat = (update_part.get("message") or update_part).get("chat") or {}
        sender = update_part.get("from") or {}
        chat_id = chat.get("id", sender.get("id"))
        return chat_id in self._allowed

    def note_stranger(self, message: dict) -> None:
        chat = message.get("chat", {})
        chat_id = chat.get("id")
        if chat_id in self._strangers:
            return
        self._strangers.add(chat_id)
        who = chat.get("username") or chat.get("first_name") or "someone"
        print(
            f"\n  Telegram: ignored a message from {who} (chat id {chat_id}).\n"
            f"  If that was you, add {chat_id} to [telegram] allowed_chat_ids in "
            f"config/local.toml and restart.\n"
        )

    # --- The chat a turn belongs to ---------------------------------------

    @property
    def turn_chat(self) -> int | None:
        return getattr(self._turn, "chat", None)

    def owner_chat(self) -> int | None:
        """Where to send something: the chat that asked, else the first allowed one."""
        if self.turn_chat is not None:
            return self.turn_chat
        return next(iter(self._config.allowed_chat_ids), None)


_active: Bot | None = None


def active() -> Bot | None:
    """The running bot, if Telegram is configured and started."""
    return _active


def token(config: TelegramConfig) -> str:
    import os

    return (config.token or "").strip() or os.environ.get("TALOS_TELEGRAM_TOKEN", "").strip()


class Server:
    """Answers the bot's messages on a background thread, one turn at a time."""

    def __init__(self, bot: Bot, assistant, describe_action, sample_rate: int) -> None:
        self._bot = bot
        self._assistant = assistant
        self._describe = describe_action
        self._sample_rate = sample_rate
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="telegram", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        backoff = 2
        while not self._stop.is_set():
            try:
                updates = self._bot.updates()
                backoff = 2
            except TelegramError as exc:
                if "Conflict" in str(exc):
                    # Another copy of Talos is already polling this bot, and
                    # Telegram delivers each message to only one of them.
                    print("  Telegram: another Talos is already using this bot. Stopping here.")
                    return
                log.warning("Telegram: %s", exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            except OSError as exc:
                # Offline, asleep, or a dropped connection. Keep trying, more
                # slowly, rather than dying for the rest of the session.
                log.warning("Telegram unreachable: %s", exc)
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            for update in updates:
                try:
                    self.handle(update)
                except Exception:  # noqa: BLE001 — one bad message is not the end of the bot
                    log.exception("Telegram message failed")

    def handle(self, update: dict) -> None:
        bot = self._bot
        query = update.get("callback_query")
        if query:
            # A button from a question that is no longer being waited for.
            if bot._is_allowed(query):
                bot.call("answerCallbackQuery", {"callback_query_id": query["id"], "text": "That question has expired."})
            return

        message = update.get("message")
        if not message:
            return
        if not bot._is_allowed(message):
            bot.note_stranger(message)
            return

        chat_id = message["chat"]["id"]
        text = str(message.get("text", "")).strip()
        if text == "/start":
            bot.send_text(chat_id, "Talos here. Write or send a voice message.")
            return

        bot.typing(chat_id)
        voice = message.get("voice") or message.get("audio")
        if voice:
            text = self._transcribe(voice["file_id"])
            if not text:
                bot.send_text(chat_id, "I couldn't make out that voice message.")
                return
            # Shown back, so a mishearing is visible before the answer to it.
            bot.send_text(chat_id, f"Heard: {text}")
        if not text:
            bot.send_text(chat_id, "I can read text and voice messages.")
            return

        print(f"telegram> {text}")
        bot._turn.chat = chat_id
        try:
            turn = self._assistant.respond_in_writing(
                text, lambda name, permission, arguments: self._confirm(chat_id, name, arguments)
            )
        finally:
            bot._turn.chat = None
        print(f"  {turn.reply}\n")
        bot.send_text(chat_id, turn.reply)

    def _confirm(self, chat_id: int, name: str, arguments: dict) -> bool:
        return self._bot.ask(chat_id, f"May I run this?\n{self._describe(name, arguments)}")

    def _transcribe(self, file_id: str) -> str:
        from faster_whisper import decode_audio

        data = self._bot.download(file_id)
        audio = decode_audio(io.BytesIO(data), sampling_rate=self._sample_rate)
        return self._assistant.hear(audio)


def start(config: TelegramConfig, assistant, describe_action, sample_rate: int) -> Server | None:
    """Start answering on Telegram, if it is configured. Returns None when it is not."""
    global _active

    secret = token(config)
    if not secret:
        return None
    bot = Bot(secret, config)
    try:
        me = bot.call("getMe")
    except (TelegramError, OSError) as exc:
        print(f"  telegram  not started: {exc}")
        return None
    bot.name = me.get("username", "")
    _active = bot
    server = Server(bot, assistant, describe_action, sample_rate)
    server.start()
    if not config.allowed_chat_ids:
        print(f"  telegram  @{bot.name} — no chats allowed yet; message it once to see your chat id")
    else:
        print(f"  telegram  @{bot.name}")
    return server
