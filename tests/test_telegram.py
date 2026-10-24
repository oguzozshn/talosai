"""The Telegram channel, against a fake Bot API.

Nothing here reaches Telegram. `Bot.call` is replaced by a script of what the
API would answer, and everything sent is recorded, so the tests can check who
gets answered, what permission looks like, and where files go.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import pytest

from talos import telegram
from talos.config import TelegramConfig
from talos.skills import _net

OWNER = 111
STRANGER = 999


class FakeBot(telegram.Bot):
    def __init__(self, config: TelegramConfig, updates: list[list[dict]] | None = None) -> None:
        super().__init__("token", config)
        self.sent: list[tuple[str, dict]] = []
        self.files: list[tuple[int, str]] = []
        self._script = list(updates or [])

    def call(self, method, params=None, *, timeout=20):
        self.sent.append((method, params or {}))
        if method == "getUpdates":
            return self._script.pop(0) if self._script else []
        if method == "sendMessage":
            return {"message_id": len(self.sent)}
        return True

    def send_file(self, chat_id, path):
        self.files.append((chat_id, path.name))

    def texts(self, chat_id: int) -> list[str]:
        return [p["text"] for m, p in self.sent if m == "sendMessage" and p.get("chat_id") == chat_id]


@dataclass
class FakeTurn:
    reply: str


class FakeAssistant:
    def __init__(self, reply: str = "Done.") -> None:
        self.reply = reply
        self.asked: list[str] = []

    def respond_in_writing(self, text, confirm, heard_as=None):
        self.asked.append(text)
        return FakeTurn(self.reply)


def message(chat_id: int, text: str, update_id: int = 1) -> dict:
    return {"update_id": update_id, "message": {"chat": {"id": chat_id, "username": "x"}, "text": text}}


def button(chat_id: int, data: str, update_id: int = 2) -> dict:
    return {
        "update_id": update_id,
        "callback_query": {"id": "q", "data": data, "message": {"chat": {"id": chat_id}}, "from": {"id": chat_id}},
    }


@pytest.fixture
def config() -> TelegramConfig:
    return TelegramConfig(allowed_chat_ids=(OWNER,), confirm_seconds=1)


class TestWhoIsAnswered:
    def test_the_owner_gets_an_answer(self, config):
        bot, assistant = FakeBot(config), FakeAssistant("Paris.")
        telegram.Server(bot, assistant, str, 16000).handle(message(OWNER, "Capital of France?"))
        assert assistant.asked == ["Capital of France?"]
        assert bot.texts(OWNER) == ["Paris."]

    def test_a_stranger_gets_nothing_and_their_id_is_printed(self, config, capsys):
        bot, assistant = FakeBot(config), FakeAssistant()
        telegram.Server(bot, assistant, str, 16000).handle(message(STRANGER, "list my files"))
        assert assistant.asked == []
        assert bot.sent == []
        assert str(STRANGER) in capsys.readouterr().out

    def test_no_allowed_chats_means_nobody(self):
        bot, assistant = FakeBot(TelegramConfig()), FakeAssistant()
        telegram.Server(bot, assistant, str, 16000).handle(message(OWNER, "hi"))
        assert assistant.asked == []


class TestPermission:
    def test_yes_button_allows(self, config):
        bot = FakeBot(config, [[button(OWNER, "PLACEHOLDER")]])
        original = bot.call

        # The button's data carries a token made when the question is sent,
        # so the scripted press is given it then.
        def call(method, params=None, *, timeout=20):
            result = original(method, params, timeout=timeout)
            if method == "sendMessage" and "reply_markup" in (params or {}):
                data = params["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
                bot._script[0][0]["callback_query"]["data"] = data
            return result

        bot.call = call
        assert bot.ask(OWNER, "May I?") is True

    def test_a_typed_yes_counts(self, config):
        bot = FakeBot(config, [[message(OWNER, "evet", update_id=5)]])
        assert bot.ask(OWNER, "May I?") is True

    def test_silence_is_no(self, config):
        assert FakeBot(config).ask(OWNER, "May I?") is False

    def test_a_strangers_button_does_not_count(self, config):
        bot = FakeBot(config, [[button(STRANGER, "anything")]])
        assert bot.ask(OWNER, "May I?") is False

    def test_other_messages_are_kept_for_later(self, config):
        bot = FakeBot(config, [[message(OWNER, "something else", update_id=7)]])
        bot.ask(OWNER, "May I?")
        assert [u["update_id"] for u in bot.updates()] == [7]


class TestSendFile:
    @pytest.fixture
    def desktop(self, tmp_path, monkeypatch):
        from talos.skills import _workspace

        (tmp_path / "notes.pdf").write_bytes(b"%PDF")
        (tmp_path / ".env").write_text("SECRET=1", encoding="utf-8")
        monkeypatch.setattr(_workspace, "WORKSPACES", (tmp_path,))
        monkeypatch.setattr(_workspace, "FOLDERS", ())
        return tmp_path

    def test_without_telegram_it_says_so(self, desktop, monkeypatch):
        from talos.skills.send import send_file

        monkeypatch.setattr(telegram, "_active", None)
        assert "isn't set up" in send_file("notes.pdf")

    def test_a_file_goes_to_the_chat_that_asked(self, desktop, monkeypatch, config):
        from talos.skills.send import send_file

        bot = FakeBot(config)
        monkeypatch.setattr(telegram, "_active", bot)
        bot._turn.chat = 222
        assert send_file("notes.pdf") == "Sent notes.pdf to Telegram."
        assert bot.files == [(222, "notes.pdf")]

    def test_asked_by_voice_it_goes_to_the_owner(self, desktop, monkeypatch, config):
        from talos.skills.send import send_file

        bot = FakeBot(config)
        monkeypatch.setattr(telegram, "_active", bot)
        send_file("notes.pdf")
        assert bot.files == [(OWNER, "notes.pdf")]

    def test_credentials_are_never_sent(self, desktop, monkeypatch, config):
        from talos.skills.send import send_file

        bot = FakeBot(config)
        monkeypatch.setattr(telegram, "_active", bot)
        assert "credentials" in send_file(".env")
        assert bot.files == []


def test_ipv4_patch_survives_overlapping_threads():
    import socket

    original = socket.getaddrinfo
    inside = threading.Event()
    release = threading.Event()

    def hold():
        with _net.ipv4_only():
            inside.set()
            release.wait(2)

    worker = threading.Thread(target=hold)
    worker.start()
    inside.wait(2)
    with _net.ipv4_only():
        pass
    # The short call ended first; the long one must still be patched.
    assert socket.getaddrinfo is _net._ipv4
    release.set()
    worker.join()
    assert socket.getaddrinfo is original


def test_an_upload_keeps_a_turkish_file_name(tmp_path, config, monkeypatch):
    path = tmp_path / "Ders Notu çalışma.pdf"
    path.write_bytes(b"%PDF-1.4")
    bot = telegram.Bot("token", config)
    sent = {}
    monkeypatch.setattr(bot, "_send", lambda request, timeout: sent.setdefault("body", request.data))
    bot._upload("sendDocument", {"chat_id": OWNER}, "document", path)
    body = sent["body"]
    assert b"filename*=UTF-8''Ders%20Notu%20%C3%A7al%C4%B1%C5%9Fma.pdf" in body
    assert b"%PDF-1.4" in body and f'name="chat_id"\r\n\r\n{OWNER}'.encode() in body
