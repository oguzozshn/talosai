"""The reasoning layer: a local model behind Ollama.

Streams its reply as complete sentences rather than tokens, because the
consumer is text-to-speech and a synthesiser needs a whole clause to get the
prosody right. Splitting here rather than in the speech layer keeps the seam
in one place.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator

import ollama

from talos.config import AssistantConfig, LlmConfig
from talos.skills import ConfirmFn, Registry

log = logging.getLogger(__name__)

# Split after . ! ? : ; and newlines, but not on the dot of an abbreviation or
# a decimal. Turkish ordinals ("3." as in "3. toplantı") also end in a dot, so
# require the next character to be whitespace followed by something that is not
# a digit.
_SENTENCE_END = re.compile(r"(?<=[.!?:;])\s+(?=[^\d\s])|\n+")

# Long clauses still need to be broken somewhere, or the first spoken audio
# waits for a full paragraph.
_MAX_CLAUSE_CHARS = 180

# The character definition tells the model not to use markdown, and it mostly
# obeys — but "mostly" is not good enough when the failure is a synthesiser
# reading asterisks aloud. Strip it here as well, because a rule enforced only
# by a prompt is not enforced.
_LIST_MARKER = re.compile(r"^[ \t]*(?:[-*+•]|\d+[.)])[ \t]+", re.MULTILINE)
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]*", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*{1,3}|_{2,3})(.+?)\1", re.DOTALL)
_CODE = re.compile(r"`+([^`]*)`+")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
# Left over once markers are gone: a line that was only "1." or "-".
_EMPTY_ITEM = re.compile(r"^[\s\-*•.]*$")


def _strip_line_markers(text: str) -> str:
    """Remove list and heading markers. Line-local, so safe on a partial buffer.

    Done before splitting rather than after: "1." followed by a newline reads
    as a finished sentence, so leaving the marker in produces a fragment that
    gets synthesised on its own — the assistant literally saying "one".
    """
    return _HEADING.sub("", _LIST_MARKER.sub("", text))


def _strip_inline(text: str) -> str:
    """Remove emphasis, code spans and links from a complete sentence.

    After splitting, because these are paired and a half-arrived pair would be
    mangled if stripped from a buffer still being filled.
    """
    text = _LINK.sub(r"\1", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = _CODE.sub(r"\1", text)
    return " ".join(text.split())


def _split(buffer: str) -> tuple[list[str], str]:
    """Pull complete sentences out of a buffer, returning them and the rest."""
    sentences: list[str] = []
    while True:
        match = _SENTENCE_END.search(buffer)
        if match:
            head, buffer = buffer[: match.start()], buffer[match.end() :]
        elif len(buffer) > _MAX_CLAUSE_CHARS:
            # No boundary in sight. Break at the last space so a word is not
            # split across two synthesiser calls.
            cut = buffer.rfind(" ", 0, _MAX_CLAUSE_CHARS)
            if cut <= 0:
                break
            head, buffer = buffer[:cut], buffer[cut + 1 :]
        else:
            break
        head = _strip_inline(head)
        # A line that was nothing but a list marker leaves nothing behind.
        # Yielding it would have the assistant say "one" on its own.
        if head and not _EMPTY_ITEM.match(head):
            sentences.append(head)
    return sentences, buffer


def _normalise_tool_call(call: object) -> tuple[str, dict] | None:
    """Read a tool call from either a dict or the client's typed object."""
    function = getattr(call, "function", None)
    if function is None and isinstance(call, dict):
        function = call.get("function")
    if function is None:
        return None

    name = getattr(function, "name", None)
    arguments = getattr(function, "arguments", None)
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")

    if not name:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return name, dict(arguments or {})


class Brain:
    def __init__(
        self,
        config: LlmConfig,
        assistant: AssistantConfig,
        character: str,
        registry: Registry | None = None,
        confirm: ConfirmFn | None = None,
        max_rounds: int = 3,
    ) -> None:
        self._config = config
        self._assistant = assistant
        self._character = character
        self._registry = registry
        self._confirm = confirm
        self._max_rounds = max_rounds
        self._history: list[dict[str, str]] = []

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self._character},
            *self._history,
            {"role": "user", "content": prompt},
        ]

    def warm_up(self) -> None:
        """Force the model into VRAM before the first real request.

        A cold load is around a minute for a 5 GB model. keep_alive stops it
        being unloaded again mid-conversation.
        """
        try:
            ollama.chat(
                model=self._config.model,
                messages=[{"role": "user", "content": "hi"}],
                think=self._config.think,
                keep_alive=self._config.keep_alive,
                options={"num_predict": 1},
            )
        except Exception as exc:  # noqa: BLE001 — warm-up is best effort
            log.warning("Model warm-up failed: %s", exc)

    def _tools(self) -> list[dict] | None:
        if self._registry is None or not len(self._registry):
            return None
        return self._registry.schemas()

    def reply(self, prompt: str) -> Iterator[str]:
        """Stream the answer sentence by sentence, calling skills as needed."""
        messages = self._messages(prompt)
        spoken: list[str] = []

        for round_number in range(self._max_rounds + 1):
            # Past the ceiling, offer no tools at all. Telling a looping model
            # "you are out of attempts" while still handing it the tools is an
            # invitation to try once more; removing them forces an answer.
            last_round = round_number == self._max_rounds
            tools = None if last_round else self._tools()

            buffer = ""
            tool_calls: list[tuple[str, dict]] = []
            said_something = False

            stream = ollama.chat(
                model=self._config.model,
                messages=messages,
                stream=True,
                tools=tools,
                think=self._config.think,
                keep_alive=self._config.keep_alive,
                options={
                    "num_ctx": self._config.context,
                    "temperature": self._config.temperature,
                },
            )

            for chunk in stream:
                message = chunk.get("message", {})
                for call in message.get("tool_calls") or []:
                    normalised = _normalise_tool_call(call)
                    if normalised:
                        tool_calls.append(normalised)

                buffer += message.get("content", "")
                buffer = _strip_line_markers(buffer)
                sentences, buffer = _split(buffer)
                for sentence in sentences:
                    said_something = True
                    spoken.append(sentence)
                    yield sentence

            tail = _strip_inline(_strip_line_markers(buffer))
            if tail and not _EMPTY_ITEM.match(tail):
                said_something = True
                spoken.append(tail)
                yield tail

            # Models emit either tool calls or an answer, not both. If it has
            # already started answering, running the tools now would mean
            # speaking twice about the same question.
            if not tool_calls or said_something:
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": name, "arguments": arguments}}
                        for name, arguments in tool_calls
                    ],
                }
            )
            for name, arguments in tool_calls:
                log.info("calling skill %s(%s)", name, arguments)
                result = self._registry.execute(name, arguments, self._confirm)
                messages.append({"role": "tool", "name": name, "content": result})

        self._remember(prompt, " ".join(spoken))

    def _remember(self, prompt: str, answer: str) -> None:
        self._history.append({"role": "user", "content": prompt})
        self._history.append({"role": "assistant", "content": answer})
        # Two entries per turn. Trimming from the front keeps the most recent
        # exchanges, which are the ones pronouns and follow-ups refer back to.
        limit = self._assistant.history_turns * 2
        if len(self._history) > limit:
            self._history = self._history[-limit:]

    def forget(self) -> None:
        self._history.clear()
