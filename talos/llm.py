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
from datetime import datetime

import ollama

from talos.config import AssistantConfig, LlmConfig
from talos.memory import Exchange, Memory, embed, load_profile
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
        memory: Memory | None = None,
    ) -> None:
        self._config = config
        self._assistant = assistant
        self._character = character
        self._registry = registry
        self._confirm = confirm
        self._max_rounds = max_rounds
        self._memory = memory
        self._history: list[dict[str, str]] = []

    def _system_prompt(self) -> str:
        """Character plus profile, read fresh so edits apply without a restart."""
        profile = load_profile()
        if not profile:
            return self._character
        return f"{self._character}\n\n## Who you are talking to\n\n{profile}"

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self._system_prompt()},
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

    # Roughly what each skill is doing, so the filler is about the right thing
    # rather than a generic noise. Matched on a prefix of the skill name.
    _ACKNOWLEDGEMENTS = {
        "recall": "Let me check what we said.",
        "last_conversations": "Let me check what we said.",
        "read_notes": "Let me look at your notes.",
        "take_note": "Writing that down.",
        "find_file": "Let me look for that.",
        "read_file": "Let me read that.",
        "project_status": "Let me check that project.",
        "list_projects": "Let me look.",
        "recent_work": "Let me check your recent work.",
    }

    def _acknowledge(self, tool_calls: list[tuple[str, dict]]) -> str:
        """Something to say while a skill runs.

        A skill call costs a full extra round trip — measured at nine seconds
        against one for a plain answer. That gap will not optimise away, and
        nine seconds of silence sounds like the assistant has stopped working
        rather than like it is working.
        """
        if not self._config.acknowledge_skills or not tool_calls:
            return ""
        name = tool_calls[0][0]
        return self._ACKNOWLEDGEMENTS.get(name, "One moment.")

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
                    "num_predict": self._config.max_tokens,
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

            if not tool_calls:
                break

            # Speaking and then calling a tool used to end the turn here, on
            # the reasoning that a model emits one or the other. That was true
            # until the character definition started asking it to say what it
            # is about to do — at which point obeying that instruction
            # silently cancelled the tool call it was announcing.
            if not said_something:
                filler = self._acknowledge(tool_calls)
                if filler:
                    # Spoken but not remembered: it is a noise to cover a gap,
                    # not part of the answer.
                    yield filler

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
        if self._memory is not None and answer.strip():
            # Runs after the last sentence has been handed over, so the
            # embedding call happens while the reply is still being spoken
            # rather than before it starts.
            try:
                exchange = Exchange(datetime.now(), prompt, answer)
                self._memory.record(prompt, answer, embed(exchange.as_text()))
            except Exception:  # noqa: BLE001 — never lose a turn over bookkeeping
                log.exception("could not record exchange")

        self._history.append({"role": "user", "content": prompt})
        self._history.append({"role": "assistant", "content": answer})
        # Two entries per turn. Trimming from the front keeps the most recent
        # exchanges, which are the ones pronouns and follow-ups refer back to.
        limit = self._assistant.history_turns * 2
        if len(self._history) > limit:
            self._history = self._history[-limit:]

    def forget(self) -> None:
        self._history.clear()
