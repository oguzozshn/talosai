"""The reasoning layer: a local model behind Ollama.

Streams its reply as complete sentences rather than tokens, because the
consumer is text-to-speech and a synthesiser needs a whole clause to get the
prosody right. Splitting here rather than in the speech layer keeps the seam
in one place.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

import ollama

from talos.config import AssistantConfig, LlmConfig

log = logging.getLogger(__name__)

# Split after . ! ? : ; and newlines, but not on the dot of an abbreviation or
# a decimal. Turkish ordinals ("3." as in "3. toplantı") also end in a dot, so
# require the next character to be whitespace followed by something that is not
# a digit.
_SENTENCE_END = re.compile(r"(?<=[.!?:;])\s+(?=[^\d\s])|\n+")

# Long clauses still need to be broken somewhere, or the first spoken audio
# waits for a full paragraph.
_MAX_CLAUSE_CHARS = 180


class Brain:
    def __init__(self, config: LlmConfig, assistant: AssistantConfig, character: str) -> None:
        self._config = config
        self._assistant = assistant
        self._character = character
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

    def reply(self, prompt: str) -> Iterator[str]:
        """Stream the answer sentence by sentence, then record the exchange."""
        buffer = ""
        spoken: list[str] = []

        stream = ollama.chat(
            model=self._config.model,
            messages=self._messages(prompt),
            stream=True,
            think=self._config.think,
            keep_alive=self._config.keep_alive,
            options={
                "num_ctx": self._config.context,
                "temperature": self._config.temperature,
            },
        )

        for chunk in stream:
            buffer += chunk.get("message", {}).get("content", "")

            while True:
                match = _SENTENCE_END.search(buffer)
                if match:
                    sentence, buffer = buffer[: match.start()], buffer[match.end() :]
                elif len(buffer) > _MAX_CLAUSE_CHARS:
                    # No boundary in sight. Break at the last space so a word
                    # is not split across two synthesiser calls.
                    cut = buffer.rfind(" ", 0, _MAX_CLAUSE_CHARS)
                    if cut <= 0:
                        break
                    sentence, buffer = buffer[:cut], buffer[cut + 1 :]
                else:
                    break

                sentence = sentence.strip()
                if sentence:
                    spoken.append(sentence)
                    yield sentence

        tail = buffer.strip()
        if tail:
            spoken.append(tail)
            yield tail

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
