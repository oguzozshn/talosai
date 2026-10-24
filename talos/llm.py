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
from talos.coach import CONVERSATION, correction_for
from talos.memory import Exchange, Memory, embed, load_profile
from talos.router import Router
from talos.session import current as session
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
        router: Router | None = None,
    ) -> None:
        self._config = config
        self._assistant = assistant
        self._character = character
        self._registry = registry
        self._confirm = confirm
        self._max_rounds = max_rounds
        self._memory = memory
        self._router = router
        # The skills offered to the model this turn, or None for all of them.
        # Chosen once per request, from the request: see talos/router.py.
        self.last_offered: list[str] | None = None
        self._history: list[dict[str, str]] = []
        # A running précis of what fell out of the window. Kept per session:
        # anything older than that is in the database and reachable by recall.
        self._summary = ""
        # Skills requested during the last reply, in order. Exposed because
        # which tool a model reaches for is the thing worth measuring about it,
        # and it is otherwise only visible in the log.
        self.last_tool_calls: list[tuple[str, dict]] = []
        # What each of those calls actually returned, same order as above.
        # Together the two are the whole story of what a turn did — for
        # --diagnostic to print directly instead of it only living in the log.
        self.last_tool_results: list[str] = []
        # Reasoning content, when the model emits any. Empty whenever `think`
        # is off, which is the default — see ROADMAP.md phase 0 for why
        # thinking stays off (12.6s against 0.45s, none of it spoken).
        self.last_thinking: str = ""
        # How many model <-> tool round trips the last reply took. 1 is a
        # plain answer; more means at least one skill call, which is where
        # the ~9s of extra latency documented in ROADMAP.md actually goes.
        # Counted separately from the coaching correction call below, which
        # is not a round trip in this sense — it never sees the tools.
        self.last_rounds: int = 0
        # Which mode the history was built under. Changing mode has to clear
        # it: the model copies the shape of the last few turns far more
        # readily than it re-reads its instructions, so leaving corrections in
        # the history means it keeps correcting after being told to stop.
        self._history_mode = session.coaching

    def _system_prompt(self) -> str:
        """Character plus profile, read fresh so edits apply without a restart."""
        profile = load_profile()
        prompt = self._character
        if profile:
            heading = "## Who you are talking to"
            prompt = f"{prompt}\n\n{heading}\n\n{profile}"
        # While practising, the prompt says only how to converse.
        # Correcting is a separate call assembled afterwards: asking one
        # call to do both reliably produced one at the expense of the other.
        if session.coaching:
            prompt = f"{prompt}\n{CONVERSATION}"
        return prompt

    @staticmethod
    def _now() -> str:
        """The date and time, for a model that has neither.

        Sent as its own message after the character rather than inside it, so
        the long stable prefix — character, profile, tool schemas — stays
        identical between requests and stays prefilled. Rounded to the minute
        for the same reason: a timestamp that changes every second would
        invalidate the cache on every request.
        """
        now = datetime.now()
        return (
            f"Right now it is {now:%A %d %B %Y, %H:%M}. "
            f"Work out dates from this rather than guessing, and say "
            f"\"today\" or \"tomorrow\" when that is what you mean."
        )

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "system", "content": self._now()},
        ]
        if self._summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Earlier in this conversation: {self._summary}",
                }
            )
        messages += self._history
        messages.append({"role": "user", "content": prompt})
        return messages

    def warm_up(self) -> None:
        """Force the model into VRAM, and prefill the part of every prompt
        that never changes.

        Loading the weights is only half of it. The character definition, the
        profile and nine tool schemas come to well over a thousand tokens that
        are identical on every request, and prefilling them is most of the
        cost of a first turn — measured at 9.8 s against 1.2 s for the turns
        after it, on a model that was already resident.

        So the warm-up sends the real system prompt and the real tools rather
        than a bare "hi". A warm-up that does not resemble the request it is
        warming up for is not warming anything.
        """
        try:
            ollama.chat(
                model=self._config.model,
                messages=self._messages("hello"),
                tools=self._tools(),
                think=self._config.think,
                keep_alive=self._config.keep_alive,
                options={
                    "num_ctx": self._config.context,
                    "num_predict": 1,
                },
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

    def set_confirm(self, confirm: ConfirmFn) -> None:
        self._confirm = confirm

    @property
    def confirm(self) -> ConfirmFn | None:
        return self._confirm

    def _tools(self) -> list[dict] | None:
        if self._registry is None or not len(self._registry):
            return None
        if self.last_offered is None:
            return self._registry.schemas()
        return [self._registry.get(name).schema() for name in self.last_offered]

    def _record_if_asked(self, prompt: str, messages: list[dict]) -> None:
        """Write the note when told to remember something, whatever the model thinks.

        Told "remember my girlfriend's name is Dilara", the model answered
        "I'll remember that" and called nothing — before and after take_note's
        description was rewritten to say this is exactly what it is for, and
        after three attempts at saying so in the character definition. The
        promise is the worst possible failure here: it sounds like the thing
        working, so nobody checks, and the fact is gone by morning.

        The result is then handed to the model as a tool call it made itself,
        so what it says out loud is built on what was actually written.
        """
        if self._registry is None:
            return
        try:
            from talos.skills.notes import note_request
        except ImportError:
            # Skills are files, and a deleted file takes its capability with
            # it. That has to include this.
            return
        if self._registry.get("take_note") is None:
            return

        what = note_request(prompt)
        if what is None:
            return

        result = self._registry.execute("take_note", {"text": what}, self._confirm)
        log.info("recorded on request: %s", what)

        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "take_note", "arguments": {"text": what}}}
                ],
            }
        )
        messages.append({"role": "tool", "name": "take_note", "content": result})
        self.last_tool_calls.append(("take_note", {"text": what}))
        self.last_tool_results.append(result)

    def reply(self, prompt: str, note: str = "") -> Iterator[str]:
        """Stream the answer sentence by sentence, calling skills as needed.

        `note` is context about this turn rather than about the conversation —
        currently which words the recogniser struggled with. It is not stored,
        because it describes one recording and would be meaningless attached to
        the transcript later.
        """
        if session.coaching != self._history_mode:
            self._history.clear()
            self._summary = ""
            self._history_mode = session.coaching

        messages = self._messages(prompt)
        if note:
            messages.insert(-1, {"role": "system", "content": note})
        spoken: list[str] = []
        self.last_tool_calls = []
        self.last_tool_results = []
        self.last_thinking = ""
        self.last_rounds = 0
        # Left over only if the last reply failed part way, and then it would
        # attach this answer to that turn's question.
        session.after_reply.clear()
        self.last_offered = self._router.choose(prompt) if self._router else None
        self._record_if_asked(prompt, messages)

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
                thinking = message.get("thinking")
                if thinking:
                    self.last_thinking += thinking
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
            self.last_tool_calls += tool_calls
            for name, arguments in tool_calls:
                log.info("calling skill %s(%s)", name, arguments)
                result = self._registry.execute(name, arguments, self._confirm)
                self.last_tool_results.append(result)
                messages.append({"role": "tool", "name": name, "content": result})

        self.last_rounds = round_number + 1
        if self._router is not None:
            self._router.used([name for name, _ in self.last_tool_calls])

        # After the conversation, never inside it. The reply is already being
        # spoken by the time this runs, so the second call costs nothing the
        # listener notices.
        if session.coaching:
            correction = correction_for(
                prompt, self._config.model, self._config.keep_alive
            )
            if correction:
                spoken.append(correction)
                yield correction

        self._remember(prompt, " ".join(spoken))
        session.reply_finished(" ".join(spoken))

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
            dropped, self._history = self._history[:-limit], self._history[-limit:]
            self._condense(dropped)

    def _condense(self, dropped: list[dict[str, str]]) -> None:
        """Fold what fell out of the window into the running summary.

        Without this, a long conversation forgets its own beginning without
        saying so: the assistant answers confidently about a subject it can no
        longer see. A précis is not as good as the transcript, but it is the
        difference between a vague memory and none.
        """
        if not self._assistant.summarise or not dropped:
            return

        transcript = "\n".join(
            f"{'User' if m['role'] == 'user' else 'You'}: {m['content']}" for m in dropped
        )
        # The existing notes go in the material to be rewritten, not in the
        # instruction. Appending "fold this in" to the system prompt was
        # ignored on the third pass in testing: the model summarised only the
        # new transcript and silently dropped a fact from two rounds earlier.
        instruction = (
            "You are keeping running notes on a conversation, for your own "
            "later reference. Rewrite the notes so they cover both what you "
            "already noted and the new part. Nothing already noted may be "
            "dropped unless the new part supersedes it. Keep names, numbers, "
            "decisions and anything the user asked to be remembered; drop "
            "pleasantries. At most three sentences. Output only the notes."
        )
        material = (
            f"Existing notes:\n{self._summary or '(none yet)'}\n\n"
            f"New part of the conversation:\n{transcript}"
        )

        try:
            response = ollama.chat(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": material},
                ],
                think=False,
                keep_alive=self._config.keep_alive,
                options={"num_ctx": self._config.context, "temperature": 0.2,
                         "num_predict": 160},
            )
        except Exception as exc:  # noqa: BLE001 — a lost summary is not a lost turn
            log.warning("could not summarise dropped history: %s", exc)
            return

        summary = (response.get("message", {}).get("content") or "").strip()
        if not summary:
            return

        # Summarisers concatenate. Asked to keep everything and stay short, the
        # model reliably chooses "keep everything", so the note grows by a
        # clause per fold and would eventually crowd out the conversation it
        # exists to support. The sentence limit is a request; this is the bound.
        if len(summary) > self._assistant.summary_chars:
            summary = self._compress(summary)

        self._summary = summary
        log.info("condensed %d message(s): %s", len(dropped), summary)

    def _compress(self, notes: str) -> str:
        """Squeeze over-long notes back under budget."""
        budget = self._assistant.summary_chars
        try:
            response = ollama.chat(
                model=self._config.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Shorten these notes to under {budget} characters. "
                            "Keep every name, number and decision. Cut wording, "
                            "not facts. Output only the shortened notes."
                        ),
                    },
                    {"role": "user", "content": notes},
                ],
                think=False,
                keep_alive=self._config.keep_alive,
                options={"num_ctx": self._config.context, "temperature": 0.2,
                         "num_predict": 200},
            )
            shorter = (response.get("message", {}).get("content") or "").strip()
        except Exception as exc:  # noqa: BLE001
            log.warning("could not compress notes: %s", exc)
            shorter = ""

        if shorter and len(shorter) < len(notes):
            return shorter

        # Last resort: cut at a sentence boundary rather than mid-word, keeping
        # the earliest notes, which hold the facts the later ones refer back to.
        cut = notes.rfind(". ", 0, budget)
        return notes[: cut + 1] if cut > 0 else notes[:budget]

    def forget(self) -> None:
        self._history.clear()
