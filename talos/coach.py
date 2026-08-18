"""Practising English out loud, and being told what went wrong.

Two different signals, and they are worth keeping apart because one is solid
and the other is a hint.

**Grammar and phrasing** come from the model reading the transcript. That is
what a model is good at, and the transcript is what you actually said.

**Pronunciation** comes from Whisper's per-word confidence. When the recogniser
is unsure of a word, it is often because the word was not said clearly. It is
a hint, not a verdict: background noise, an unusual name, and a word the model
simply does not expect all look the same as a mispronunciation from here. So
shaky words are offered as "these were hard to make out" rather than as "you
said these wrong", and the coaching prompt is told not to invent a phonetic
diagnosis it cannot actually hear.

Nobody can hear their own accent. A recogniser that stumbles on the same word
every time is still telling you something useful.

**Repeat flags** lean on the conversation history already in context — no new
storage, nothing persisted across sessions. That means it is a heuristic, not
a ledger: it only catches a pattern if both occurrences happen to survive in
the same window, and it forgets everything the moment the session ends or
history gets summarised. Good enough to say "that's the same one as before"
mid-conversation; not good enough to answer "what do I keep getting wrong"
across the last month. That would mean writing corrections into memory.py
instead of just hoping the model notices twice in one prompt — worth doing if
this turns out to matter, not done here.
"""

from __future__ import annotations

from dataclasses import dataclass

# Below this, Whisper was genuinely unsure. Chosen high rather than low: the
# cost of naming a word that was said perfectly well is that the learner
# distrusts the feedback, which is worse than missing one.
SHAKY_BELOW = 0.55

# Words that carry no pronunciation information worth commenting on, and which
# Whisper is unsure about constantly because they are short and unstressed.
FILLER = {
    "a", "an", "the", "of", "to", "and", "or", "is", "it", "in", "on",
    "at", "so", "um", "uh", "i", "you", "we", "my", "me", "that", "this",
}

# Two prompts instead of one, because one could not hold the shape.
#
# Asked in a single prompt to reply and then correct, an 8B model reliably did
# one of: correcting without replying, labelling the parts out loud, inventing
# a correction for a correct sentence, or answering as though it were the
# learner. Every fix moved the failure somewhere else, and growing the prompt
# made it worse — the model copies what it is shown more readily than it
# follows what it is told, so more examples meant more parroting.
#
# So the structure is assembled in code. The conversation is one call that
# knows nothing about correcting; the correction is a separate call that sees
# only the sentence and can return nothing. Neither can eat the other.

CONVERSATION = """
You are talking with Oğuz, who is practising his spoken English with you.

Just talk. Reply to what he actually said the way a person would, and ask
something back so there is somewhere to go next. If he tells you he went to
the cinema, ask what he saw.

Two or three sentences. Never correct his English here and never comment on
it — that is handled elsewhere, and doing it here produces it twice.

He is practising, not you. This inverts if it is not said plainly, so:

- Never offer to speak so that he can correct you.
- Never say "I'll speak and you respond", "ready?", or anything resembling it.
- Never explain how the practice will work. He asked for it; it has started.
- If he asks you a question, answer it as yourself, briefly, and hand the
  conversation back.
"""

GRADER = """
You check one sentence of a learner's English and return at most one
correction. You are not talking to him — your output is inserted into someone
else's reply.

Answer with the corrected wording first, then what he said, like this:

    "I went", not "I go" — it already happened.
    "for two years", not "since two years".
    "I want to improve", not "I want improve".
    "a very interesting film", not "very interesting movie".

Nothing else. No greeting, no encouragement, no question, no explanation
beyond the short clause after the dash, and never the whole sentence quoted
back.

Answer exactly NONE — the word on its own — when there is nothing worth
correcting. That includes all of these:

    "What did you do yesterday?"          NONE, it is correct
    "Today I am gonna go to work."        NONE, informal is not wrong
    "Yeah, I finished it."                NONE, informal is not wrong
    "The weather is quite nice today."    NONE, correct
    "I finished the report, it was hard." NONE, correct

Grammar only. Not style, not flow, not punctuation, not word choice, not
whether a fluent speaker would have phrased it differently. If the grammar is
sound, the answer is NONE however plain the sentence is.

One thing only, the most useful one, never a list.
"""

# What the grader says when there is nothing to say.
NOTHING = "NONE"


@dataclass
class Word:
    text: str
    confidence: float


def shaky_words(words: list[Word], limit: int = 3) -> list[str]:
    """Words the recogniser was least sure of, worst first."""
    candidates = [
        w for w in words
        if w.confidence < SHAKY_BELOW
        and w.text.strip(".,!?;:'\"").casefold() not in FILLER
        and len(w.text.strip(".,!?;:'\"")) > 2
    ]
    candidates.sort(key=lambda w: w.confidence)

    seen: set[str] = set()
    unique: list[str] = []
    for word in candidates:
        cleaned = word.text.strip(".,!?;:'\"")
        key = cleaned.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(cleaned)
        if len(unique) >= limit:
            break
    return unique


def pronunciation_note(words: list[Word]) -> str:
    """A line to hand the model, or nothing if the speech was clear."""
    shaky = shaky_words(words)
    if not shaky:
        return ""
    listed = ", ".join(f"'{w}'" for w in shaky)
    return (
        f"[The recogniser was unsure of {listed}. That may be pronunciation or "
        f"may be noise — treat it as something to try again, not as a mistake.]"
    )


def correction_for(sentence: str, model: str, keep_alive: str) -> str:
    """Ask the grader about one sentence. Empty string means nothing to say.

    A separate call rather than part of the reply: a grader that cannot see
    the conversation cannot be talked out of its job, and one that returns
    only a clause cannot swallow the reply it is meant to accompany.
    """
    import logging

    import ollama

    log = logging.getLogger(__name__)
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": GRADER},
                {"role": "user", "content": sentence},
            ],
            think=False,
            keep_alive=keep_alive,
            # Low temperature and a tight cap: this is a judgement, not
            # writing, and a long answer here is a wrong answer.
            options={"temperature": 0.1, "num_predict": 60},
        )
    except Exception as exc:  # noqa: BLE001 — no correction is fine
        log.warning("grader failed: %s", exc)
        return ""

    verdict = (response.get("message", {}).get("content") or "").strip()

    # Models say NONE in more ways than one.
    stripped = verdict.strip().strip('."\'').upper()
    if not verdict or stripped.startswith(NOTHING):
        return ""
    # A grader that starts explaining has stopped grading.
    # Any line break at all means it stopped following the format — the shape
    # asked for is one clause. The remaining false positive was a two-line
    # answer offering "gonna" as an error, which informality is not.
    if len(verdict) > 160 or chr(10) in verdict:
        log.debug("grader rambled, ignoring: %s", verdict[:80])
        return ""

    # The most common false positive is the sentence handed straight back,
    # sometimes with different punctuation. A sentence is not its own
    # correction, and this is cheaper to check than to prompt away.
    if _same_words(verdict, sentence):
        log.debug("grader echoed the sentence, ignoring")
        return ""

    return verdict


def _same_words(verdict: str, sentence: str) -> bool:
    """Whether the verdict is just the sentence again.

    Compared as sets of letters-only words: a rewrite that changes nothing but
    a comma, a full stop or the capitalisation is not a correction anyone can
    act on.
    """
    import re

    def words(text: str) -> set:
        return set(re.findall(r"[a-z']+", text.lower()))

    said = words(sentence)
    if not said:
        return False
    offered = words(verdict)
    # One direction only. A real correction quotes the original *and* the fix,
    # so it always contains every word of the sentence — testing that way
    # round threw away five good corrections out of six. An echo is a verdict
    # that introduces nothing: no word the sentence did not already have.
    return offered <= said

