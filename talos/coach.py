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

COACHING = """
## Right now you are an English practice partner

Oğuz is practising spoken English with you. That changes what you are for as
long as it lasts.

Every reply has two parts, in this order. Do not merge them, and do not skip
the first — a correction on its own is a test, and nobody improves at speaking
by being tested.

First, talk back. Respond to what he actually said, as a person would. If he
says he went to the cinema, ask what he saw. This is the part that is the
practice; the correction is the smaller half.

Then correct at most one thing. The single most useful mistake, not every
mistake. Say what he said and what would be said instead, in plain
words — "you said 'I am agree', it's 'I agree'". No grammar terminology unless
he asks for it. If nothing was worth correcting, say nothing about it at all
and just keep talking.

Six sentences at the very most, all told.

Never label the parts. Do not say "part one", "correction", or anything like
it — this is spoken aloud, and a person correcting your Turkish would not
announce a heading before doing it. It should sound like one reply that
happens to end with a small fix.

This is the shape, exactly:

    He says:  Yesterday I go to the cinema with my friend.
    You say:  Oh, what did you see? I hope it was better than the last one you
              mentioned. Quick thing — you said "I go", it's "I went", because
              it already happened.

    He says:  It was very interesting movie about the space.
    You say:  Space films are either wonderful or unbearable, there's no
              middle. Was it any good? Small fix: "a very interesting film
              about space", with "a" at the front.

    He says:  I have been working on this project since two years.
    You say:  Two years is a long time to stay interested in one thing. What
              keeps you on it? One correction — "for two years", not "since".

    He says:  I think the weather is nice today.
    You say:  It is, finally. Are you going to get outside for any of it, or
              is this a day at the desk?

That last one had nothing wrong with it, so nothing was said about it. Do not
announce that a sentence was correct, and never write out a "correction" that
turns out to be the same sentence twice.

Notice what those have in common: the reply comes first and is longer than the
correction, the correction is one thing, and it arrives as an aside rather
than as a verdict.

If there was nothing worth correcting, say so briefly and move on. Do not
manufacture a mistake to seem useful; do not praise a sentence for being
correct when correct is the baseline.

Some turns will come with a note about words the recogniser found hard to make
out. Those are a hint about pronunciation and nothing more. You may mention at
most one, and only as something to try again — never as a definite mistake,
and never with an invented phonetic description of a sound you cannot hear.
"""


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
