"""Which language something was said in: Turkish or English.

Talos answers in the language it was spoken to in, and says Turkish with the
Turkish voice. Two places decide which: Whisper, which names the language of
what it heard, and this, for text that never went through Whisper — a typed
line, a Telegram message — and for choosing a voice per sentence.

Only Turkish and English are told apart, because those are the two spoken
here. Anything that is not clearly Turkish is English.
"""

from __future__ import annotations

import re

TURKISH = "tr"
ENGLISH = "en"

# Letters English does not use. ö and ü are also German, which is not spoken
# here; ç also French, ditto.
_LETTERS = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
# Common words with no special letters, so "Bugun ne var" typed on an English
# keyboard still counts.
_WORDS = re.compile(
    r"\b(ve|bir|bu|ne|nedir|neler|nasil|mi|mu|var|yok|icin|ama|cok|bana|beni|"
    r"benim|sana|bugun|yarin|saat|kac|lazim|merhaba|tesekkurler|evet|hayir|tamam|"
    r"nerede|demek|olan|olarak|daha|gibi|kadar)\b",
    re.IGNORECASE,
)


def detect(text: str) -> str:
    """TURKISH or ENGLISH, from the text alone."""
    words = text.split()
    if not words:
        return ENGLISH
    turkish = 0
    for i, word in enumerate(words):
        if _WORDS.fullmatch(word.strip(".,!?;:'\"").casefold()):
            turkish += 1
        elif _LETTERS.search(word):
            # A capitalised word mid-sentence is a name: "what did Oğuz say
            # about Gülşah" is English with two Turkish names in it.
            mid_sentence = i > 0 and not words[i - 1].endswith((".", "!", "?"))
            if not (word[0].isupper() and mid_sentence):
                turkish += 1
    return TURKISH if turkish / len(words) >= 0.2 else ENGLISH


def resolve(heard_as: str | None, text: str) -> str:
    """The language to answer in: what Whisper heard, if it was one of ours, else from the text."""
    if heard_as in (TURKISH, ENGLISH):
        return heard_as
    return detect(text)


NAMES = {TURKISH: "Turkish", ENGLISH: "English"}
