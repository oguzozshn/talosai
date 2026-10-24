"""Choosing which skills the model sees for one request.

Every skill handed to the model is its schema in the prompt, before anyone has
said anything. At 40 skills that was 3.6k of 4096 tokens; merging brought it to
2.8k, but every skill added since puts it back up, and a small model choosing
between forty options picks wrong more often than between eight.

So each request is matched against the skills by meaning, and only the closest
few are offered, plus a handful that are always there. The matching uses the
same embedding model memory already loads, so it costs no extra graphics
memory, only one embedding per request.

What it risks: a skill the request needed but did not resemble is not offered
at all, and the model cannot call what it cannot see. That is why the skills
just used stay offered for follow-ups ("now open it"), why some are always
offered, and why `scripts/check_router.py` measures how often the right skill
makes the cut.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np

from talos.memory import EMBED_MODEL
from talos.skills import Registry

log = logging.getLogger(__name__)

# Offered on every request. Each is a fallback for a whole class of question:
# anything about Talos, anything current, anything about the day, anything to
# be written down. Without them a vague request could be routed to nothing.
ALWAYS = ("about_me", "web_search", "my_day", "take_note")

# nomic-embed-text was trained with these prefixes, and scores noticeably worse
# without them.
DOCUMENT = "search_document: "
QUERY = "search_query: "


def _skill_text(registry: Registry, name: str) -> str:
    skill = registry.get(name)
    args = "; ".join(
        f"{arg}: {spec.get('description', '')}"
        for arg, spec in skill.parameters.get("properties", {}).items()
    )
    examples = ". ".join(skill.examples)
    return f"{name.replace('_', ' ')}. {skill.description} {args}. {examples}"


class Router:
    def __init__(self, registry: Registry, top: int = 6, embed=None) -> None:
        self._registry = registry
        self._top = top
        self._embed = embed or _embed
        self._names: list[str] = []
        self._vectors: np.ndarray | None = None
        # Skills used in the last few turns, so a follow-up such as "send it
        # to me" still finds the skill the previous answer was built on.
        self._recent: deque[set[str]] = deque(maxlen=2)

    def prepare(self) -> bool:
        """Embed every skill once. False if the embedding model is unavailable."""
        names = self._registry.names
        vectors = self._embed([DOCUMENT + _skill_text(self._registry, n) for n in names])
        if vectors is None:
            log.warning("skill router off: could not embed the skills")
            return False
        self._names = names
        self._vectors = _normalised(vectors)
        return True

    def choose(self, request: str) -> list[str] | None:
        """The skills to offer for this request, or None to offer all of them."""
        if self._vectors is None:
            return None
        query = self._embed([QUERY + request])
        if query is None:
            return None
        scores = self._vectors @ _normalised(query)[0]
        ranked = [self._names[i] for i in np.argsort(-scores)]
        chosen = set(ranked[: self._top])
        chosen.update(n for n in ALWAYS if n in self._registry)
        for used in self._recent:
            chosen.update(used)
        # Sorted, so the same set always produces the same prompt and the
        # model's cache of it can be reused.
        return sorted(chosen)

    def used(self, names: list[str]) -> None:
        """Record which skills a turn called."""
        self._recent.append(set(names))

    def reset(self) -> None:
        """Forget recent turns, for a conversation that starts over."""
        self._recent.clear()


def _normalised(vectors) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.where(norms == 0, 1, norms)


def _embed(texts: list[str]):
    try:
        import ollama

        response = ollama.embed(model=EMBED_MODEL, input=texts)
        return response["embeddings"]
    except Exception as exc:  # noqa: BLE001 — routing degrades to offering everything
        log.warning("could not embed for routing: %s", exc)
        return None
