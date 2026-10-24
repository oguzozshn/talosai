"""Studying: answers from the course material, and a glossary of what was asked.

The course is taken from what is asked ("in requirements engineering"), else
from today's study session in Argos — asking what a word means in the middle
of a Requirements session is asking about Requirements.
"""

from __future__ import annotations

import random
from datetime import date

from talos import study
from talos.session import current as session
from talos.skills.base import Permission, skill

NOT_SET_UP = "Course material isn't set up. Set [study] folder in settings to the folder of courses."
# Characters of each passage read back. The model needs enough to answer from;
# the whole passage would crowd the context with three of them.
QUOTE = 600


def _course(lib: study.Library, named: str) -> str | None:
    if named.strip():
        return lib.match_course(named)
    # Nothing named: the course of today's study session, if there is one.
    from talos.skills import argos

    data = argos._data_dir()
    if data is None:
        return None
    plans = argos._load(data / argos.MARKER) or {}
    names = {p["id"]: p.get("ad", "") for p in plans.get("planlar", [])}
    for entry in plans.get("takvim", {}).get(date.today().isoformat(), []):
        if not entry.get("yapilmadi"):
            course = lib.match_course(names.get(entry.get("planId"), ""))
            if course:
                return course
    return None


def _passages(found: list[study.Passage]) -> str:
    return "\n\n".join(f"[{p.cite()}]\n{p.text[:QUOTE]}" for p in found)


@skill(Permission.OPEN)
def course_notes(question: str, course: str = "") -> str:
    """Answer a question from the owner's course material — slides, lecture notes, textbooks — saying which file and page.

    Args:
        question: What to look up, in a few words
        course: Part of the course name, such as requirements; empty for the one being studied today, or all

    Examples:
        what do the slides say about elicitation
        what does my textbook say about business requirements
        ders notlarında risk değerlendirmesi hakkında ne yazıyor
        find the part of the lecture about stakeholders
    """
    lib = study.library()
    if lib is None:
        return NOT_SET_UP
    chosen = _course(lib, course)
    found = lib.search(question, chosen)
    if found is None:
        return "I can't search the course material right now: the embedding model isn't answering."
    if not found:
        if lib.indexing:
            return "The course material is still being indexed. Ask again in a minute."
        return "There's nothing indexed in the course folder yet."
    return (
        f"Closest passages from {chosen or 'all courses'}:\n\n{_passages(found)}\n\n"
        "(Answer from these and say which file they came from. If they do not "
        "answer the question, say so rather than filling the gap.)"
    )


@skill(Permission.OPEN)
def explain_term(term: str, course: str = "") -> str:
    """Explain a term or concept: call this for every "what is X" or "what does X mean", even one you already know.

    It checks the course material for the course's own sense of the term, and
    keeps the term in the owner's glossary to revise from — which answering
    from memory would skip.

    Args:
        term: The word or phrase, such as RAG or traceability matrix
        course: Part of the course name; empty for the one being studied today

    Examples:
        what does traceability mean
        what is a use case
        what's a functional requirement
        I don't understand what elicitation means
        RAG ne demek
        explain the term stakeholder
    """
    lib = study.library()
    if lib is None:
        return NOT_SET_UP
    chosen = _course(lib, course)
    found = lib.search(f"{term} definition", chosen, limit=3) or []
    # Only passages that actually use the term. The nearest passage by
    # meaning, for a term the material never mentions, is a confident
    # explanation of something else.
    mentioned = [p for p in found if term.casefold() in p.text.casefold()]

    # Kept once the reply is finished, with what was actually said.
    session.after_reply.append(
        lambda answer: lib.remember_term(term, chosen or "General", answer)
    )

    if mentioned:
        return (
            f"{term} in the course material:\n\n{_passages(mentioned)}\n\n"
            "(Explain it in two or three sentences, in the course's sense, and "
            "say which file it is from.)"
        )
    return (
        f"{term} isn't in the indexed course material"
        f"{' for ' + chosen if chosen else ''}. (Explain it briefly from general "
        "knowledge, and say that it is not from the course notes.)"
    )


@skill(Permission.OPEN)
def glossary(course: str = "", quiz: bool = False) -> str:
    """List the terms the owner asked about while studying, or quiz them on one.

    Args:
        course: Part of the course name; empty for every course
        quiz: true to ask the owner one term instead of listing them

    Examples:
        which terms have I asked about
        quiz me on my requirements terms
        sorduğum terimlerden beni sına
        what words did I look up this week
    """
    lib = study.library()
    if lib is None:
        return NOT_SET_UP
    chosen = lib.match_course(course) if course.strip() else None
    terms = lib.terms(chosen)
    if not terms:
        return f"No terms asked about{' in ' + chosen if chosen else ''} yet."

    if quiz:
        term, where, definition, _ = random.choice(terms)
        return (
            f"Quiz term: {term} ({where}). (Ask the owner to explain {term} in "
            f"their own words. Do not give the answer first. When they have "
            f"answered, compare it with this, which is what you told them "
            f"before: {definition})"
        )

    shown = terms[:15]
    names = ", ".join(t[0] for t in shown)
    more = f", and {len(terms) - len(shown)} more" if len(terms) > len(shown) else ""
    return f"{len(terms)} term(s){' in ' + chosen if chosen else ''}: {names}{more}."
