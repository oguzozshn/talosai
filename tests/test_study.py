"""Course material and the glossary, with a fake embedding: one dimension per word it knows."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from talos import study
from talos.session import current as session
from talos.skills import study as skills

WORDS = ["elicitation", "traceability", "risk", "stakeholder", "definition"]


def fake_embed(texts):
    return [[1e-3 + text.lower().count(w) for w in WORDS] for text in texts]


def slides(path: Path, *texts: str) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for n, text in enumerate(texts, 1):
            archive.writestr(f"ppt/slides/slide{n}.xml", f"<p:sld><a:t>{text}</a:t></p:sld>")


@pytest.fixture
def lib(tmp_path, monkeypatch) -> study.Library:
    course = tmp_path / "Dersler" / "Requirements"
    course.mkdir(parents=True)
    slides(
        course / "week3.pptx",
        "Elicitation is the process of discovering what stakeholders need from the system.",
        "Traceability links each requirement to its origin and to the code that implements it.",
    )
    (course / "notes.txt").write_text("Risk: anything that may stop the project. " * 3, encoding="utf-8")
    made = study.Library(tmp_path / "Dersler", tmp_path / "study.db", embed=fake_embed)
    made.update()
    session.after_reply.clear()
    monkeypatch.setattr(study, "_library", made)
    monkeypatch.setattr(skills, "_course", lambda library, named: library.match_course(named))
    return made


def test_slides_are_read_one_by_one(tmp_path):
    path = tmp_path / "deck.pptx"
    slides(path, "A first slide that says enough to be worth keeping.", "And a second &amp; last one, also long enough.")
    assert study.passages_of(path) == [
        ("slide 1", "A first slide that says enough to be worth keeping."),
        ("slide 2", "And a second & last one, also long enough."),
    ]


def test_a_question_finds_its_slide_and_says_where(lib):
    found = lib.search("what is elicitation", "Requirements")
    assert found[0].cite() == "Requirements — week3.pptx, slide 1"


def test_unchanged_files_are_not_read_again(lib):
    assert lib.update() == (0, 0)


def test_a_deleted_file_leaves_the_index(lib):
    (lib.folder / "Requirements" / "notes.txt").unlink()
    lib.update()
    assert all(p.file != "notes.txt" for p in lib.search("risk", None, limit=10))


def test_a_course_is_found_from_part_of_its_name(lib):
    assert lib.match_course("requirements engineering") == "Requirements"
    assert lib.match_course("physics") is None


def test_a_term_in_the_material_is_quoted_with_its_source(lib):
    text = skills.explain_term("traceability", "requirements")
    assert "week3.pptx, slide 2" in text


def test_a_term_not_in_the_material_says_so(lib):
    assert "isn't in the indexed course material" in skills.explain_term("RAG", "requirements")


def test_the_glossary_keeps_what_was_said(lib):
    skills.explain_term("traceability", "requirements")
    session.reply_finished("Traceability links a requirement to where it came from.")
    assert lib.terms("Requirements") == [
        ("traceability", "Requirements", "Traceability links a requirement to where it came from.", 1)
    ]
    skills.explain_term("traceability", "requirements")
    session.reply_finished("Again.")
    assert lib.terms()[0][3] == 2
    assert "1 term(s)" in skills.glossary()


def test_a_quiz_does_not_give_the_answer_away_first(lib):
    lib.remember_term("elicitation", "Requirements", "Finding out what stakeholders need.")
    text = skills.glossary(quiz=True)
    assert "Do not give the answer first" in text
