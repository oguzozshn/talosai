"""Finding files and folders, with the Windows index stubbed out."""

from __future__ import annotations

from pathlib import Path

import pytest

from talos.skills import _index, _workspace, find_file


@pytest.fixture
def desktop(tmp_path: Path, monkeypatch) -> Path:
    projects = tmp_path / "Projects"
    (projects / "Talos").mkdir(parents=True)
    (projects / "Talos" / "node_modules" / "talos").mkdir(parents=True)
    (tmp_path / "Dersler" / "RAG notes").mkdir(parents=True)
    (tmp_path / "Dersler" / "drag race.txt").write_text("x", encoding="utf-8")
    (tmp_path / "Dersler" / "rag.pdf").write_text("x", encoding="utf-8")
    (tmp_path / "Dersler" / ".env").write_text("x", encoding="utf-8")

    monkeypatch.setattr(_workspace, "WORKSPACES", (projects,))
    monkeypatch.setattr(_workspace, "FOLDERS", (tmp_path,))
    return tmp_path


def index_returning(paths):
    return lambda pattern, roots: list(paths)


def test_only_the_outermost_root_is_searched(desktop):
    assert _workspace.search_roots() == (desktop,)


def test_a_project_the_index_skipped_is_still_found(desktop, monkeypatch):
    monkeypatch.setattr(_index, "find", index_returning([]))
    assert "Projects/Talos (folder)" in find_file.find_file("Talos")


def test_whole_words_rank_above_letters_inside_a_word(desktop, monkeypatch):
    d = desktop / "Dersler"
    monkeypatch.setattr(_index, "find", index_returning([d / "drag race.txt", d / "RAG notes", d / "rag.pdf"]))
    lines = find_file.find_file("rag").splitlines()[1:]
    assert lines[0].endswith("rag.pdf")
    assert lines[1].endswith("RAG notes (folder)")
    assert lines[2].endswith("drag race.txt")


def test_ignored_folders_and_credentials_are_left_out(desktop, monkeypatch):
    monkeypatch.setattr(
        _index, "find",
        index_returning([desktop / "Projects" / "Talos" / "node_modules" / "talos", desktop / "Dersler" / ".env"]),
    )
    assert "node_modules" not in find_file.find_file("talos")
    assert "No file or folder" in find_file.find_file(".env")


def test_without_the_index_the_folders_are_walked(desktop, monkeypatch):
    monkeypatch.setattr(_index, "find", lambda pattern, roots: None)
    assert "rag.pdf" in find_file.find_file("rag.pdf")


def test_explorer_is_not_asked_to_go_outside(desktop, tmp_path_factory):
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    assert "outside my workspace" in find_file.show_in_explorer(str(elsewhere))


@pytest.mark.parametrize(
    ("glob", "like"),
    [("*rag*", "%rag%"), ("a_b?", "a[_]b_"), ("o'neil", "o''neil"), ("100%", "100[%]")],
)
def test_globs_become_index_patterns(glob, like):
    assert _index._like(glob) == like


def test_several_words_match_in_any_order(desktop, monkeypatch):
    letter = desktop / "Dersler" / "Letter of motivation.txt"
    letter.write_text("x", encoding="utf-8")
    monkeypatch.setattr(_index, "find", index_returning([letter]))
    assert "Letter of motivation.txt" in find_file.find_file("motivation letter")


def test_nothing_found_suggests_the_turkish_word(desktop, monkeypatch):
    monkeypatch.setattr(_index, "find", index_returning([]))
    assert "Turkish" in find_file.find_file("graduation certificate")


def test_a_folder_in_the_pattern_narrows_the_search(desktop, monkeypatch):
    other = desktop / "Projects" / "Other"
    other.mkdir()
    (other / "README.md").write_text("x", encoding="utf-8")
    (desktop / "Projects" / "Talos" / "README.md").write_text("x", encoding="utf-8")
    monkeypatch.setattr(_index, "find", index_returning([]))
    text = find_file.find_file("Talos/README.md")
    assert "Talos" in text and "Other" not in text
