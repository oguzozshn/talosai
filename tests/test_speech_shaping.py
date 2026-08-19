"""Turning a model's reply into something a synthesiser can say.

Two jobs, both of which have failed in ways that were audible.

Splitting decides when a clause is complete enough to start speaking. Get it
wrong in one direction and the assistant waits for the whole paragraph; wrong
in the other and it speaks half-sentences.

Stripping removes markup the model was told not to produce and produced
anyway. The character file has forbidden markdown since the first day; the
synthesiser was still handed `**bold**` and a bare `1.` — which it read aloud
as the word "one" — until this was enforced in code.
"""

from __future__ import annotations

from talos.llm import _split, _strip_inline, _strip_line_markers


def split_all(text: str) -> list[str]:
    """Everything a buffer yields, including whatever is left at the end."""
    sentences, rest = _split(_strip_line_markers(text))
    rest = _strip_inline(rest)
    return sentences + ([rest] if rest else [])


class TestSplitting:
    def test_a_sentence_is_held_until_it_is_complete(self):
        sentences, rest = _split("Hello there")
        assert sentences == []
        assert rest == "Hello there"

    def test_a_finished_sentence_is_released(self):
        sentences, rest = _split("Hello there. And then")
        assert sentences == ["Hello there."]
        assert rest == "And then"

    def test_a_long_clause_is_broken_at_a_word(self):
        # Without this the first audio waits for a full stop that may never
        # arrive. The break must not land inside a word.
        sentences, rest = _split("word " * 60)
        assert sentences, "a very long clause should be released"
        assert not sentences[0].endswith("wor")

    def test_a_decimal_is_not_a_sentence_ending(self):
        sentences, _rest = _split("It costs 3.5 million and rising. Next")
        assert sentences == ["It costs 3.5 million and rising."]


class TestStripping:
    def test_emphasis_is_removed_but_the_word_is_kept(self):
        assert _strip_inline("that is **important** here") == "that is important here"

    def test_code_spans_lose_their_backticks(self):
        assert _strip_inline("run `git status` now") == "run git status now"

    def test_a_link_becomes_its_text(self):
        assert _strip_inline("see [the docs](http://x.y) for more") == (
            "see the docs for more"
        )

    def test_list_markers_are_removed_before_splitting(self):
        # Removed before rather than after, because "1." followed by a newline
        # reads as a finished sentence: leaving the marker in produces a
        # fragment that gets synthesised on its own.
        assert "1." not in _strip_line_markers("1. first thing\n2. second thing")

    def test_a_numbered_list_never_yields_a_bare_number(self):
        spoken = split_all("Here they are:\n1. **aspnetcore** — a project\n2. BACnet")
        assert all(part.strip(". ") for part in spoken)
        assert not any(part.strip() in {"1.", "2.", "1", "2"} for part in spoken)

    def test_headings_do_not_reach_the_synthesiser(self):
        assert _strip_line_markers("## Projects\ntext") == "Projects\ntext"

    def test_a_real_reply_survives_untouched(self):
        # Stripping must not damage ordinary prose, which is most replies.
        plain = "You worked on Talos today. Would you like more detail?"
        assert split_all(plain) == ["You worked on Talos today.",
                                    "Would you like more detail?"]
