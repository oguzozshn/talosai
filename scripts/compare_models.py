"""Measure which local model picks the right skill, and how fast.

    python scripts/compare_models.py
    python scripts/compare_models.py --models qwen3:8b llama3.1:8b
    python scripts/compare_models.py --repeat 3

The roadmap has said since phase 3 that the model choice turns on tool-calling
reliability and should be decided by measuring. This is the measurement.

Three things are scored, and the third is the one people forget:

  chose the right skill   — asked about projects, called the projects skill
  passed valid arguments  — the registry validates them, as it would live
  left it alone           — asked something no skill answers, called nothing

A model that calls a skill for every question is not being helpful, it is being
expensive: each call costs a full extra round trip. Restraint counts as much as
recall, so half the cases here expect no call at all.

Skills are never executed. Only the choice is under test, and running them
would make the score depend on what happens to be in the projects folder.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from talos.config import load  # noqa: E402
from talos.llm import Brain  # noqa: E402
from talos.skills import Registry  # noqa: E402
from talos.skills import load as load_skills  # noqa: E402


@dataclass
class Case:
    prompt: str
    # The skill that should be called, or None for "answer without one".
    expects: str | None


CASES = [
    # Should reach for a skill.
    Case("What projects have I worked on recently?", "recent_work"),
    Case("What's the status of the Talos project?", "project_status"),
    Case("List my projects.", "list_projects"),
    Case("Find the README file in the Talos project.", "find_file"),
    Case("Take a note that I need to fix the parser tomorrow.", "take_note"),
    Case("What did we say about BACnet earlier?", "recall"),
    Case("Read my notes back to me.", "read_notes"),
    # Should not. A skill call here is a wrong answer that also costs a round
    # trip, so these carry the same weight as the ones above.
    Case("What is the capital of France?", None),
    Case("Thanks, that's all for now.", None),
    Case("Explain what a race condition is, briefly.", None),
    Case("How are you doing today?", None),
    Case("What is twelve times eleven?", None),
]


@dataclass
class Score:
    model: str
    correct: int = 0
    wrong_skill: int = 0
    missed: int = 0
    unwanted: int = 0
    bad_arguments: int = 0
    latencies: list[float] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.correct + self.wrong_skill + self.missed + self.unwanted

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def median_latency(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0


def evaluate(model: str, registry: Registry, repeat: int, context: int | None = None) -> Score:
    config = load()
    llm = replace(config.llm, model=model, context=context or config.llm.context)
    score = Score(model=model)

    print(f"\n{model}")
    print("-" * len(model))

    for case in CASES:
        for _ in range(repeat):
            # A fresh brain per case: history would let an earlier answer
            # decide a later one, and each case is meant to stand alone.
            brain = Brain(llm, config.assistant, config.character, registry=registry)

            started = time.perf_counter()
            try:
                for _sentence in brain.reply(case.prompt):
                    pass
            except Exception as exc:  # noqa: BLE001
                score.failures.append(f"{case.prompt!r}: {type(exc).__name__}: {exc}")
                continue
            score.latencies.append(time.perf_counter() - started)

            called = [name for name, _ in brain.last_tool_calls]
            chosen = called[0] if called else None

            if case.expects is None:
                if chosen is None:
                    score.correct += 1
                    mark = "ok  "
                else:
                    score.unwanted += 1
                    mark = "call"
                    score.failures.append(f"{case.prompt!r}: called {chosen}, expected none")
            elif chosen is None:
                score.missed += 1
                mark = "none"
                score.failures.append(f"{case.prompt!r}: called nothing, expected {case.expects}")
            elif chosen == case.expects:
                score.correct += 1
                mark = "ok  "
                # The registry validates arguments the same way it would live,
                # so a right skill with malformed arguments is still a failure.
                problem = registry._validate(registry.get(chosen), brain.last_tool_calls[0][1])
                if problem:
                    score.bad_arguments += 1
                    score.failures.append(f"{case.prompt!r}: {problem}")
            else:
                score.wrong_skill += 1
                mark = "wrong"
                score.failures.append(
                    f"{case.prompt!r}: called {chosen}, expected {case.expects}"
                )

            print(f"  [{mark:<5}] {case.prompt[:52]:<54} -> {chosen or '(none)'}")

    return score


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["qwen3:8b", "gemma3:4b", "llama3.1:8b"])
    parser.add_argument("--repeat", type=int, default=1, help="runs per case")
    parser.add_argument("--context", type=int, help="num_ctx, instead of the one in settings")
    args = parser.parse_args()

    registry = load_skills()
    print(f"{len(registry)} skills: {', '.join(registry.names)}")
    print(f"{len(CASES)} cases, {args.repeat} run(s) each")

    scores = []
    for model in args.models:
        try:
            scores.append(evaluate(model, registry, args.repeat, args.context))
        except Exception as exc:  # noqa: BLE001
            print(f"\n{model}: could not evaluate — {exc}")

    print("\n\nResults")
    print("=======\n")
    header = f"{'model':<16}{'correct':>9}{'wrong':>7}{'missed':>8}{'unwanted':>10}{'bad args':>10}{'median':>9}"
    print(header)
    print("-" * len(header))
    for s in sorted(scores, key=lambda s: s.accuracy, reverse=True):
        print(
            f"{s.model:<16}{s.correct:>4}/{s.total:<4}{s.wrong_skill:>7}"
            f"{s.missed:>8}{s.unwanted:>10}{s.bad_arguments:>10}{s.median_latency:>8.2f}s"
        )

    for s in scores:
        if s.failures:
            print(f"\n{s.model} — what went wrong")
            for failure in s.failures[:12]:
                print(f"  {failure}")

    print(
        "\nWrong and missed are failures of recall; unwanted is a failure of\n"
        "restraint, and costs a full extra round trip every time it happens."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
