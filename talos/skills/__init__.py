"""Skill discovery and execution.

Every module in this package is imported at startup and scanned for functions
carrying the `@skill` decorator. Adding a capability means adding a file here;
nothing else in the codebase changes.

Execution is deliberately forgiving in one direction and strict in the other.
Bad arguments come back as readable errors the model can correct on the next
attempt, because small local models do get arguments wrong and a stack trace
teaches them nothing. But a skill that changes or destroys something does not
run without permission, whatever the model asked for.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Callable
from pathlib import Path

from talos.skills.base import Permission, Skill, skill

__all__ = ["Permission", "Registry", "Skill", "skill"]

log = logging.getLogger(__name__)

ConfirmFn = Callable[[str, Permission], bool]


def a_or_an(word: str) -> str:
    """For error messages the model reads — "an integer", "a string"."""
    return f"{'an' if word[:1] in 'aeiou' else 'a'} {word}"


class Registry:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    @property
    def names(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def discover(self) -> Registry:
        """Import every sibling module and collect the decorated functions."""
        package = Path(__file__).parent
        for info in pkgutil.iter_modules([str(package)]):
            if info.name.startswith("_") or info.name == "base":
                continue
            try:
                module = importlib.import_module(f"{__name__}.{info.name}")
            except Exception:  # noqa: BLE001 — one bad skill must not sink the rest
                log.exception("Could not load skill module '%s'", info.name)
                continue

            for attribute in vars(module).values():
                definition = getattr(attribute, "_skill", None)
                if isinstance(definition, Skill):
                    if definition.name in self._skills:
                        log.warning("Duplicate skill '%s', keeping the first", definition.name)
                        continue
                    self._skills[definition.name] = definition
        return self

    def schemas(self) -> list[dict]:
        return [s.schema() for s in self._skills.values()]

    def execute(self, name: str, arguments: dict, confirm: ConfirmFn | None = None) -> str:
        """Run a skill by name. Never raises — the model reads the result."""
        definition = self._skills.get(name)
        if definition is None:
            return (
                f"There is no skill called '{name}'. "
                f"Available: {', '.join(self.names) or 'none'}."
            )

        problem = self._validate(definition, arguments)
        if problem:
            return problem

        arguments, problem = self._coerce(definition, arguments)
        if problem:
            return problem

        if definition.permission is not Permission.OPEN:
            if confirm is None:
                return (
                    f"'{name}' changes something and needs confirmation, but there "
                    f"is no way to ask right now. It was not run."
                )
            if not confirm(name, definition.permission):
                return f"Permission refused. '{name}' was not run."

        try:
            result = definition.function(**arguments)
        except Exception as exc:  # noqa: BLE001 — surface it, do not crash the turn
            log.exception("Skill '%s' failed", name)
            return f"'{name}' failed: {type(exc).__name__}: {exc}"

        return str(result)

    @staticmethod
    def _coerce(definition: Skill, arguments: dict) -> tuple[dict, str | None]:
        """Convert arguments to the types the schema declares.

        Models send `"5"` where the schema says integer. Llama 3.1 did it on
        three separate skills, and because nothing checked types, the string
        reached the function and crashed it on a comparison — a skill failing
        at runtime over something the schema had already specified.

        Converting is right rather than rejecting: the model expressed the
        correct intent in the wrong notation, and refusing would cost a round
        trip to be told what it already meant. Only genuinely unconvertible
        values are refused.
        """
        properties = definition.parameters.get("properties", {})
        coerced: dict = {}

        for name, value in arguments.items():
            wanted = properties.get(name, {}).get("type", "string")
            try:
                if wanted == "integer" and not isinstance(value, bool):
                    coerced[name] = int(str(value).strip())
                elif wanted == "number" and not isinstance(value, bool):
                    coerced[name] = float(str(value).strip())
                elif wanted == "boolean":
                    if isinstance(value, bool):
                        coerced[name] = value
                    else:
                        text = str(value).strip().lower()
                        if text not in {"true", "false", "1", "0", "yes", "no"}:
                            raise ValueError(text)
                        coerced[name] = text in {"true", "1", "yes"}
                elif wanted == "string":
                    coerced[name] = value if isinstance(value, str) else str(value)
                else:
                    coerced[name] = value
            except (TypeError, ValueError):
                return arguments, (
                    f"'{definition.name}' expects {name} to be {a_or_an(wanted)}, "
                    f"but got {value!r}."
                )

        return coerced, None

    @staticmethod
    def _validate(definition: Skill, arguments: dict) -> str | None:
        """Check arguments against the schema, returning a correctable message."""
        properties = definition.parameters.get("properties", {})
        required = definition.parameters.get("required", [])

        missing = [name for name in required if name not in arguments]
        if missing:
            return (
                f"'{definition.name}' needs {', '.join(missing)}. "
                f"Expected: {', '.join(properties) or 'no arguments'}."
            )

        unexpected = [name for name in arguments if name not in properties]
        if unexpected:
            return (
                f"'{definition.name}' does not take {', '.join(unexpected)}. "
                f"It takes: {', '.join(properties) or 'no arguments'}."
            )
        return None


def load() -> Registry:
    return Registry().discover()
