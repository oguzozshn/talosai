"""What a skill is, and how one is declared.

A skill is a plain function with type hints and a docstring. The decorator
turns those into the JSON schema the model needs, so a capability is added by
writing a function and nothing else — no registration list to update, no schema
to hand-write and keep in sync.

    @skill(Permission.OPEN)
    def find_file(pattern: str, limit: int = 10) -> str:
        '''Search the projects directory for files matching a pattern.

        Args:
            pattern: Glob pattern, for example *.cs or README*
            limit: Maximum number of results to return
        '''

Hand-written schemas drift from the function they describe, and the model
believes the schema. Deriving it from the signature means they cannot disagree.
"""

from __future__ import annotations

import inspect
import re
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class Permission(Enum):
    """What a skill is allowed to do without being asked twice.

    The tiers exist from the start deliberately. Retrofitting a permission
    model onto an assistant that can already delete things is how a folder
    goes missing.
    """

    # Reads something and reports it. Runs without asking.
    OPEN = "open"

    # Changes something, or reaches outside the machine. Needs a spoken yes
    # before it runs.
    CONFIRM = "confirm"


# "    name: description" under an Args: heading.
_ARG_LINE = re.compile(r"^\s*(\w+)\s*:\s*(.+)$")

_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


@dataclass
class Skill:
    name: str
    description: str
    permission: Permission
    function: Callable[..., str]
    parameters: dict = field(default_factory=dict)

    def schema(self) -> dict:
        """The tool definition, in the shape Ollama expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _split_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Return the summary and the documented argument descriptions."""
    lines = doc.strip().splitlines()
    summary: list[str] = []
    args: dict[str, str] = {}
    in_args = False

    for line in lines:
        stripped = line.strip()
        if stripped.lower() in {"args:", "arguments:", "parameters:"}:
            in_args = True
            continue
        if in_args:
            match = _ARG_LINE.match(line)
            if match:
                args[match.group(1)] = match.group(2).strip()
            elif stripped and not line.startswith((" ", "\t")):
                in_args = False
        elif stripped:
            summary.append(stripped)

    return " ".join(summary), args


def _json_type(annotation: typing.Any) -> str:
    origin = typing.get_origin(annotation)
    if origin is not None:
        annotation = origin
    return _JSON_TYPES.get(annotation, "string")


def _build_parameters(function: Callable, descriptions: dict[str, str]) -> dict:
    signature = inspect.signature(function)
    try:
        hints = typing.get_type_hints(function)
    except Exception:  # noqa: BLE001 — a bad annotation should not break discovery
        hints = {}

    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, parameter in signature.parameters.items():
        if name.startswith("_"):
            continue
        spec: dict[str, typing.Any] = {"type": _json_type(hints.get(name, str))}
        if name in descriptions:
            spec["description"] = descriptions[name]
        properties[name] = spec
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    return {"type": "object", "properties": properties, "required": required}


def skill(permission: Permission = Permission.OPEN, *, name: str | None = None):
    """Mark a function as a capability the assistant can call."""

    def decorator(function: Callable[..., str]) -> Callable[..., str]:
        doc = inspect.getdoc(function) or ""
        summary, arg_descriptions = _split_docstring(doc)
        if not summary:
            raise ValueError(
                f"Skill '{function.__name__}' needs a docstring. It is what the "
                f"model reads to decide whether to call it."
            )

        function._skill = Skill(  # type: ignore[attr-defined]
            name=name or function.__name__,
            description=summary,
            permission=permission,
            function=function,
            parameters=_build_parameters(function, arg_descriptions),
        )
        return function

    return decorator
