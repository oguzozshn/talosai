"""The directories skills are allowed to touch, and the files they are not.

Every skill that takes a path goes through `resolve` here. Not because the
model is malicious, but because it is small and will occasionally produce
`../../..` or an absolute path from somewhere else, and a file search that can
walk out of its workspace can read anything on the machine.

There is more than one root because projects do not all live in one folder —
one sits on the desktop, the rest under Projects. Each root is checked after
resolving symlinks, so a link pointing outside does not slip through.

Secrets are refused inside the roots as well. Granting access to a folder is
not the same as granting access to the API keys that happen to sit in it, and
the difference matters most for an assistant that reads things out loud.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

from talos.config import load

_config = load()


def _roots() -> tuple[Path, ...]:
    resolved: list[Path] = []
    for entry in _config.skills.workspaces:
        try:
            path = Path(entry).expanduser().resolve()
        except OSError:
            continue
        if path.is_dir():
            resolved.append(path)
    return tuple(resolved)


WORKSPACES = _roots()

# Filenames that hold credentials. Refused for reading as much as for writing:
# the damage from a secret is done by it being seen, and this assistant speaks
# its answers out loud in a room.
SECRET_PATTERNS = (
    "gizli.json", ".env", ".env.*", "*.key", "*.pem", "*.pfx", "*.p12",
    "secrets.*", "secret.*", "credentials.*", "*token*", "*password*",
    "id_rsa", "id_ed25519", "*.keystore", "appsettings.*.json",
    "local.toml", "*.secrets.json",
)


class OutsideWorkspace(ValueError):
    pass


class SecretFile(ValueError):
    pass


def is_secret(path: Path) -> bool:
    name = path.name.casefold()
    return any(fnmatch.fnmatch(name, pattern.casefold()) for pattern in SECRET_PATTERNS)


def resolve(relative: str = "", *, allow_secret: bool = False) -> Path:
    """Resolve a path inside one of the workspaces, or refuse.

    Relative paths are tried against each root in turn. Absolute paths are
    allowed only if they land inside one, so a model that helpfully expands a
    filename to a full path is not punished for it.
    """
    if not WORKSPACES:
        raise OutsideWorkspace("No workspace is configured, so I can't read anything.")

    candidate = Path(relative.strip()) if relative else Path()

    if candidate.is_absolute() or not relative:
        attempts = [candidate if candidate.is_absolute() else WORKSPACES[0]]
    else:
        # First root that actually contains it wins, so "Talos/README.md" and
        # "personnal_assistant/siradaki.md" both work without being told which
        # root they belong to.
        attempts = [root / candidate for root in WORKSPACES]

    resolved = None
    for attempt in attempts:
        try:
            candidate_path = attempt.resolve()
        except OSError:
            continue
        if _inside(candidate_path):
            resolved = candidate_path
            if candidate_path.exists():
                break

    if resolved is None:
        roots = ", ".join(str(r) for r in WORKSPACES)
        raise OutsideWorkspace(
            f"'{relative}' is outside my workspace. I can only reach inside: {roots}."
        )

    if not allow_secret and is_secret(resolved):
        raise SecretFile(
            f"'{resolved.name}' looks like it holds credentials, so I won't open it."
        )
    return resolved


def _inside(path: Path) -> bool:
    return any(path == root or root in path.parents for root in WORKSPACES)


def describe(path: Path) -> str:
    """Path as the user would recognise it — relative to whichever root holds it."""
    for root in WORKSPACES:
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        # Name the root when there is more than one, or "data/W34.json" is
        # ambiguous between projects.
        return f"{root.name}/{relative}" if len(WORKSPACES) > 1 else str(relative)
    return str(path)


def roots_description() -> str:
    return ", ".join(str(r) for r in WORKSPACES) or "(none)"
