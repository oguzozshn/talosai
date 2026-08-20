"""Talos describing itself.

The tool schemas are already in every prompt, so in a narrow sense the model
always knew what it could do. But a schema is a machine-readable contract, not
an explanation — asked "what can you do", it produced either a shrug or an
invented list. Neither is the answer.

The list is built from the registry rather than written down, so it cannot go
stale: add a skill file and it appears here, remove one and it leaves.

Answers are grouped by area rather than enumerated, because thirty-four
function names read aloud is not information. Asking about one area gets the
detail; asking in general gets the shape.
"""

from __future__ import annotations

from talos.skills.base import Permission, skill

# Which module a skill lives in decides its area, so this map is the only
# thing to touch when a new file appears.
AREAS = {
    "project_status": ("your projects", "what you have, what state it is in, what you touched recently"),
    "find_file": ("your files", "finding and reading anything in the folders I can see"),
    "edit": ("changing code", "editing, creating, and undoing — only where git can undo it"),
    "review": ("looking over code", "fetching a diff or a commit so we can talk about it"),
    "notes": ("notes", "writing things down and reading them back"),
    "recall": ("memory", "searching everything we have said before"),
    "planner": ("your week", "the plan you keep in the personal assistant project"),
    "todoist": ("Todoist", "what is due, adding tasks, marking them done"),
    "weather": ("weather", "current conditions anywhere"),
    "search": ("the web", "looking up anything current"),
    "screen": ("your screen", "reading the text on either monitor"),
    "usage": ("your time", "which applications you have been in and for how long"),
    "gpu": ("the graphics card", "how much is in use, and letting go of it"),
    "practice": ("English practice", "conversation with one correction per reply"),
    "listening": ("how I listen", "switching between the wake word and the hotkey"),
    "about_self": ("myself", "what I can do and how I work"),
}

HOW_IT_WORKS = (
    "I run entirely on this machine. Whisper turns your speech into text on the "
    "graphics card, a model called qwen3 does the thinking, and Piper says the "
    "answer out loud. Nothing goes over the network except three skills you can "
    "name — weather, Todoist and web search. "
    "I hear you because I transcribe everything and check whether my name is in "
    "the first two words, so you can say the name and the question in one breath. "
    "I remember our conversations in a database here, and I can only read the "
    "folders you listed in my settings."
)


def _grouped() -> dict[str, list[str]]:
    """Every registered skill, bucketed by the module it came from."""
    from talos.skills import load

    grouped: dict[str, list[str]] = {}
    for name in load().names:
        definition = load().get(name)
        module = definition.function.__module__.rsplit(".", 1)[-1]
        grouped.setdefault(module, []).append(name)
    return grouped


@skill(Permission.OPEN)
def what_can_you_do(about: str = "") -> str:
    """Answer "what can you do", "what are your skills", "what can you help with".

    Always call this rather than answering from memory. It counts the skills
    actually loaded right now, which is the only accurate answer — you cannot
    see how many there are or what they are grouped into.

    Args:
        about: An area to go into detail on, such as projects, memory or Todoist
    """
    grouped = _grouped()
    wanted = about.strip().casefold()

    if wanted:
        # Match on the area name or on the module, so "projects", "my
        # projects" and "project_status" all land in the same place.
        for module, (area, detail) in AREAS.items():
            if module not in grouped:
                continue
            if wanted in area.casefold() or wanted in module or area.casefold() in wanted:
                names = ", ".join(n.replace("_", " ") for n in grouped[module])
                return f"For {area} I can help with {detail}. The skills are: {names}."
        return (
            f"I have nothing for {about}. Ask what I can do in general and I will "
            f"tell you the areas I cover."
        )

    covered = [AREAS[m][0] for m in grouped if m in AREAS]
    total = sum(len(v) for v in grouped.values())
    listed = ", ".join(covered[:-1]) + f", and {covered[-1]}" if len(covered) > 1 else covered[0]

    # Deliberately not a list of thirty-four names. Spoken, that is noise.
    return (
        f"I have {total} skills across {len(covered)} areas: {listed}. "
        f"Ask about any one of those and I will go into detail. "
        f"You never need to name a skill — say what you want and I will pick."
    )


@skill(Permission.OPEN)
def how_do_you_work() -> str:
    """Explain how Talos works: what runs where, and what leaves the machine."""
    return HOW_IT_WORKS


@skill(Permission.OPEN)
def what_are_your_limits() -> str:
    """Report honestly what Talos is bad at, so it is not relied on for those."""
    return (
        "Three things worth knowing. I am a small model, so I am good at fetching "
        "and poor at judging — check anything that matters. I can only see about "
        "three thousand words at once, so I cannot read a whole source file and "
        "reason about it. And I cannot tell your voice from anyone else's, so in "
        "the seconds after I answer, whatever is said in the room is treated as "
        "meant for me."
    )
