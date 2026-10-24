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
    "notes": ("notes", "writing things down and reading them back"),
    "recall": ("memory", "searching everything we have said before"),
    "day": ("your day", "what is planned, due and next, from Argos, the weekly plan and Todoist"),
    "argos": ("your studies and goals", "hours left on each study plan, and the goals you keep in Argos"),
    "todoist": ("Todoist", "what is due, adding tasks, marking them done"),
    "weather": ("weather", "current conditions anywhere"),
    "clock": ("the time", "the time anywhere in the world, worked out here without the internet"),
    "search": ("the web", "looking up anything current"),
    "send": ("sending files", "sending a file to your phone over Telegram"),
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
    "answer out loud. Nothing goes over the network except what you can name: "
    "weather, Todoist, web search, and Telegram when it is set up. "
    "I hear you because I transcribe everything and check whether my name is in "
    "the first two words, so you can say the name and the question in one breath. "
    "I remember our conversations in a database here, and I can only read the "
    "folders you listed in my settings."
)


def _area_for(wanted: str, grouped: dict[str, list[str]]) -> str | None:
    """The module an area question is about, matched loosely.

    On the area name or the module first, so "projects", "my projects" and
    "project_status" land in the same place; then on the skills' own names,
    so "sending a file" finds send_file.
    """
    for module, (area, _) in AREAS.items():
        if module in grouped and (
            wanted in area.casefold() or wanted in module or area.casefold() in wanted
        ):
            return module
    # Then by words, scored, so "sending a file" goes to send_file (two
    # words) rather than find_file (one), and "my day" finds "your day".
    words = {_stem(w) for w in wanted.replace("_", " ").split() if len(w) > 2} - {"my", "the", "your"}
    best, best_score = None, 0
    for module, names in grouped.items():
        if module not in AREAS:
            continue
        vocabulary = {_stem(w) for w in AREAS[module][0].split()}
        vocabulary |= {_stem(part) for name in names for part in name.split("_")}
        score = len(words & vocabulary)
        if score > best_score:
            best, best_score = module, score
    return best


def _stem(word: str) -> str:
    word = word.casefold()
    for ending in ("ing", "es", "s"):
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            return word[: -len(ending)]
    return word


def _how_to_ask(names: list[str]) -> str:
    """What to say to use these skills, from the examples written in each.

    Asked "how do I use that", reading out function names tells nobody what
    to say. The examples are things a person would say, so they are the
    answer — two per skill, in English, which is enough to show the shape.
    """
    from talos.skills import load

    phrases: list[str] = []
    for name in names:
        examples = [e for e in load().get(name).examples if e.isascii()]
        phrases += [f'"{e}"' for e in examples[:2]]
    if not phrases:
        return ""
    return "You could say " + ", ".join(phrases[:-1]) + f", or {phrases[-1]}."


def _grouped() -> dict[str, list[str]]:
    """Every registered skill, bucketed by the module it came from."""
    from talos.skills import load

    grouped: dict[str, list[str]] = {}
    for name in load().names:
        definition = load().get(name)
        module = definition.function.__module__.rsplit(".", 1)[-1]
        grouped.setdefault(module, []).append(name)
    return grouped


LIMITS = (
    "Three things worth knowing. I am a small model, so I am good at fetching "
    "and poor at judging — check anything that matters. I can only see a few "
    "thousand words at once, so I cannot read a whole source file and reason "
    "about it. And I cannot tell your voice from anyone else's, so in the "
    "seconds after I answer, whatever is said in the room is treated as meant "
    "for me."
)


# One skill for every question about Talos itself. Three of them — what, how,
# limits — were three entries in every prompt for questions asked rarely.
@skill(Permission.OPEN)
def about_me(topic: str = "skills") -> str:
    """Answer questions about Talos itself: what it can do, how it works, what it is bad at.

    Always call this rather than answering from memory. It counts the skills
    actually loaded right now, which is the only accurate answer — you cannot
    see how many there are or what they are grouped into.

    Args:
        topic: "skills" for what Talos can do, an area such as files or Todoist for what to say to use it, "how" for how it works, "limits" for what it is bad at

    Examples:
        what can you do
        how do you work
        what are you bad at
        what can you do with Todoist
        how do I use the file features
        how do I ask you to send a file
        neler yapabilirsin
    """
    wanted = topic.strip().casefold()
    if wanted in {"skills", "skill", "abilities", "everything", "all"}:
        wanted = ""
    if wanted in {"how", "how it works", "how you work", "works"}:
        return HOW_IT_WORKS
    if wanted in {"limits", "limit", "weaknesses", "bad at"}:
        return LIMITS

    grouped = _grouped()

    if wanted:
        module = _area_for(wanted, grouped)
        if module is not None:
            area, detail = AREAS[module]
            return f"For {area} I can help with {detail}. {_how_to_ask(grouped[module])}"
        return (
            f"I have nothing for {topic}. Ask what I can do in general and I will "
            f"tell you the areas I cover."
        )

    covered = [AREAS[m][0] for m in grouped if m in AREAS]
    total = sum(len(v) for v in grouped.values())
    listed = ", ".join(covered[:-1]) + f", and {covered[-1]}" if len(covered) > 1 else covered[0]

    # Deliberately not a list of every name. Spoken, that is noise.
    #
    # The last sentence is for the model, not the owner. Handed only the
    # area names, it filled them in from imagination — "find, copy, move and
    # send files" — and Talos cannot copy or move anything.
    return (
        f"I have {total} skills across {len(covered)} areas: {listed}. "
        f"Ask how to use any one of them and I will tell you what to say. "
        f"You never need to name a skill — say what you want and I will pick. "
        f"(Say these areas as they are. Do not describe what each one does "
        f"beyond its name; that is what asking about one area is for.)"
    )
