"""Load settings and the character definition.

Two files, with different jobs. `config/settings.toml` is tracked and holds the
defaults; `config/local.toml` is git-ignored and overrides them key by key, so
a machine-specific tweak never shows up as a diff.

The character definition lives in its own markdown file rather than in the TOML
because it is prose that gets edited constantly during tuning, and prose inside
a config value is miserable to edit.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = ROOT / "config" / "settings.toml"
LOCAL_SETTINGS = ROOT / "config" / "local.toml"


@dataclass(frozen=True)
class AssistantConfig:
    name: str = "Talos"
    character_file: str = "config/character.md"
    history_turns: int = 6
    # Whether exchanges are written to data/memory.db and can be searched
    # later. Turning this off makes every conversation end when it ends.
    remember: bool = True
    # Condense exchanges that fall out of the window instead of dropping them.
    # Costs one extra model call each time the window overflows.
    summarise: bool = True
    # Hard bound on those notes. Summarisers concatenate rather than compress,
    # so without a ceiling the note grows a clause per fold until it crowds out
    # the conversation it exists to support.
    summary_chars: int = 600


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    hotkey: str = "f9"
    min_seconds: float = 0.4
    silence_threshold: float = 0.01
    # Device index, or any part of a device name. Empty follows the Windows
    # default, which moves when you plug in a headset — usually what you want,
    # occasionally not.
    input_device: str = ""
    output_device: str = ""


@dataclass(frozen=True)
class WakeConfig:
    word: str = "Talos"
    # Silero's speech probability above which a chunk counts as speech.
    speech_threshold: float = 0.5
    # Silence that means the sentence ended rather than a breath.
    silence_seconds: float = 0.8
    # Shorter than this is a cough, a door, a chair.
    min_seconds: float = 0.5
    # Longer than this is transcribed anyway, so one long monologue cannot
    # stall the listener indefinitely.
    max_seconds: float = 15.0
    # After the name is said on its own, how long the next utterance counts as
    # a command without repeating it.
    follow_up_seconds: float = 8.0
    # Listen while answering, so "Talos" or "stop" said over a reply cuts it
    # off. Talos's own voice in the microphone is told apart by comparing it
    # with what it is saying; with speakers rather than a headset, turn this
    # off if it ever stops itself.
    interruptible: bool = True
    # The same, after Talos has answered — so a reply that ends in a question
    # can simply be answered. Set to 0 to require the name every time, which
    # is safer in a room with a television and no speaker verification.
    continue_seconds: float = 10.0
    # Spoken when the name is heard alone, so there is some sign it is
    # listening. Silence is indistinguishable from not having heard.
    acknowledgement: str = "Yes?"
    # How long to keep discarding microphone input after Talos finishes
    # speaking, so the tail of its own voice is not heard as a command.
    settle_seconds: float = 0.25


@dataclass(frozen=True)
class SttConfig:
    model: str = "large-v3-turbo"
    # A language code pins the decoder; empty lets it detect one per utterance.
    language: str = "en"
    beam_size: int = 1
    vad_filter: bool = True
    # Words Whisper should expect. Technical vocabulary is exactly what it gets
    # wrong — "serialization" came back as "sterilization" — and telling the
    # decoder to expect a word works far better than correcting it afterwards.
    vocabulary: tuple[str, ...] = ()
    # Add the project folder names too, so repositories are transcribed as
    # themselves rather than as whatever English word they resemble.
    include_project_names: bool = True


@dataclass(frozen=True)
class LlmConfig:
    model: str = "qwen3:8b"
    think: bool = False
    keep_alive: str = "30m"
    context: int = 4096
    temperature: float = 0.7
    # Runaway guard only. Length is the character definition's job; a cap tight
    # enough to enforce brevity truncates real answers mid-sentence.
    max_tokens: int = 400
    # Say something before a skill runs. A skill call costs a full extra round
    # trip — about nine seconds — and silence that long sounds like a crash.
    acknowledge_skills: bool = True


@dataclass(frozen=True)
class SkillsConfig:
    enabled: bool = True
    # The directories skills may read. Everything outside is refused. A list
    # because projects do not all live in one folder.
    workspaces: tuple[str, ...] = ()
    # Folders skills may find and read in, beyond the projects. Kept apart
    # from workspaces so that a folder of documents is not listed as projects.
    folders: tuple[str, ...] = ()
    # How many times the model may call a skill and be handed the result
    # before it has to answer. Without a ceiling, a model that misreads a
    # result can call the same skill forever.
    max_rounds: int = 3
    # Offer the model only the skills closest to each request, plus a few
    # always-on ones, instead of all of them. See talos/router.py.
    router: bool = True
    # How many of the closest skills to offer.
    router_top: int = 6


@dataclass(frozen=True)
class ProfileConfig:
    """Learning facts about the owner — proposed by Talos, promoted by hand."""

    # Whether the extraction pass runs at all. It only ever runs when asked,
    # from `run main.py --profile`; nothing happens during a conversation.
    learn: bool = True
    # Hard bound on the profile body, in characters. The body is 828 today and
    # every character of it is in the prompt of every single reply, competing
    # with the conversation for room. Over this, promoting a new fact means
    # dropping an old one — which is the point, because the hard part of a
    # profile that learns is not adding, it is pruning.
    max_chars: int = 1400
    # Exchanges per extraction call, sized against llm.context.
    batch: int = 10


@dataclass(frozen=True)
class UsageConfig:
    # Record which application has focus, so it can be asked about later.
    # Only the executable name is stored, never window titles, and only while
    # Talos is running. Turn it off and nothing is written at all.
    track: bool = True


@dataclass(frozen=True)
class TodoistConfig:
    # Leave empty here. The token belongs in config/local.toml, which is
    # git-ignored, or in TODOIST_API_TOKEN. Nothing writes it — Talos only
    # reads whichever you filled in.
    token: str = ""


@dataclass(frozen=True)
class TtsConfig:
    engine: str = "piper"
    # Name fragment of the voice file to use. Empty takes whatever is present.
    voice: str = ""
    # The voice for sentences in Turkish. Empty reads everything in `voice`.
    turkish_voice: str = "tr_TR"
    length_scale: float = 1.0
    noise_w: float = 0.8
    noise_scale: float = 0.667
    sentence_silence: float = 0.15


@dataclass(frozen=True)
class ProactiveConfig:
    """Speaking without being asked. See talos/proactive.py."""

    enabled: bool = True
    # The morning brief: given the first time Talos is running between these.
    brief: bool = True
    brief_time: str = "08:30"
    brief_until: str = "12:00"
    # How long before something with a clock time to remind about it.
    remind_minutes: int = 10
    # Nothing is said in this window, whatever is due.
    quiet_from: str = "23:00"
    quiet_until: str = "08:00"
    # Where it is said: out loud, on Telegram, or both.
    speak: bool = True
    telegram: bool = True


@dataclass(frozen=True)
class StudyConfig:
    # One folder per course inside, holding its slides, notes and books.
    # Empty turns the course search and glossary off.
    folder: str = ""


@dataclass(frozen=True)
class TelegramConfig:
    # Leave empty here, like the Todoist token: it goes in config/local.toml
    # or TALOS_TELEGRAM_TOKEN. With no token, Telegram is simply off.
    token: str = ""
    # The only chats Talos answers. Everyone else is ignored without a reply,
    # and their chat id is printed so the owner can add their own.
    allowed_chat_ids: tuple[int, ...] = ()
    # How long a yes/no button waits before it counts as no.
    confirm_seconds: int = 120
    # The Bot API refuses uploads over 50 MB.
    max_file_mb: int = 50


@dataclass(frozen=True)
class Config:
    assistant: AssistantConfig = field(default_factory=AssistantConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    wake: WakeConfig = field(default_factory=WakeConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    todoist: TodoistConfig = field(default_factory=TodoistConfig)
    usage: UsageConfig = field(default_factory=UsageConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    study: StudyConfig = field(default_factory=StudyConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)

    @property
    def character(self) -> str:
        """The system prompt, read fresh from disk.

        Not cached: during tuning the character gets edited between runs, and
        a stale prompt is a confusing thing to debug.
        """
        path = ROOT / self.assistant.character_file
        if not path.exists():
            return f"You are {self.assistant.name}, a local voice assistant."

        text = path.read_text(encoding="utf-8")
        # The file is written for humans too, with a title and an explanatory
        # preamble above a horizontal rule. Everything below the rule is the
        # prompt itself.
        _, separator, body = text.partition("\n---\n")
        return (body if separator else text).strip()


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _section(data: dict[str, Any], name: str, cls: type) -> Any:
    """Build one config dataclass, ignoring keys it does not define.

    Unknown keys are skipped rather than raising: a settings file left over
    from a later version should degrade to defaults, not refuse to start.
    """
    values = data.get(name, {})
    known = {f for f in cls.__dataclass_fields__}
    return cls(**{k: v for k, v in values.items() if k in known})


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge one level deep — sections merge, keys within them replace."""
    merged = dict(base)
    for section, values in override.items():
        if isinstance(values, dict) and isinstance(merged.get(section), dict):
            merged[section] = {**merged[section], **values}
        else:
            merged[section] = values
    return merged


def load() -> Config:
    data = _merge(_read(SETTINGS), _read(LOCAL_SETTINGS))
    return Config(
        assistant=_section(data, "assistant", AssistantConfig),
        audio=_section(data, "audio", AudioConfig),
        wake=_section(data, "wake", WakeConfig),
        stt=_section(data, "stt", SttConfig),
        llm=_section(data, "llm", LlmConfig),
        tts=_section(data, "tts", TtsConfig),
        skills=_section(data, "skills", SkillsConfig),
        todoist=_section(data, "todoist", TodoistConfig),
        usage=_section(data, "usage", UsageConfig),
        profile=_section(data, "profile", ProfileConfig),
        telegram=_section(data, "telegram", TelegramConfig),
        study=_section(data, "study", StudyConfig),
        proactive=_section(data, "proactive", ProactiveConfig),
    )
