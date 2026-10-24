"""Talos — push to talk.

    run main.py                  hold the hotkey, speak, release
    run main.py --text           type instead of speaking
    run main.py --telegram       answer only on Telegram, no microphone
    run main.py --index-courses  index the course folder now, then exit
    run main.py --say "merhaba"  one exchange, then exit
    run main.py --mute           run the whole chain but stay silent
    run main.py --diagnostic     also print what was heard, which skills ran,
                                  and what they returned, after every turn

--mute exists so the chain can be exercised without filling the room with
speech, which matters more than it sounds when the machine is next to someone
asleep. --diagnostic is the equivalent for "why did it do that" — it does not
touch logging.basicConfig, so it works whether or not --verbose is also on.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from talos.assistant import Assistant  # noqa: E402
from talos.audio import Recorder  # noqa: E402
from talos.config import load  # noqa: E402
from talos.session import current as session  # noqa: E402

log = logging.getLogger("talos")

# Where --diagnostic leaves audio that produced no reply. One file,
# overwritten each time: it exists to be re-run through the decoder, not to
# accumulate.
UNRECOGNISED_AUDIO = Path(__file__).resolve().parent / "data" / "unrecognised.wav"


def resolve_hotkey(name: str):
    from pynput import keyboard

    key = getattr(keyboard.Key, name.lower(), None)
    if key is None:
        raise SystemExit(
            f"Unknown hotkey '{name}'. Use a named key such as f9, ctrl_r, or alt_r."
        )
    return key


def show_memory(erase: bool = False) -> int:
    """Read the memory from outside the assistant.

    Whether Talos remembered something is otherwise only visible through its
    own answer, which is exactly the thing being tested — so there needs to be
    a way to look at the record directly.
    """
    from talos.memory import DATABASE, Memory, load_profile

    memory = Memory()

    if erase:
        removed = memory.forget_all()
        print(f"Erased {removed} exchange(s) from {DATABASE}")
        return 0

    profile = load_profile()
    print("Profile")
    print("-------")
    print(profile if profile else "  (none — config/profile.md is empty or missing)")

    total = memory.count()
    print(f"\n\nRemembered exchanges: {total}")
    print(f"Database: {DATABASE}")
    print(f"Semantic search: {'available' if memory._vec_available else 'unavailable'}")

    if total:
        print("\nMost recent")
        print("-----------")
        for exchange in memory.recent(10):
            print(f"  {exchange.summary()}")
    return 0


def review_profile(config) -> int:
    """Propose facts about the owner, and let the owner decide.

    Talos never writes config/profile.md. It goes into every prompt unchanged,
    so a wrong fact there is invisible and permanent, and a profile that grows
    on its own gets slower and worse with nothing to show for it. Extraction
    proposes; this is where a person promotes.

    Deliberately does not build an Assistant: this needs Ollama and SQLite,
    not Whisper and Piper.
    """
    from talos.memory import Memory
    from talos.profile import Candidates, drop, extract, learned, promote, size

    budget = config.profile.max_chars
    memory = Memory()
    candidates = Candidates(memory)

    print(f"Profile: {size()} of {budget} characters, in the prompt of every reply.")
    print(f"Log: {memory.count()} exchange(s), scanned to {candidates.scanned_to}.")
    print()

    kept = extract(memory, candidates, config.llm, config.profile, report=print)
    print()
    print(f"{kept} new candidate(s).")
    print()

    pending = candidates.pending()
    if not pending:
        print("Nothing to review.")
        return 0

    for candidate in pending:
        print(f"  {candidate.fact}")
        print(f'    from [{candidate.proposed:%d %B}] "{candidate.quote}"')
        try:
            answer = input("    [a]ccept  [r]eject  [s]kip  [q]uit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if answer.startswith("q"):
            break
        if answer.startswith("r"):
            candidates.reject(candidate.id)
            print("    rejected — it will not be proposed again.")
            print()
            continue
        if not answer.startswith("a"):
            print("    skipped.")
            print()
            continue

        if promote(candidate.fact, budget):
            candidates.accept(candidate.id)
            print(f"    added. Profile is now {size()} of {budget} characters.")
            print()
            continue

        # Over budget, which is the interesting case: the hard part of a
        # profile that learns is not adding, it is choosing what to lose.
        print(f"    Would not fit — the profile is {size()} of {budget} characters.")
        existing = learned()
        if not existing:
            print("    Nothing promoted to drop. Trim config/profile.md by hand.")
            print()
            continue
        for number, fact in enumerate(existing, start=1):
            print(f"      {number}. {fact}")
        try:
            choice = input("    Drop which, to make room? [number, enter to skip]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not choice.isdigit() or not 1 <= int(choice) <= len(existing):
            print("    skipped.")
            print()
            continue
        drop(existing[int(choice) - 1])
        if promote(candidate.fact, budget):
            candidates.accept(candidate.id)
            print(f"    swapped. Profile is now {size()} of {budget} characters.")
        else:
            print("    still would not fit.")
        print()

    print(f"Profile: {size()} of {budget} characters.")
    return 0


def describe_action(skill_name: str, arguments: dict) -> str:
    """What is about to happen, in one line, for a confirmation prompt."""
    if not arguments:
        return skill_name
    parts = []
    for key, value in arguments.items():
        text = str(value)
        if len(text) > 60:
            text = text[:57] + "..."
        parts.append(f"{key}={text!r}")
    return f"{skill_name}  {'  '.join(parts)}"


def console_confirm(skill_name: str, permission, arguments: dict) -> bool:
    """Ask at the terminal before a skill that changes something runs."""
    print(f"\n  Permission needed ({permission.value}):")
    print(f"    {describe_action(skill_name, arguments)}")
    try:
        answer = input("  Allow? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"y", "yes", "e", "evet"}


def start_up(config, mute: bool) -> Assistant:
    print("Talos")
    print("=====\n")
    print("Warming up — this is the cold-start cost, paid once so the first")
    print("sentence does not pay it.\n")

    from talos.devices import describe, probe_input, resolve_or_default

    assistant = Assistant(config, mute=mute, confirm=console_confirm)
    mic, fell_back = probe_input(config.audio.input_device)
    unavailable = f" — '{config.audio.input_device}' unavailable" if fell_back else ""
    print(f"  mic       {describe(mic, 'input')}{unavailable}")
    print(f"  speaker   {describe(resolve_or_default(config.audio.output_device, kind='output'), 'output')}")
    print(f"  speech    {config.stt.model} on {assistant.ears.device}")
    print(f"  reasoning {config.llm.model}")
    if assistant.voice is not None:
        print(f"  voice     {assistant.voice.voice_name}")
    else:
        print("  voice     unavailable — replies will be printed only")
    if assistant.skills:
        print(f"  skills    {len(assistant.skills)}: {', '.join(assistant.skills.names)}")
    print()

    assistant.warm_up()
    print("Ready.\n")
    return assistant


def run_text_mode(assistant: Assistant, diagnostic: bool = False) -> int:
    print("Type to talk. Empty line or Ctrl+C to quit.\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            return 0
        turn = assistant.respond_to_text(text)
        if diagnostic:
            print(turn.diagnostic())
        print(f"  [{turn.timings.summary()}]\n")


def run_push_to_talk(assistant: Assistant, config, diagnostic: bool = False) -> int:
    from pynput import keyboard

    hotkey = resolve_hotkey(config.audio.hotkey)
    # A microphone change is found as the key goes down, so it is printed then
    # and spoken after the reply: said at once, it would be recorded as the
    # start of the question.
    unspoken: list[str] = []

    def microphone_changed(message: str) -> None:
        print(f"  {message}")
        unspoken.append(message)

    recorder = Recorder(config.audio, announce=microphone_changed)
    busy = threading.Lock()
    keyboard_listener = None

    print(f"Hold {config.audio.hotkey.upper()} to talk. Ctrl+C to quit.")
    print("Say start listening, while holding the key, to switch to the wake word.\n")

    def dead_end(audio, reason: str) -> None:
        """A turn that produced no reply, and everything known about why.

        These two paths are where --diagnostic is needed most and where it
        said nothing at all: both return before a Turn exists, so
        Turn.diagnostic() never runs. A muted microphone, a headset handing
        over unusable audio, and a decoder pinned to the wrong language are
        three different faults that all print the same one line here.
        """
        print(f"  {reason}")
        if diagnostic:
            print(f"  audio      {recorder.describe(audio)}")
            saved = recorder.save(audio, UNRECOGNISED_AUDIO)
            if saved is not None:
                print(f"  saved      {saved}")
        print()

    def handle(audio) -> None:
        # Runs off the listener thread: a turn takes seconds, and blocking the
        # listener would swallow the next key press.
        try:
            if not recorder.is_speech(audio):
                dead_end(audio, "(nothing heard)")
                return
            turn = assistant.respond_to_audio(audio)
            if turn is None:
                # Past the gate, so the audio was loud enough and long enough
                # and Whisper still returned nothing. That is a decode problem
                # rather than a level problem, and the saved file is the only
                # way to tell which.
                dead_end(audio, "(nothing recognised)")
                return
            if diagnostic:
                print(turn.diagnostic())
            print(f"  [{turn.timings.summary()}]\n")
        except Exception:  # noqa: BLE001 — one bad turn must not kill the loop
            log.exception("turn failed")
        finally:
            while unspoken:
                message = unspoken.pop(0)
                if assistant.voice is not None:
                    assistant.voice.say(message)
            busy.release()
            # Checked after the turn rather than during it: the skill that
            # asks for the switch runs inside the turn, and stopping the
            # listener underneath it would cut the reply off mid-sentence.
            if session.switch_requested and keyboard_listener is not None:
                keyboard_listener.stop()

    def on_press(key) -> None:
        if key == hotkey and not recorder.recording and not busy.locked():
            try:
                recorder.start()
            except Exception as exc:  # noqa: BLE001 — this is the listener thread
                # Not even the default microphone opened. Raising here would
                # kill the key listener, and with it the assistant; the next
                # press tries again, by which time one may be plugged in.
                print(f"  no microphone available: {exc}\n")
                return
            print("listening...")

    def on_release(key) -> None:
        if key != hotkey or not recorder.recording:
            return
        audio = recorder.stop()
        busy.acquire()
        threading.Thread(target=handle, args=(audio,), daemon=True).start()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        keyboard_listener = listener
        try:
            listener.join()
        except KeyboardInterrupt:
            return 0
    # Reached either because the key listener was stopped for a mode switch,
    # or because it died. Either way the caller decides what happens next.
    return 0


_AGREEMENT = {"yes", "yeah", "yep", "sure", "ok", "okay", "go", "do", "please", "confirm", "evet"}
_REFUSAL = {"no", "nope", "don't", "dont", "stop", "cancel", "wait", "hayir", "hayır"}


def spoken_confirm(assistant: Assistant, listener, skill_name: str, permission, arguments: dict) -> bool:
    """Ask out loud, and listen for the answer.

    A confirmation that can only be given at the keyboard is not a
    confirmation for someone standing across the room, which is the whole
    situation this assistant exists for.
    """
    question = f"I'm about to run {describe_action(skill_name, arguments)}. Should I?"
    print(f"\n  Permission needed ({permission.value}):")
    print(f"    {describe_action(skill_name, arguments)}")

    if assistant.voice is not None:
        assistant.voice.say(question)

    answer = listener.ask()
    print(f"    you said: {answer or '(nothing)'}")

    words = {w.strip(".,!?").lower() for w in answer.split()}
    if words & _REFUSAL:
        return False
    if words & _AGREEMENT:
        return True

    # Anything else, including silence, is a no. An assistant that reads
    # ambiguity as consent is one you cannot leave running.
    if assistant.voice is not None:
        assistant.voice.say("I'll leave it.")
    return False


def run_wake_word(assistant: Assistant, config, diagnostic: bool = False) -> int:
    from talos.wake import WakeListener

    def announce(message: str) -> None:
        # Spoken as well as printed: in this mode nobody is looking at the
        # screen, and a microphone that changed without a word is the kind of
        # fault that looks like the assistant ignoring you.
        print(f"  {message}")
        if assistant.voice is not None:
            assistant.voice.say(message)

    listener = WakeListener(
        config.audio, config.wake, assistant.ears.transcribe, announce=announce
    )
    assistant.hold_microphone = listener.hold
    # Rebound now that the listener exists: confirmations are spoken and heard
    # through the same stream the wake word uses.
    assistant.set_confirm(
        lambda name, permission, arguments: spoken_confirm(
            assistant, listener, name, permission, arguments
        )
    )
    print(f"Say \"{config.wake.word}\" to wake me. Ctrl+C to quit.")
    print("Say stop listening to switch to the hotkey.\n")

    try:
        for command, _audio in listener.listen():
            if not command:
                # The name on its own. Say something: on screen there is a
                # line to read, but the person is looking at neither the
                # screen nor a microphone light, and silence reads as failure.
                print(f"  {config.wake.acknowledgement}")
                if assistant.voice is not None:
                    assistant.voice.say(config.wake.acknowledgement)
                continue
            print(f"you> {command}")
            turn = assistant.respond_to_text(command)
            if diagnostic:
                print(turn.diagnostic())
            print(f"  [{turn.timings.summary()}]")
            # Say so, because the window is invisible otherwise and the whole
            # point of it is that you can just keep talking.
            if config.wake.continue_seconds > 0:
                print(f"  (still listening for {config.wake.continue_seconds:.0f}s)\n")
            else:
                print()

            # After the reply has been spoken, not during it: the skill that
            # asks to stop listening runs inside the turn, and closing the
            # stream there would cut off the sentence saying so.
            if session.switch_requested:
                listener.stop()
                break
    except KeyboardInterrupt:
        listener.stop()
        print()
    assistant.hold_microphone = None
    return 0


def index_courses(library) -> int:
    if library is None:
        print("No course folder. Set [study] folder in config/settings.toml.")
        return 1
    print(f"Indexing {library.folder}. Only new and changed files are read.\n")
    read, added = library.update(progress=lambda name, count: print(f"  {name}: {count} passages"))
    print(f"\n{read} file(s) read, {added} passages added.")
    return 0


def run_telegram_only() -> int:
    """Nothing to do here but wait: the bot answers on its own thread."""
    print("Answering on Telegram only. Ctrl+C to quit.\n")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wake", action="store_true", help="listen for the wake word")
    parser.add_argument("--text", action="store_true", help="type instead of speaking")
    parser.add_argument(
        "--telegram", action="store_true", help="answer only on Telegram, with no microphone"
    )
    parser.add_argument("--say", help="run one exchange with this text, then exit")
    parser.add_argument("--mute", action="store_true", help="run the chain but do not play audio")
    parser.add_argument("--devices", action="store_true", help="list audio devices and exit")
    parser.add_argument("--memory", action="store_true", help="show what is remembered and exit")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="review facts Talos proposes about you, and promote the true ones",
    )
    parser.add_argument("--forget", action="store_true", help="erase everything remembered")
    parser.add_argument(
        "--index-courses", action="store_true", help="index the course folder now, showing progress, and exit"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="print what was heard, which skills ran, and what they returned",
    )
    args = parser.parse_args()

    # Line by line even when stdout is a pipe. Started in the background, as
    # a service or by another program, Python buffers stdout in blocks, and
    # "Ready." sits in the buffer while whoever started it concludes nothing
    # is happening — or that it is running when it has already crashed.
    sys.stdout.reconfigure(line_buffering=True)

    if args.devices:
        from talos.devices import listing

        print(listing())
        return 0

    if args.memory or args.forget:
        return show_memory(erase=args.forget)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    # --verbose is for watching Talos, not its dependencies. At DEBUG these
    # libraries emit every HTTP header and every decoding attempt, which buries
    # the one line you were trying to read.
    for noisy in ("httpx", "httpcore", "faster_whisper", "urllib3", "speechbrain", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    config = load()

    if args.profile:
        return review_profile(config)

    from talos import study

    library = study.open_library(config.study.folder)
    if args.index_courses:
        return index_courses(library)

    # Started before the assistant, so the minute spent warming models is
    # itself recorded. It only counts while Talos runs, and every answer
    # built on it says so.
    if config.usage.track:
        from talos.usage import Sampler

        Sampler().start()

    assistant = start_up(config, mute=args.mute)
    # After warming up, so the first file read does not compete with it.
    if library is not None and not args.say:
        study.index_in_background(library)

    # Alongside whichever mode runs, not instead of it: the phone and the
    # microphone are two ways into the same assistant.
    if not args.say:
        from talos import telegram

        server = telegram.start(config.telegram, assistant, describe_action, config.audio.sample_rate)
        # After Telegram, so a reminder can be sent there as well as said.
        from talos import proactive

        # Answering only on Telegram means nobody is at the machine, so
        # nothing is said out loud there.
        proactive.start(assistant, speak=not args.telegram)
        if args.telegram:
            if server is None:
                print("Telegram is not set up. USING.md says how.")
                return 1
            return run_telegram_only()

    if args.say:
        turn = assistant.respond_to_text(args.say)
        if args.diagnostic:
            print(turn.diagnostic())
        print(f"  [{turn.timings.summary()}]")
        return 0
    if args.text:
        return run_text_mode(assistant, diagnostic=args.diagnostic)

    # The voice modes can hand over to each other while running, so which one
    # is active lives in the session rather than in this function.
    session.mode = "wake" if args.wake else "hotkey"
    while True:
        session.switch_requested = False
        if session.mode == "wake":
            run_wake_word(assistant, config, diagnostic=args.diagnostic)
        else:
            run_push_to_talk(assistant, config, diagnostic=args.diagnostic)

        if not session.switch_requested:
            return 0
        print(f"\n--- switching to {session.mode} ---\n")


if __name__ == "__main__":
    raise SystemExit(main())
