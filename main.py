"""Talos — push to talk.

    run main.py                  hold the hotkey, speak, release
    run main.py --text           type instead of speaking
    run main.py --say "merhaba"  one exchange, then exit
    run main.py --mute           run the whole chain but stay silent

--mute exists so the chain can be exercised without filling the room with
speech, which matters more than it sounds when the machine is next to someone
asleep.
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

log = logging.getLogger("talos")


def resolve_hotkey(name: str):
    from pynput import keyboard

    key = getattr(keyboard.Key, name.lower(), None)
    if key is None:
        raise SystemExit(
            f"Unknown hotkey '{name}'. Use a named key such as f9, ctrl_r, or alt_r."
        )
    return key


def console_confirm(skill_name: str, permission) -> bool:
    """Ask at the terminal before a skill that changes something runs.

    A stopgap. The point of the permission tiers is that the answer comes by
    voice, and for the OWNER tier that the voice is verified — neither of which
    exists until phase 2. Asking at the keyboard is honest in the meantime:
    it does block the action, it just blocks it in the wrong place.
    """
    print(f"\n  '{skill_name}' needs permission ({permission.value}).")
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

    from talos.devices import describe, resolve

    assistant = Assistant(config, mute=mute, confirm=console_confirm)
    print(f"  mic       {describe(resolve(config.audio.input_device, kind='input'), 'input')}")
    print(f"  speaker   {describe(resolve(config.audio.output_device, kind='output'), 'output')}")
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


def run_text_mode(assistant: Assistant) -> int:
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
        print(f"  [{turn.timings.summary()}]\n")


def run_push_to_talk(assistant: Assistant, config) -> int:
    from pynput import keyboard

    hotkey = resolve_hotkey(config.audio.hotkey)
    recorder = Recorder(config.audio)
    busy = threading.Lock()

    print(f"Hold {config.audio.hotkey.upper()} to talk. Ctrl+C to quit.\n")

    def handle(audio) -> None:
        # Runs off the listener thread: a turn takes seconds, and blocking the
        # listener would swallow the next key press.
        try:
            if not recorder.is_speech(audio):
                print("  (nothing heard)\n")
                return
            turn = assistant.respond_to_audio(audio)
            if turn is None:
                print("  (nothing recognised)\n")
                return
            print(f"  [{turn.timings.summary()}]\n")
        except Exception:  # noqa: BLE001 — one bad turn must not kill the loop
            log.exception("turn failed")
        finally:
            busy.release()

    def on_press(key) -> None:
        if key == hotkey and not recorder.recording and not busy.locked():
            recorder.start()
            print("listening...")

    def on_release(key) -> None:
        if key != hotkey or not recorder.recording:
            return
        audio = recorder.stop()
        busy.acquire()
        threading.Thread(target=handle, args=(audio,), daemon=True).start()

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            pass
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", action="store_true", help="type instead of speaking")
    parser.add_argument("--say", help="run one exchange with this text, then exit")
    parser.add_argument("--mute", action="store_true", help="run the chain but do not play audio")
    parser.add_argument("--devices", action="store_true", help="list audio devices and exit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.devices:
        from talos.devices import listing

        print(listing())
        return 0

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load()
    assistant = start_up(config, mute=args.mute)

    if args.say:
        turn = assistant.respond_to_text(args.say)
        print(f"  [{turn.timings.summary()}]")
        return 0
    if args.text:
        return run_text_mode(assistant)
    return run_push_to_talk(assistant, config)


if __name__ == "__main__":
    raise SystemExit(main())
