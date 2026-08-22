"""Choosing which microphone and which speaker to use.

By default everything follows the Windows default device, which is usually
what you want — plug in a headset and the assistant moves with it. But the
default also moves when you did not mean it to, and an assistant that starts
listening to the wrong microphone gives no clue why it stopped hearing you.

So a device can be pinned in `config/settings.toml`, by index or by any part
of its name. Names are matched rather than required to be exact, because the
full string Windows reports is long and changes between driver versions.
"""

from __future__ import annotations

import logging

import sounddevice as sd

log = logging.getLogger(__name__)


class DeviceNotFound(ValueError):
    pass


# The last ambiguous match warned about, so it is not repeated verbatim.
_warned_about: tuple | None = None

# The last pinned device reported missing, for the same reason.
_warned_missing: tuple | None = None


def refresh() -> None:
    """Re-read the device list after something was plugged in or pulled out.

    PortAudio enumerates devices when it initialises and then caches that
    list. Unplug a headset and it keeps reporting a device that is no longer
    there, and keeps resolving "the default" to it — so an assistant that
    followed the default at startup goes deaf, and reopening the stream is not
    enough to fix it because the answer it gets is stale.

    Terminating and initialising again is the documented way to make it look
    afresh. Any stream still open at that point is invalid, so callers close
    theirs first.
    """
    try:
        sd._terminate()
        sd._initialize()
    except Exception as exc:  # noqa: BLE001 — worst case the list stays stale
        log.debug("could not reinitialise the audio backend: %s", exc)


def _matches(device: dict, wanted: str) -> bool:
    return wanted.casefold() in str(device.get("name", "")).casefold()


def resolve(wanted: str | int | None, *, kind: str) -> int | None:
    """Turn a config value into a device index. None means the system default.

    Args:
        wanted: index, name fragment, or None/"" for the default
        kind: "input" or "output"
    """
    global _warned_about

    if wanted is None or wanted == "":
        return None

    channels_key = f"max_{kind}_channels"
    devices = sd.query_devices()

    if isinstance(wanted, int) or str(wanted).isdigit():
        index = int(wanted)
        if not 0 <= index < len(devices):
            raise DeviceNotFound(f"No device at index {index}. Run: run main.py --devices")
        if devices[index][channels_key] < 1:
            raise DeviceNotFound(f"Device {index} ({devices[index]['name']}) has no {kind}.")
        return index

    candidates = [
        i for i, d in enumerate(devices) if d[channels_key] >= 1 and _matches(d, str(wanted))
    ]
    if not candidates:
        raise DeviceNotFound(
            f"No {kind} device matching '{wanted}'. Run: run main.py --devices"
        )
    if len(candidates) > 1:
        names = ", ".join(f"{i}: {devices[i]['name']}" for i in candidates[:4])
        # Once per ambiguity, not once per recording. resolve() runs again on
        # every key press, and a warning that repeats on every key press is
        # one you stop reading — including the turn where it mattered.
        seen = (str(wanted), kind, tuple(candidates))
        if seen != _warned_about:
            log.warning(
                "'%s' matches several %s devices, using the first — %s", wanted, kind, names
            )
            _warned_about = seen
    return candidates[0]


def resolve_or_default(wanted: str | int | None, *, kind: str) -> int | None:
    """The pinned device, or the system default when it is not there.

    A pinned headset is not a fixture. It is off, or flat, or it has gone
    to a phone, and none of that is a reason for the assistant to refuse to
    start or to die on the next key press — which is what a DeviceNotFound
    out of Recorder.start() does, on the listener thread, where nothing
    catches it.

    So it falls back. But quietly falling back to a different microphone is
    the exact failure this module was written to prevent, so it says so —
    once per device that goes missing, because the alternative is a warning
    on every recording.
    """
    global _warned_missing

    try:
        return resolve(wanted, kind=kind)
    except DeviceNotFound as exc:
        if _warned_missing != (wanted, kind):
            log.warning("%s — falling back to the system default %s", exc, kind)
            _warned_missing = (wanted, kind)
        return None


def describe(index: int | None, kind: str) -> str:
    devices = sd.query_devices()
    if index is None:
        default = sd.default.device[0 if kind == "input" else 1]
        try:
            return f"{devices[default]['name']}  (system default)"
        except (IndexError, TypeError):
            return "system default"
    return str(devices[index]["name"])


def listing() -> str:
    """Every device, marked up for the settings file."""
    devices = sd.query_devices()
    default_in, default_out = sd.default.device

    lines = ["Audio devices", "=" * 13, ""]
    for index, device in enumerate(devices):
        directions = []
        if device["max_input_channels"] > 0:
            directions.append("in")
        if device["max_output_channels"] > 0:
            directions.append("out")
        if not directions:
            continue

        marks = []
        if index == default_in:
            marks.append("default in")
        if index == default_out:
            marks.append("default out")
        suffix = f"   <- {', '.join(marks)}" if marks else ""
        lines.append(f"  {index:>3}  [{'/'.join(directions):<7}] {device['name']}{suffix}")

    lines += [
        "",
        "To pin one, put the index or any part of the name in config/settings.toml:",
        "",
        "  [audio]",
        '  input_device = "Realtek"',
        "  output_device = 5",
        "",
        "Leave them empty to follow the Windows default.",
    ]
    return "\n".join(lines)
