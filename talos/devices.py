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


def _matches(device: dict, wanted: str) -> bool:
    return wanted.casefold() in str(device.get("name", "")).casefold()


def resolve(wanted: str | int | None, *, kind: str) -> int | None:
    """Turn a config value into a device index. None means the system default.

    Args:
        wanted: index, name fragment, or None/"" for the default
        kind: "input" or "output"
    """
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
        log.warning("'%s' matches several %s devices, using the first — %s", wanted, kind, names)
    return candidates[0]


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
