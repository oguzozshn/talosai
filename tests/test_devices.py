"""What happens when the microphone you pinned is not there.

Pinning is the right answer to a Windows default that wanders, but a pinned
Bluetooth headset is not a fixture: it is off, or flat, or it has gone to a
phone. Before this, that raised DeviceNotFound out of Recorder.start() — on
the pynput listener thread, where nothing catches it, so taking the headset
off and pressing the key killed the assistant with a traceback.

Falling back is not free either. Quietly listening to a different microphone
is the failure this module was written to prevent, so the fallback has to
announce itself.
"""

from __future__ import annotations

import logging

import pytest

from talos import devices
from talos.devices import DeviceNotFound, resolve, resolve_or_default


@pytest.fixture(autouse=True)
def forget_warnings():
    """The warn-once state is module level and would leak between tests."""
    devices._warned_missing = None
    devices._warned_about = None
    yield
    devices._warned_missing = None
    devices._warned_about = None


class TestStrictResolve:
    def test_nothing_pinned_means_the_system_default(self):
        assert resolve("", kind="input") is None
        assert resolve(None, kind="input") is None

    def test_a_name_that_matches_nothing_is_an_error(self):
        with pytest.raises(DeviceNotFound):
            resolve("no-such-microphone", kind="input")

    def test_an_index_past_the_end_is_an_error(self):
        with pytest.raises(DeviceNotFound):
            resolve(9999, kind="input")


class TestFallingBack:
    def test_a_missing_device_becomes_the_system_default(self):
        assert resolve_or_default("no-such-microphone", kind="input") is None

    def test_it_says_so(self, caplog):
        with caplog.at_level(logging.WARNING, logger="talos.devices"):
            resolve_or_default("no-such-microphone", kind="input")
        assert "no-such-microphone" in caplog.text
        assert "falling back" in caplog.text

    def test_it_says_so_once_and_not_on_every_recording(self, caplog):
        with caplog.at_level(logging.WARNING, logger="talos.devices"):
            for _ in range(5):
                resolve_or_default("no-such-microphone", kind="input")
        assert caplog.text.count("falling back") == 1

    def test_a_second_missing_device_is_still_reported(self, caplog):
        with caplog.at_level(logging.WARNING, logger="talos.devices"):
            resolve_or_default("no-such-microphone", kind="input")
            resolve_or_default("nor-this-one", kind="input")
        assert caplog.text.count("falling back") == 2

    def test_a_device_that_is_there_is_returned_untouched(self, monkeypatch):
        monkeypatch.setattr(devices, "resolve", lambda wanted, *, kind: 7)
        assert resolve_or_default("whatever", kind="input") == 7

    def test_nothing_pinned_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="talos.devices"):
            assert resolve_or_default("", kind="input") is None
        assert caplog.text == ""
