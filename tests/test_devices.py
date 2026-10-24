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


class FakeStream:
    """Stands in for sd.InputStream: opens, or refuses the way PortAudio does."""

    opened: list = []
    refuse: set = set()

    def __init__(self, device=None, **_kwargs):
        if device in FakeStream.refuse:
            raise devices.sd.PortAudioError("Unanticipated host error (GLE=0x48F)")
        self.device = device
        FakeStream.opened.append(device)

    def start(self):
        pass

    def close(self):
        pass


@pytest.fixture
def fake_streams(monkeypatch):
    FakeStream.opened = []
    FakeStream.refuse = set()
    monkeypatch.setattr(devices.sd, "InputStream", FakeStream)
    monkeypatch.setattr(devices, "describe", lambda index, kind: "Laptop mic  (system default)")
    return FakeStream


class TestInputSelector:
    """The pinned microphone is preferred, not required."""

    def test_the_pinned_device_is_used_when_it_opens(self, fake_streams, monkeypatch):
        monkeypatch.setattr(devices, "resolve", lambda wanted, *, kind: 32)
        selector = devices.InputSelector("SOUNDPEATS")
        selector.open()
        assert fake_streams.opened == [32]
        assert selector.device == 32
        assert not selector.fallen_back

    def test_listed_but_not_connected_falls_back(self, fake_streams, monkeypatch):
        # The case that crashed wake mode: a paired headset in its case is
        # still listed, so resolve() finds it, and the open is what fails.
        monkeypatch.setattr(devices, "resolve", lambda wanted, *, kind: 32)
        fake_streams.refuse = {32}
        selector = devices.InputSelector("SOUNDPEATS")
        selector.open()
        assert fake_streams.opened == [None]
        assert selector.device is None
        assert selector.fallen_back

    def test_not_listed_at_all_falls_back(self, fake_streams):
        selector = devices.InputSelector("no-such-microphone")
        selector.open()
        assert fake_streams.opened == [None]
        assert selector.fallen_back

    def test_the_change_is_announced_once_each_way(self, fake_streams, monkeypatch):
        monkeypatch.setattr(devices, "resolve", lambda wanted, *, kind: 32)
        said: list[str] = []
        selector = devices.InputSelector("SOUNDPEATS", said.append)

        fake_streams.refuse = {32}
        for _ in range(3):
            selector.open()
        assert len(said) == 1
        assert "default" in said[0]

        fake_streams.refuse = set()
        for _ in range(3):
            selector.open()
        assert len(said) == 2
        assert "back" in said[1]
        assert selector.device == 32

    def test_nothing_pinned_is_not_a_fallback(self, fake_streams):
        said: list[str] = []
        selector = devices.InputSelector("", said.append)
        selector.open()
        assert fake_streams.opened == [None]
        assert not selector.fallen_back
        assert said == []

    def test_no_working_microphone_at_all_is_raised(self, fake_streams, monkeypatch):
        monkeypatch.setattr(devices, "resolve", lambda wanted, *, kind: 32)
        fake_streams.refuse = {32, None}
        with pytest.raises(devices.sd.PortAudioError):
            devices.InputSelector("SOUNDPEATS").open()


class TestHostApiPreference:
    """WDM-KS lists headsets that are off; the others do not."""

    def test_a_wdm_ks_entry_loses_to_the_same_device_on_mme(self, monkeypatch):
        listed = [
            {"name": "Headset (SOUNDPEATS)", "hostapi": 3, "max_input_channels": 1},
            {"name": "Headset (SOUNDPEATS)", "hostapi": 0, "max_input_channels": 1},
        ]
        apis = ["MME", "Windows DirectSound", "Windows WASAPI", "Windows WDM-KS"]
        monkeypatch.setattr(devices.sd, "query_devices", lambda: listed)
        monkeypatch.setattr(devices.sd, "query_hostapis", lambda i: {"name": apis[i]})
        assert resolve("SOUNDPEATS", kind="input") == 1

    def test_a_wdm_ks_only_match_is_still_returned(self, monkeypatch):
        # Tried, and when it will not open, InputSelector falls back.
        listed = [{"name": "Headset (SOUNDPEATS)", "hostapi": 3, "max_input_channels": 1}]
        monkeypatch.setattr(devices.sd, "query_devices", lambda: listed)
        monkeypatch.setattr(devices.sd, "query_hostapis", lambda i: {"name": "Windows WDM-KS"})
        assert resolve("SOUNDPEATS", kind="input") == 0


class TestProbeInput:
    """The startup banner names the microphone that will be used, not the pinned one."""

    def test_a_pinned_device_that_opens_is_named(self, fake_streams, monkeypatch):
        monkeypatch.setattr(devices, "resolve", lambda wanted, *, kind: 32)
        assert devices.probe_input("SOUNDPEATS") == (32, False)

    def test_listed_but_not_connected_is_the_default(self, fake_streams, monkeypatch):
        monkeypatch.setattr(devices, "resolve", lambda wanted, *, kind: 32)
        fake_streams.refuse = {32}
        assert devices.probe_input("SOUNDPEATS") == (None, True)

    def test_not_listed_is_the_default(self, fake_streams):
        assert devices.probe_input("no-such-microphone") == (None, True)

    def test_nothing_pinned_is_the_default_without_opening(self, fake_streams):
        assert devices.probe_input("") == (None, False)
        assert fake_streams.opened == []

    def test_it_says_nothing(self, fake_streams, monkeypatch, caplog):
        # The selector reports the fallback once the listener opens; a second
        # warning from the banner would be the same news twice.
        monkeypatch.setattr(devices, "resolve", lambda wanted, *, kind: 32)
        fake_streams.refuse = {32}
        with caplog.at_level(logging.WARNING, logger="talos.devices"):
            devices.probe_input("SOUNDPEATS")
        assert caplog.text == ""
