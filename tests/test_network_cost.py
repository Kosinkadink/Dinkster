from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pytest
from dinkster_server import NWPath, detect_network_cost, network_cost


@dataclass(frozen=True)
class Path:
    is_expensive: bool = False
    is_constrained: bool = False


class Runner:
    def __init__(self, output: str, *, error: bool = False) -> None:
        self.output = output
        self.error = error
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...]) -> str | None:
        self.commands.append(command)
        if self.error:
            return None
        return self.output


class PathReader:
    def __init__(self, path: NWPath) -> None:
        self.path = path

    def __call__(self) -> NWPath:
        return self.path


@pytest.mark.parametrize(
    ("output", "expected"),
    (
        ("GENERAL.STATE:100 (connected)\nGENERAL.METERED:yes\n", "metered"),
        ("GENERAL.STATE:100 (connected)\nGENERAL.METERED:yes (guessed)\n", "metered"),
        ("GENERAL.STATE:100 (connected)\nGENERAL.METERED:no\n", "unmetered"),
        ("GENERAL.STATE:100 (connected)\nGENERAL.METERED:no (guessed)\n", "unmetered"),
        ("GENERAL.STATE:100 (connected)\nGENERAL.METERED:unknown\n", "unknown"),
        ("GENERAL.STATE:100 (connected)\n", "unknown"),
        (
            "GENERAL.STATE:30 (disconnected)\nGENERAL.METERED:yes\n\n"
            "GENERAL.STATE:100 (connected)\nGENERAL.METERED:no\n",
            "unmetered",
        ),
        (
            "GENERAL.STATE:30 (disconnected)\nGENERAL.METERED:no\n\n"
            "GENERAL.STATE:100 (connected)\nGENERAL.METERED:unknown\n",
            "unknown",
        ),
    ),
)
def test_linux_network_manager_metered_detection(output: str, expected: str) -> None:
    runner = Runner(output)
    assert detect_network_cost(platform_name="linux", runner=runner) == expected
    assert runner.commands == [
        (
            "nmcli",
            "--terse",
            "--fields",
            "GENERAL.STATE,GENERAL.METERED",
            "device",
            "show",
        )
    ]


@pytest.mark.parametrize(
    ("output", "expected"),
    (
        ("Unrestricted\n", "unmetered"),
        ("Fixed\n", "metered"),
        ("Variable\n", "metered"),
        ("Unknown\n", "unknown"),
    ),
)
def test_windows_connection_cost_detection(output: str, expected: str) -> None:
    runner = Runner(output)
    assert detect_network_cost(platform_name="win32", runner=runner) == expected
    assert runner.commands[0][0:4] == (
        "powershell.exe",
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
    )


@pytest.mark.parametrize(
    ("path", "expected"),
    (
        (Path(is_expensive=True), "metered"),
        (Path(is_constrained=True), "metered"),
        (Path(), "unmetered"),
    ),
)
def test_macos_network_path_detection(path: NWPath, expected: str) -> None:
    assert detect_network_cost(platform_name="darwin", nwpath_reader=PathReader(path)) == expected


def test_macos_native_network_path_monitor(monkeypatch: pytest.MonkeyPatch) -> None:
    class Network:
        def __init__(self) -> None:
            self.handler: Callable[[object], None] | None = None
            self.cancelled = False

        @staticmethod
        def nw_path_monitor_create() -> object:
            return object()

        def nw_path_monitor_set_update_handler(
            self, _monitor: object, handler: Callable[[object], None]
        ) -> None:
            self.handler = handler

        @staticmethod
        def nw_path_monitor_set_queue(_monitor: object, _queue: object) -> None:
            pass

        def nw_path_monitor_start(self, _monitor: object) -> None:
            assert self.handler is not None
            self.handler(object())

        def nw_path_monitor_cancel(self, _monitor: object) -> None:
            self.cancelled = True

        @staticmethod
        def nw_path_is_expensive(_path: object) -> bool:
            return False

        @staticmethod
        def nw_path_is_constrained(_path: object) -> bool:
            return True

    class Dispatch:
        DISPATCH_QUEUE_SERIAL = None

        @staticmethod
        def dispatch_queue_create(_label: bytes, _attribute: object) -> object:
            return object()

    native = Network()
    monkeypatch.setattr(
        network_cost,
        "_module",
        lambda name: native if name == "Network" else Dispatch(),
    )

    assert detect_network_cost(platform_name="darwin") == "metered"
    assert native.cancelled is True


@pytest.mark.parametrize("platform", ("linux", "win32"))
def test_network_detection_failures_are_unknown(platform: str) -> None:
    assert detect_network_cost(platform_name=platform, runner=Runner("", error=True)) == "unknown"


def test_unsupported_platform_and_missing_macos_path_are_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(network_cost, "_read_macos_nwpath", lambda: None)
    assert detect_network_cost(platform_name="freebsd") == "unknown"
    assert detect_network_cost(platform_name="darwin") == "unknown"
