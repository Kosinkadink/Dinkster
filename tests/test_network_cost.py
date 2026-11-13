from __future__ import annotations

from dataclasses import dataclass

import pytest
from dinkster_server import NWPath, detect_network_cost


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


@pytest.mark.parametrize("platform", ("linux", "win32"))
def test_network_detection_failures_are_unknown(platform: str) -> None:
    assert detect_network_cost(platform_name=platform, runner=Runner("", error=True)) == "unknown"
