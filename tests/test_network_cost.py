from __future__ import annotations

import pytest
from dinkster_server import detect_network_cost


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
