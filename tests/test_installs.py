"""The install registry format (installs.toml; see packages/dinkster-supervisor/README.md).

One format, two writers' worth of discipline: dinkster-installs writes it,
the station reads it, and both directions go through
dinkster_supervisor.installs. Parsing is strict - a typo fails station
startup loudly instead of quietly serving defaults - and the writer is
deterministic, so a config round-trips exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from dinkster_supervisor import (
    InstallDef,
    InstallsError,
    dump_installs,
    load_installs,
    parse_installs,
)


def test_roundtrip_and_deterministic_writer() -> None:
    """dump -> parse is exact, entries sort by name, and optional fields
    are omitted at their defaults so the file stays as small as what was
    actually configured."""
    installs = (
        InstallDef(
            name="zeta",
            root=Path("/opt/dinkster/zeta"),
            port=8201,
            autostart=True,
            engine=("/venvs/old/bin/python", "-m", "dinkster.serve"),
        ),
        InstallDef(name="alpha", root=Path("/opt/dinkster/alpha"), port=8200),
    )
    text = dump_installs(installs)
    # deterministic order: alpha before zeta, defaults omitted
    assert text.index("[installs.alpha]") < text.index("[installs.zeta]")
    assert "autostart" not in text.split("[installs.zeta]")[0]
    parsed = parse_installs(text)
    assert set(parsed) == set(installs)
    # a second dump of the parse is byte-identical
    assert dump_installs(parsed) == text


def test_windows_paths_survive_the_writer() -> None:
    """TOML basic-string escaping keeps backslashes exact - the config a
    Windows machine writes is the config it reads back."""
    install = InstallDef(
        name="win",
        root=Path(r"C:\Users\kosin\dinkster installs\main"),
        port=8200,
        engine=(r"C:\venvs\dinkster\Scripts\python.exe", "-m", "dinkster.serve"),
    )
    parsed = parse_installs(dump_installs((install,)))
    assert parsed == (install,)


def test_missing_file_is_an_empty_registry(tmp_path: Path) -> None:
    """A station with nothing to manage is valid, not an error."""
    assert load_installs(tmp_path / "absent.toml") == ()


def test_load_reads_the_file(tmp_path: Path) -> None:
    config = tmp_path / "installs.toml"
    config.write_text('[installs.main]\nroot = "/opt/main"\nport = 8200\n')
    (install,) = load_installs(config)
    assert install == InstallDef(name="main", root=Path("/opt/main"), port=8200)


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("not toml [", "invalid TOML"),
        ('title = "x"\n', "unknown top-level keys"),
        ("installs = 3\n", "must be a table"),
        ('[installs.BadName]\nroot = "/r"\nport = 8200\n', "lowercase"),
        ('[installs.-bad]\nroot = "/r"\nport = 8200\n', "lowercase"),
        ('[installs.a]\nroot = "/r"\nport = 8200\nextra = 1\n', "unknown keys"),
        ("[installs.a]\nport = 8200\n", "'root' must be"),
        ('[installs.a]\nroot = ""\nport = 8200\n', "'root' must be"),
        ('[installs.a]\nroot = "/r"\n', "'port' must be"),
        ('[installs.a]\nroot = "/r"\nport = 0\n', "'port' must be"),
        ('[installs.a]\nroot = "/r"\nport = 65536\n', "'port' must be"),
        ('[installs.a]\nroot = "/r"\nport = true\n', "'port' must be"),
        ('[installs.a]\nroot = "/r"\nport = 8200\nautostart = 1\n', "'autostart'"),
        ('[installs.a]\nroot = "/r"\nport = 8200\nengine = "python"\n', "'engine'"),
        ('[installs.a]\nroot = "/r"\nport = 8200\nengine = [""]\n', "'engine'"),
        ('[installs.a]\nroot = "/r"\nport = 8200\nengine = [1]\n', "'engine'"),
    ],
)
def test_malformed_configs_refuse_loudly(text: str, fragment: str) -> None:
    with pytest.raises(InstallsError, match=fragment.replace("[", r"\[")):
        parse_installs(text)


def test_port_collision_names_both_installs() -> None:
    text = '[installs.a]\nroot = "/a"\nport = 8200\n\n[installs.b]\nroot = "/b"\nport = 8200\n'
    with pytest.raises(InstallsError, match=r"port 8200 is already used by \[installs.a\]"):
        parse_installs(text)


def test_root_collision_names_both_installs() -> None:
    """Two engines on one install root would fight over its generations -
    refused at parse time, not discovered at runtime."""
    text = (
        '[installs.a]\nroot = "/same"\nport = 8200\n\n[installs.b]\nroot = "/same"\nport = 8201\n'
    )
    with pytest.raises(InstallsError, match=r"root '/same' is already used"):
        parse_installs(text)


def test_duplicate_names_are_a_toml_error() -> None:
    """TOML itself rejects a redefined table; it surfaces as InstallsError
    like every other malformation."""
    text = '[installs.a]\nroot = "/a"\nport = 8200\n\n[installs.a]\nroot = "/b"\nport = 8201\n'
    with pytest.raises(InstallsError, match="invalid TOML"):
        parse_installs(text)
