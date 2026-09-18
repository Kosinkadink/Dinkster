"""The install registry: the station's input contract.

``installs.toml`` names every managed installation - its install root,
its public port, whether the station starts it automatically, and
(optionally) the engine command that serves it:

    [installs.main]
    root = "/home/user/.dinkster/installs/main"
    port = 8200
    autostart = true

    [installs.experiments]
    root = "/home/user/.dinkster/installs/experiments"
    port = 8201
    engine = ["/some/venv/bin/python", "-m", "dinkster.serve"]

The file is WRITTEN by ``dinkster-installs`` (which lives in the umbrella
and can initialize install roots) and READ by the station (which lives
here and deliberately imports no dinkster engine code - the layer that
manages several engine versions cannot share a Python environment with
any one of them). Both directions go through this module so there is
exactly one format.

Parsing is strict: unknown keys, malformed ports, and colliding ports
are load errors, never silently ignored - a typo in an install's config
should fail the station's startup loudly, not quietly serve defaults.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

__all__ = [
    "DEFAULT_INSTALLS_FILE",
    "InstallDef",
    "IngressDef",
    "StationConfig",
    "InstallsError",
    "dump_installs",
    "dump_station_config",
    "load_installs",
    "load_station_config",
    "parse_installs",
    "parse_station_config",
]

DEFAULT_INSTALLS_FILE = Path.home() / ".dinkster" / "installs.toml"
DEFAULT_MANAGEMENT_PORT = 3649

_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_ENTRY_KEYS = frozenset({"root", "port", "autostart", "engine"})


class InstallsError(Exception):
    """The installs config is malformed - named loudly, never defaulted."""


@dataclass(frozen=True)
class InstallDef:
    """One managed installation, as configured."""

    name: str
    root: Path
    port: int
    autostart: bool = False
    engine: tuple[str, ...] = field(default_factory=tuple)
    """Engine command override (interpreter first). Empty = the station's
    default engine command. Either way the station appends
    ``--install-root <root>`` (and the supervisor protocol appends
    ``--host/--port``), so a custom engine must accept all three."""


@dataclass(frozen=True)
class IngressDef:
    port: int
    members: tuple[str, ...]
    primary: str
    state_path: Path | None = None


@dataclass(frozen=True)
class StationConfig:
    installs: tuple[InstallDef, ...]
    ingress: IngressDef | None = None


def parse_station_config(text: str, source: str = "installs config") -> StationConfig:
    try:
        data: dict[str, object] = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise InstallsError(f"{source}: invalid TOML: {exc}") from exc
    unknown = set(data) - {"installs", "ingress"}
    if unknown:
        raise InstallsError(f"{source}: unknown top-level keys {sorted(unknown)}")
    installs = _parse_install_tables(data.get("installs", {}), source)
    raw = data.get("ingress")
    if raw is None:
        return StationConfig(installs)
    if not isinstance(raw, dict):
        raise InstallsError(f"{source}: [ingress] must be a table")
    entry = cast("dict[object, object]", raw)
    extra = {str(key) for key in entry} - {"port", "members", "primary", "state_path"}
    if extra:
        raise InstallsError(f"{source}: [ingress]: unknown keys {sorted(extra)}")
    port = entry.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise InstallsError(f"{source}: [ingress]: 'port' must be an integer in 1..65535")
    members_raw = entry.get("members")
    if not isinstance(members_raw, list) or not members_raw:
        raise InstallsError(f"{source}: [ingress]: 'members' must be a non-empty list")
    members_objects = cast("list[object]", members_raw)
    members = tuple(item for item in members_objects if isinstance(item, str))
    if (
        len(members) != len(members_objects)
        or any(not item or not _NAME_PATTERN.fullmatch(item) for item in members)
        or len(set(members)) != len(members)
    ):
        raise InstallsError(f"{source}: [ingress]: members must be unique install names")
    names = {install.name for install in installs}
    if not set(members) <= names:
        raise InstallsError(f"{source}: [ingress]: unknown members {sorted(set(members) - names)}")
    primary = entry.get("primary", members[0])
    if not isinstance(primary, str) or primary not in members:
        raise InstallsError(f"{source}: [ingress]: primary must name a member")
    state_raw = entry.get("state_path")
    if state_raw is not None and (not isinstance(state_raw, str) or not state_raw):
        raise InstallsError(f"{source}: [ingress]: state_path must be a non-empty path string")
    collisions = [install.name for install in installs if install.port == port]
    if collisions:
        raise InstallsError(
            f"{source}: [ingress]: port {port} collides with [installs.{collisions[0]}]"
        )
    if port == DEFAULT_MANAGEMENT_PORT:
        raise InstallsError(
            f"{source}: [ingress]: port {port} collides with the station management port"
        )
    return StationConfig(
        installs,
        IngressDef(port, members, primary, Path(state_raw) if state_raw else None),
    )


def parse_installs(text: str, source: str = "installs config") -> tuple[InstallDef, ...]:
    """Decode and validate installs config text. Order follows the file."""
    try:
        data: dict[str, object] = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise InstallsError(f"{source}: invalid TOML: {exc}") from exc
    unknown_top = set(data) - {"installs", "ingress"}
    if unknown_top:
        raise InstallsError(
            f"{source}: unknown top-level keys {sorted(unknown_top)} "
            f"(everything lives under [installs.<name>])"
        )
    return _parse_install_tables(data.get("installs", {}), source)


def _parse_install_tables(tables: object, source: str) -> tuple[InstallDef, ...]:
    if not isinstance(tables, dict):
        raise InstallsError(f"{source}: 'installs' must be a table of installs")
    installs: list[InstallDef] = []
    ports: dict[int, str] = {}
    roots: dict[str, str] = {}
    for name_raw, entry_raw in cast("dict[object, object]", tables).items():
        name = str(name_raw)
        where = f"{source}: [installs.{name}]"
        if not _NAME_PATTERN.match(name):
            raise InstallsError(
                f"{where}: install names are lowercase alphanumerics and "
                f"hyphens, starting alphanumeric"
            )
        if not isinstance(entry_raw, dict):
            raise InstallsError(f"{where}: must be a table")
        entry = cast("dict[object, object]", entry_raw)
        unknown = {str(key) for key in entry} - _ENTRY_KEYS
        if unknown:
            raise InstallsError(f"{where}: unknown keys {sorted(unknown)}")
        root = entry.get("root")
        if not isinstance(root, str) or not root:
            raise InstallsError(f"{where}: 'root' must be a non-empty path string")
        root_holder = roots.get(root)
        if root_holder is not None:
            raise InstallsError(
                f"{where}: root {root!r} is already used by "
                f"[installs.{root_holder}] - two engines on one install "
                f"root would fight over its generations"
            )
        roots[root] = name
        port = entry.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise InstallsError(f"{where}: 'port' must be an integer in 1..65535")
        holder = ports.get(port)
        if holder is not None:
            raise InstallsError(f"{where}: port {port} is already used by [installs.{holder}]")
        ports[port] = name
        autostart = entry.get("autostart", False)
        if not isinstance(autostart, bool):
            raise InstallsError(f"{where}: 'autostart' must be a boolean")
        engine_raw = entry.get("engine", [])
        if not isinstance(engine_raw, list):
            raise InstallsError(f"{where}: 'engine' must be a list of non-empty strings")
        engine: list[str] = []
        for item in cast("list[object]", engine_raw):
            if not isinstance(item, str) or not item:
                raise InstallsError(f"{where}: 'engine' must be a list of non-empty strings")
            engine.append(item)
        installs.append(
            InstallDef(
                name=name,
                root=Path(root),
                port=port,
                autostart=autostart,
                engine=tuple(engine),
            )
        )
    return tuple(installs)


def load_station_config(path: Path) -> StationConfig:
    if not path.is_file():
        return StationConfig(())
    config = parse_station_config(path.read_text(), str(path))
    ingress = config.ingress
    if ingress is not None and ingress.state_path is None:
        ingress = IngressDef(
            ingress.port, ingress.members, ingress.primary, path.with_name("ingress.sqlite3")
        )
    return StationConfig(config.installs, ingress)


def load_installs(path: Path) -> tuple[InstallDef, ...]:
    """Parse the installs config file; a missing file is an empty registry
    (a station with nothing to manage is valid, not an error)."""
    if not path.is_file():
        return ()
    return load_station_config(path).installs


def _toml_string(value: str) -> str:
    """A TOML basic string. JSON string escaping is a subset of TOML's
    basic-string escapes, so this is exact, including Windows paths."""
    return json.dumps(value)


def dump_installs(installs: tuple[InstallDef, ...] | list[InstallDef]) -> str:
    """Deterministic config text (sorted by name); ``parse_installs`` of
    the output round-trips exactly. Optional fields are omitted at their
    defaults - the file stays as small as what was actually configured."""
    chunks: list[str] = []
    for install in sorted(installs, key=lambda entry: entry.name):
        lines = [
            f"[installs.{install.name}]",
            f"root = {_toml_string(str(install.root))}",
            f"port = {install.port}",
        ]
        if install.autostart:
            lines.append("autostart = true")
        if install.engine:
            listed = ", ".join(_toml_string(part) for part in install.engine)
            lines.append(f"engine = [{listed}]")
        chunks.append("\n".join(lines) + "\n")
    return "\n".join(chunks)


def dump_station_config(config: StationConfig) -> str:
    text = dump_installs(config.installs)
    if config.ingress is None:
        return text
    ingress = config.ingress
    lines = ["[ingress]", f"port = {ingress.port}"]
    lines.append("members = [" + ", ".join(_toml_string(item) for item in ingress.members) + "]")
    if ingress.primary != ingress.members[0]:
        lines.append(f"primary = {_toml_string(ingress.primary)}")
    if ingress.state_path is not None:
        lines.append(f"state_path = {_toml_string(str(ingress.state_path))}")
    return text + ("\n" if text else "") + "\n".join(lines) + "\n"
