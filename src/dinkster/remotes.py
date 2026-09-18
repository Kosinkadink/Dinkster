"""Strict persisted remote-worker configuration.

``remotes.toml`` names the ``dinkster_workers.service`` daemons a served
engine composes at startup: endpoint, pre-shared token file, optional
node-type allowlist, and memory budgets for the remote's devices. A
missing file means no remotes; a present malformed file fails loudly
instead of silently serving without the intended workers.

One table per remote::

    [worker.upscale-box]
    endpoint = "192.168.1.53:5151"
    token_file = "/home/op/.config/dinkster/upscale-box.token"
    # tls_ca_file = "/home/op/.config/dinkster/upscale-box.pem"  # optional;
    #   the daemon's --tls-cert certificate, pinned - set it iff the
    #   daemon serves TLS
    # nodes = ["esrgan.upscale"]   # optional; default: everything announced

    [worker.upscale-box.memory]
    ram = "24G"
    "vram:cuda:0" = "20G"

The table name is the remote's identity: the ``@<name>`` qualifier on
every device fact it reports, its pack id on the composed surface, and
its diagnostics key. Budgets are declared in the remote's own device
namespace and qualified with ``@<name>`` when they reach the governor.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from dinkster_memory import parse_size

__all__ = ["RemoteSpec", "RemotesError", "load_remotes", "parse_remotes"]

_WORKER_KEYS = {"endpoint", "token_file", "tls_ca_file", "nodes", "trust_reserved", "memory"}


class RemotesError(Exception):
    """A persisted remote-worker configuration is malformed."""


@dataclass(frozen=True, slots=True)
class RemoteSpec:
    """One configured remote worker, validated but not yet connected."""

    name: str
    host: str
    port: int
    token_file: Path
    tls_ca_file: Path | None = None
    """Optional pinned trust anchor for server-authenticating TLS: the
    daemon's ``--tls-cert`` certificate (or the CA that signed it). None
    means plaintext TCP; there is no unverified-TLS mode."""
    nodes: tuple[str, ...] | None = None
    """Optional node-type allowlist; None routes everything the service's
    hello announces."""
    trust_reserved: bool = False
    """Allow the remote to announce node types under reserved namespace
    roots (``std``, ``comfy``, ``core``, ``dinkster``) - the same host-vouches
    escape hatch as PackSpec.trust_reserved."""
    memory_budgets: Mapping[str, int] = field(default_factory=dict)
    """Budgets in the REMOTE's device namespace (``ram``, ``vram:cuda:0``);
    the host qualifies each key with ``@<name>`` before the governor sees
    it."""


def _parse_endpoint(raw: object, where: str) -> tuple[str, int]:
    if not isinstance(raw, str) or not raw:
        raise RemotesError(f"{where}: 'endpoint' must be a HOST:PORT string")
    host, _, port_text = raw.rpartition(":")
    if not host or not port_text.isdigit():
        raise RemotesError(f"{where}: malformed endpoint (want HOST:PORT): {raw!r}")
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise RemotesError(f"{where}: endpoint port must be 1-65535, got {port}")
    return host, port


def _parse_memory(raw: object, where: str) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        raise RemotesError(f"{where}: 'memory' must be a table of device = size")
    budgets: dict[str, int] = {}
    for device_raw, size_raw in cast("Mapping[object, object]", raw).items():
        device = str(device_raw)
        if not device or any(character.isspace() for character in device):
            raise RemotesError(
                f"{where}: memory device keys must be non-empty strings "
                f"without whitespace, got {device_raw!r}"
            )
        if "@" in device:
            raise RemotesError(
                f"{where}: memory device {device!r} must be unqualified - "
                "budgets are declared in the remote's own namespace and the "
                "host adds the @name qualifier"
            )
        entry = f"{where}: memory budget for {device!r}"
        if isinstance(size_raw, bool):
            raise RemotesError(f"{entry}: size must be a string or integer bytes")
        if isinstance(size_raw, int):
            if size_raw < 0:
                raise RemotesError(f"{entry}: integer bytes must be >= 0")
            budgets[device] = size_raw
            continue
        if not isinstance(size_raw, str):
            raise RemotesError(f"{entry}: size must be a string or integer bytes")
        try:
            budgets[device] = parse_size(size_raw)
        except ValueError as exc:
            raise RemotesError(f"{entry}: {exc}") from None
    return budgets


def parse_remotes(data: object, source: str) -> tuple[RemoteSpec, ...]:
    """Validate a TOML-decoded ``[worker.NAME]`` table set."""
    if not isinstance(data, Mapping):
        raise RemotesError(f"{source}: config must be a table")
    top = cast("Mapping[object, object]", data)
    unknown_top = {str(key) for key in top} - {"worker"}
    if unknown_top:
        raise RemotesError(
            f"{source}: unknown top-level keys {sorted(unknown_top)} "
            f"(remotes live under [worker.NAME])"
        )
    workers = top.get("worker", {})
    if not isinstance(workers, Mapping):
        raise RemotesError(f"{source}: 'worker' must be a table of [worker.NAME] entries")

    specs: list[RemoteSpec] = []
    for name_raw, entry_raw in cast("Mapping[object, object]", workers).items():
        name = str(name_raw)
        where = f"{source}: [worker.{name}]"
        if not name or any(character.isspace() for character in name) or "@" in name:
            raise RemotesError(
                f"{source}: worker names must be non-empty strings without "
                f"whitespace or '@', got {name_raw!r}"
            )
        if name.casefold() == "local":
            raise RemotesError(f"{where}: worker name 'local' is reserved")
        if not isinstance(entry_raw, Mapping):
            raise RemotesError(f"{where}: must be a table")
        entry = cast("Mapping[str, object]", entry_raw)
        unknown = set(entry) - _WORKER_KEYS
        if unknown:
            raise RemotesError(f"{where}: unknown keys {sorted(unknown)}")
        host, port = _parse_endpoint(entry.get("endpoint"), where)
        token_raw = entry.get("token_file")
        if not isinstance(token_raw, str) or not token_raw:
            raise RemotesError(f"{where}: 'token_file' must be a non-empty path string")
        tls_ca_raw = entry.get("tls_ca_file")
        tls_ca_file: Path | None = None
        if tls_ca_raw is not None:
            if not isinstance(tls_ca_raw, str) or not tls_ca_raw:
                raise RemotesError(f"{where}: 'tls_ca_file' must be a non-empty path string")
            tls_ca_file = Path(tls_ca_raw)
        nodes: tuple[str, ...] | None = None
        nodes_raw = entry.get("nodes")
        if nodes_raw is not None:
            if not isinstance(nodes_raw, list) or not all(
                isinstance(value, str) and value for value in cast("list[object]", nodes_raw)
            ):
                raise RemotesError(f"{where}: 'nodes' must be a list of non-empty strings")
            nodes = tuple(cast("list[str]", nodes_raw))
        trust_raw = entry.get("trust_reserved", False)
        if not isinstance(trust_raw, bool):
            raise RemotesError(f"{where}: 'trust_reserved' must be a boolean")
        memory_raw = entry.get("memory", {})
        specs.append(
            RemoteSpec(
                name=name,
                host=host,
                port=port,
                token_file=Path(token_raw),
                tls_ca_file=tls_ca_file,
                nodes=nodes,
                trust_reserved=trust_raw,
                memory_budgets=_parse_memory(memory_raw, where),
            )
        )
    return tuple(specs)


def load_remotes(path: Path) -> tuple[RemoteSpec, ...]:
    """Load remote-worker specs; a missing file means no remotes."""
    if not path.exists():
        return ()
    try:
        data = tomllib.loads(path.read_text("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise RemotesError(f"{path}: invalid TOML: {exc}") from exc
    return parse_remotes(data, str(path))
