"""Installation-owned pack sandbox policy, applied only when an engine starts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict, cast

from dinkster_store.egress import normalize_egress_origin
from filelock import FileLock

from .environment_state import (
    EnvironmentError,
    read_current,
    read_document,
    write_document,
    writer_lock,
)
from .home_context import installation_context


class SandboxSettings(TypedDict):
    sandboxPacks: bool
    added: dict[str, list[str]]
    removed: dict[str, list[str]]


def read_sandbox_settings(root: Path) -> SandboxSettings:
    if (context := installation_context(root)) is not None:
        return cast("SandboxSettings", context.call(root, "settings"))
    with FileLock(str(root / ".pack-sandbox.lock"), timeout=30):
        return _read_sandbox_settings(root)


def _read_sandbox_settings(root: Path) -> SandboxSettings:
    path = root / "pack-sandbox.json"
    if not path.exists():
        return {"sandboxPacks": False, "added": {}, "removed": {}}
    value = read_document(path)
    if (
        set(value) != {"sandboxPacks", "added", "removed"}
        or type(value["sandboxPacks"]) is not bool
    ):
        raise EnvironmentError("invalid installation sandbox settings")
    for key in ("added", "removed"):
        rows = value[key]
        if not isinstance(rows, dict):
            raise EnvironmentError("invalid installation sandbox addresses")
        for origins in cast("dict[str, object]", rows).values():
            if not isinstance(origins, list):
                raise EnvironmentError("invalid installation sandbox addresses")
            for origin in cast("list[object]", origins):
                if not isinstance(origin, str) or normalize_egress_origin(origin) != origin:
                    raise EnvironmentError("invalid installation sandbox address")
    return cast("SandboxSettings", value)


def set_sandbox_switch(root: Path, enabled: bool) -> dict[str, object]:
    if (context := installation_context(root)) is not None:
        result = cast("dict[str, object]", context.call(root, "switch", enabled))
        result["root"] = str(root)
        return result
    with writer_lock(root):
        settings = read_sandbox_settings(root)
        settings["sandboxPacks"] = enabled
        with FileLock(str(root / ".pack-sandbox.lock"), timeout=30):
            write_document(root / "pack-sandbox.json", dict(settings))
    return {"root": str(root), "sandboxPacks": enabled}


def internet_addresses(
    pack: str, requested: list[str], settings: SandboxSettings
) -> list[dict[str, str]]:
    removed = settings["removed"].get(pack, [])
    return [
        {"address": address, "source": "pack" if address in requested else "user"}
        for address in dict.fromkeys([*requested, *settings["added"].get(pack, [])])
        if address not in removed
    ]


def change_internet_address(
    root: Path, pack: str, address: str, requested: list[str], snapshot: str, *, allow: bool
) -> list[dict[str, str]]:
    if (context := installation_context(root)) is not None:
        return cast(
            "list[dict[str, str]]",
            context.call(root, "internet", pack, address, requested, snapshot, allow),
        )
    address = normalize_egress_origin(address)
    with writer_lock(root):
        if str(read_current(root)) != snapshot:
            raise EnvironmentError("installation changed; read its packs again")
        settings = read_sandbox_settings(root)
        include, exclude = ("added", "removed") if allow else ("removed", "added")
        included = settings[include].setdefault(pack, [])
        if address not in included:
            included.append(address)
        excluded = settings[exclude].get(pack, [])
        if address in excluded:
            excluded.remove(address)
        with FileLock(str(root / ".pack-sandbox.lock"), timeout=30):
            write_document(root / "pack-sandbox.json", dict(settings))
        return internet_addresses(pack, requested, settings)


def sandbox_command_arguments(packs: list[dict[str, object]]) -> list[str]:
    """Only the managed engine launcher translates saved policy into authority."""
    arguments = ["--sandbox-packs"]
    network: dict[str, list[str]] = {}
    for row in packs:
        pack = str(row["pack"])
        addresses = row["internetAddresses"]
        if addresses is None:
            raise EnvironmentError(f"cannot determine sandbox requests for {pack}")
        network[pack] = [item["address"] for item in cast("list[dict[str, str]]", addresses)]
        graphics = cast("dict[str, object]", row["graphicsCard"])
        if graphics["requested"] is True:
            arguments.extend(("--sandbox-grant-gpu", pack))
    arguments.extend(("--sandbox-network-policy", json.dumps(network, ensure_ascii=True)))
    return arguments
