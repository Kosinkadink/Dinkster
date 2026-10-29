"""One registered data directory per project volume, with explicit project references."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

from .environment_state import (
    EnvironmentError,
    atomic_write,
    durable_unlink,
    read_document,
    write_document,
)


def registry_path() -> Path:
    if os.name == "nt":
        parent = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
    else:
        parent = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    if not parent.is_absolute():
        raise EnvironmentError("user data directory must be absolute")
    return parent / "dinkster" / "stores.json"


def volume(path: Path) -> str:
    parent = path.resolve()
    while not parent.exists():
        parent = parent.parent
    return str(parent.stat().st_dev)


def registered_stores(path: Path | None = None) -> dict[str, str]:
    path = path or registry_path()
    if not path.exists():
        return {}
    document = read_document(path)
    if document.get("format") != "dinkster.stores/1" or not isinstance(
        document.get("volumes"), dict
    ):
        raise EnvironmentError("invalid store registry")
    entries = cast("dict[str, object]", document["volumes"])
    if any(
        not key.isdecimal() or not isinstance(value, str) or not Path(value).is_absolute()
        for key, value in entries.items()
    ):
        raise EnvironmentError("store registry requires volume ids and absolute paths")
    return cast("dict[str, str]", entries)


def select_store(root: Path, requested: Path | None) -> Path:
    """Read-only selection; the caller rechecks under the registry writer lock."""
    device = volume(root)
    registered = registered_stores().get(device)
    if requested is not None and not requested.is_absolute():
        raise EnvironmentError("--data must be absolute")
    selected = requested or (Path(registered) if registered else root.parent / "dinkster-data")
    selected = selected.resolve()
    root = root.resolve()
    if registered is not None and selected != Path(registered).resolve():
        raise EnvironmentError(f"volume already uses store {registered}")
    if volume(selected) != device:
        raise EnvironmentError("project and wheel store must be on the same volume")
    if selected.is_relative_to(root) or root.is_relative_to(selected):
        raise EnvironmentError("project and wheel store must not contain each other")
    return selected


def store_projects(store: Path) -> list[str]:
    path = store / "projects.json"
    if not path.exists():
        return []
    document = read_document(path)
    projects = document.get("projects")
    if document.get("format") != "dinkster.store/1" or not isinstance(projects, list):
        raise EnvironmentError("invalid store project references")
    values = cast("list[object]", projects)
    if any(not isinstance(value, str) or not Path(value).is_absolute() for value in values):
        raise EnvironmentError("store project references must be absolute")
    return sorted(set(cast("list[str]", values)))


def register_project(root: Path, store: Path) -> None:
    """Caller holds registry, store and project locks, after journal publication."""
    if select_store(root, store) != store:
        raise EnvironmentError("store selection changed after planning")
    projects = store_projects(store)
    projects.append(str(root.resolve()))
    write_document(
        store / "projects.json", {"format": "dinkster.store/1", "projects": sorted(set(projects))}
    )
    entries = registered_stores()
    entries[volume(root)] = str(store)
    write_document(registry_path(), {"format": "dinkster.stores/1", "volumes": entries})


def setup_registration(root: Path, store: Path) -> dict[str, object]:
    """Snapshot only this setup's metadata ownership while all writer locks are held."""
    registry = registry_path()
    return {
        "store": str(store),
        "registry": str(registry),
        "project": (root / "project.json").read_bytes().decode("utf-8")
        if (root / "project.json").exists()
        else None,
        "pointer": (root / "store").read_bytes().decode("utf-8")
        if (root / "store").exists()
        else None,
        "addedProject": str(root) not in store_projects(store),
        "addedVolume": volume(root) not in registered_stores(),
        "hadProjects": (store / "projects.json").exists(),
        "hadRegistry": registry.exists(),
    }


def restore_setup_registration(root: Path, snapshot: dict[str, object]) -> None:
    """Undo this setup's additions without restoring stale shared-document snapshots."""
    store = Path(cast("str", snapshot["store"]))
    registry = Path(cast("str", snapshot["registry"]))
    projects = store_projects(store)
    if snapshot["addedProject"]:
        projects = [project for project in projects if project != str(root)]
        if projects or snapshot["hadProjects"]:
            write_document(
                store / "projects.json", {"format": "dinkster.store/1", "projects": projects}
            )
        else:
            durable_unlink(store / "projects.json")
    entries = registered_stores(registry)
    device = volume(root)
    if snapshot["addedVolume"] and not projects and entries.get(device) == str(store):
        del entries[device]
    if entries or snapshot["hadRegistry"]:
        write_document(registry, {"format": "dinkster.stores/1", "volumes": entries})
    else:
        durable_unlink(registry)
    for key, name in (("project", "project.json"), ("pointer", "store")):
        previous = snapshot[key]
        if previous is None:
            durable_unlink(root / name)
        else:
            atomic_write(root / name, cast("str", previous))
