"""Admit and select pack sources belonging to one explicit engine generation."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from dinkster_registry import LockedPack, Lockfile, build_artifact, unpack_artifact
from dinkster_schema import canonical_claim, canonical_name
from dinkster_workers import PackManifest, load_manifest

from .compose import CompositionError, PackSpec
from .packs import pack_info_from_manifest


def _manifest(root: Path, entry: LockedPack) -> PackManifest:
    manifest = load_manifest(root / "dinkster-pack.toml")
    if canonical_name(manifest.name) != entry.pack:
        raise CompositionError(f"generation pack {entry.pack!r} manifest names {manifest.name!r}")
    if tuple(canonical_claim(claim) for claim in manifest.namespaces) != entry.claims:
        raise CompositionError(f"generation pack {entry.pack!r} manifest claims differ from lock")
    if manifest.publisher and canonical_name(manifest.publisher) != entry.publisher:
        raise CompositionError(f"generation pack {entry.pack!r} publisher differs from lock")
    return manifest


def stage_generation_packs(generation: Path, lock: Lockfile, artifacts: Mapping[str, Path]) -> None:
    """Admit verified sources after dependency rebuild, before the owner seals the target."""
    if (generation / "generation.json").exists() or (generation / "dinkster.lock").exists():
        raise CompositionError("pack sources require a fresh, unsealed generation")
    sources = generation / "pack-sources"
    sources.mkdir(exist_ok=False)
    for entry in lock.packs:
        root = sources / entry.pack
        unpack_artifact(artifacts[entry.pack], root, entry.artifact_digest)
        _manifest(root, entry)
    (generation / "dinkster.lock").write_text(lock.record_json() + "\n", encoding="utf-8")


def generation_pack_specs(generation: Path) -> tuple[PackSpec, ...]:
    """Select only this generation's admitted sources and recorded hosting placement."""
    if not generation.is_absolute():
        raise CompositionError("--generation requires an absolute path")
    generation = generation.resolve()
    record = json.loads((generation / "generation.json").read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("format") != "dinkster.generation/1":
        raise CompositionError("unsupported generation format")
    hosting = record.get("inProcessPacks", [])
    if (
        not isinstance(hosting, list)
        or any(not isinstance(pack, str) for pack in hosting)
        or len(set(hosting)) != len(hosting)
    ):
        raise CompositionError("generation inProcessPacks must contain distinct pack ids")
    lock = Lockfile.from_record_json((generation / "dinkster.lock").read_text(encoding="utf-8"))
    if set(hosting) - {entry.pack for entry in lock.packs}:
        raise CompositionError("generation hosting names packs absent from dinkster.lock")
    specs: list[PackSpec] = []
    for entry in lock.packs:
        root = generation / "pack-sources" / entry.pack
        if not root.resolve().is_relative_to(generation / "pack-sources"):
            raise CompositionError(f"generation pack {entry.pack!r} source escapes generation")
        with tempfile.TemporaryDirectory(prefix="dinkster-generation-verify-") as directory:
            actual = build_artifact(root, Path(directory) / "pack.zip")
        if actual != entry.artifact_digest:
            raise CompositionError(
                f"generation pack {entry.pack!r} artifact digest differs from lock"
            )
        manifest = _manifest(root, entry)
        in_process = entry.pack in hosting
        python = (
            generation
            / "packs"
            / entry.pack
            / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )
        if not in_process and not python.is_file():
            raise CompositionError(
                f"generation pack {entry.pack!r} interpreter is missing: {python}"
            )
        info = replace(
            pack_info_from_manifest(manifest),
            version=entry.version,
            artifact_digest=entry.artifact_digest,
            source=entry.source,
            publisher=entry.publisher,
        )
        specs.append(
            PackSpec(
                manifest=manifest.path,
                python=None if in_process else str(python),
                env={"PYTHONPATH": str(root)},
                packs={entry.pack: info},
                in_process=in_process,
                trust_reserved=entry.publisher == "dinkster",
                require_catalog=True,
                asset_vault_write=(
                    entry.pack == "dinkster-nodes-remote"
                    and entry.publisher == "dinkster"
                    and not in_process
                ),
            )
        )
    return tuple(specs)
