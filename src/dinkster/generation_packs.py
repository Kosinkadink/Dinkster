"""Admit and select pack sources belonging to one explicit engine generation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from dinkster_registry import (
    ArtifactError,
    InstallError,
    LockedPack,
    Lockfile,
    build_artifact,
    unpack_artifact,
)
from dinkster_schema import canonical_claim, canonical_name
from dinkster_workers import ManifestError, PackManifest, load_manifest
from dinkster_workers.accelerator import AcceleratorError, resolve_accelerator
from dinkster_workers.pack_lock import resolve_pack_lock
from dinkster_workers.provision import ProvisionError

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
    (generation / "dinkster.lock").write_text(
        lock.record_json() + "\n", encoding="utf-8", newline="\n"
    )


def prepare_pack_locks(
    generation: Path,
    lock: Lockfile,
    artifacts: Mapping[str, Path],
    output: Path,
    *,
    variant: str,
    indexes: Sequence[str] = (),
    find_links: Sequence[Path] = (),
    accelerator: str | None = None,
    uv: str = "uv",
) -> tuple[Path, ...]:
    """Resolve selected archives against an unsealed custom generation's constraints."""
    if (generation / "generation.json").exists() or (generation / "dinkster.lock").exists():
        raise CompositionError("pack locks require a fresh, unsealed generation")
    constraints = generation / "constraints.txt"
    if not constraints.is_file():
        raise CompositionError("custom pack locks require effective generation constraints")
    output.mkdir(exist_ok=False)
    paths: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="dinkster-custom-pack-sources-") as directory:
        manifests = []
        for entry in lock.packs:
            source = unpack_artifact(
                artifacts[entry.pack], Path(directory) / entry.pack, entry.artifact_digest
            )
            manifests.append(_manifest(source, entry))
        for manifest in manifests:
            dependency_lock = resolve_pack_lock(
                pack=manifest.name,
                variant=variant,
                requirements=(
                    manifest.requires_for(accelerator) if accelerator else manifest.requires
                ),
                python=Path(sys.executable),
                constraints=constraints,
                indexes=indexes,
                find_links=find_links,
                uv=uv,
            )
            path = output / f"{manifest.name}.pack-lock.json"
            dependency_lock.write(path)
            paths.append(path)
    return tuple(paths)


def load_generation_packs(generation: Path) -> tuple[str, tuple[PackSpec, ...]]:
    """Read the selected variant and admitted sources from this generation only."""
    if not generation.is_absolute():
        raise CompositionError("--generation requires an absolute path")
    generation = generation.resolve()
    record = json.loads((generation / "generation.json").read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("format") != "dinkster.generation/1":
        raise CompositionError("unsupported generation format")
    variant = record.get("variant")
    if not isinstance(variant, str) or not variant.strip():
        raise CompositionError("generation variant must be a non-empty string")
    hosting = record.get("inProcessPacks", [])
    if (
        not isinstance(hosting, list)
        or any(not isinstance(pack, str) for pack in hosting)
        or len(set(hosting)) != len(hosting)
    ):
        raise CompositionError("generation inProcessPacks must contain distinct pack ids")
    return variant, generation_pack_specs(generation, tuple(hosting))


def generation_pack_specs(
    generation: Path, in_process_packs: tuple[str, ...] = ()
) -> tuple[PackSpec, ...]:
    """Select verified admitted sources and their already-provisioned interpreters."""
    if not generation.is_absolute():
        raise CompositionError("--generation requires an absolute path")
    generation = generation.resolve()
    lock = Lockfile.from_record_json((generation / "dinkster.lock").read_text(encoding="utf-8"))
    if len(set(in_process_packs)) != len(in_process_packs):
        raise CompositionError("generation inProcessPacks must contain distinct pack ids")
    if set(in_process_packs) - {entry.pack for entry in lock.packs}:
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
        in_process = entry.pack in in_process_packs
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--catalog-lock", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument(
        "--prepare-locks", type=Path, help="new scratch output for custom pack locks"
    )
    parser.add_argument("--variant")
    parser.add_argument("--index", action="append", default=[])
    parser.add_argument("--find-links", type=Path, action="append", default=[])
    parser.add_argument("--accelerator")
    parser.add_argument("--uv", default="uv")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    packs: list[str] = []
    locks: tuple[Path, ...] = ()
    errors: list[str] = []
    try:
        if not all(
            path.is_absolute() for path in (args.generation, args.catalog_lock, args.artifacts)
        ):
            raise CompositionError(
                "--generation, --catalog-lock and --artifacts require absolute paths"
            )
        lock = Lockfile.from_record_json(args.catalog_lock.read_text(encoding="utf-8"))
        artifacts = {
            entry.pack: args.artifacts / f"{entry.artifact_digest.removeprefix('blake3:')}.zip"
            for entry in lock.packs
        }
        if args.prepare_locks is not None:
            if not args.prepare_locks.is_absolute() or not args.variant:
                raise CompositionError("custom pack locks require absolute output and --variant")
            locks = prepare_pack_locks(
                args.generation,
                lock,
                artifacts,
                args.prepare_locks,
                variant=args.variant,
                indexes=args.index,
                find_links=args.find_links,
                accelerator=resolve_accelerator(args.accelerator) if args.accelerator else None,
                uv=args.uv,
            )
        else:
            stage_generation_packs(args.generation, lock, artifacts)
        packs = [entry.pack for entry in lock.packs]
    except (
        AcceleratorError,
        ArtifactError,
        CompositionError,
        InstallError,
        ManifestError,
        OSError,
        ProvisionError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        errors.append(str(exc))
        print(str(exc), file=sys.stderr)
    if args.json:
        result: dict[str, object] = {"packs": packs, "errors": errors}
        if args.prepare_locks is not None:
            result["locks"] = [str(path) for path in locks]
        print(json.dumps(result))
    elif packs:
        print("\n".join(packs))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
