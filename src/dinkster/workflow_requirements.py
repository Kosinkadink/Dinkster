"""Derive portable requirements from saved workflow documents."""

from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from zipfile import BadZipFile, ZipFile

from dinkster_assets import AssetRef, AssetSource, PackagedSource, RemoteSource
from dinkster_workers import PackManifest
from packaging.version import Version

VARIANT_DEVICES = {
    "linux-cpu": "cpu",
    "win-cpu": "cpu",
    "win-nvidia": "cuda",
    "win-nvidia-cu126": "cuda",
}


class RequirementError(ValueError):
    """A workflow cannot provide an unambiguous requirement set."""


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise RequirementError(f"{label} must be an object")
    return cast("dict[str, object]", value)


@dataclass(frozen=True)
class PackRequirement:
    pack: str
    version: str | None
    node_types: tuple[str, ...] = ()

    def to_wire(self) -> dict[str, object]:
        return {"pack": self.pack, "version": self.version}


@dataclass(frozen=True)
class WorkflowRequirements:
    packs: tuple[PackRequirement, ...] = ()
    assets: tuple[AssetRef, ...] = ()
    unattributed_nodes: tuple[str, ...] = ()

    def to_wire(self) -> dict[str, object]:
        return {
            "packs": [pack.to_wire() for pack in self.packs],
            "assets": [asset.to_wire() for asset in self.assets],
            "unattributedNodes": list(self.unattributed_nodes),
        }


def derive_workflow_requirements(document: object) -> WorkflowRequirements:
    """Collect instantiated types and asset literals, never view or extension data.

    Saved attribution is a record, not a namespace grant. Unknown attribution
    and versions stay unknown so callers cannot quietly install guessed packs.
    """
    doc = _object(document, "workflow")
    if doc.get("format") != "dinkster-workflow" or doc.get("formatVersion") != 1:
        raise RequirementError("expected a dinkster-workflow formatVersion 1 document")
    graphs = _object(doc.get("graphs"), "workflow graphs")
    environment = _object(doc.get("environment", {}), "workflow environment")
    stamped_nodes = _object(environment.get("nodes", {}), "environment nodes")
    stamped_packs = _object(environment.get("packs", {}), "environment packs")
    used_types: set[str] = set()
    assets: dict[str, AssetRef] = {}

    def collect_assets(value: object) -> None:
        pending = [value]
        while pending:
            item = pending.pop()
            if isinstance(item, list):
                pending.extend(cast("list[object]", item))
            elif isinstance(item, dict):
                obj = cast("dict[str, object]", item)
                if "digest" in obj and "name" in obj and "size" in obj:
                    asset = AssetRef.from_wire(obj)
                    assets.setdefault(asset.digest, asset)
                else:
                    pending.extend(obj.values())

    for graph_id, raw_graph in graphs.items():
        graph = _object(raw_graph, f"graph {graph_id}")
        nodes = _object(graph.get("nodes"), f"graph {graph_id} nodes")
        for node_id, raw_node in nodes.items():
            node = _object(raw_node, f"node {node_id}")
            node_type = node.get("type")
            if not isinstance(node_type, str) or not node_type:
                raise RequirementError(f"node {node_id} requires a type")
            if not node_type.startswith("#") and node.get("virtual") is not True:
                used_types.add(node_type)
            collect_assets(_object(node.get("values", {}), f"node {node_id} values"))
        for source_id, raw_source in _object(
            graph.get("valueSources", {}), f"graph {graph_id} valueSources"
        ).items():
            source = _object(raw_source, f"value source {source_id}")
            collect_assets(source.get("value"))

    packs: dict[str, PackRequirement] = {}
    unattributed: list[str] = []
    for node_type in sorted(used_types):
        stamp = _object(stamped_nodes.get(node_type, {}), f"stamp for {node_type}")
        pack = stamp.get("pack")
        if not isinstance(pack, str) or not pack:
            unattributed.append(node_type)
            continue
        provenance = _object(stamped_packs.get(pack, {}), f"stamp for pack {pack}")
        version = provenance.get("version")
        if version is not None and (not isinstance(version, str) or not version):
            raise RequirementError(f"saved version for {pack} must be a non-empty string")
        previous = packs.get(pack)
        packs[pack] = PackRequirement(
            pack,
            cast("str | None", version),
            (*previous.node_types, node_type) if previous is not None else (node_type,),
        )
    return WorkflowRequirements(
        tuple(packs[key] for key in sorted(packs)),
        tuple(assets[key] for key in sorted(assets)),
        tuple(unattributed),
    )


def project_requirements(
    workflows: Iterable[object], *, additions: WorkflowRequirements | None = None
) -> WorkflowRequirements:
    """Union workflow needs and explicit user additions without changing documents.

    Different saved versions are a conflict, not an implicit upgrade. A caller
    must let the user choose before installing one version for both workflows.
    """
    packs: dict[str, PackRequirement] = {}
    assets: dict[str, AssetRef] = {}
    unattributed: set[str] = set()
    for requirements in (
        additions or WorkflowRequirements(),
        *(derive_workflow_requirements(doc) for doc in workflows),
    ):
        for pack in requirements.packs:
            previous = packs.get(pack.pack)
            if previous is not None and previous.version != pack.version:
                raise RequirementError(
                    f"conflicting saved versions for {pack.pack}: "
                    f"{previous.version!r} and {pack.version!r}"
                )
            packs[pack.pack] = PackRequirement(
                pack.pack,
                pack.version,
                tuple(sorted(set(pack.node_types) | set(previous.node_types if previous else ()))),
            )
        for asset in requirements.assets:
            assets.setdefault(asset.digest, asset)
        unattributed.update(requirements.unattributed_nodes)
    return WorkflowRequirements(
        tuple(packs[key] for key in sorted(packs)),
        tuple(assets[key] for key in sorted(assets)),
        tuple(sorted(unattributed)),
    )


@dataclass(frozen=True)
class WorkflowPlan:
    packs: tuple[PackRequirement, ...]
    assets: tuple[dict[str, object], ...]
    errors: tuple[str, ...]

    @property
    def empty(self) -> bool:
        return not (self.packs or self.assets or self.errors)

    def to_wire(self) -> dict[str, object]:
        return {
            "packs": [pack.to_wire() for pack in self.packs],
            "assets": list(self.assets),
            "errors": list(self.errors),
        }


def generation_device(generation: Path, variant: str) -> str:
    """Read a custom target's device family from its verified base, not its name or host."""
    record = _object(json.loads((generation / "generation.json").read_bytes()), "generation")
    if record.get("format") != "dinkster.generation/1":
        raise RequirementError("unsupported generation format")
    if record.get("variant") != variant:
        raise RequirementError("generation variant differs from workflow target")
    files = _object(record.get("files"), "generation files")
    base_path = generation / "base-manifest.json"
    base_data = base_path.read_bytes()
    if base_path.is_symlink() or "sha256:" + hashlib.sha256(base_data).hexdigest() != files.get(
        "base-manifest.json"
    ):
        raise RequirementError("generation base manifest differs from its inventory")
    base = _object(json.loads(base_data), "generation base")
    if (
        base.get("format") != "dinkster.base/1"
        or base.get("variant") != variant
        or base.get("id") != record.get("base")
    ):
        raise RequirementError("generation base identity differs from its record")
    wheels = cast("list[dict[str, object]]", base["wheels"])
    torch = next((wheel for wheel in wheels if wheel["name"] == "torch"), None)
    if torch is None:
        raise RequirementError("generation base has no torch device declaration")
    system = cast("dict[str, object]", base["platform"])["os"]
    python = Version(cast("str", base["python"]))
    library = (
        "Lib/site-packages"
        if system == "windows"
        else f"lib/python{python.major}.{python.minor}/site-packages"
    )
    name = f"python/{library}/torch/version.py"
    path = generation / name
    data = path.read_bytes()
    if path.is_symlink() or "sha256:" + hashlib.sha256(data).hexdigest() != files.get(name):
        raise RequirementError("generation torch build metadata differs from its inventory")
    return _torch_device(data, version=cast("str", torch["version"]), system=cast("str", system))


def acquired_base_device(base: Mapping[str, object], wheel: Path) -> str:
    """Derive preflight device from the verified acquired base's exact retained torch wheel."""
    wheels = cast("list[dict[str, object]]", base["wheels"])
    rows = [row for row in wheels if row["name"] == "torch"]
    if len(rows) != 1:
        raise RequirementError("acquired base must declare one torch wheel")
    row = rows[0]
    if (
        wheel.is_symlink()
        or not wheel.is_file()
        or wheel.name != row["filename"]
        or wheel.stat().st_size != row["size"]
    ):
        raise RequirementError("acquired torch wheel differs from its manifest")
    with wheel.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
            raise RequirementError("acquired torch wheel differs from its manifest")
        stream.seek(0)
        try:
            with ZipFile(stream) as archive:
                members = [row for row in archive.infolist() if row.filename == "torch/version.py"]
                if len(members) != 1:
                    raise RequirementError("acquired torch wheel must contain one torch/version.py")
                data = archive.read(members[0])
        except BadZipFile as exc:
            raise RequirementError("acquired torch wheel metadata is invalid") from exc
    system = cast("dict[str, object]", base["platform"])["os"]
    return _torch_device(data, version=cast("str", row["version"]), system=cast("str", system))


def _torch_device(data: bytes, *, version: str, system: str) -> str:
    """Read literal torch build declarations without executing the metadata module."""
    metadata: dict[str, object] = {}
    fields = {"__version__", "cuda", "hip", "rocm", "xpu"}
    try:
        module = ast.parse(data)
    except (SyntaxError, ValueError) as exc:
        raise RequirementError("torch build metadata is not valid Python") from exc
    declarations = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in fields
    ]
    for statement in module.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            target, value = statement.target, statement.value
        elif (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            target, value = statement.targets[0], statement.value
        else:
            continue
        if target.id not in fields:
            continue
        if target.id in metadata:
            raise RequirementError("torch build metadata has repeated declarations")
        if not isinstance(value, ast.Constant):
            raise RequirementError("torch build metadata requires literal declarations")
        metadata[target.id] = value.value
    if len(declarations) != len(metadata):
        raise RequirementError(
            "torch build metadata requires unique top-level literal declarations"
        )
    if metadata.get("__version__") != version:
        raise RequirementError("torch version differs from its base")
    devices: set[str] = set()
    for field, device in (("hip", "rocm"), ("rocm", "rocm"), ("cuda", "cuda"), ("xpu", "xpu")):
        value = metadata.get(field)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise RequirementError("torch build metadata has an invalid device declaration")
            devices.add(device)
    if len(devices) > 1 or (
        metadata.get("hip") and metadata.get("rocm") and metadata["hip"] != metadata["rocm"]
    ):
        raise RequirementError("torch build metadata has conflicting device declarations")
    if devices:
        return devices.pop()
    if "cuda" in metadata and "hip" in metadata:
        return "mps" if system == "macos" else "cpu"
    raise RequirementError("torch build has no declared device family")


def manifest_unavailability(
    requirement: PackRequirement, manifest: PackManifest, variant: str, *, device: str | None = None
) -> str | None:
    """Only manifest-declared resource needs can make portable packs hardware-specific."""
    if device is None and variant not in VARIANT_DEVICES:
        raise RequirementError(f"unknown engine variant: {variant}")
    device = VARIANT_DEVICES[variant] if device is None else device
    if manifest.sandbox.gpu and device == "cpu":
        return "pack.sandbox.gpu requires a GPU"
    for provider in manifest.vision_providers:
        if provider.node in requirement.node_types and device not in provider.devices:
            return (
                f"node {provider.node} declares devices {', '.join(provider.devices)}; "
                f"{device} is not supported"
            )
    return None


def plan_workflow_requirements(
    requirements: WorkflowRequirements,
    *,
    installed_packs: Mapping[str, str | None],
    asset_available: Callable[[AssetRef], bool],
    sources_for: Callable[[str], Sequence[AssetSource]],
    pack_unavailable: Callable[[PackRequirement], str | None],
) -> WorkflowPlan:
    """Read-only missing-item plan. Callbacks are host facts, never workflow authority.

    The availability callback must only inspect local, authorized bytes; it must
    not invoke a fetching resolver. Source leads are returned without fetching.
    Pack provisioning owns dependency-lock compatibility and executes only after
    the caller confirms this plan.
    """
    packs: list[PackRequirement] = []
    assets: list[dict[str, object]] = []
    errors = [f"node {node}: no saved pack attribution" for node in requirements.unattributed_nodes]
    for requirement in requirements.packs:
        if requirement.version is None:
            errors.append(f"pack {requirement.pack}: no saved version")
        reason = pack_unavailable(requirement)
        if reason is not None:
            errors.append(f"pack {requirement.pack}: {reason}")
        installed = requirement.pack in installed_packs
        if not installed or installed_packs[requirement.pack] != requirement.version:
            packs.append(requirement)
    for asset in requirements.assets:
        if not asset_available(asset):
            sources = sources_for(asset.digest)
            entry: dict[str, object] = {
                **asset.to_wire(),
                "sources": list(
                    dict.fromkeys(
                        source.url for source in sources if isinstance(source, RemoteSource)
                    )
                ),
            }
            packaged = sorted(
                {source.pack for source in sources if isinstance(source, PackagedSource)}
            )
            if packaged:
                entry["packagedFrom"] = packaged
            assets.append(entry)
    return WorkflowPlan(tuple(packs), tuple(assets), tuple(errors))


def main(argv: list[str] | None = None) -> int:
    """Print a server's read-only plan for one or more saved documents."""
    import argparse
    import json
    import sys
    import urllib.error
    import urllib.request
    from pathlib import Path
    from zipfile import BadZipFile

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] in (["project-plan"], ["project-apply"]):
        from dinkster_assets import AssetError
        from dinkster_assets.vault import AssetPublicationError
        from dinkster_registry.artifact import ArtifactError
        from dinkster_registry.install import InstallError
        from dinkster_workers import ManifestError

        from .compose import CompositionError
        from .project_requirements import apply_project_requirements, plan_project_requirements

        parser = argparse.ArgumentParser(description="Read a project's workflow requirement plan")
        parser.add_argument("--project-root", required=True, type=Path)
        parser.add_argument("--source-generation", type=Path)
        parser.add_argument("--generation", type=Path)
        parser.add_argument("--catalog-lock", type=Path)
        parser.add_argument("--artifacts", type=Path)
        parser.add_argument("--variant", required=True)
        parser.add_argument("--json", action="store_true", required=True)
        parser.add_argument("--yes", action="store_true")
        args = parser.parse_args(arguments[1:])
        try:
            for path in (
                args.project_root,
                args.source_generation,
                args.generation,
                args.catalog_lock,
                args.artifacts,
            ):
                if path is not None and not path.is_absolute():
                    raise RequirementError("project and generation/catalog paths must be absolute")
            if arguments[0] == "project-apply":
                if not args.yes or args.generation is None:
                    raise RequirementError("project-apply requires --yes and --generation")
                approved = dict(_object(json.load(sys.stdin), "approved project plan"))
                result = apply_project_requirements(
                    args.project_root,
                    approved=approved,
                    generation=args.generation,
                    variant=args.variant,
                    source_generation=args.source_generation,
                    catalog_lock=args.catalog_lock,
                    artifacts=args.artifacts,
                )
            else:
                result = plan_project_requirements(
                    args.project_root,
                    variant=args.variant,
                    source_generation=args.source_generation,
                    target_generation=args.generation,
                    catalog_lock=args.catalog_lock,
                    artifacts=args.artifacts,
                )
            print(json.dumps(result))
            plan = cast("dict[str, object]", result["plan"])
            if plan["errors"]:
                print("project requirement plan has blocking errors", file=sys.stderr)
            return int(bool(plan["errors"]))
        except (
            OSError,
            ValueError,
            AssetError,
            AssetPublicationError,
            ArtifactError,
            InstallError,
            ManifestError,
            CompositionError,
            BadZipFile,
        ) as exc:
            print(f"project plan failed: {exc}", file=sys.stderr)
            return 1

    parser = argparse.ArgumentParser(description="Print workflow requirements without installing")
    parser.add_argument("workflows", nargs="+", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:3639")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--project")
    args = parser.parse_args(arguments)
    try:
        documents = [json.loads(path.read_text("utf-8")) for path in args.workflows]
        body = {"workflows": documents, "variant": args.variant}
        if args.project is not None:
            body["project"] = args.project
        request = urllib.request.Request(
            args.server.rstrip("/") + "/api/workflows/plan",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        print(json.dumps(result, indent=2))
        return int(bool(result["errors"]))
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"workflow plan failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
