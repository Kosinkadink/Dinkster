"""Derive portable requirements from saved workflow documents."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from dinkster_assets import AssetRef


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
        packs[pack] = PackRequirement(pack, cast("str | None", version))
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
            packs[pack.pack] = pack
        for asset in requirements.assets:
            assets.setdefault(asset.digest, asset)
        unattributed.update(requirements.unattributed_nodes)
    return WorkflowRequirements(
        tuple(packs[key] for key in sorted(packs)),
        tuple(assets[key] for key in sorted(assets)),
        tuple(sorted(unattributed)),
    )
