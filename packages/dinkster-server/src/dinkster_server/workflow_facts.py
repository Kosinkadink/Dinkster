"""Read saved-document facts in the engine; the library never interprets graphs."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from dinkster_assets import AssetError, AssetRef, LocalAssetRef, is_digest
from dinkster_assets.model import NamedAssetResolver
from dinkster_schema import AssetWidget, NodeSchema, WidgetRepresentations

if TYPE_CHECKING:
    from .library import ServerLibrary


def workflow_facts(
    data: bytes, digest: str, schemas: Mapping[str, NodeSchema], library: ServerLibrary
) -> dict[str, object] | None:
    try:
        value: object = json.loads(data)
    except (ValueError, UnicodeError):
        return None
    if not isinstance(value, dict):
        return None
    document = cast("dict[str, object]", value)
    if document.get("format") != "dinkster-workflow" or document.get("formatVersion") != 1:
        return None
    graphs = document.get("graphs")
    environment = document.get("environment", {})
    if not isinstance(graphs, dict) or not isinstance(environment, dict):
        return None
    stamps = cast("dict[str, object]", environment).get("packs", {})
    if not isinstance(stamps, dict):
        return None
    packs: list[dict[str, object]] = []
    for pack, raw in cast("dict[str, object]", stamps).items():
        if not isinstance(raw, dict):
            return None
        stamp = cast("dict[str, object]", raw)
        packs.append(
            {
                "pack": pack,
                **{
                    key: stamp.get(key)
                    for key in ("version", "artifactDigest", "source", "publisher")
                },
            }
        )
    models: list[dict[str, object]] = []
    for graph_id, raw_graph in cast("dict[str, object]", graphs).items():
        if not isinstance(raw_graph, dict):
            return None
        nodes = cast("dict[str, object]", raw_graph).get("nodes")
        if not isinstance(nodes, dict):
            return None
        for node_id, raw_node in cast("dict[str, object]", nodes).items():
            if not isinstance(raw_node, dict):
                return None
            node = cast("dict[str, object]", raw_node)
            node_type, values = node.get("type"), node.get("values", {})
            schema = schemas.get(node_type) if isinstance(node_type, str) else None
            if not isinstance(node_type, str) or not isinstance(values, dict):
                return None
            if schema is None:
                if node_type.startswith("#") or node.get("virtual") is True:
                    continue
                return None
            inputs = cast("dict[str, object]", values)
            for spec in schema.inputs:
                widget = spec.widget
                if isinstance(widget, WidgetRepresentations):
                    widget = widget.representations[0].widget
                if not isinstance(widget, AssetWidget) or not widget.kind.startswith("model/"):
                    continue
                selected = inputs.get(spec.id)
                if isinstance(selected, dict):
                    wire = cast("dict[str, object]", selected)
                    name, identity = wire.get("name"), wire.get("digest")
                    identity = (
                        identity if isinstance(identity, str) and is_digest(identity) else None
                    )
                    if identity is None and isinstance(library.resolver, NamedAssetResolver):
                        try:
                            ref = library.resolver.bind_name(str(name or ""), widget.kind)
                            verified = ref.verified() if isinstance(ref, LocalAssetRef) else ref
                            identity = verified.digest if isinstance(verified, AssetRef) else None
                        except (AssetError, OSError):
                            pass
                elif isinstance(selected, str) and selected:
                    name, identity = selected, None
                    if isinstance(library.resolver, NamedAssetResolver):
                        try:
                            ref = library.resolver.bind_name(selected, widget.kind)
                            verified = ref.verified() if isinstance(ref, LocalAssetRef) else ref
                            identity = verified.digest if isinstance(verified, AssetRef) else None
                        except (AssetError, OSError):
                            pass
                else:
                    continue
                if isinstance(name, str) and name:
                    models.append(
                        {
                            "location": json.dumps([graph_id, node_id, spec.id]),
                            "name": name,
                            "kind": widget.kind,
                            "digest": identity,
                        }
                    )
    pending: list[object] = [document]
    machine_paths = False
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(cast("dict[str, object]", item).values())
        elif isinstance(item, list):
            pending.extend(cast("list[object]", item))
        elif isinstance(item, str) and re.match(r"^(?:/|[A-Za-z]:[\\/]|\\\\|local:)", item):
            machine_paths = True
    return {"digest": digest, "packs": packs, "models": models, "machinePaths": machine_paths}
