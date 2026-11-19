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
    data: bytes,
    digest: str,
    schemas: Mapping[str, NodeSchema],
    library: ServerLibrary,
    *,
    node_packs: Mapping[str, str] | None = None,
    include_models: bool = True,
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
    stamped_nodes = cast("dict[str, object]", environment).get("nodes", {})
    if not isinstance(stamps, dict) or not isinstance(stamped_nodes, dict):
        return None
    used_packs: set[str] = set()
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
            if node_type.startswith("#") or node.get("virtual") is True:
                continue
            if schema is None and include_models:
                return None
            raw_stamp = cast("dict[str, object]", stamped_nodes).get(node_type, {})
            if not isinstance(raw_stamp, dict):
                return None
            pack = cast("dict[str, object]", raw_stamp).get("pack")
            if pack is None and node_packs is not None and schema is not None:
                pack = node_packs.get(node_type, "core")
            if not isinstance(pack, str) or not pack:
                return None
            used_packs.add(pack)
            if not include_models:
                continue
            assert schema is not None
            inputs = cast("dict[str, object]", values)
            for spec in schema.inputs:
                widget = spec.widget
                if isinstance(widget, WidgetRepresentations):
                    widget = widget.representations[0].widget
                if not isinstance(widget, AssetWidget) or not widget.kind.startswith("model/"):
                    continue
                selected = inputs.get(spec.id)
                selections: list[tuple[object, list[str | int]]] = [
                    (selected, [graph_id, node_id, spec.id])
                ]
                while selections:
                    selected, location = selections.pop()
                    if isinstance(selected, list):
                        selections.extend(
                            (item, [*location, index])
                            for index, item in reversed(
                                list(enumerate(cast("list[object]", selected)))
                            )
                        )
                        continue
                    if isinstance(selected, dict):
                        wire = cast("dict[str, object]", selected)
                        name, identity = wire.get("name"), wire.get("digest")
                        identity = (
                            identity if isinstance(identity, str) and is_digest(identity) else None
                        )
                    elif isinstance(selected, str):
                        name, identity = selected, None
                    else:
                        continue
                    if not isinstance(name, str) or not name:
                        continue
                    if identity is None and isinstance(library.resolver, NamedAssetResolver):
                        try:
                            ref = library.resolver.bind_name(name, widget.kind)
                            verified = ref.verified() if isinstance(ref, LocalAssetRef) else ref
                            identity = verified.digest if isinstance(verified, AssetRef) else None
                        except (AssetError, OSError):
                            pass
                    models.append(
                        {
                            "location": json.dumps(location),
                            "name": name,
                            "kind": widget.kind,
                            "digest": identity,
                        }
                    )
    packs: list[dict[str, object]] = []
    for pack in sorted(used_packs - {"core"}):
        raw = cast("dict[str, object]", stamps).get(pack, {})
        if not isinstance(raw, dict):
            return None
        stamp = cast("dict[str, object]", raw)
        if any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in (
                stamp.get(key) for key in ("version", "artifactDigest", "source", "publisher")
            )
        ):
            return None
        packs.append(
            {
                "pack": pack,
                **{
                    key: stamp.get(key)
                    for key in ("version", "artifactDigest", "source", "publisher")
                },
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
