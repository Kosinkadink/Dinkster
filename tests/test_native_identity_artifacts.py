from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from dinkster_schema import TypeExpr, type_expr_from_wire

ROOT = Path(__file__).parent.parent
IDENTITIES = {
    "comfy.AUDIO": "AUDIO",
    "comfy.VIDEO": "VIDEO",
    "comfy.VIDEO_EDIT": "VIDEO_EDIT",
    "comfy.MOGE_MODEL": "MOGE_MODEL",
    "comfy.MOGE_GEOMETRY": "MOGE_GEOMETRY",
    "comfy.BACKGROUND_REMOVAL": "BACKGROUND_REMOVAL",
    "comfy.VOXEL": "VOXEL",
    "comfy.MESH": "MESH",
    "comfy.SHAPE_SUBDIVIDES": "SHAPE_SUBDIVIDES",
    "comfy.BOUNDING_BOX": "BOUNDING_BOX",
    "comfy.MODEL_PATCH": "MODEL_PATCH",
    "comfy.LATENT_UPSCALE_MODEL": "LATENT_UPSCALE_MODEL",
    "comfy.STYLE_MODEL": "STYLE_MODEL",
}
CORE_SCHEMA_IDENTITIES = {
    **IDENTITIES,
    "comfy.CONTROL_NET": "CONTROL_NET",
    "comfy.GLIGEN": "GLIGEN",
}
CORE_SCHEMA_PATH = "packages/dinkster-compat-comfy/src/dinkster_compat_comfy/core_schemas.json"
ARTIFACT_PATHS = (
    CORE_SCHEMA_PATH,
    "packages/dinkster-compat-comfy/comfy-aliases.inactive.json",
    "packages/dinkster-nodes-generation/comfy-aliases.inactive.json",
    "packages/dinkster-nodes-generation/comfy-groups.inactive.json",
    "packages/dinkster-nodes-image/comfy-aliases.json",
    "packages/dinkster-nodes-image/comfy-groups.json",
    "packages/dinkster-nodes-media-io/comfy-aliases.json",
)
TYPE_EXPR_KINDS = frozenset(
    {"concrete", "union", "wildcard", "variable", "list", "asset", "stream"}
)


def _type_atoms(type_expression: TypeExpr) -> Iterator[str]:
    yield from type_expression.types
    if type_expression.element is not None:
        yield from _type_atoms(type_expression.element)


def _artifact_type_atoms(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "type" and isinstance(child, dict) and child.get("kind") in TYPE_EXPR_KINDS:
                yield from _type_atoms(type_expr_from_wire(child))
            else:
                yield from _artifact_type_atoms(child)
    elif isinstance(value, list):
        for child in value:
            yield from _artifact_type_atoms(child)


def test_retained_artifacts_use_bare_approved_native_type_identities() -> None:
    type_atoms: set[str] = set()
    opaque_types: set[str] = set()

    for path in ARTIFACT_PATHS:
        artifact = json.loads((ROOT / path).read_text(encoding="utf-8"))
        type_atoms.update(_artifact_type_atoms(artifact))
        if "opaqueTypes" in artifact:
            opaque_types.update(artifact["opaqueTypes"])

    for prefixed, canonical in IDENTITIES.items():
        assert prefixed not in type_atoms
        assert prefixed not in opaque_types
        assert canonical in type_atoms or canonical in opaque_types

    core_schema = json.loads((ROOT / CORE_SCHEMA_PATH).read_text(encoding="utf-8"))
    core_type_atoms = set(_artifact_type_atoms(core_schema))
    core_opaque_types = set(core_schema["opaqueTypes"])
    for prefixed, canonical in CORE_SCHEMA_IDENTITIES.items():
        assert prefixed not in core_type_atoms
        assert prefixed not in core_opaque_types
        assert canonical in core_type_atoms or canonical in core_opaque_types


def test_core_schema_control_net_leaves_are_canonical_and_retained_metadata_is_unchanged() -> None:
    core_schema = json.loads((ROOT / CORE_SCHEMA_PATH).read_text(encoding="utf-8"))

    assert core_schema["sourceRepository"] == "https://github.com/Comfy-Org/ComfyUI"
    assert core_schema["sourceCommit"] == "15eb748b3ec5f8a0a2d470b7fb280e2d7579f916"
    assert core_schema["opaqueTypes"].count("CONTROL_NET") == 1
    assert "comfy.CONTROL_NET" not in core_schema["opaqueTypes"]

    expected_leaves = {
        "comfy.ControlNetApply": ("control_net",),
        "comfy.ControlNetApplyAdvanced": ("control_net",),
        "comfy.ControlNetApplySD3": ("control_net",),
        "comfy.ControlNetInpaintingAliMamaApply": ("control_net",),
        "comfy.ControlNetLoader": ("control_net",),
        "comfy.DiffControlNetLoader": ("control_net",),
        "comfy.SetUnionControlNetType": ("control_net", "CONTROL_NET"),
    }
    actual_leaves: dict[str, tuple[str, ...]] = {}
    for node_id, schema in core_schema["schemas"].items():
        ids = tuple(
            item["id"]
            for item in schema["interface"]
            if item.get("type", {}).get("types") == ["CONTROL_NET"]
        )
        if ids:
            actual_leaves[node_id] = ids

    assert actual_leaves == expected_leaves


def test_core_schema_gligen_leaves_are_canonical_and_retained_metadata_is_unchanged() -> None:
    core_schema = json.loads((ROOT / CORE_SCHEMA_PATH).read_text(encoding="utf-8"))

    assert core_schema["sourceRepository"] == "https://github.com/Comfy-Org/ComfyUI"
    assert core_schema["sourceCommit"] == "15eb748b3ec5f8a0a2d470b7fb280e2d7579f916"
    assert core_schema["opaqueTypes"].count("GLIGEN") == 1
    assert "comfy.GLIGEN" not in core_schema["opaqueTypes"]

    actual_leaves = {
        node_id: tuple(
            item["id"]
            for item in schema["interface"]
            if item.get("type", {}).get("types") == ["GLIGEN"]
        )
        for node_id, schema in core_schema["schemas"].items()
    }
    assert {node_id: inputs for node_id, inputs in actual_leaves.items() if inputs} == {
        "comfy.GLIGENLoader": ("gligen",),
        "comfy.GLIGENTextBoxApply": ("gligen_textbox_model",),
    }
