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
ARTIFACT_PATHS = (
    "packages/dinkster-compat-comfy/src/dinkster_compat_comfy/core_schemas.json",
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
