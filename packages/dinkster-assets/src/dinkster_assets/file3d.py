"""Original-format 3D files with explicitly granted source and sidecar identities."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import PurePosixPath
from typing import cast

from dinkster_values import TypeRegistry, TypeSpec, stable_hash
from dinkster_values.builtin_types import (
    FILE_3D_FBX_TYPE,
    FILE_3D_GLB_TYPE,
    FILE_3D_GLTF_TYPE,
    FILE_3D_KSPLAT_TYPE,
    FILE_3D_OBJ_TYPE,
    FILE_3D_PLY_TYPE,
    FILE_3D_POINT_CLOUD_ANY_TYPE,
    FILE_3D_SPLAT_ANY_TYPE,
    FILE_3D_SPLAT_TYPE,
    FILE_3D_SPZ_TYPE,
    FILE_3D_STL_TYPE,
    FILE_3D_TYPE,
    FILE_3D_USDZ_TYPE,
)

from .identity import AssetError
from .model import AssetRef, AssetResolver, LocalAssetRef, LocalAssetResolver

FILE3D_FORMATS = {
    FILE_3D_TYPE: None,
    FILE_3D_GLB_TYPE: frozenset({"glb"}),
    FILE_3D_GLTF_TYPE: frozenset({"gltf"}),
    FILE_3D_FBX_TYPE: frozenset({"fbx"}),
    FILE_3D_OBJ_TYPE: frozenset({"obj"}),
    FILE_3D_STL_TYPE: frozenset({"stl"}),
    FILE_3D_USDZ_TYPE: frozenset({"usdz"}),
    FILE_3D_PLY_TYPE: frozenset({"ply"}),
    FILE_3D_SPLAT_TYPE: frozenset({"splat"}),
    FILE_3D_SPZ_TYPE: frozenset({"spz"}),
    FILE_3D_KSPLAT_TYPE: frozenset({"ksplat"}),
    FILE_3D_SPLAT_ANY_TYPE: frozenset({"ply", "splat", "spz", "ksplat"}),
    FILE_3D_POINT_CLOUD_ANY_TYPE: frozenset({"ply"}),
}


def register_file3d_types(
    registry: TypeRegistry, resolver: AssetResolver | None = None
) -> tuple[TypeSpec, ...]:
    """Bind original file identities to the receiving process's asset authority."""

    def asset(value: object) -> AssetRef | LocalAssetRef:
        if isinstance(value, LocalAssetRef):
            if value.resolver is None and isinstance(resolver, LocalAssetResolver):
                value = replace(value, resolver=resolver)
            value.local_path()
            return value.verified() or value
        if isinstance(value, AssetRef):
            return replace(value, resolver=resolver) if value.resolver is None else value
        if not isinstance(value, Mapping):
            raise AssetError("FILE_3D sources require asset identities, not filesystem paths")
        wire = cast("Mapping[str, object]", value)
        if set(wire) - {
            "digest",
            "name",
            "size",
            "mediaType",
            "virtualPath",
            "verification",
            "localBinding",
        }:
            raise AssetError("FILE_3D source contains unknown fields")
        if wire.get("verification") == "unverified":
            return asset(
                LocalAssetRef.from_wire(
                    wire, resolver if isinstance(resolver, LocalAssetResolver) else None
                )
            )
        return AssetRef.from_wire(wire, resolver)

    def coerce(value: object) -> dict[str, object]:
        if not isinstance(value, Mapping):
            raise AssetError("FILE_3D requires format, source and optional resources")
        obj = cast("Mapping[str, object]", value)
        if set(obj) - {"format", "source", "resources"}:
            raise AssetError("FILE_3D contains unknown fields")
        format_id = obj.get("format")
        if not isinstance(format_id, str) or re.fullmatch(r"[a-z0-9]*", format_id) is None:
            raise AssetError("FILE_3D format must be a lowercase file extension or empty")
        resources = obj.get("resources", {})
        if not isinstance(resources, Mapping):
            raise AssetError("FILE_3D resources must map contained relative names to assets")
        bound: dict[str, AssetRef | LocalAssetRef] = {}
        for name, ref in cast("Mapping[object, object]", resources).items():
            if (
                not isinstance(name, str)
                or not name
                or name == "."
                or PurePosixPath(name).is_absolute()
                or str(PurePosixPath(name)) != name
                or ".." in PurePosixPath(name).parts
                or any(char in name for char in "\\:%?#\x00")
            ):
                raise AssetError(
                    "FILE_3D resource name must be a canonical contained relative path"
                )
            bound[name] = asset(ref)
        return {"format": format_id, "source": asset(obj.get("source")), "resources": bound}

    def wire(value: object) -> dict[str, object]:
        obj = coerce(value)
        source = cast("AssetRef | LocalAssetRef", obj["source"])
        resources = cast("dict[str, AssetRef | LocalAssetRef]", obj["resources"])
        return {
            "format": obj["format"],
            "source": source.to_wire(),
            "resources": {name: ref.to_wire() for name, ref in sorted(resources.items())},
        }

    def meta(value: object) -> dict[str, object]:
        obj = coerce(value)
        refs = [obj["source"], *cast("dict[str, object]", obj["resources"]).values()]
        return {
            "format": obj["format"],
            "asset_refs": [ref.to_wire() for ref in refs if isinstance(ref, AssetRef)],
        }

    def register(type_id: str, formats: frozenset[str] | None) -> TypeSpec:
        def typed(value: object) -> dict[str, object]:
            obj = coerce(value)
            if formats is not None and obj["format"] and obj["format"] not in formats:
                raise AssetError(f"{type_id} does not accept format {obj['format']!r}")
            return obj

        def encode(value: object) -> bytes:
            return json.dumps(wire(typed(value)), sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )

        def validate(data: bytes, metadata: Mapping[str, object]) -> None:
            obj = typed(json.loads(data))
            if dict(metadata) != meta(obj):
                raise AssetError("FILE_3D metadata does not match encoded source identities")

        spec = registry.register(
            type_id,
            encode=encode,
            decode=lambda data: typed(json.loads(data)),
            coerce=typed,
            fingerprint=lambda value: stable_hash([type_id.encode("utf-8"), encode(value)]),
            meta=meta,
            validate_encoded=validate,
        )
        registry.register_rendition(type_id, "manifest", mime="application/json", render=encode)
        return spec

    return tuple(
        registry.spec(type_id) if type_id in registry else register(type_id, formats)
        for type_id, formats in FILE3D_FORMATS.items()
    )
