"""Original-format 3D files with explicitly granted source and sidecar identities."""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import struct
from collections.abc import Mapping
from dataclasses import replace
from pathlib import PurePosixPath
from typing import cast
from urllib.parse import unquote, urlsplit

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

from .declared import MOUNTS_ENV
from .identity import AssetError
from .model import AssetRef, AssetResolver, LocalAssetRef, LocalAssetResolver
from .mounts import MountSnapshotResolver, MountTable

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
    FILE_3D_SPLAT_ANY_TYPE: frozenset({"", "ply", "splat", "spz", "ksplat"}),
    FILE_3D_POINT_CLOUD_ANY_TYPE: frozenset({"", "ply"}),
}


def _resource_name(name: object) -> str:
    if (
        not isinstance(name, str)
        or not name
        or name == "."
        or PurePosixPath(name).is_absolute()
        or str(PurePosixPath(name)) != name
        or ".." in PurePosixPath(name).parts
        or any(char in name for char in "\\:%?#")
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
    ):
        raise AssetError("FILE_3D resource name must be a canonical contained relative path")
    return name


def _gltf_resource_name(uri: object) -> str:
    if not isinstance(uri, str) or any(ord(char) < 32 or ord(char) == 127 for char in uri):
        raise AssetError("FILE_3D resource URI must be a contained relative path")
    parsed = urlsplit(uri)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise AssetError("FILE_3D resource URI must be a contained relative path")
    return _resource_name(unquote(parsed.path, errors="strict"))


def _mtl_texture_name(declaration: str) -> str:
    stream = io.StringIO(declaration)
    lexer = shlex.shlex(stream, posix=True)
    lexer.whitespace_split = True
    tokens: list[tuple[str, int]] = []
    while True:
        start = stream.tell()
        token = lexer.get_token()
        if token is None:
            break
        tokens.append((token, start))
    fixed = {
        "-blendu": 1,
        "-blendv": 1,
        "-boost": 1,
        "-mm": 2,
        "-texres": 1,
        "-clamp": 1,
        "-bm": 1,
        "-imfchan": 1,
        "-type": 1,
        "-cc": 1,
    }
    index = 0
    while index < len(tokens):
        option = tokens[index][0]
        if option in {"-o", "-s", "-t"}:
            index += 1
            count = 0
            while index < len(tokens) and count < 3:
                try:
                    float(tokens[index][0])
                except ValueError:
                    break
                index += 1
                count += 1
            if count == 0:
                raise AssetError(f"MTL texture option {option} requires a numeric argument")
        elif option in fixed:
            index += 1 + fixed[option]
        else:
            break
    if index >= len(tokens):
        raise AssetError("MTL texture declaration requires a filename")
    remainder = declaration[tokens[index][1] :].split("#", 1)[0].strip()
    if remainder.startswith(("'", '"')):
        names = shlex.split(remainder)
        if len(names) != 1:
            raise AssetError("MTL texture declaration requires one filename")
        remainder = names[0]
    return _resource_name(remainder)


def load_file3d(
    source: AssetRef | LocalAssetRef | str | None,
    *,
    resources: Mapping[str, AssetRef | LocalAssetRef] | None = None,
) -> dict[str, object] | None:
    """Preserve a selected file and bind its declared sidecars within existing grants.

    Explicit resources support uploaded groups and sources outside mounted folders.
    Otherwise only the selected file's existing input/model3d grant can supply them.
    Neither resource names nor the source's display name introduce filesystem access.
    """
    if source is None or isinstance(source, str) and source in {"", "none"}:
        if resources:
            raise AssetError("an empty FILE_3D cannot have resources")
        return None
    if not isinstance(source, (AssetRef, LocalAssetRef)):
        raise AssetError("FILE_3D loading requires a bound asset, not a filename")
    path = source.local_path()
    format_id = PurePosixPath(source.name).suffix.lower().removeprefix(".")
    if re.fullmatch(r"[a-z0-9]*", format_id) is None:
        raise AssetError("FILE_3D format must be a lowercase file extension or empty")
    supplied = dict(resources) if resources is not None else None
    if supplied is not None:
        for name in supplied:
            _resource_name(name)
    bound: dict[str, AssetRef | LocalAssetRef] = {}
    snapshot = os.environ.get(MOUNTS_ENV)
    resolver = (
        source.resolver
        if isinstance(source.resolver, (MountTable, MountSnapshotResolver))
        else MountSnapshotResolver(snapshot)
        if snapshot
        else None
    )

    def bind(name: str) -> AssetRef | LocalAssetRef:
        name = _resource_name(name)
        if name in bound:
            return bound[name]
        ref = supplied.get(name) if supplied is not None else None
        if ref is None and supplied is None and resolver is not None:
            for kind in ("media/model3d", "directory/input"):
                for root in resolver.roots_for_kind(kind):
                    root = root.resolve()
                    if path.is_relative_to(root):
                        relative = path.parent.relative_to(root) / name
                        if isinstance(resolver, MountTable):
                            ref = resolver.bind_relative_file(kind, root, relative.as_posix())
                        else:
                            ref = resolver.bind_relative_file(
                                kind, root, relative.as_posix(), require_host_grant=True
                            )
                        break
                if ref is not None:
                    break
        if ref is None:
            raise AssetError(f"FILE_3D resource {name!r} requires an explicit asset or mount grant")
        with ref.open():
            pass
        bound[name] = ref
        return ref

    if format_id in {"gltf", "glb"}:
        with source.open() as handle:
            if format_id == "glb":
                header = handle.read(20)
                if len(header) != 20:
                    raise AssetError("GLB source has an incomplete header")
                magic, version, _length, chunk_length, chunk_kind = struct.unpack("<4sIIII", header)
                if magic != b"glTF" or version != 2 or chunk_kind != 0x4E4F534A:
                    raise AssetError("GLB source requires a version2 JSON chunk")
                document = json.loads(handle.read(chunk_length))
            else:
                document = json.load(handle)
        for section in ("buffers", "images"):
            for entry in document.get(section, []):
                uri = entry.get("uri")
                if uri is not None and not (isinstance(uri, str) and uri.startswith("data:")):
                    bind(_gltf_resource_name(uri))
    elif format_id == "obj":
        text = source.read_bytes().decode("utf-8-sig").replace("\r\n", "\n").replace("\\\n", "")
        for line in text.splitlines():
            words = line.split(maxsplit=1)
            if len(words) != 2 or words[0] != "mtllib":
                continue
            for name in shlex.split(words[1], comments=True):
                material = bind(name)
                text = (
                    material.read_bytes()
                    .decode("utf-8-sig")
                    .replace("\r\n", "\n")
                    .replace("\\\n", "")
                )
                for declaration in text.splitlines():
                    parts = declaration.split(maxsplit=1)
                    if len(parts) == 2 and (
                        parts[0].startswith("map_")
                        or parts[0] in {"bump", "disp", "decal", "norm", "refl"}
                    ):
                        texture = _mtl_texture_name(parts[1])
                        bind((PurePosixPath(name).parent / texture).as_posix())
    if supplied is not None and supplied.keys() != bound.keys():
        raise AssetError("FILE_3D resources contain undeclared files")
    return {"format": format_id, "source": source, "resources": bound}


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

    def coerce(value: object) -> dict[str, object] | None:
        if value is None:
            return None
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
            bound[_resource_name(name)] = asset(ref)
        return {"format": format_id, "source": asset(obj.get("source")), "resources": bound}

    def wire(value: object) -> dict[str, object] | None:
        obj = coerce(value)
        if obj is None:
            return None
        source = cast("AssetRef | LocalAssetRef", obj["source"])
        resources = cast("dict[str, AssetRef | LocalAssetRef]", obj["resources"])
        return {
            "format": obj["format"],
            "source": source.to_wire(),
            "resources": {name: ref.to_wire() for name, ref in sorted(resources.items())},
        }

    def meta(value: object) -> dict[str, object]:
        obj = coerce(value)
        if obj is None:
            return {"asset_refs": []}
        refs = [obj["source"], *cast("dict[str, object]", obj["resources"]).values()]
        return {
            "format": obj["format"],
            "asset_refs": [ref.to_wire() for ref in refs if isinstance(ref, AssetRef)],
        }

    def register(type_id: str, formats: frozenset[str] | None) -> TypeSpec:
        def typed(value: object) -> dict[str, object] | None:
            obj = coerce(value)
            if obj is not None and formats is not None and obj["format"] not in formats:
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
