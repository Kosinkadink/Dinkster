"""Durable adoption of asset-backed results before producer acknowledgement."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import cast

from dinkster_assets import AssetError, AssetRef, AssetResolver, AssetVault, open_verified
from dinkster_values import (
    ASSET_BASE_TYPE,
    MEBIBYTE,
    Value,
    default_decode,
    iter_value_tree,
    parse_asset_type_id,
)

_COPY_BYTES = MEBIBYTE


def _references(raw: object, resolver: AssetResolver | None = None) -> Iterator[AssetRef]:
    if not isinstance(raw, list):
        raise AssetError("asset_refs must be a list")
    for item in cast("list[object]", raw):
        if not isinstance(item, Mapping):
            raise AssetError("asset reference must be a mapping")
        yield AssetRef.from_wire(cast("Mapping[str, object]", item), resolver)


def result_asset_digests(header: Mapping[str, object], blobs: Sequence[bytes]) -> set[str]:
    """Pin source blobs on frame receipt, before another query can replace transfer pins."""
    digests: set[str] = set()

    def visit(raw: object) -> None:
        if not isinstance(raw, Mapping):
            raise AssetError("result value must be a mapping")
        wire = cast("Mapping[str, object]", raw)
        index = wire.get("metaBlob")
        if type(index) is not int or not 0 <= index < len(blobs):
            raise AssetError("result metadata blob index is invalid")
        try:
            metadata = default_decode(blobs[index])
        except Exception as exc:
            raise AssetError("result metadata blob is invalid") from exc
        if not isinstance(metadata, Mapping):
            raise AssetError("result metadata must be a mapping")
        digests.update(
            ref.digest
            for ref in _references(cast("Mapping[str, object]", metadata).get("asset_refs", []))
        )
        children = wire.get("elements", [])
        if not isinstance(children, list):
            raise AssetError("result value elements must be a list")
        for child in cast("list[object]", children):
            visit(child)

    outputs = header.get("outputs", {})
    if not isinstance(outputs, Mapping):
        raise AssetError("result outputs must be a mapping")
    for wire in cast("Mapping[str, object]", outputs).values():
        visit(wire)
    return digests


def _source_references(
    values: Mapping[str, Value], *, include_asset_inputs: bool = False
) -> dict[str, AssetRef]:
    references: dict[str, AssetRef] = {}
    for value in values.values():
        for child in iter_value_tree(value):
            refs = list(_references(child.meta.get("asset_refs", [])))
            if include_asset_inputs and (
                child.type_id == ASSET_BASE_TYPE or parse_asset_type_id(child.type_id) is not None
            ):
                refs.append(AssetRef.from_wire(child.meta.entries))
            for ref in refs:
                previous = references.setdefault(ref.digest, ref)
                if previous.size != ref.size:
                    raise AssetError("source references disagree about byte size")
    return references


def source_asset_files(
    outputs: Mapping[str, Value], resolver: AssetResolver | None
) -> list[tuple[str, Path]]:
    """Resolve declared source identities without decoding value payloads."""
    files: dict[str, Path] = {}
    for output in outputs.values():
        for value in iter_value_tree(output):
            for ref in _references(value.meta.get("asset_refs", []), resolver):
                if ref.digest not in files:
                    files[ref.digest] = ref.local_path()
    return list(files.items())


class ProducedAssetAuthority:
    """Copy digest-addressed results from configured stores into the engine vault."""

    def __init__(
        self,
        vault: AssetVault | None,
        source: AssetResolver | None = None,
        existing: AssetResolver | None = None,
    ) -> None:
        self._vault = vault
        self._source = source
        self._existing = existing or vault

    def capture(
        self,
        outputs: Mapping[str, Value],
        transferred: AssetResolver | None = None,
        inputs: Mapping[str, Value] | None = None,
    ) -> None:
        incoming = _source_references(inputs or {}, include_asset_inputs=True)
        for ref in _source_references(outputs).values():
            if ref.digest in incoming:
                if incoming[ref.digest].size != ref.size:
                    raise AssetError("source references disagree about byte size")
                # Returning a caller-owned source does not create a producer-owned asset.
                continue
            held = self._existing.resolve(ref.digest) if self._existing is not None else None
            if held is not None:
                with open_verified(held, ref.digest) as handle:
                    if os.fstat(handle.fileno()).st_size != ref.size:
                        raise AssetError("source byte size does not match engine vault")
                continue
            if self._vault is None:
                raise AssetError("produced sources require an engine DINKSTER_ASSET_VAULT")
            path = transferred.resolve(ref.digest) if transferred is not None else None
            if path is None and self._source is not None:
                path = self._source.resolve(ref.digest)
            if path is None:
                raise AssetError(f"source {ref.digest} did not reach the engine")
            with path.open("rb") as handle, self._vault.writer(ref.digest) as writer:
                count = 0
                while chunk := handle.read(_COPY_BYTES):
                    count += len(chunk)
                    if count > ref.size:
                        raise AssetError("source transfer exceeds its declared byte size")
                    writer.write(chunk)
                if count != ref.size:
                    raise AssetError("source transfer is truncated")
                writer.commit()
