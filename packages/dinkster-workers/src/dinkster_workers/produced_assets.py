"""Durable adoption of asset-backed results before producer acknowledgement."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import cast

from dinkster_assets import (
    AssetError,
    AssetRef,
    AssetResolver,
    AssetVault,
    LocalAssetRef,
    open_verified,
)
from dinkster_assets.declared import offer_produced_outputs
from dinkster_assets.produced import ProducedAssetInventory
from dinkster_assets.scratch import hold_produced_vault
from dinkster_assets.value import value_asset_references
from dinkster_values import (
    MEBIBYTE,
    Value,
    default_decode,
    iter_value_tree,
)

_COPY_BYTES = MEBIBYTE


def _references(raw: object, resolver: AssetResolver | None = None) -> Iterator[AssetRef]:
    if not isinstance(raw, list):
        raise AssetError("asset_refs must be a list")
    for item in cast("list[object]", raw):
        if not isinstance(item, Mapping):
            raise AssetError("asset reference must be a mapping")
        wire = cast("Mapping[str, object]", item)
        if wire.get("verification") == "unverified":
            LocalAssetRef.from_wire(wire)
            continue
        yield AssetRef.from_wire(wire, resolver)


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
    algebra = header.get("resultAlgebra")
    if algebra is not None:
        if not isinstance(algebra, Mapping):
            raise AssetError("result algebra must be a mapping")
        values = cast("Mapping[str, object]", algebra).get("values")
        if not isinstance(values, list):
            raise AssetError("result algebra values must be a list")
        for wire in cast("list[object]", values):
            visit(wire)
    from .boundary import decode_saved_file_previews

    for claim in decode_saved_file_previews(header):
        ref = AssetRef.from_wire(cast("Mapping[str, object]", claim["preview"]))
        digests.add(ref.digest)
    return digests


def offer_source_assets(outputs: Mapping[str, Value]) -> None:
    """Record publication handoff before transfer or direct engine adoption."""
    offer_produced_outputs(value_asset_references(outputs, include_asset_values=False))


def source_asset_files(
    outputs: Mapping[str, Value],
    resolver: AssetResolver | None,
    *,
    previews: Sequence[Mapping[str, object]] = (),
) -> list[tuple[str, Path]]:
    """Resolve declared source identities without decoding value payloads."""
    files: dict[str, Path] = {}
    for output in outputs.values():
        for value in iter_value_tree(output):
            for ref in _references(value.meta.get("asset_refs", []), resolver):
                if ref.digest not in files:
                    files[ref.digest] = ref.local_path()
    for claim in previews:
        try:
            ref = AssetRef.from_wire(cast("Mapping[str, object]", claim["preview"]), resolver)
            files.setdefault(ref.digest, ref.local_path())
        except Exception:
            continue
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

    def capture_previews(
        self,
        claims: Sequence[Mapping[str, object]],
        transferred: AssetResolver | None = None,
    ) -> tuple[Mapping[str, object], ...]:
        """Adopt optional file-backed previews before the producer is acknowledged."""
        accepted: list[Mapping[str, object]] = []
        for claim in claims:
            try:
                ref = AssetRef.from_wire(cast("Mapping[str, object]", claim["preview"]))
                if self._vault is None:
                    continue
                inventory = ProducedAssetInventory(self._vault.root)
                held = self._vault.resolve(ref.digest)
                if held is not None:
                    with open_verified(held, ref.digest) as source:
                        if os.fstat(source.fileno()).st_size != ref.size:
                            continue
                        inventory.confirm_existing(ref.digest, ref.size)
                    accepted.append(claim)
                    continue
                path = transferred.resolve(ref.digest) if transferred is not None else None
                if path is None and self._source is not None:
                    path = self._source.resolve(ref.digest)
                if path is None:
                    continue
                with (
                    open_verified(path, ref.digest) as source,
                    closing(hold_produced_vault(self._vault.root)),
                ):
                    if os.fstat(source.fileno()).st_size != ref.size:
                        continue
                    inventory.pending(ref.digest, ref.size)
                    with self._vault.writer(ref.digest) as writer:
                        while chunk := source.read(_COPY_BYTES):
                            writer.write(chunk)
                        writer.commit()
                        inventory.adopted(ref.digest, ref.size)
                accepted.append(claim)
            except Exception:
                # Optional media cannot invalidate successfully produced sources.
                continue
        return tuple(accepted)

    def capture(
        self,
        outputs: Mapping[str, Value],
        transferred: AssetResolver | None = None,
        inputs: Mapping[str, Value] | None = None,
    ) -> None:
        incoming = value_asset_references(inputs or {})
        offer_source_assets(outputs)
        for ref in value_asset_references(outputs, include_asset_values=False).values():
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
                    if self._vault is not None and self._vault.resolve(ref.digest) == held:
                        ProducedAssetInventory(self._vault.root).confirm_existing(
                            ref.digest, ref.size
                        )
                continue
            if self._vault is None:
                raise AssetError("produced sources require an engine DINKSTER_ASSET_VAULT")
            path = transferred.resolve(ref.digest) if transferred is not None else None
            if path is None and self._source is not None:
                path = self._source.resolve(ref.digest)
            if path is None:
                raise AssetError(f"source {ref.digest} did not reach the engine")
            inventory = ProducedAssetInventory(self._vault.root)
            inventory.pending(ref.digest, ref.size)
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
                inventory.adopted(ref.digest, ref.size)
