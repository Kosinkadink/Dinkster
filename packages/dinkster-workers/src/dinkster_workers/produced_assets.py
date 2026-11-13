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
    digest_bytes,
    open_verified,
)
from dinkster_assets.declared import offer_produced_outputs
from dinkster_assets.produced import ProducedAssetInventory
from dinkster_assets.scratch import hold_produced_vault
from dinkster_assets.value import value_asset_references
from dinkster_schema.reporting import ImageDocumentArtifact
from dinkster_values import (
    IMAGE_DOCUMENT_TYPE,
    MEBIBYTE,
    SAVED_ARTIFACT_CLAIM_LIMIT,
    EncodedPayload,
    Rendition,
    TypeRegistry,
    Value,
    ValueMeta,
    default_decode,
    iter_value_tree,
)

_COPY_BYTES = MEBIBYTE
DOCUMENT_RENDITION_MIME = "application/vnd.dinkster.image-document+json"


def rendition_values(registry: TypeRegistry, rendition: Rendition) -> dict[str, Value]:
    """Declare the returned modern document, never restamp its legacy source."""
    if rendition.mime != DOCUMENT_RENDITION_MIME:
        return {}
    spec = registry.spec(IMAGE_DOCUMENT_TYPE)
    if not rendition.data.startswith(b"{") or spec.validate_encoded is None:
        raise AssetError("document rendition requires the declared modern document codec")
    value = registry.wrap(IMAGE_DOCUMENT_TYPE, spec.decode(rendition.data))
    if spec.encode(value.resolve()) != rendition.data:
        raise AssetError("document rendition does not match its declared envelope")
    spec.validate_encoded(rendition.data, value.meta.entries)
    return {"document": value}


def image_document_values(documents: Sequence[ImageDocumentArtifact]) -> dict[str, Value]:
    """Validate canonical attachments independently of a pack's output codecs."""
    from dinkster_image_document.document import (
        ImageDocument,
        document_meta,
        validate_image_document_encoded,
    )

    values: dict[str, Value] = {}
    try:
        for index, document in enumerate(documents):
            decoded = ImageDocument(document.data)
            if decoded.data != document.data:
                raise AssetError("original image document is not canonical")
            metadata = document_meta(decoded)
            validate_image_document_encoded(document.data, metadata)
            values[str(index)] = Value(
                IMAGE_DOCUMENT_TYPE,
                digest_bytes(document.data),
                ValueMeta(metadata),
                EncodedPayload(IMAGE_DOCUMENT_TYPE, document.data, None),
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise AssetError(f"invalid original image document: {exc}") from exc
    return values


def image_document_sources(
    header: Mapping[str, object], documents: Sequence[ImageDocumentArtifact]
) -> dict[str, Value]:
    """Read required source declarations without relying on a local pack codec."""
    sources = header.get("imageDocumentSources", {})
    if not isinstance(sources, Mapping) or set(cast("Mapping[str, object]", sources)) != {
        str(index) for index in range(len(documents))
    }:
        raise AssetError("image document sources do not match the attachment batch")
    declarations = cast("Mapping[str, object]", sources)
    values: dict[str, Value] = {}
    for index, document in enumerate(documents):
        refs = [ref.to_wire() for ref in _document_source_references(declarations[str(index)])]
        values[str(index)] = Value(
            IMAGE_DOCUMENT_TYPE,
            digest_bytes(document.data),
            ValueMeta({"asset_refs": refs}),
            EncodedPayload(IMAGE_DOCUMENT_TYPE, document.data, None),
        )
    return values


def _document_source_references(raw: object) -> Iterator[AssetRef]:
    if type(raw) is not list or len(cast("list[object]", raw)) > SAVED_ARTIFACT_CLAIM_LIMIT:
        raise AssetError("image document sources require a bounded reference list")
    for item in cast("list[object]", raw):
        if not isinstance(item, Mapping) or set(cast("Mapping[str, object]", item)) != {
            "digest",
            "name",
            "size",
            "mediaType",
        }:
            raise AssetError("image document source reference has invalid fields")
        yield AssetRef.from_wire(cast("Mapping[str, object]", item))


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
    sources = header.get("imageDocumentSources", {})
    if (
        not isinstance(sources, Mapping)
        or len(cast("Mapping[str, object]", sources)) > SAVED_ARTIFACT_CLAIM_LIMIT
    ):
        raise AssetError("image document sources must be a mapping")
    for refs in cast("Mapping[str, object]", sources).values():
        digests.update(ref.digest for ref in _document_source_references(refs))
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
