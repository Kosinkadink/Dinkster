"""Host-owned publication of encoded invocation previews into the asset vault."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import cast

from dinkster_assets import (
    AssetError,
    AssetRef,
    AssetVault,
    classify_media,
    digest_bytes,
    resolver_from_env,
)
from dinkster_assets.declared import produced_output_scratch
from dinkster_assets.vault import VaultWriter
from dinkster_protocol import SavedArtifact
from dinkster_schema.reporting import (
    EncodedPreviewArtifact,
    attach_saved_artifact,
    capture_preview_artifacts,
    capture_saved_artifacts,
    publish_preview_artifact,
    report_value_diagnostic,
    use_saved_file_preview_sink,
)
from dinkster_values import MEBIBYTE, SAVED_ARTIFACT_CLAIM_LIMIT

SAVED_FILE_PREVIEW_LIMIT_BYTES = 16 * MEBIBYTE


@contextmanager
def capture_saved_file_previews(*, enabled: bool) -> Generator[list[Mapping[str, object]]]:
    """Serialize lazy encoders and spool each item into the existing worker vault.

    Only small references survive a call. Committed content has normal vault
    retention, including when a node fails; partial writes always roll back.
    The host must still authorize the exact saved claim before using a preview.
    """
    claims: list[Mapping[str, object]] = []
    lock = threading.Lock()
    closed = threading.Event()

    def capture(
        ref: Mapping[str, object], encode: Callable[[], bytes], name: str, media_type: str
    ) -> bool:
        with lock:
            if not enabled or closed.is_set():
                return False
            try:
                if len(claims) >= SAVED_ARTIFACT_CLAIM_LIMIT:
                    raise ValueError("saved-file preview count exceeds invocation limits")
                with capture_saved_artifacts() as validated:
                    attach_saved_artifact(ref)
                vault = produced_output_scratch()
                if vault is None:
                    raise AssetError("saved-file preview requires host staging storage")
                data = encode()
                if closed.is_set():
                    return False
                if type(data) is not bytes or len(data) > SAVED_FILE_PREVIEW_LIMIT_BYTES:
                    raise ValueError("saved-file preview exceeds encoded item limit")
                with capture_preview_artifacts():
                    publish_preview_artifact(data, name=name, media_type=media_type)
                classification = classify_media(data)
                if classification.media_type != media_type or not name.lower().endswith(
                    "." + classification.extension
                ):
                    raise AssetError("saved-file preview does not match encoded content")
                digest = digest_bytes(data)
                with vault.writer(digest) as writer:
                    writer.write(data)
                    if closed.is_set():
                        return False
                    writer.commit()
                claims.append(
                    MappingProxyType(
                        {
                            "saved": validated[0],
                            "preview": MappingProxyType(
                                {
                                    "digest": digest,
                                    "name": name,
                                    "size": len(data),
                                    "mediaType": media_type,
                                    "virtualPath": f"previews/{digest[7:]}/{name}",
                                }
                            ),
                        }
                    )
                )
                return True
            except Exception as exc:
                report_value_diagnostic("saved-file-preview-unavailable", {"reason": str(exc)})
                return False

    with use_saved_file_preview_sink(capture):
        try:
            yield claims
        finally:
            # An async node may abandon a to_thread encoder on cancellation.
            # Closing its context must not block the event loop on that encoder.
            closed.set()


@dataclass
class _StagedPreviews:
    stack: ExitStack
    writers: list[VaultWriter]
    artifacts: tuple[SavedArtifact, ...]


class PreviewArtifactPublisher:
    """Stage off-loop, then commit one successful invocation's preview batch.

    Cancellation before commit discards staging. Each committed digest has the
    vault's normal retention even if a later commit fails: another ingest may
    already share it. Failed batches return no descriptors. There is no output mount.
    """

    def __init__(self, vault: AssetVault | None) -> None:
        self._vault = vault
        self._lock = asyncio.Lock()

    async def saved_files(
        self, artifacts: Sequence[SavedArtifact], claims: Sequence[Mapping[str, object]]
    ) -> tuple[SavedArtifact, ...]:
        """Associate optional media only with an exact host-authorized saved file."""
        from .blobs import await_file_operation
        from .boundary import decode_saved_file_previews

        by_ref = {
            (item.digest, item.name, item.size, item.media_type, item.virtual_path): index
            for index, item in enumerate(artifacts)
        }
        result = list(artifacts)
        for claim in decode_saved_file_previews({"savedFilePreviews": list(claims)}):
            try:
                saved = AssetRef.from_wire(cast("Mapping[str, object]", claim["saved"]))
                index = by_ref.get(
                    (saved.digest, saved.name, saved.size, saved.media_type, saved.virtual_path)
                )
                if index is None:
                    raise AssetError("preview does not match an authorized saved file")
                preview = AssetRef.from_wire(
                    cast("Mapping[str, object]", claim["preview"]), resolver_from_env()
                )

                def read(ref: AssetRef = preview) -> bytes:
                    with ref.open() as source:
                        data = source.read(SAVED_FILE_PREVIEW_LIMIT_BYTES + 1)
                    if len(data) != ref.size:
                        raise AssetError("saved-file preview bytes do not match declared size")
                    return data

                data = await await_file_operation(asyncio.create_task(asyncio.to_thread(read)))
                published = await self(
                    artifacts[index].node_id,
                    (EncodedPreviewArtifact(data, preview.name, preview.media_type),),
                )
                result[index] = replace(artifacts[index], preview=published[0])
                del data
            except Exception as exc:
                logging.getLogger(__name__).warning("saved-file preview unavailable: %s", exc)
        return tuple(result)

    def _stage(self, node_id: str, claims: Sequence[EncodedPreviewArtifact]) -> _StagedPreviews:
        if self._vault is None:
            raise AssetError("preview publication requires a host asset vault")
        with capture_preview_artifacts() as validated:
            for claim in claims:
                publish_preview_artifact(claim.data, name=claim.name, media_type=claim.media_type)
        artifacts: list[SavedArtifact] = []
        for claim in validated:
            classification = classify_media(claim.data)
            if classification.media_type != claim.media_type:
                raise AssetError("preview artifact media_type does not match encoded content")
            if not claim.name.lower().endswith("." + classification.extension):
                raise AssetError("preview artifact extension does not match encoded content")
            digest = digest_bytes(claim.data)
            artifacts.append(
                SavedArtifact(
                    node_id,
                    digest,
                    claim.name,
                    len(claim.data),
                    claim.media_type,
                    f"previews/{digest[7:]}/{claim.name}",
                )
            )
        stack = ExitStack()
        writers: list[VaultWriter] = []
        try:
            for claim, artifact in zip(validated, artifacts, strict=True):
                writer = stack.enter_context(self._vault.writer(artifact.digest))
                writer.write(claim.data)
                writers.append(writer)
            return _StagedPreviews(stack, writers, tuple(artifacts))
        except BaseException:
            stack.close()
            raise

    async def __call__(
        self, node_id: str, claims: Sequence[EncodedPreviewArtifact]
    ) -> tuple[SavedArtifact, ...]:
        if not claims:
            return ()
        async with self._lock:
            task = asyncio.create_task(asyncio.to_thread(self._stage, node_id, claims))
            try:
                staged = await asyncio.shield(task)
            except asyncio.CancelledError:
                while not task.done():
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await asyncio.shield(task)
                if not task.cancelled() and task.exception() is None:
                    task.result().stack.close()
                raise
            try:
                # No await splits the publication commit from its result handoff.
                for writer in staged.writers:
                    writer.commit_with_result()
                return staged.artifacts
            finally:
                staged.stack.close()
