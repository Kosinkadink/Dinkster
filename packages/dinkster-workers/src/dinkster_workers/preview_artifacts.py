"""Host-owned publication of encoded invocation previews into the asset vault."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from types import MappingProxyType

from dinkster_assets import (
    AssetError,
    AssetVault,
    classify_media,
    digest_bytes,
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
    active = True

    def capture(
        ref: Mapping[str, object], encode: Callable[[], bytes], name: str, media_type: str
    ) -> bool:
        with lock:
            if not enabled or not active:
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
            with lock:
                active = False


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
