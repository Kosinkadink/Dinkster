"""Host-owned publication of encoded invocation previews into the asset vault."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass

from dinkster_assets import AssetError, AssetVault, classify_media, digest_bytes
from dinkster_assets.vault import VaultWriter
from dinkster_protocol import SavedArtifact
from dinkster_schema.reporting import (
    EncodedPreviewArtifact,
    capture_preview_artifacts,
    publish_preview_artifact,
)


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
