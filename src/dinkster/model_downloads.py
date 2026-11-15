"""Shared ComfyUI model folders and public provider acquisition."""

from __future__ import annotations

from collections.abc import Callable
from http.client import HTTPException
from pathlib import Path

from dinkster_assets import (
    AssetError,
    AssetNeed,
    AssetRef,
    AssetWriter,
    MountDef,
    RemoteSource,
    SaveTarget,
)
from dinkster_assets.acquire import AcquisitionResult
from dinkster_assets.huggingface import HuggingFaceFile, select_huggingface_file
from dinkster_assets.provenance import ProvenanceRecord, ProvenanceStore
from dinkster_assets.vault import AssetPublicationError

from .model_settings import ModelSettings
from .mounts_api import MountService

QUICKSTART_CHECKPOINT = AssetNeed(
    name="v1-5-pruned-emaonly-fp16.safetensors",
    digest="blake3:4c50ebc6e2a5cb19e8d19626d5ede1fb64755562085ce7383d86c72d1d03eb7e",
    kind="model/checkpoint",
    size=2132696762,
    sources=(
        RemoteSource(
            "https://huggingface.co/Comfy-Org/stable-diffusion-1.5/resolve/"
            "9cfd069101959ca3828bf9c04a4419870832b74f/v1-5-pruned-emaonly-fp16.safetensors"
        ),
    ),
    metadata={
        "license": "creativeml-openrail-m",
        "sha256": "e9476a13728cd75d8279f6ec8bad753a66a1957ca375a1464dc63b37db6e3916",
    },
)


def shared_model_mounts() -> tuple[MountDef, ...]:
    """Load model locations independently of the currently selected project library."""
    return ModelSettings.load().prepare()


class ModelDownloads:
    def __init__(self, mounts: MountService, provenance: ProvenanceStore) -> None:
        self.mounts = mounts
        self.provenance = provenance

    def _target(self, need: AssetNeed, project: str) -> MountDef | None:
        table = self.mounts.table
        candidates = [
            mount
            for mount_id in self.mounts.project_context(project).mounts
            if table.kind(mount_id) == need.kind
            and (mount := table.get(mount_id)) is not None
            and mount.mode == "readwrite"
        ]
        return min(
            candidates,
            key=lambda mount: (table.source(mount.id) != "shared-models", mount.priority),
            default=None,
        )

    def destination(self, need: AssetNeed, project: str) -> str | None:
        target = self._target(need, project)
        return str(target.path) if target is not None else None

    def acquire(
        self,
        need: AssetNeed,
        project: str,
        before_publish: Callable[[AssetRef, str], None],
    ) -> AcquisitionResult:
        table = self.mounts.table
        target = self._target(need, project)
        if target is None:
            return AcquisitionResult(need, "failed", detail="Select a writable model folder first")
        mount_id = target.id
        for source in need.sources:
            if not isinstance(source, RemoteSource):
                continue
            try:
                selection = select_huggingface_file(source.url)
                name = Path(selection.file).name

                def publish(ref: AssetRef, selected: HuggingFaceFile = selection) -> None:
                    try:
                        before_publish(ref, selected.url)
                        self.provenance.add(
                            ProvenanceRecord(
                                digest=ref.digest,
                                sources=(selected.url,),
                                license=selected.license,
                                metadata={
                                    "provider": "huggingface",
                                    "repo": selected.repo,
                                    "revision": selected.revision,
                                    "file": selected.file,
                                    "sha256": selected.sha256,
                                    "size": selected.size,
                                },
                            )
                        )
                    except Exception as exc:
                        raise AssetPublicationError(
                            "asset publication metadata is unavailable"
                        ) from exc

                ref = selection.download(
                    AssetWriter(table),
                    SaveTarget(mount_id, name),
                    digest=need.digest,
                    before_publish=publish,
                )
            except (AssetError, OSError, HTTPException):
                continue
            table.refresh_publication(mount_id)
            return AcquisitionResult(need, "acquired", path=table.resolve(ref.digest))
        return AcquisitionResult(
            need,
            "failed",
            detail="No public Hugging Face source verified; existing model files were preserved",
        )
