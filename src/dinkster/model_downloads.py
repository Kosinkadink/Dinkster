"""Shared ComfyUI model folders and public provider acquisition."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
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
from dinkster_assets.civitai import CivitaiFile
from dinkster_assets.fetch import FetchAborted
from dinkster_assets.huggingface import HuggingFaceFile, select_huggingface_file
from dinkster_assets.model import model_is_hidden, model_ref_is_hidden
from dinkster_assets.mounts import ModelFileSelection
from dinkster_assets.provenance import ProvenanceRecord, ProvenanceStore
from dinkster_assets.vault import AssetPublicationError
from dinkster_library import LibraryClient
from dinkster_values.asset_folders import (
    AssetFolderContext,
    current_asset_folders,
    use_asset_folders,
)

from .model_settings import ModelSettings
from .mounts_api import MountService
from .provider_credentials import ProviderCredentials

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


def shared_model_mounts(library: LibraryClient | None = None) -> tuple[MountDef, ...]:
    """Load model locations independently of the currently selected project library."""
    return ModelSettings.load(library).prepare()


class ModelDownloads:
    def __init__(self, mounts: MountService, provenance: ProvenanceStore) -> None:
        self.mounts = mounts
        self.provenance = provenance
        self.credentials = ProviderCredentials()

    def disabled_locations(
        self, digest: str, project: str, *, kind: str | None = None
    ) -> tuple[str, ...]:
        enabled = self.mounts.project_context(project).mounts
        matches: set[str] = set()
        for mount in self.mounts.table.ready_snapshot():
            if kind is not None and mount.asset_kind != kind:
                continue
            for ref in mount.refs:
                if ref.digest != digest:
                    continue
                if mount.mount_id not in enabled:
                    matches.add(mount.mount_id)
                else:
                    try:
                        resolved = self.mounts.table.ref(ref.virtual_path)
                        if resolved.digest != digest:
                            continue
                        with resolved.open():
                            return ()
                    except (AssetError, OSError):
                        pass
        if not matches:
            return ()
        settings = self.mounts.model_settings()
        if settings is None:
            return ()
        return tuple(
            location.id
            for location in settings.locations
            if matches.intersection(mount.id for mount in location.mounts(0))
        )

    def disabled_name_locations(self, name: str, kind: str) -> tuple[str, ...]:
        context = current_asset_folders()
        settings = self.mounts.model_settings()
        if context is None or settings is None:
            return ()
        matches: list[str] = []
        for location in settings.locations:
            disabled = tuple(
                mount.id for mount in location.mounts(0) if mount.id not in context.mounts
            )
            if not disabled:
                continue
            with use_asset_folders(AssetFolderContext(disabled)):
                if self.mounts.table.bind_name(name, kind) is not None:
                    matches.append(location.id)
        return tuple(matches)

    def hidden_name(self, name: str, kind: str) -> bool:
        context = current_asset_folders()
        if context is None or not context.hidden_models:
            return False
        with use_asset_folders(replace(context, hidden_models=())):
            ref = self.mounts.table.bind_name(name, kind)
        return ref is not None and model_ref_is_hidden(ref)

    def hidden_digest(self, digest: str, project: str) -> bool:
        context = self.mounts.project_context(project)
        if not context.hidden_models:
            return False
        with use_asset_folders(context):
            return model_is_hidden(digest=digest)

    def target(self, kind: str, project: str) -> MountDef | None:
        table = self.mounts.table
        candidates = [
            mount
            for mount_id in self.mounts.project_context(project).mounts
            if table.kind(mount_id) == kind
            and (mount := table.get(mount_id)) is not None
            and mount.mode == "readwrite"
        ]
        return min(
            candidates,
            key=lambda mount: (table.source(mount.id) != "shared-models", mount.priority),
            default=None,
        )

    def destination(self, need: AssetNeed, project: str) -> str | None:
        target = self.target(need.kind, project)
        return str(target.path) if target is not None else None

    def download_selected(
        self,
        selected: HuggingFaceFile | CivitaiFile,
        target: MountDef,
        project: str,
        before_publish: Callable[[AssetRef, str], None],
        *,
        digest: str | None = None,
        progress: Callable[[str, int], None] | None = None,
        expected_root: Path | None = None,
        expected_credential_revision: str | None = None,
    ) -> AssetRef:
        table = self.mounts.table
        root = expected_root if expected_root is not None else target.path.resolve()
        token, credential_revision = (
            (None, None) if isinstance(selected, CivitaiFile) else self.credentials.snapshot()
        )
        if expected_credential_revision is not None and (
            credential_revision != expected_credential_revision
        ):
            raise AssetError("provider credentials changed; review the download again")

        def require_credentials() -> None:
            if (
                credential_revision is not None
                and self.credentials.snapshot()[1] != credential_revision
            ):
                raise FetchAborted()

        last_credential_check = 0.0

        def checked_progress(status: str, received: int) -> None:
            nonlocal last_credential_check
            now = time.monotonic()
            if now - last_credential_check >= 0.25:
                require_credentials()
                last_credential_check = now
            if progress is not None:
                progress(status, received)

        def require_target() -> None:
            require_credentials()
            context = self.mounts.project_context(project)
            with use_asset_folders(context):
                if (
                    self.hidden_digest(digest, project)
                    if digest is not None
                    else self.hidden_name(Path(selected.file).name, target.kind)
                ):
                    raise AssetError("Model hidden for this project; unhide it in model settings")
            if (
                table.get(target.id) != target
                or target.path.resolve() != root
                or target.id not in context.mounts
            ):
                raise AssetError("model destination changed; review the download again")

        def publish(ref: AssetRef) -> None:
            require_target()
            relative = ref.virtual_path.split("/", 2)[2]
            path = (search_root if ref.virtual_path.startswith("model-files/") else root) / relative
            with use_asset_folders(self.mounts.project_context(project)):
                if model_is_hidden(path, ref.digest):
                    raise AssetError("Model hidden for this project; unhide it in model settings")
            try:
                before_publish(ref, selected.url)
                self.provenance.add(
                    ProvenanceRecord(
                        digest=ref.digest,
                        sources=(selected.url,),
                        license=selected.license,
                        metadata={
                            "provider": "civitai"
                            if isinstance(selected, CivitaiFile)
                            else "huggingface",
                            "repo": selected.repo,
                            "revision": selected.revision,
                            "file": selected.file,
                            "sha256": selected.sha256,
                            "size": selected.size,
                        },
                    )
                )
            except FetchAborted:
                raise
            except Exception as exc:
                raise AssetPublicationError("asset publication metadata is unavailable") from exc
            require_target()

        require_target()
        writer = AssetWriter(table)
        settings = self.mounts.model_settings()
        search_root = target.path.resolve()
        if settings is not None:
            for location in settings.locations:
                if any(mount.id == target.id for mount in location.mounts(0)):
                    search_root = location.path.resolve()
                    break

        def admit_file(selection: ModelFileSelection) -> None:
            require_target()
            if settings is None:
                raise AssetError("model location is unavailable")
            settings.remember_file_selection(selection)
            self.mounts.project_context(project)
            require_target()

        destination = SaveTarget(target.id, Path(selected.file).name)
        with use_asset_folders(self.mounts.project_context(project)):
            ref = writer.reuse_model(
                destination,
                sha256=selected.sha256,
                size=selected.size,
                digest=digest,
                before_publish=publish,
                progress=checked_progress,
                search_root=search_root,
                kind=target.kind,
                admit_file=admit_file,
            )
        if ref is not None:
            return ref
        require_target()
        with use_asset_folders(self.mounts.project_context(project)):
            ref = selected.download(
                writer,
                destination,
                digest=digest,
                before_publish=publish,
                progress=checked_progress,
                token=token,
            )
        return ref

    def acquire(
        self,
        need: AssetNeed,
        project: str,
        before_publish: Callable[[AssetRef, str], None],
    ) -> AcquisitionResult:
        table = self.mounts.table
        target = self.target(need.kind, project)
        if target is None:
            return AcquisitionResult(need, "failed", detail="Select a writable model folder first")
        for source in need.sources:
            if not isinstance(source, RemoteSource):
                continue
            try:
                selection = select_huggingface_file(source.url, token=self.credentials.token())
                ref = self.download_selected(
                    selection, target, project, before_publish, digest=need.digest
                )
            except (AssetError, OSError, HTTPException):
                continue
            table.refresh_publication(target.id)
            return AcquisitionResult(need, "acquired", path=table.resolve(ref.digest))
        return AcquisitionResult(
            need,
            "failed",
            detail="No public Hugging Face source verified; existing model files were preserved",
        )
