"""Shared ComfyUI model folders and public provider acquisition."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from http.client import HTTPException
from pathlib import Path
from typing import cast

from dinkster_assets import (
    AssetError,
    AssetNeed,
    AssetRef,
    AssetWriter,
    MountDef,
    MountsError,
    RemoteSource,
    SaveTarget,
)
from dinkster_assets.acquire import AcquisitionResult
from dinkster_assets.huggingface import select_huggingface_file
from dinkster_assets.provenance import ProvenanceRecord, ProvenanceStore
from dinkster_assets.vault import AssetPublicationError

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

MODEL_FOLDERS = {
    "model/checkpoint": "checkpoints",
    "model/lora": "loras",
    "model/vae": "vae",
    "model/text-encoder": "text_encoders",
    "model/diffusion": "diffusion_models",
    "model/controlnet": "controlnet",
    "model/upscale": "upscale_models",
    "model/upscaler": "upscale_models",
    "model/embedding": "embeddings",
    "model/clip-vision": "clip_vision",
}


def shared_model_mounts() -> tuple[MountDef, ...]:
    """Load model locations independently of the currently selected project library."""
    home = Path(os.environ.get("DINKSTER_HOME", Path.home() / ".dinkster")).expanduser()
    settings = home / "model-settings.json"
    if not settings.exists():
        home.mkdir(parents=True, exist_ok=True)
        default = {
            "version": 1,
            "downloadLocation": "models",
            "locations": [
                {
                    "id": "models",
                    "path": str(Path.home() / "Dinkster" / "Models"),
                    "mode": "readwrite",
                }
            ],
        }
        descriptor, temporary = tempfile.mkstemp(prefix=".model-settings-", dir=home)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(default, output, indent=2)
                output.flush()
                os.fsync(output.fileno())
            try:
                os.link(temporary, settings)
            except FileExistsError:
                pass
        finally:
            Path(temporary).unlink(missing_ok=True)
    try:
        raw: object = json.loads(settings.read_text("utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("expected model settings object")
        data = cast("dict[str, object]", raw)
        locations = data.get("locations")
        if data.get("version") != 1 or not isinstance(locations, list):
            raise ValueError("expected version 1 model locations")
        download_location = data.get("downloadLocation")
        if not isinstance(download_location, str):
            raise ValueError("expected a default download location id")
        mounts: list[MountDef] = []
        writable: set[str] = set()
        for index, item in enumerate(cast("list[object]", locations)):
            if not isinstance(item, dict):
                raise ValueError("expected model location object")
            row = cast("dict[str, object]", item)
            location, root, mode = row["id"], row["path"], row["mode"]
            if not all(isinstance(value, str) for value in (location, root, mode)):
                raise ValueError("model location fields must be strings")
            root = Path(cast("str", root)).expanduser()
            if not root.is_absolute():
                raise ValueError("model location path must be absolute")
            priority = 0 if location == download_location else 2 * (index + 1)
            if mode == "readwrite":
                writable.add(cast("str", location))
            for kind, folder in MODEL_FOLDERS.items():
                mounts.append(
                    MountDef(
                        f"{location}-{kind.split('/')[1]}",
                        root / folder,
                        cast("str", mode),
                        priority=priority,
                        kind=kind,
                    )
                )
            mounts.append(
                MountDef(
                    f"{location}-clip",
                    root / "clip",
                    cast("str", mode),
                    priority=priority + 1,
                    kind="model/text-encoder",
                )
            )
        if len({mount.id for mount in mounts}) != len(mounts):
            raise ValueError("duplicate model location ids")
        if download_location not in writable:
            raise ValueError("default download location must name a writable model location")
        for mount in mounts:
            if mount.mode == "readwrite":
                mount.path.mkdir(parents=True, exist_ok=True)
        return tuple(mounts)
    except (OSError, ValueError, KeyError, MountsError) as exc:
        raise MountsError(f"cannot load shared model settings: {exc}") from exc


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

                def publish(ref: AssetRef, url: str = selection.url) -> None:
                    try:
                        before_publish(ref, url)
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
            self.provenance.add(
                ProvenanceRecord(
                    digest=ref.digest,
                    sources=(selection.url,),
                    license=selection.license,
                    metadata={
                        "provider": "huggingface",
                        "repo": selection.repo,
                        "revision": selection.revision,
                        "file": selection.file,
                        "sha256": selection.sha256,
                        "size": selection.size,
                    },
                )
            )
            return AcquisitionResult(need, "acquired", path=table.resolve(ref.digest))
        return AcquisitionResult(
            need,
            "failed",
            detail="No public Hugging Face source verified; existing model files were preserved",
        )
