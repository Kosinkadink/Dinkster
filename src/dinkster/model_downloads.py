"""Shared ComfyUI model folders and public provider acquisition."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
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

from .mounts_api import MountService

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
        mounts: list[MountDef] = []
        for item in cast("list[object]", locations):
            if not isinstance(item, dict):
                raise ValueError("expected model location object")
            row = cast("dict[str, object]", item)
            location, root, mode = row["id"], row["path"], row["mode"]
            if not all(isinstance(value, str) for value in (location, root, mode)):
                raise ValueError("model location fields must be strings")
            root = Path(cast("str", root)).expanduser()
            if not root.is_absolute():
                raise ValueError("model location path must be absolute")
            for kind, folder in MODEL_FOLDERS.items():
                path = root / folder
                if mode == "readwrite":
                    path.mkdir(parents=True, exist_ok=True)
                    (root / "clip").mkdir(exist_ok=True)
                mounts.append(
                    MountDef(f"{location}-{kind.split('/')[1]}", path, cast("str", mode), kind=kind)
                )
            mounts.append(
                MountDef(
                    f"{location}-clip", root / "clip", cast("str", mode), kind="model/text-encoder"
                )
            )
        if len({mount.id for mount in mounts}) != len(mounts):
            raise ValueError("duplicate model location ids")
        return tuple(mounts)
    except (OSError, ValueError, KeyError, MountsError) as exc:
        raise MountsError(f"cannot load shared model settings: {exc}") from exc


class ModelDownloads:
    def __init__(self, mounts: MountService, provenance: ProvenanceStore) -> None:
        self.mounts = mounts
        self.provenance = provenance

    def acquire(
        self,
        need: AssetNeed,
        project: str,
        before_publish: Callable[[AssetRef, str], None],
    ) -> AcquisitionResult:
        context = self.mounts.project_context(project)
        table = self.mounts.table
        candidates = [
            mount_id
            for mount_id in context.mounts
            if table.kind(mount_id) == need.kind
            and (mount := table.get(mount_id)) is not None
            and mount.mode == "readwrite"
        ]
        if not candidates:
            return AcquisitionResult(need, "failed", detail="Select a writable model folder first")
        mount_id = candidates[0]
        for source in need.sources:
            if not isinstance(source, RemoteSource):
                continue
            try:
                selection = select_huggingface_file(source.url)
                name = Path(selection.file).name
                ref = selection.download(
                    AssetWriter(table),
                    SaveTarget(mount_id, name),
                    digest=need.digest,
                    before_publish=lambda ref, url=selection.url: before_publish(ref, url),
                )
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
            except (AssetError, OSError, TimeoutError):
                continue
        return AcquisitionResult(
            need,
            "failed",
            detail="No public Hugging Face source verified; existing model files were preserved",
        )
