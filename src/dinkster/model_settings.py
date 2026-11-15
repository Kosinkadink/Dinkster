"""Machine-wide model locations, independent of project libraries and credentials."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dinkster_assets import MountDef, MountsError

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

FOLDER_MODES = frozenset({"always-on", "default-on", "opt-in"})


@dataclass(frozen=True)
class ModelLocation:
    id: str
    path: Path
    mode: str
    visibility: str = "default-on"

    def mounts(self, priority: int) -> tuple[MountDef, ...]:
        return (
            *(
                MountDef(
                    f"{self.id}-{kind.split('/')[1]}",
                    self.path / folder,
                    self.mode,
                    priority=priority,
                    kind=kind,
                )
                for kind, folder in MODEL_FOLDERS.items()
            ),
            MountDef(
                f"{self.id}-clip",
                self.path / "clip",
                self.mode,
                priority=priority + 1,
                kind="model/text-encoder",
            ),
        )


@dataclass(frozen=True)
class ModelSettings:
    locations: tuple[ModelLocation, ...]
    download_location: str

    @classmethod
    def load(cls) -> ModelSettings:
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
                        "visibility": "default-on",
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
            rows = data.get("locations")
            if data.get("version") != 1 or not isinstance(rows, list):
                raise ValueError("expected version 1 model locations")
            download_location = data.get("downloadLocation")
            if not isinstance(download_location, str):
                raise ValueError("expected a default download location id")
            locations: list[ModelLocation] = []
            for item in cast("list[object]", rows):
                if not isinstance(item, dict):
                    raise ValueError("expected model location object")
                row = cast("dict[str, object]", item)
                location, root, mode = row["id"], row["path"], row["mode"]
                visibility = row.get("visibility", "default-on")
                if not all(isinstance(value, str) for value in (location, root, mode, visibility)):
                    raise ValueError("model location fields must be strings")
                root = Path(cast("str", root)).expanduser()
                if not root.is_absolute():
                    raise ValueError("model location path must be absolute")
                if visibility not in FOLDER_MODES:
                    raise ValueError(
                        "model folder visibility must be always-on, default-on or opt-in"
                    )
                locations.append(
                    ModelLocation(
                        cast("str", location), root, cast("str", mode), cast("str", visibility)
                    )
                )
            result = cls(tuple(locations), download_location)
            mounts = result.mounts()
            if len({mount.id for mount in mounts}) != len(mounts):
                raise ValueError("duplicate model location ids")
            if len({location.path.resolve() for location in locations}) != len(locations):
                raise ValueError("duplicate model location paths")
            if download_location not in {row.id for row in locations if row.mode == "readwrite"}:
                raise ValueError("default download location must name a writable model location")
            return result
        except (OSError, ValueError, KeyError, MountsError) as exc:
            raise MountsError(f"cannot load shared model settings: {exc}") from exc

    def mounts(self) -> tuple[MountDef, ...]:
        return tuple(
            mount
            for index, location in enumerate(self.locations)
            for mount in location.mounts(
                0 if location.id == self.download_location else 2 * (index + 1)
            )
        )

    def folder_modes(self) -> dict[str, str]:
        return {
            mount.id: location.visibility
            for location in self.locations
            for mount in location.mounts(0)
        }

    def prepare(self) -> tuple[MountDef, ...]:
        mounts = self.mounts()
        for mount in mounts:
            if mount.mode == "readwrite":
                mount.path.mkdir(parents=True, exist_ok=True)
        return mounts
