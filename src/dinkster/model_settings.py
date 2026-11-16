"""Machine-wide model locations, independent of project libraries and credentials."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dinkster_assets import MountDef, MountsError

from .storelock import hold_lock

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


class ModelSettingsConflict(MountsError):
    """A different project changed shared settings after this edit was opened."""


@dataclass(frozen=True)
class ModelLocation:
    id: str
    path: Path
    mode: str
    visibility: str = "default-on"
    kind: str = ""

    def mounts(self, priority: int) -> tuple[MountDef, ...]:
        if self.kind:
            return (
                MountDef(
                    f"{self.id}-{self.kind.split('/')[1]}",
                    self.path,
                    self.mode,
                    priority=priority,
                    kind=self.kind,
                ),
            )
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
    download_location: str | None

    @staticmethod
    def path() -> Path:
        return (
            Path(os.environ.get("DINKSTER_HOME", Path.home() / ".dinkster")).expanduser()
            / "model-settings.json"
        )

    @classmethod
    def load(cls) -> ModelSettings:
        settings = cls.path()
        if not settings.exists():
            settings.parent.mkdir(parents=True, exist_ok=True)
            default = cls(
                (ModelLocation("models", Path.home() / "Dinkster" / "Models", "readwrite"),),
                "models",
            )
            descriptor, temporary = tempfile.mkstemp(prefix=".model-settings-", dir=settings.parent)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                    json.dump(default.to_wire(), output, indent=2)
                    output.flush()
                    os.fsync(output.fileno())
                try:
                    os.link(temporary, settings)
                except FileExistsError:
                    pass
            finally:
                Path(temporary).unlink(missing_ok=True)
        try:
            return cls.from_wire(json.loads(settings.read_text("utf-8")))
        except (OSError, ValueError) as exc:
            raise MountsError(f"cannot load shared model settings: {exc}") from exc

    @classmethod
    def from_wire(cls, raw: object) -> ModelSettings:
        try:
            if not isinstance(raw, dict) or set(raw) != {
                "version",
                "downloadLocation",
                "locations",
            }:
                raise ValueError("expected version, downloadLocation and locations")
            data = cast("dict[str, object]", raw)
            rows = data["locations"]
            if (
                type(data["version"]) is not int
                or data["version"] != 1
                or not isinstance(rows, list)
            ):
                raise ValueError("expected version 1 model locations")
            download_location = data["downloadLocation"]
            if download_location is not None and not isinstance(download_location, str):
                raise ValueError("expected a default download location id or null")
            locations: list[ModelLocation] = []
            for item in cast("list[object]", rows):
                if not isinstance(item, dict) or set(item) - {
                    "id",
                    "path",
                    "mode",
                    "visibility",
                    "kind",
                }:
                    raise ValueError("expected model location fields")
                row = cast("dict[str, object]", item)
                location, root, mode = row["id"], row["path"], row["mode"]
                visibility, kind = row.get("visibility", "default-on"), row.get("kind", "")
                if not all(
                    isinstance(value, str) for value in (location, root, mode, visibility, kind)
                ):
                    raise ValueError("model location fields must be strings")
                if not root:
                    raise ValueError("model location path must not be empty")
                root = Path(cast("str", root)).expanduser()
                if not root.is_absolute():
                    raise ValueError("model location path must be absolute")
                if visibility not in FOLDER_MODES:
                    raise ValueError(
                        "model folder visibility must be always-on, default-on or opt-in"
                    )
                if kind and kind not in MODEL_FOLDERS:
                    raise ValueError("model location kind must be a supported model kind")
                locations.append(
                    ModelLocation(
                        cast("str", location),
                        root,
                        cast("str", mode),
                        cast("str", visibility),
                        cast("str", kind),
                    )
                )
            result = cls(tuple(locations), download_location)
            mounts = result.mounts()
            if len({mount.id for mount in mounts}) != len(mounts):
                raise ValueError("duplicate model location ids")
            if len({location.id for location in locations}) != len(locations):
                raise ValueError("duplicate model location ids")
            if len({location.path.resolve() for location in locations}) != len(locations):
                raise ValueError("duplicate model location paths")
            writable = {row.id for row in locations if row.mode == "readwrite"}
            if download_location not in writable and (writable or download_location is not None):
                raise ValueError("default download location must name a writable model location")
            return result
        except (OSError, ValueError, KeyError) as exc:
            raise MountsError(f"invalid shared model settings: {exc}") from exc

    def to_wire(self) -> dict[str, object]:
        return {
            "version": 1,
            "downloadLocation": self.download_location,
            "locations": [
                {
                    "id": location.id,
                    "path": str(location.path),
                    "mode": location.mode,
                    "visibility": location.visibility,
                    **({"kind": location.kind} if location.kind else {}),
                }
                for location in self.locations
            ],
        }

    @property
    def revision(self) -> str:
        return hashlib.sha256(json.dumps(self.to_wire(), sort_keys=True).encode()).hexdigest()

    def save(self, expected_revision: str) -> None:
        settings = self.path()
        with hold_lock(settings.with_suffix(".lock"), timeout=5):
            if self.load().revision != expected_revision:
                raise ModelSettingsConflict("model settings changed; refresh before saving")
            self.prepare()
            descriptor, temporary = tempfile.mkstemp(prefix=".model-settings-", dir=settings.parent)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                    json.dump(self.to_wire(), output, indent=2)
                    output.write("\n")
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, settings)
            finally:
                Path(temporary).unlink(missing_ok=True)

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
