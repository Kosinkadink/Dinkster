"""Machine-wide model locations, independent of project libraries and credentials."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

from dinkster_assets import AssetError, MountDef, MountsError
from dinkster_assets.kind import is_asset_kind
from dinkster_assets.mounts import ModelFileSelection
from dinkster_library import LibraryClient, LibraryError

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
    "model/audio-encoder": "audio_encoders",
    "model/background-removal": "background_removal",
    "model/config": "configs",
    "model/detection": "detection",
    "model/diffusers": "diffusers",
    "model/frame-interpolation": "frame_interpolation",
    "model/geometry-estimation": "geometry_estimation",
    "model/gligen": "gligen",
    "model/latent-upscaler": "latent_upscale_models",
    "model/optical-flow": "optical_flow",
    "model/patch": "model_patches",
    "model/style": "style_models",
    "model/hypernetwork": "hypernetworks",
    "model/photomaker": "photomaker",
}

MODEL_FOLDER_ALIASES = {
    "clip": "model/text-encoder",
    "unet": "model/diffusion",
}

FOLDER_MODES = frozenset({"always-on", "default-on", "opt-in"})


def filesystem_space(path: Path) -> dict[str, object]:
    """Observe a configured destination's filesystem, without creating missing folders."""
    try:
        resolved = path.resolve()
        for candidate in (resolved, *resolved.parents):
            try:
                device = candidate.stat().st_dev
                identity = f"device:{device}"
                if sys.platform == "win32":
                    import ctypes
                    from ctypes import wintypes

                    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                    kernel.GetDiskFreeSpaceExW.argtypes = (
                        wintypes.LPCWSTR,
                        ctypes.POINTER(ctypes.c_ulonglong),
                        ctypes.c_void_p,
                        ctypes.c_void_p,
                    )
                    kernel.GetDiskFreeSpaceExW.restype = wintypes.BOOL
                    available = ctypes.c_ulonglong()
                    if not kernel.GetDiskFreeSpaceExW(
                        str(candidate), ctypes.byref(available), None, None
                    ):
                        raise ctypes.WinError(ctypes.get_last_error())
                    free = available.value
                else:
                    if sys.platform == "linux":
                        import errno
                        import fcntl

                        descriptor = os.open(candidate, os.O_RDONLY | os.O_DIRECTORY)
                        try:
                            info = bytearray(1024)
                            try:
                                # BTRFS_IOC_FS_INFO identifies the pool, unlike subvolume st_dev.
                                fcntl.ioctl(descriptor, 0x8400941F, info, True)
                            except OSError as error:
                                if error.errno != errno.ENOTTY:
                                    raise
                            else:
                                identity = f"btrfs:{info[16:32].hex()}"
                        finally:
                            os.close(descriptor)
                    free = shutil.disk_usage(candidate).free
            except FileNotFoundError:
                continue
            return {
                "status": "available",
                "filesystem": hashlib.sha256(identity.encode("ascii")).hexdigest(),
                "freeBytes": free,
            }
    except (OSError, RuntimeError):
        pass
    return {"status": "unavailable"}


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
        folders = dict(MODEL_FOLDERS)
        # Unrecognized folders use the open kind vocabulary: custom_model -> model/custom-model.
        present: set[str] | None = None
        try:
            children = sorted(self.path.iterdir())
        except OSError:
            # Failed root reads still declare grants so scanning exposes the failure.
            children = []
        else:
            if self.mode == "read":
                present = {child.name for child in children}
        known = {*folders.values(), *MODEL_FOLDER_ALIASES}
        for child in children:
            kind = f"model/{child.name.replace('_', '-')}"
            if child.is_dir() and child.name not in known and is_asset_kind(kind):
                folders.setdefault(kind, child.name)
        mounts = (
            *(
                MountDef(
                    f"{self.id}-{kind.split('/')[1]}",
                    self.path / folder,
                    self.mode,
                    priority=priority,
                    kind=kind,
                )
                for kind, folder in folders.items()
            ),
            *(
                MountDef(
                    f"{self.id}-{folder}",
                    self.path / folder,
                    self.mode,
                    priority=priority + 1,
                    kind=kind,
                )
                for folder, kind in MODEL_FOLDER_ALIASES.items()
            ),
        )
        return tuple(mount for mount in mounts if present is None or mount.path.name in present)


@dataclass(frozen=True)
class ModelSettings:
    locations: tuple[ModelLocation, ...]
    download_location: str | None
    library: LibraryClient | None = field(default=None, compare=False, repr=False)
    _shared_revision: int | None = field(default=None, compare=False, repr=False)

    @staticmethod
    def _shared(library: LibraryClient, **changes: object) -> dict[str, object]:
        try:
            if "folders.models" not in cast("list[str]", library.hello["capabilities"]):
                raise LibraryError("api", "library helper does not support account model settings")
            return cast(
                "dict[str, object]",
                library.call(
                    "folders.models.patch" if changes else "folders.models.get", **changes
                )["folders"],
            )
        except LibraryError as exc:
            error = ModelSettingsConflict if exc.code == "stale" else MountsError
            raise error(str(exc)) from exc

    @staticmethod
    def path() -> Path:
        return (
            Path(os.environ.get("DINKSTER_HOME", Path.home() / ".dinkster")).expanduser()
            / "model-settings.json"
        )

    @classmethod
    def load(cls, library: LibraryClient | None = None) -> ModelSettings:
        if library is not None:
            record = cls._shared(library)
            values = cast("dict[str, object]", record["values"])
            if "locations" not in values:
                result = cls(
                    (ModelLocation("models", Path.home() / "Dinkster" / "Models", "readwrite"),),
                    "models",
                )
            else:
                try:
                    locations = values["locations"]
                    if not isinstance(locations, dict):
                        raise ValueError("model locations must be an object")
                    rows = cast("dict[str, object]", locations)
                    order = values.get("order", list(rows))
                    if (
                        not isinstance(order, list)
                        or any(not isinstance(key, str) for key in cast("list[object]", order))
                        or len(order) != len(rows)
                        or set(cast("list[str]", order)) != set(rows)
                    ):
                        raise ValueError("model location order must contain every location once")
                    projected: list[dict[str, object]] = []
                    for key in cast("list[str]", order):
                        row = rows[key]
                        if not isinstance(row, dict):
                            raise ValueError("model location must be an object")
                        projected.append(
                            {
                                "id": key,
                                **{
                                    name: value
                                    for name, value in cast("dict[str, object]", row).items()
                                    if name in {"path", "mode", "visibility", "kind"}
                                },
                            }
                        )
                    result = cls.from_wire(
                        {
                            "version": 1,
                            "locations": projected,
                            "downloadLocation": values.get("downloadLocation"),
                        }
                    )
                except (TypeError, ValueError) as exc:
                    raise MountsError(f"invalid shared model settings: {exc}") from exc
            return replace(result, library=library, _shared_revision=cast(int, record["revision"]))
        with hold_lock(cls.path().with_suffix(".lock"), timeout=5):
            return cls._load_legacy()

    @classmethod
    def _load_legacy(cls) -> ModelSettings:
        settings = cls.path()
        if any(
            settings.with_name(name).exists()
            for name in ("model-settings.sqlite", "model-settings-migration.json")
        ):
            raise MountsError(
                "account model settings require a library helper; JSON writes refused"
            )
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
                kind = cast("str", kind)
                if kind and (not kind.startswith("model/") or not is_asset_kind(kind)):
                    raise ValueError("model location kind must be a valid model kind")
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
        if self._shared_revision is not None:
            return f"shared:{self._shared_revision}"
        return hashlib.sha256(json.dumps(self.to_wire(), sort_keys=True).encode()).hexdigest()

    def save(self, expected_revision: str) -> None:
        if self.library is not None:
            current = self.load(self.library)
            if current.revision != expected_revision:
                raise ModelSettingsConflict("model settings changed; refresh before saving")
            self.prepare()
            locations = {
                str(row["id"]): {key: value for key, value in row.items() if key != "id"}
                for row in cast("list[dict[str, object]]", self.to_wire()["locations"])
            }
            removed = [
                ["locations", row.id] for row in current.locations if row.id not in locations
            ]
            removed.extend(["locations", row.id, "kind"] for row in self.locations if not row.kind)
            self._shared(
                self.library,
                revision=current._shared_revision,
                patch={
                    "locations": locations,
                    "order": list(locations),
                    "downloadLocation": self.download_location,
                },
                remove=removed,
            )
            return
        settings = self.path()
        with hold_lock(settings.with_suffix(".lock"), timeout=5):
            if self._load_legacy().revision != expected_revision:
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

    def file_selections(self) -> tuple[ModelFileSelection, ...]:
        """Read engine-established file authority, never imported provenance."""
        if self.library is not None:
            values = cast("dict[str, object]", self._shared(self.library)["values"])
            rows = values.get("files", {})
            if not isinstance(rows, dict):
                raise MountsError("invalid shared model file selections")
            raw: object = list(cast("dict[str, object]", rows).values())
        else:
            path = self.path().with_name("model-file-selections.json")
            try:
                raw = json.loads(path.read_text("utf-8"))
            except (OSError, ValueError):
                return ()
        if not isinstance(raw, list):
            return ()
        selected: list[ModelFileSelection] = []
        for row in cast("list[object]", raw):
            try:
                entry = ModelFileSelection.from_wire(row)
            except (AssetError, KeyError, TypeError, ValueError):
                continue
            if any(
                location.path.resolve() == entry.root
                and any(
                    mount.id == entry.mount_id
                    and mount.path.resolve() == entry.mount_root
                    and mount.kind == entry.kind
                    for mount in location.mounts(0)
                )
                for location in self.locations
            ):
                selected.append(entry)
        return tuple(selected)

    def remember_file_selection(self, selected: ModelFileSelection) -> None:
        if self.library is not None:
            current = self.load(self.library)
            if current.to_wire() != self.to_wire() or selected.resolve() is None:
                raise ModelSettingsConflict("model location or selected file changed")
            prior = next(
                (
                    row
                    for row in current.file_selections()
                    if row.ref.virtual_path == selected.ref.virtual_path
                ),
                None,
            )
            if prior is not None and prior.ref.digest == selected.ref.digest:
                selected = replace(
                    selected, aliases=tuple(dict.fromkeys((*prior.aliases, *selected.aliases)))
                )
            self._shared(
                self.library,
                revision=current._shared_revision,
                patch={"files": {selected.ref.virtual_path: selected.to_wire()}},
            )
            return
        path = self.path().with_name("model-file-selections.json")
        with hold_lock(self.path().with_suffix(".lock"), timeout=5):
            current = self._load_legacy()
            if current.revision != self.revision or selected.resolve() is None:
                raise ModelSettingsConflict("model location or selected file changed")
            rows = {row.ref.virtual_path: row for row in current.file_selections()}
            prior = rows.get(selected.ref.virtual_path)
            if prior is not None and prior.ref.digest == selected.ref.digest:
                selected = replace(
                    selected, aliases=tuple(dict.fromkeys((*prior.aliases, *selected.aliases)))
                )
            rows[selected.ref.virtual_path] = selected
            descriptor, temporary = tempfile.mkstemp(prefix=".model-files-", dir=path.parent)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                    json.dump([row.to_wire() for row in rows.values()], output)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, path)
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
