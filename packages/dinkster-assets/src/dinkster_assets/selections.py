"""Source option names bound to live exact-kind grants, never filesystem defaults."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from .declared import MOUNTS_ENV, mount_roots_for_kind
from .identity import AssetError
from .model import AssetRef, LocalAssetRef
from .mounts import MountSnapshotResolver

SELECTION_KINDS = frozenset(
    {"model/vae", "model/diffusers", "directory/input", "directory/dataset"}
)
_MODEL_EXTENSIONS = {".ckpt", ".pt", ".pt2", ".bin", ".pth", ".safetensors", ".pkl", ".sft"}
_IMAGE_TAES = ("taesd", "taesdxl", "taesd3", "taef1", "taef2", "taeqi2_1")
_VIDEO_TAES = ("taehv", "lighttaew2_2", "lighttaew2_1", "lighttaehy1_5", "taeltx_2", "taeh3")


@dataclass(frozen=True)
class _Choice:
    kind: str
    root: Path
    paths: tuple[Path, ...]
    shape: str


def _name(name: str, *, root_allowed: bool = False) -> None:
    if root_allowed and name == ".":
        return
    if (
        type(name) is not str
        or not name
        or "\\" in name
        or re.match(r"^[A-Za-z]:", name)
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise AssetError("selection requires a safe source-relative option name")


def _walk(root: Path, *, follow_links: bool = True) -> Iterator[tuple[Path, list[str], list[str]]]:
    seen: set[tuple[int, int]] = set()
    for directory, subdirs, files in os.walk(root, followlinks=follow_links):
        path = Path(directory)
        try:
            resolved = path.resolve(strict=True)
            stat = resolved.stat()
            identity = (stat.st_dev, stat.st_ino)
            if not resolved.is_relative_to(root) or identity in seen:
                subdirs[:] = []
                continue
            seen.add(identity)
            subdirs[:] = sorted(
                child for child in subdirs if (path / child).resolve().is_relative_to(root)
            )
            files = sorted(
                child for child in files if (path / child).resolve().is_relative_to(root)
            )
        except (OSError, RuntimeError):
            subdirs[:] = []
            continue
        yield path, subdirs, files


def _files(root: Path) -> dict[str, Path]:
    return {
        path.relative_to(root).as_posix(): path
        for directory, _subdirs, files in _walk(root)
        for name in files
        if (path := directory / name).suffix.lower() in _MODEL_EXTENSIONS and path.is_file()
    }


def _choices(kind: str, env: Mapping[str, str] | None) -> dict[str, _Choice]:
    if kind not in SELECTION_KINDS:
        raise AssetError(f"unsupported asset selection kind: {kind!r}")
    choices: dict[str, _Choice] = {}
    for granted in mount_roots_for_kind(kind, env=env):
        root = granted.resolve()
        if kind == "model/vae":
            for name, path in sorted(_files(root).items()):
                choices.setdefault(name, _Choice(kind, root, (path,), "file"))
            continue
        for directory, subdirs, files in _walk(root, follow_links=kind != "directory/input"):
            name = directory.relative_to(root).as_posix()
            matched = (
                kind == "directory/input"
                and name != "."
                or kind == "model/diffusers"
                and "model_index.json" in files
                or kind == "directory/dataset"
                and name != "."
                and (
                    "metadata.json" in files or any(file.endswith(".safetensors") for file in files)
                )
            )
            if matched:
                choices.setdefault(name, _Choice(kind, root, (directory,), "directory"))
                if kind == "directory/dataset":
                    subdirs[:] = []
    if kind != "model/vae":
        return dict(sorted(choices.items()))
    pairs: dict[str, _Choice] = {}
    for granted in mount_roots_for_kind("model/vae-approx", env=env):
        root = granted.resolve()
        files = dict(sorted(_files(root).items()))
        for name, path in files.items():
            if name.startswith(_VIDEO_TAES):
                choices.setdefault(name, _Choice("model/vae-approx", root, (path,), "file"))
        for name in _IMAGE_TAES:
            encoder = next(
                (path for file, path in files.items() if file.startswith(f"{name}_encoder.")), None
            )
            decoder = next(
                (path for file, path in files.items() if file.startswith(f"{name}_decoder.")), None
            )
            if encoder is not None and decoder is not None:
                pairs.setdefault(
                    name, _Choice("model/vae-approx", root, (encoder, decoder), "pair")
                )
    # Source literals and derived names take precedence over equally named files.
    choices.update(pairs)
    choices["pixel_space"] = _Choice(kind, Path(), (), "literal")
    return choices


def asset_selection_names(kind: str, *, env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """Enumerate source option names only inside the current exact-kind grants.

    VAE includes the file-free pixel_space literal even without grants. Other
    choices require a directory grant; a library, vault or guessed path never
    supplies authority. Pair members must coexist in one vae-approx root.
    """
    return tuple(_choices(kind, env))


def asset_selection_options(
    kind: str, *, env: Mapping[str, str] | None = None
) -> tuple[dict[str, str], ...]:
    """Picker descriptors contain source names and shapes, never host paths."""
    return tuple(
        {"name": name, "shape": choice.shape} for name, choice in _choices(kind, env).items()
    )


def _check_subtree(path: Path) -> None:
    root = path.resolve(strict=True)

    def visit(directory: Path, ancestors: frozenset[tuple[int, int]]) -> None:
        stat = directory.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in ancestors:
            raise AssetError("selected directory contains a link cycle")
        for child in directory.iterdir():
            resolved = child.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise AssetError("selected directory contains a link outside its subtree")
            if resolved.is_dir():
                visit(resolved, ancestors | {identity})
            elif not resolved.is_file():
                raise AssetError("selected directory contains a non-file member")

    visit(root, frozenset())


def _file_ref(choice: _Choice, path: Path, env: Mapping[str, str] | None) -> LocalAssetRef:
    environment = os.environ if env is None else env
    snapshot = environment.get(MOUNTS_ENV)
    if not snapshot:
        raise AssetError("selected file has no configured mounts snapshot")
    return MountSnapshotResolver(snapshot).bind_relative_file(
        choice.kind, choice.root, path.relative_to(choice.root).as_posix()
    )


def resolve_asset_selection(
    kind: str, name: str | AssetRef | LocalAssetRef, *, env: Mapping[str, str] | None = None
) -> AssetRef | LocalAssetRef | tuple[LocalAssetRef, LocalAssetRef] | Path | None:
    """Resolve one current source option, rejecting stale or ungranted choices.

    Existing file refs retain their verification policy but must resolve to
    an admitted file inside the current exact-kind grants. Returns that ref,
    two local file refs (encoder, decoder), a checked directory subtree, or
    None for pixel_space. Directory paths are invocation-scoped read authority,
    never values to store in a graph.
    """
    if kind not in SELECTION_KINDS:
        raise AssetError(f"unsupported asset selection kind: {kind!r}")
    if isinstance(name, (AssetRef, LocalAssetRef)):
        path = name.local_path().resolve(strict=True)
        if kind == "model/vae":
            for file_kind in ("model/vae", "model/vae-approx"):
                for root in mount_roots_for_kind(file_kind, env=env):
                    if not path.is_relative_to(root.resolve()):
                        continue
                    relative = path.relative_to(root.resolve()).as_posix()
                    if (
                        path.is_file()
                        and path.suffix.lower() in _MODEL_EXTENSIONS
                        and (file_kind == "model/vae" or relative.startswith(_VIDEO_TAES))
                    ):
                        return name
        raise AssetError("file reference is not an admitted selection in the current grants")
    _name(name, root_allowed=kind == "model/diffusers")
    choice = _choices(kind, env).get(name)
    if choice is None:
        raise AssetError(f"asset selection {name!r} is not granted for {kind!r}")
    try:
        if choice.shape == "literal":
            return None
        if choice.shape == "directory":
            path = choice.paths[0]
            if kind == "directory/dataset":
                # Source discovery uses markers, but lookup takes the first
                # existing same-named directory, even without its own marker.
                for root in mount_roots_for_kind(kind, env=env):
                    candidate = root.resolve() / name
                    if not candidate.resolve().is_relative_to(root.resolve()):
                        raise AssetError("selected dataset directory escapes its grant")
                    if candidate.is_dir():
                        path = candidate
                        break
            _check_subtree(path)
            return path
        if choice.shape == "pair":
            return _file_ref(choice, choice.paths[0], env), _file_ref(choice, choice.paths[1], env)
        if kind == "model/vae":
            # Source discovery accepts video prefixes, but loading routes
            # only exact video stems to vae-approx, even on name collisions.
            file_kind = "model/vae-approx" if os.path.splitext(name)[0] in _VIDEO_TAES else kind
            for root in mount_roots_for_kind(file_kind, env=env):
                root = root.resolve()
                path = _files(root).get(name)
                if path is not None:
                    return _file_ref(_Choice(file_kind, root, (path,), "file"), path, env)
            raise AssetError(f"asset selection {name!r} is not granted for {file_kind!r}")
        return _file_ref(choice, choice.paths[0], env)
    except (OSError, RuntimeError) as exc:
        raise AssetError(f"asset selection {name!r} is no longer readable") from exc
