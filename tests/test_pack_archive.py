from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
import tomllib
import unicodedata
import zipfile
from pathlib import Path

import pytest
from blake3 import blake3
from dinkster_workers import ManifestError

from dinkster import manager
from dinkster.pack_archive import (
    MAX_ARCHIVE_BYTES,
    MAX_ENTRIES,
    MAX_EXPANDED_BYTES,
    MAX_FILE_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_PATH_BYTES,
    PackArchiveError,
    _enforce_limit,
    build_pack_archive,
)
from scripts.build_official_packs import build as build_official_packs
from tests.platform_support import symlink_or_skip

MANIFEST = '[pack]\nname = "archive-test"\n\n[pack.entry]\nnodes = "nodes:NODES"\n'


def write_pack(root: Path) -> Path:
    root.mkdir()
    (root / "dinkster-pack.toml").write_bytes(MANIFEST.encode())
    (root / "nodes.py").write_bytes(b"NODES = []\n")
    return root


def test_archive_is_deterministic_and_has_registry_metadata(tmp_path: Path) -> None:
    first = write_pack(tmp_path / "first")
    second = write_pack(tmp_path / "second")
    os.utime(second / "nodes.py", (0, 0))
    (second / "nodes.py").chmod(0o755)

    first_output = tmp_path / "first.zip"
    second_output = tmp_path / "second.zip"
    first_digest = build_pack_archive(first, first_output)
    second_digest = build_pack_archive(second, second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    assert (
        first_digest == second_digest == f"blake3:{blake3(first_output.read_bytes()).hexdigest()}"
    )
    assert first_digest == "blake3:ccbd5bd1ece43a67a6256b92b58c4787036e85129b24bf08f59070c285ed3fe6"
    with zipfile.ZipFile(first_output) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        assert archive.comment == b""
        for info in archive.infolist():
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.create_system == 3
            assert info.external_attr == 0o644 << 16
            assert info.extra == b""
            assert info.comment == b""


def test_archive_normalizes_paths_and_omits_excluded_directories(tmp_path: Path) -> None:
    pack = write_pack(tmp_path / "pack")
    decomposed = unicodedata.normalize("NFD", f"caf{chr(0xE9)}.py")
    (pack / decomposed).write_text("content")
    for excluded in (
        ".git",
        ".hg",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    ):
        directory = pack / "nested" / excluded
        directory.mkdir(parents=True)
        (directory / "ignored").write_text("ignored")

    output = tmp_path / "pack.zip"
    build_pack_archive(pack, output)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    assert unicodedata.normalize("NFC", decomposed) in names
    assert all(not name.startswith("nested/") for name in names)


def test_archive_rejects_symlinks_and_normalized_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = write_pack(tmp_path / "pack")
    symlink_or_skip(pack / "link.py", pack / "nodes.py")
    with pytest.raises(PackArchiveError, match="symlink"):
        build_pack_archive(pack, tmp_path / "symlink.zip")

    (pack / "link.py").unlink()
    entries = list(pack.rglob("*"))
    composed = f"{chr(0xE9)}.py"
    decomposed = unicodedata.normalize("NFD", composed)
    (pack / composed).write_text("one")
    (pack / decomposed).write_text("two")
    # Normalization-insensitive filesystems cannot store both directory entries.
    monkeypatch.setattr(
        Path, "rglob", lambda self, pattern: iter([*entries, pack / composed, pack / decomposed])
    )
    with pytest.raises(PackArchiveError, match="collides"):
        build_pack_archive(pack, tmp_path / "collision.zip")


def test_archive_validates_manifest_before_writing(tmp_path: Path) -> None:
    pack = write_pack(tmp_path / "pack")
    (pack / "dinkster-pack.toml").write_text("not toml = [")
    output = tmp_path / "pack.zip"
    with pytest.raises(ManifestError):
        build_pack_archive(pack, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("limit", "subject"),
    [
        (MAX_ARCHIVE_BYTES, "archive"),
        (MAX_FILE_BYTES, "file"),
        (MAX_EXPANDED_BYTES, "expanded archive"),
        (MAX_PATH_BYTES, "archive path"),
        (MAX_MANIFEST_BYTES, "pack manifest"),
    ],
)
def test_byte_limits_accept_exact_and_reject_plus_one(limit: int, subject: str) -> None:
    _enforce_limit(limit, limit, subject)
    with pytest.raises(PackArchiveError, match="limit"):
        _enforce_limit(limit + 1, limit, subject)


def test_entry_limit_accepts_exact_and_rejects_plus_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = write_pack(tmp_path / "pack")
    monkeypatch.setattr("dinkster.pack_archive.MAX_ENTRIES", 2)
    build_pack_archive(pack, tmp_path / "exact.zip")
    (pack / "third.py").write_text("")
    with pytest.raises(PackArchiveError, match="3 entries; limit is 2"):
        build_pack_archive(pack, tmp_path / "too-many.zip")
    assert MAX_ENTRIES == 10_000


def test_file_manifest_expanded_and_archive_limits_are_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = write_pack(tmp_path / "pack")
    largest = pack / "largest.bin"
    largest.write_bytes(b"x" * 100)
    monkeypatch.setattr("dinkster.pack_archive.MAX_FILE_BYTES", 100)
    build_pack_archive(pack, tmp_path / "file-exact.zip")
    monkeypatch.setattr("dinkster.pack_archive.MAX_FILE_BYTES", 99)
    with pytest.raises(PackArchiveError, match="file 'largest.bin'"):
        build_pack_archive(pack, tmp_path / "file.zip")

    monkeypatch.setattr("dinkster.pack_archive.MAX_FILE_BYTES", MAX_FILE_BYTES)
    monkeypatch.setattr("dinkster.pack_archive.MAX_MANIFEST_BYTES", len(MANIFEST))
    build_pack_archive(pack, tmp_path / "manifest-exact.zip")
    monkeypatch.setattr("dinkster.pack_archive.MAX_MANIFEST_BYTES", len(MANIFEST) - 1)
    with pytest.raises(PackArchiveError, match="pack manifest"):
        build_pack_archive(pack, tmp_path / "manifest.zip")

    monkeypatch.setattr("dinkster.pack_archive.MAX_MANIFEST_BYTES", MAX_MANIFEST_BYTES)
    expanded = len(MANIFEST.encode()) + len(b"NODES = []\n") + 100
    monkeypatch.setattr("dinkster.pack_archive.MAX_EXPANDED_BYTES", expanded)
    build_pack_archive(pack, tmp_path / "expanded-exact.zip")
    monkeypatch.setattr("dinkster.pack_archive.MAX_EXPANDED_BYTES", expanded - 1)
    with pytest.raises(PackArchiveError, match="expanded archive"):
        build_pack_archive(pack, tmp_path / "expanded.zip")

    monkeypatch.setattr("dinkster.pack_archive.MAX_EXPANDED_BYTES", MAX_EXPANDED_BYTES)
    baseline = tmp_path / "archive-baseline.zip"
    build_pack_archive(pack, baseline)
    archive_size = baseline.stat().st_size
    monkeypatch.setattr("dinkster.pack_archive.MAX_ARCHIVE_BYTES", archive_size)
    build_pack_archive(pack, tmp_path / "archive-exact.zip")
    monkeypatch.setattr("dinkster.pack_archive.MAX_ARCHIVE_BYTES", archive_size - 1)
    with pytest.raises(PackArchiveError, match="archive"):
        build_pack_archive(pack, tmp_path / "archive.zip")


def test_path_limit_counts_normalized_utf8_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = write_pack(tmp_path / "pack")
    unicode_file = pack / (chr(0xE9) * 8 + ".py")
    unicode_file.write_text("")
    normalized_bytes = len(unicode_file.name.encode("utf-8"))
    monkeypatch.setattr("dinkster.pack_archive.MAX_PATH_BYTES", normalized_bytes)
    build_pack_archive(pack, tmp_path / "exact.zip")
    longer = pack / (unicode_file.name + "x")
    unicode_file.rename(longer)
    with pytest.raises(PackArchiveError, match="archive path"):
        build_pack_archive(pack, tmp_path / "too-long.zip")


def test_cli_writes_archive_and_prints_written_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = write_pack(tmp_path / "pack")
    output = tmp_path / "output" / "pack.zip"
    monkeypatch.setattr(
        sys,
        "argv",
        ["dinkster pack", "archive", str(pack), "--output", str(output)],
    )
    manager.main()
    printed = capsys.readouterr().out.strip()
    assert printed == f"blake3:{blake3(output.read_bytes()).hexdigest()}"


def test_official_artifacts_preserve_sources_and_existing_pack_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_registry import Lockfile, unpack_artifact
    from dinkster_workers import load_manifest

    root = Path(__file__).resolve().parents[1]
    version = "0.0.2.dev20261028"
    first = build_official_packs(root, tmp_path / "first", version=version)
    write_text = Path.write_text

    def windows_write_text(path, data, encoding=None, errors=None, newline=None):
        return write_text(
            path,
            data,
            encoding=encoding,
            errors=errors,
            newline="\r\n" if newline is None else newline,
        )

    with monkeypatch.context() as platform:
        platform.setattr(Path, "write_text", windows_write_text)
        platform.setenv("GIT_CONFIG_COUNT", "2")
        platform.setenv("GIT_CONFIG_KEY_0", "core.autocrlf")
        platform.setenv("GIT_CONFIG_VALUE_0", "true")
        platform.setenv("GIT_CONFIG_KEY_1", "core.eol")
        platform.setenv("GIT_CONFIG_VALUE_1", "crlf")
        second = build_official_packs(root, tmp_path / "second", version=version)
    assert [path.name for path in first] == [path.name for path in second]
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]
    assert b"\r\n" not in second[-1].read_bytes()
    lock = Lockfile.from_record_json(first[-1].read_text())
    expected = {
        "dinkster-nodes-comfy",
        "dinkster-nodes-foundation",
        "dinkster-nodes-image",
        "dinkster-nodes-media-io",
        "dinkster-nodes-remote",
        "dinkster-vision-birefnet",
        "dinkster-vision-depth-anything-v2",
        "dinkster-vision-depth-anything-v3",
        "dinkster-vision-detr",
        "dinkster-vision-efficient-sam",
        "dinkster-vision-hed",
        "dinkster-vision-rtdetr",
        "dinkster-vision-sam31",
        "dinkster-vision-upscale",
    }
    assert {pack.pack for pack in lock.packs} == expected
    core_requirements = {
        "dinkster-nodes-comfy": (
            "simpleeval==1.0.8",
            "numpy",
            "pillow",
            "torch",
            "torchvision",
            "dinkster-inference==0.1.0",
            "av",
            "scipy",
            "einops",
            "kornia",
            "safetensors",
            "tqdm",
            "spandrel",
            "comfy-kitchen",
        ),
        "dinkster-nodes-foundation": (),
        "dinkster-nodes-image": (
            "comfy-angle==0.1.1; sys_platform != 'darwin' or platform_machine == 'arm64'",
            "numpy>=1.26",
            "opencv-contrib-python-headless>=5.0,<5.1",
            "pillow>=10",
            "PyOpenGL>=3.1.8,<4",
        ),
        "dinkster-nodes-media-io": (
            "av==18.1.0",
            "numpy>=1.26",
            "pillow>=10",
            "scipy>=1.11",
        ),
        "dinkster-nodes-remote": ("httpx>=0.28,<1",),
    }
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    for pack in lock.packs:
        assert pack.version == version
        assert pack.publisher == "dinkster"
        assert pack.source == f"git:Kosinkadink/Dinkster@{commit}"
        archive = first[0].parent / f"{pack.pack}-{version}.zip"
        with zipfile.ZipFile(archive) as contents:
            assert not any("tests" in Path(name).parts for name in contents.namelist())
        extracted = unpack_artifact(
            archive, tmp_path / "unpacked" / pack.pack, pack.artifact_digest
        )
        assert b"\r\n" not in (extracted / "dinkster-pack.toml").read_bytes()
        manifest = load_manifest(extracted / "dinkster-pack.toml")
        assert manifest.name == pack.pack
        if pack.pack in core_requirements:
            assert manifest.requires == core_requirements[pack.pack]
        module = manifest.nodes_entry.partition(":")[0]
        relative = Path(*module.split(".")) / "__init__.py"
        if pack.pack.startswith("dinkster-vision-"):
            original = Path("packages/dinkster-nodes-vision/src") / relative
        else:
            original = Path("packages") / pack.pack / "src" / relative
        expected_source = subprocess.check_output(
            ["git", "show", f"HEAD:{original.as_posix()}"], cwd=root
        )
        assert (extracted / relative).read_bytes() == expected_source
        declared_templates = tomllib.loads((extracted / "dinkster-pack.toml").read_text())[
            "pack"
        ].get("templates", [])
        assert len(manifest.templates) == len(declared_templates)
        for template in manifest.templates:
            assert template.path.is_file()


@pytest.mark.parametrize(
    "pack", ["dinkster-nodes-comfy", "dinkster-nodes-remote", "dinkster-vision-upscale"]
)
def test_official_artifacts_reject_static_errors_in_built_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pack: str
) -> None:
    from dinkster_registry import build_artifact
    from dinkster_registry.model import artifact_digest

    def build_invalid_artifact(pack_dir: Path, output: Path) -> str:
        digest = build_artifact(pack_dir, output)
        if output.name.startswith(pack + "-"):
            with zipfile.ZipFile(output, "a") as archive:
                archive.writestr("runtime_import.py", "import dinkster_engine\n")
            return artifact_digest(output.read_bytes())
        return digest

    monkeypatch.setattr("scripts.build_official_packs.build_artifact", build_invalid_artifact)
    with pytest.raises(ValueError, match=rf"{pack}.*static doctor"):
        build_official_packs(
            Path(__file__).resolve().parents[1], tmp_path / "output", version="0.0.2.dev1"
        )
    assert not (tmp_path / "output" / "official-packs.lock").exists()


@pytest.mark.parametrize("version", ["0.0.2.dev1+gabc", "../../escape"])
def test_official_artifacts_require_public_pack_versions(tmp_path: Path, version: str) -> None:
    with pytest.raises(ValueError, match="pack version"):
        build_official_packs(Path(__file__).resolve().parents[1], tmp_path, version=version)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("selection", ["all", "single", "unknown", "undeclared"])
def test_official_publication_uses_selected_manifest_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str
) -> None:
    from dinkster_registry import Lockfile, unpack_artifact
    from dinkster_workers import load_manifest

    versions = {
        "dinkster-nodes-comfy": "1.3.0",
        "dinkster-nodes-foundation": "2.7.1",
        "dinkster-nodes-image": "0.8.0",
        "dinkster-nodes-media-io": "3.0.0rc1",
        "dinkster-nodes-remote": "1.2.4",
    }
    tracked = io.BytesIO()
    with tarfile.open(fileobj=tracked, mode="w") as archive:
        for pack, version in versions.items():
            declaration = (
                ""
                if selection == "undeclared" and pack == "dinkster-nodes-foundation"
                else f'version = "{version}"\nengine = ">=0.0.2,<0.1"\n'
            )
            files = {
                "dinkster-pack.toml": (
                    f'[pack]\nname = "{pack}"\npublisher = "dinkster"\n'
                    + declaration
                    + '[pack.entry]\nnodes = "fixture_nodes:NODES"\n'
                ),
                "pyproject.toml": '[project]\nrequires-python = ">=3.12"\n',
                "src/fixture_nodes.py": "NODES = []\n",
            }
            for name, text in files.items():
                data = text.encode()
                entry = tarfile.TarInfo(f"packages/{pack}/{name}")
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))

    def read_git(command, **kwargs):
        return "a" * 40 if "rev-parse" in command else tracked.getvalue()

    monkeypatch.setattr("scripts.build_official_packs.subprocess.check_output", read_git)
    requested = () if selection == "all" else ("dinkster-nodes-foundation",)
    if selection == "unknown":
        requested = ("dinkster-nodes-not-published",)
    output = tmp_path / "published"
    if selection in {"unknown", "undeclared"}:
        message = "unknown official packs" if selection == "unknown" else "version and engine"
        with pytest.raises(ValueError, match=message):
            build_official_packs(tmp_path, output, packs=requested)
        assert not list(output.glob("*.zip"))
        assert not (output / "official-packs.lock").exists()
        return
    paths = build_official_packs(tmp_path, output, packs=requested)
    expected = versions if selection == "all" else {"dinkster-nodes-foundation": "2.7.1"}
    assert {path.name for path in paths} == {
        "official-packs.lock",
        *(f"{pack}-{version}.zip" for pack, version in expected.items()),
    }
    lock = Lockfile.from_record_json(paths[-1].read_text())
    assert {pack.pack: pack.version for pack in lock.packs} == expected
    for pack in lock.packs:
        extracted = unpack_artifact(
            output / f"{pack.pack}-{pack.version}.zip",
            tmp_path / "unpacked" / pack.pack,
            pack.artifact_digest,
        )
        manifest = load_manifest(extracted / "dinkster-pack.toml")
        assert manifest.compatibility is not None
        assert manifest.compatibility.version == expected[pack.pack]
        assert manifest.compatibility.engine == ">=0.0.2,<0.1"


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        (("src/dinkster/serve.py", "pyproject.toml"), ()),
        (("packages/dinkster-nodes-foundation/src/nodes.py",), ("foundation",)),
        (("packages/dinkster-nodes-image/deleted.py",), ("image",)),
        (("packages/dinkster-nodes-dev/src/nodes.py",), ()),
        (("packages/dinkster-nodes-generation/dinkster-pack.toml",), ()),
        (("packages/dinkster-nodes-vision/first_pack/dinkster-pack.toml",), ("first",)),
        (("packages/dinkster-nodes-vision/src/vision/first/nodes.py",), ("first",)),
        (("packages/dinkster-nodes-vision/src/vision/second/nodes.py",), ("second",)),
        (("packages/dinkster-nodes-vision/src/vision/shared.py",), ("first", "second")),
        (("packages/dinkster-nodes-vision/pyproject.toml",), ("first", "second")),
        (
            ("packages/dinkster-nodes-foundation/old.py", "packages/dinkster-nodes-image/new.py"),
            ("foundation", "image"),
        ),
    ],
)
def test_official_publication_selects_only_changed_committed_packs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: tuple[str, ...],
    expected: tuple[str, ...],
) -> None:
    from scripts.select_official_packs import select_packs

    paths = {
        "foundation": "packages/dinkster-nodes-foundation/dinkster-pack.toml",
        "image": "packages/dinkster-nodes-image/dinkster-pack.toml",
        "first": "packages/dinkster-nodes-vision/first_pack/dinkster-pack.toml",
        "second": "packages/dinkster-nodes-vision/second_pack/dinkster-pack.toml",
    }
    committed = {
        path: (
            f'[pack]\nname = "{name}"\npublisher = "dinkster"\n'
            f'version = "{version}"\nengine = ">=0.2,<1"\n'
            f'[pack.entry]\nnodes = "vision.{name}:NODES"\n'
        ).encode()
        for (name, path), version in zip(
            paths.items(), ("1.0", "7.2", "3.4rc1", "0.5"), strict=True
        )
    }

    def git(command, **kwargs):
        if "rev-parse" in command:
            return "a" * 40 + "\n"
        if "ls-tree" in command:
            return "\0".join(
                [*committed, "packages/dinkster-nodes-dev/dinkster-pack.toml", ""]
            ).encode()
        if "show" in command:
            return committed[command[-1].split(":", 1)[1]]
        if "diff" in command:
            return ("\0".join(changed) + "\0").encode()
        raise AssertionError(command)

    monkeypatch.setattr("scripts.select_official_packs.subprocess.check_output", git)
    (tmp_path / "dinkster-pack.toml").write_text("invalid uncommitted data")
    assert select_packs(tmp_path, before="b" * 40) == expected
    assert select_packs(tmp_path, before="0" * 40) == ("first", "foundation", "image", "second")
    assert select_packs(tmp_path, tag="pack/first/v3.4rc1") == ("first",)
    for tag in ("v3.4rc1", "pack/first/v3.4", "pack/unknown/v3.4rc1", "pack/first/v3.4rc1/extra"):
        with pytest.raises(ValueError, match="pack tag"):
            select_packs(tmp_path, tag=tag)
    for before in ("main", "--all", "a" * 39, "z" * 40):
        with pytest.raises(ValueError, match="full Git commit"):
            select_packs(tmp_path, before=before)
    with pytest.raises(ValueError, match="select one"):
        select_packs(tmp_path)
    with pytest.raises(ValueError, match="select one"):
        select_packs(tmp_path, before="b" * 40, tag="pack/first/v3.4rc1")
    first = paths["first"]
    committed[first] = committed[first].replace(b'version = "3.4rc1"', b'version = "2.0"')
    with pytest.raises(ValueError, match="pack tag"):
        select_packs(tmp_path, tag="pack/first/v3.4rc1")
    committed[first] = committed[first].replace(b'engine = ">=0.2,<1"', b'engine = ""')
    with pytest.raises(ManifestError, match="engine"):
        select_packs(tmp_path, tag="pack/first/v2.0")


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "prepare-only",
        "missing-base",
        "duplicate-base",
        "wrong-base",
        "missing-lock",
        "duplicate-lock",
        "extra-lock",
        "pack",
        "variant",
        "python",
        "wheel-url",
        "index-url",
        "credentials",
        "empty-wheel",
        "boolean-size",
        "archive",
        "source-version",
        "publisher",
        "declaration",
    ],
)
def test_official_publication_requires_complete_supported_base_bindings(
    tmp_path: Path, problem: str | None
) -> None:
    import hashlib
    import json

    from dinkster_registry import ArtifactError, LockedPack, Lockfile, build_artifact
    from dinkster_workers.pack_lock import LockedDistribution, LockedWheel, PackLock
    from test_installer import _serve_artifacts

    from dinkster.registries import NamedRegistry
    from scripts.publish_official_packs import publish_official_packs

    sources = tmp_path / "sources"
    sources.mkdir()
    packs = []
    for name, version in (("first-pack", "1.2.3"), ("second-pack", "7.4.0rc1")):
        tree = tmp_path / name
        tree.mkdir()
        observed = "9.9" if name == "second-pack" and problem == "source-version" else version
        publisher = "other" if name == "second-pack" and problem == "publisher" else "dinkster"
        declaration = (
            ""
            if name == "second-pack" and problem == "declaration"
            else f'version = "{observed}"\nengine = ">=0.2,<1"\n'
        )
        (tree / "dinkster-pack.toml").write_text(
            f'[pack]\nname = "{name}"\npublisher = "{publisher}"\n'
            + declaration
            + '[pack.entry]\nnodes = "nodes:NODES"\n'
        )
        (tree / "nodes.py").write_text("raise RuntimeError('must not import')\n")
        archive = sources / f"{name}-{version}.zip"
        digest = build_artifact(tree, archive)
        packs.append(LockedPack(name, version, digest, "dinkster", ()))
    (sources / "official-packs.lock").write_text(Lockfile.of(packs).record_json())
    if problem == "archive":
        (sources / "second-pack-7.4.0rc1.zip").write_bytes(b"changed")
    bases, supported, locks = [], {}, []
    expected_locks = {}
    for variant, python in (("linux-cpu", "3.12.9"), ("win-nvidia", "3.14.10")):
        base = {
            "format": "dinkster.base/1",
            "variant": variant,
            "platform": {"os": "linux" if variant == "linux-cpu" else "windows", "arch": "x86_64"},
            "python": python,
            "interpreter": {},
            "wheels": [],
            "replaces": [],
            "workerProtocol": 11,
            "nvidia": None,
            "archive": {"format": "tar+zstd"},
        }
        identity = hashlib.sha256(
            json.dumps(base, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        supported[variant] = identity
        base["id"] = "sha256:" + identity
        base["archive"]["parts"] = []
        path = tmp_path / f"{variant}.json"
        path.write_text(json.dumps(base))
        bases.append(path)
        for pack in packs:
            corrupt = pack.pack == "second-pack" and variant == "win-nvidia"
            path = tmp_path / f"{pack.pack}-{pack.version}-{variant}.pack-lock.json"
            PackLock(
                "wrong" if corrupt and problem == "pack" else pack.pack,
                "wrong" if corrupt and problem == "variant" else variant,
                "<3.14" if corrupt and problem == "python" else ">=3.12",
                ("fixture-dependency==2.0",),
                (
                    "file:///local/index"
                    if corrupt and problem == "index-url"
                    else "https://pypi.org/simple",
                ),
                (
                    LockedDistribution(
                        "fixture-dependency",
                        "2.0",
                        (
                            LockedWheel(
                                "fixture_dependency-2.0-py3-none-any.whl",
                                0
                                if corrupt and problem == "empty-wheel"
                                else True
                                if corrupt and problem == "boolean-size"
                                else 17,
                                "b" * 64,
                                "file:///local/wheel.whl"
                                if corrupt and problem == "wheel-url"
                                else "https://user:password@example.test/wheel.whl"
                                if corrupt and problem == "credentials"
                                else "https://example.test/wheel.whl",
                            ),
                        ),
                    ),
                ),
            ).write(path)
            locks.append(path)
            expected_locks[pack.pack, variant] = json.loads(path.read_bytes())
    (tmp_path / "scripts").mkdir()
    if problem == "wrong-base":
        supported["win-nvidia"] = "c" * 64
    (tmp_path / "scripts/base_bundles.json").write_text(json.dumps(supported))
    if problem == "missing-base":
        bases.pop()
    elif problem == "duplicate-base":
        bases.append(bases[-1])
    elif problem == "missing-lock":
        locks.pop()
    elif problem == "duplicate-lock":
        locks.append(locks[-1])
    elif problem == "extra-lock":
        extra = tmp_path / "unselected-9.0-win-nvidia.pack-lock.json"
        extra.write_bytes(locks[-1].read_bytes())
        locks.append(extra)
    requests = []
    server = _serve_artifacts(
        {
            f"/v1/publish/{pack.pack}/versions/{pack.version}": json.dumps(
                {"status": "queued", "candidateId": pack.pack}
            ).encode()
            for pack in packs
        },
        requests,
    )
    output = tmp_path / "publication"
    before = {path: path.read_bytes() for path in [*sources.iterdir(), *locks, *bases]}
    try:
        registry = NamedRegistry("", f"http://127.0.0.1:{server.server_port}")
        if problem not in {None, "prepare-only"}:
            with pytest.raises((ValueError, ArtifactError)):
                publish_official_packs(tmp_path, sources, bases, locks, output, registry=registry)
            assert not output.exists()
            assert requests == []
        else:
            result = publish_official_packs(
                tmp_path,
                sources,
                bases,
                locks,
                output,
                registry=registry if problem is None else None,
            )
            assert len(result) == 2
            assert [row["version"] for row in result] == ["1.2.3", "7.4.0rc1"]
            for pack, row in zip(packs, result, strict=True):
                lock_map = row["locks"]
                assert isinstance(lock_map, str)
                bindings = json.loads(Path(lock_map).read_bytes())
                assert set(bindings) == {"linux-cpu", "win-nvidia"}
                for variant, identity in supported.items():
                    assert bindings[variant] == {
                        "baseId": "sha256:" + identity,
                        "artifactDigest": pack.artifact_digest,
                        "lock": expected_locks[pack.pack, variant],
                    }
                if problem is None:
                    assert row["state"] == "queued" and row["candidateId"] == pack.pack
            assert [path for path, _ in requests] == (
                [f"/v1/publish/{pack.pack}/versions/{pack.version}" for pack in packs]
                if problem is None
                else []
            )
        assert {path: path.read_bytes() for path in before} == before
        assert not list(tmp_path.glob("official-publication-*"))
    finally:
        server.shutdown()
        server.server_close()


def test_official_sources_prepare_catalogs_and_execute_existing_node_ids(tmp_path: Path) -> None:
    import asyncio
    import json
    from dataclasses import replace

    from dinkster_graph import graph_from_wire
    from dinkster_registry import Lockfile, build_artifact, unpack_artifact
    from dinkster_workers import load_manifest
    from dinkster_workers.catalog import read_catalog
    from dinkster_workers.doctor import prepare_catalog

    from dinkster.compose import CompositionError, PackSpec, ServingComposer
    from dinkster.packs import pack_info_from_manifest

    saved = tmp_path / "workflow.json"
    original = (
        b'{"nodes":{"length":{"nodeType":"StringLength","inputs":{"string":"source packs"}},'
        b'"region":{"nodeType":"dinkster.region.make",'
        b'"inputs":{"x":2.0,"y":3.0,"width":4.0,"height":5.0}}}}'
    )
    saved.write_bytes(original)
    payloads = build_official_packs(
        Path(__file__).resolve().parents[1], tmp_path / "payloads", version="0.1.2"
    )
    lock = Lockfile.from_record_json(payloads[-1].read_text())
    specs = []
    for entry in lock.packs:
        source = unpack_artifact(
            payloads[-1].parent / f"{entry.pack}-0.1.2.zip",
            tmp_path / "sources" / entry.pack,
            entry.artifact_digest,
        )
        manifest = load_manifest(source / "dinkster-pack.toml")
        assert read_catalog(manifest) is None
        report = prepare_catalog(manifest.path, environment={"PYTHONPATH": str(source)})
        assert report.ok, report
        assert read_catalog(manifest) is not None
        assert build_artifact(source, tmp_path / "repacked.zip") == entry.artifact_digest
        info = replace(
            pack_info_from_manifest(manifest),
            version=entry.version,
            artifact_digest=entry.artifact_digest,
            source=entry.source,
            publisher=entry.publisher,
        )
        specs.append(
            PackSpec(
                manifest=manifest.path,
                python=sys.executable,
                env={
                    "PYTHONPATH": str(source),
                    "DINKSTER_REMOTE_CATALOG_BASE": "",
                    "DINKSTER_REMOTE_GATEWAY_BASE": "",
                },
                packs={entry.pack: info},
                trust_reserved=True,
                require_catalog=True,
            )
        )

    async def execute() -> None:
        composer = ServingComposer()
        try:
            ordered = composer.order_pack_entries(specs)
            composer.validate_catalogs(ordered)
            for spec in ordered:
                await composer.add_pack(spec)
            composition = composer.composition
            assert composition.node_packs["StringLength"] == "dinkster-nodes-comfy"
            assert composition.node_packs["dinkster.region.make"] == "dinkster-nodes-image"
            assert composition.packs["dinkster-nodes-comfy"].version == "0.1.2"
            assert composition.packs["dinkster-nodes-image"].version == "0.1.2"
            result = await composition.make_engine(lambda _event: None).run(
                graph_from_wire(json.loads(saved.read_bytes())), ["length", "region"]
            )
            assert result.diagnostics == ()
            assert result.outputs["length"]["length"].resolve() == 12
            assert result.outputs["region"]["region"].meta.entries == {
                "x": 2.0,
                "y": 3.0,
                "width": 4.0,
                "height": 5.0,
            }
            assert saved.read_bytes() == original
            source = tmp_path / "sources" / "dinkster-nodes-comfy"
            (source / "changed.txt").write_text("source drift")
            with pytest.raises(CompositionError, match="missing or stale"):
                composer.validate_catalogs(ordered)
        finally:
            await composer.close()

    asyncio.run(execute())
