from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
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
from dinkster_workers.pack_wheel import _enforce_limit

from dinkster import manager
from dinkster.pack_archive import (
    MAX_ARCHIVE_BYTES,
    MAX_ENTRIES,
    MAX_EXPANDED_BYTES,
    MAX_FILE_BYTES,
    MAX_MANIFEST_BYTES,
    MAX_PATH_BYTES,
    PackArchiveError,
    build_pack_archive,
)
from scripts.build_official_packs import build as build_official_packs
from tests.async_scenarios import run_scenario
from tests.platform_support import symlink_or_skip

MANIFEST = (
    '[pack]\nname = "archive-test"\nversion = "1.0"\nengine = ">=0.9,<1"\n'
    '[pack.entry]\nnodes = "archive_test:NODES"\n'
)
WHEEL_FILENAME = "archive_test-1.0-py3-none-any.whl"


@pytest.fixture(scope="module")
def official_payloads(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, ...]:
    # Consumers only read these wheels and unpack into their own test directories.
    return build_official_packs(
        Path(__file__).resolve().parents[1], tmp_path_factory.mktemp("official-payloads")
    )


def write_pack(root: Path) -> Path:
    root.mkdir()
    (root / "dinkster-pack.toml").write_bytes(MANIFEST.encode())
    (root / "archive_test").mkdir()
    (root / "archive_test/__init__.py").write_bytes(b"NODES = []\n")
    return root


def test_archive_is_deterministic_and_has_registry_metadata(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures/registry-admission/color-tools"
    first = shutil.copytree(fixture, tmp_path / "first")
    second = shutil.copytree(fixture, tmp_path / "second")
    os.utime(second / "color_tools/__init__.py", (0, 0))
    (second / "color_tools/__init__.py").chmod(0o755)

    first_output = tmp_path / "first-output/color_tools-1.2.3-py3-none-any.whl"
    second_output = tmp_path / "second-output/color_tools-1.2.3-py3-none-any.whl"
    first_digest = build_pack_archive(first, first_output)
    second_digest = build_pack_archive(second, second_output)

    assert first_output.read_bytes() == second_output.read_bytes()
    assert (
        first_digest
        == second_digest
        == f"sha256:{hashlib.sha256(first_output.read_bytes()).hexdigest()}"
    )
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
    (pack / "archive_test" / decomposed).write_text("content")
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

    output = tmp_path / WHEEL_FILENAME
    build_pack_archive(pack, output)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    assert "archive_test/" + unicodedata.normalize("NFC", decomposed) in names
    assert all("nested" not in name.split("/") for name in names)


def test_archive_rejects_symlinks_and_normalized_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = write_pack(tmp_path / "pack")
    package = pack / "archive_test"
    symlink_or_skip(package / "link.py", package / "__init__.py")
    with pytest.raises(PackArchiveError, match="symlink"):
        build_pack_archive(pack, tmp_path / WHEEL_FILENAME)

    (package / "link.py").unlink()
    entries = list(pack.rglob("*"))
    composed = f"{chr(0xE9)}.py"
    decomposed = unicodedata.normalize("NFD", composed)
    (package / composed).write_text("one")
    (package / decomposed).write_text("two")
    # Normalization-insensitive filesystems cannot store both directory entries.
    monkeypatch.setattr(
        Path,
        "rglob",
        lambda self, pattern: iter([*entries, package / composed, package / decomposed]),
    )
    with pytest.raises(PackArchiveError, match="collides"):
        build_pack_archive(pack, tmp_path / WHEEL_FILENAME)


def test_archive_validates_manifest_before_writing(tmp_path: Path) -> None:
    pack = write_pack(tmp_path / "pack")
    (pack / "dinkster-pack.toml").write_text("not toml = [")
    output = tmp_path / WHEEL_FILENAME
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
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_ENTRIES", 5)
    build_pack_archive(pack, tmp_path / WHEEL_FILENAME)
    (pack / "archive_test/third.py").write_text("")
    with pytest.raises(PackArchiveError, match="6 entries; limit is 5"):
        build_pack_archive(pack, tmp_path / WHEEL_FILENAME)
    assert MAX_ENTRIES == 10_000


def test_file_manifest_expanded_and_archive_limits_are_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = write_pack(tmp_path / "pack")
    largest = pack / "largest.bin"
    largest.write_bytes(b"x" * 1024)
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_FILE_BYTES", 1024)
    build_pack_archive(pack, tmp_path / WHEEL_FILENAME)
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_FILE_BYTES", 1023)
    with pytest.raises(PackArchiveError, match="file 'archive_test/largest.bin'"):
        build_pack_archive(pack, tmp_path / WHEEL_FILENAME)

    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_FILE_BYTES", MAX_FILE_BYTES)
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_MANIFEST_BYTES", len(MANIFEST))
    build_pack_archive(pack, tmp_path / WHEEL_FILENAME)
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_MANIFEST_BYTES", len(MANIFEST) - 1)
    with pytest.raises(PackArchiveError, match="pack manifest"):
        build_pack_archive(pack, tmp_path / WHEEL_FILENAME)

    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_MANIFEST_BYTES", MAX_MANIFEST_BYTES)
    baseline = tmp_path / WHEEL_FILENAME
    build_pack_archive(pack, baseline)
    with zipfile.ZipFile(baseline) as archive:
        expanded = sum(info.file_size for info in archive.infolist())
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_EXPANDED_BYTES", expanded)
    build_pack_archive(pack, tmp_path / WHEEL_FILENAME)
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_EXPANDED_BYTES", expanded - 1)
    with pytest.raises(PackArchiveError, match="expanded archive"):
        build_pack_archive(pack, tmp_path / WHEEL_FILENAME)

    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_EXPANDED_BYTES", MAX_EXPANDED_BYTES)
    build_pack_archive(pack, baseline)
    archive_size = baseline.stat().st_size
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_ARCHIVE_BYTES", archive_size)
    build_pack_archive(pack, tmp_path / WHEEL_FILENAME)
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_ARCHIVE_BYTES", archive_size - 1)
    with pytest.raises(PackArchiveError, match="archive"):
        build_pack_archive(pack, tmp_path / WHEEL_FILENAME)


def test_path_limit_counts_normalized_utf8_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pack = write_pack(tmp_path / "pack")
    unicode_file = pack / "archive_test" / (chr(0xE9) * 32 + ".py")
    unicode_file.write_text("")
    normalized_bytes = len(("archive_test/" + unicode_file.name).encode("utf-8"))
    monkeypatch.setattr("dinkster_workers.pack_wheel.MAX_PATH_BYTES", normalized_bytes)
    build_pack_archive(pack, tmp_path / WHEEL_FILENAME)
    longer = unicode_file.with_name(unicode_file.name + "x")
    unicode_file.rename(longer)
    with pytest.raises(PackArchiveError, match="archive path"):
        build_pack_archive(pack, tmp_path / WHEEL_FILENAME)


def test_cli_writes_archive_and_prints_written_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pack = write_pack(tmp_path / "pack")
    output = tmp_path / "output" / WHEEL_FILENAME
    monkeypatch.setattr(
        sys,
        "argv",
        ["dinkster pack", "archive", str(pack), "--output", str(output)],
    )
    manager.main()
    printed = capsys.readouterr().out.strip()
    assert printed == f"sha256:{hashlib.sha256(output.read_bytes()).hexdigest()}"


def test_official_artifacts_preserve_sources_and_existing_pack_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, official_payloads: tuple[Path, ...]
) -> None:
    from dinkster_registry import Lockfile, unpack_artifact
    from dinkster_workers import load_manifest

    root = Path(__file__).resolve().parents[1]
    first = official_payloads
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
        second = build_official_packs(root, tmp_path / "second")
    assert [path.name for path in first] == [path.name for path in second]
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]
    assert b"\r\n" not in second[-1].read_bytes()
    lock = Lockfile.from_record_json(first[-1].read_text())
    expected = {
        "dinkster-audio",
        "dinkster-audio-models",
        "dinkster-conditioning",
        "dinkster-cosmos",
        "dinkster-flux",
        "dinkster-hidream",
        "dinkster-hunyuan",
        "dinkster-latent",
        "dinkster-llm",
        "dinkster-ltx",
        "dinkster-minimax",
        "dinkster-misc-models",
        "dinkster-models",
        "dinkster-3d",
        "dinkster-3d-models",
        "dinkster-image",
        "dinkster-utilities",
        "dinkster-nodes-remote",
        "dinkster-pack-helpers",
        "dinkster-qwen",
        "dinkster-sampling",
        "dinkster-sd",
        "dinkster-seedvr",
        "dinkster-text",
        "dinkster-upscale",
        "dinkster-video",
        "dinkster-vision-background-removal",
        "dinkster-vision-depth-anything-3",
        "dinkster-vision-marigold",
        "dinkster-vision-mediapipe",
        "dinkster-vision-moge",
        "dinkster-vision-rtdetr",
        "dinkster-vision-sam3",
        "dinkster-vision-sam3d-body",
        "dinkster-vision-sdpose",
        "dinkster-void",
        "dinkster-wan",
    }
    assert {pack.pack for pack in lock.packs} == expected
    core_requirements = {
        "dinkster-image": (
            "comfy-angle==0.1.1; sys_platform != 'darwin' or platform_machine == 'arm64'",
            "numpy>=1.26",
            "opencv-contrib-python-headless>=5.0,<5.1",
            "pillow>=10",
            "PyOpenGL>=3.1.8,<4",
            "av==18.1.0",
            "torch",
            "torchvision",
            "scipy",
            "kornia",
            "dinkster-inference",
        ),
        "dinkster-nodes-remote": ("httpx>=0.28,<1",),
    }
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    for pack in lock.packs:
        assert pack.publisher == "dinkster"
        assert pack.source == f"git:Kosinkadink/Dinkster@{commit}"
        package_name = pack.pack.replace("-", "_")
        archive = first[0].parent / f"{package_name}-{pack.version}-py3-none-any.whl"
        with zipfile.ZipFile(archive) as contents:
            assert not any("tests" in Path(name).parts for name in contents.namelist())
            if pack.pack == "dinkster-pack-helpers":
                assert pack.claims == ()
                assert not any(name.endswith("dinkster-pack.toml") for name in contents.namelist())
                for name in contents.namelist():
                    if name.startswith(f"{package_name}/"):
                        expected_source = subprocess.check_output(
                            ["git", "show", f"HEAD:packages/{pack.pack}/src/{name}"], cwd=root
                        )
                        assert contents.read(name) == expected_source
                continue
        extracted = (
            unpack_artifact(archive, tmp_path / "unpacked" / pack.pack, pack.artifact_digest)
            / package_name
        )
        assert b"\r\n" not in (extracted / "dinkster-pack.toml").read_bytes()
        manifest = load_manifest(extracted / "dinkster-pack.toml")
        assert manifest.name == pack.pack
        if pack.pack in core_requirements:
            assert manifest.requires == core_requirements[pack.pack]
        module = manifest.nodes_entry.partition(":")[0]
        relative = Path(*module.split(".")) / "__init__.py"
        if not pack.pack.startswith("dinkster-nodes-"):
            original = (
                Path("packages")
                / pack.pack.replace("dinkster-", "dinkster-nodes-", 1)
                / "src"
                / relative
            )
        else:
            original = Path("packages") / pack.pack / "src" / relative
        original_manifest = subprocess.check_output(
            ["git", "show", f"HEAD:{original.parents[2].as_posix()}/dinkster-pack.toml"],
            cwd=root,
        )
        assert pack.version == tomllib.loads(original_manifest.decode())["pack"]["version"]
        expected_source = subprocess.check_output(
            ["git", "show", f"HEAD:{original.as_posix()}"], cwd=root
        )
        assert (extracted / "__init__.py").read_bytes() == expected_source
        declared_templates = tomllib.loads((extracted / "dinkster-pack.toml").read_text())[
            "pack"
        ].get("templates", [])
        assert len(manifest.templates) == len(declared_templates)
        for template in manifest.templates:
            assert template.path.is_file()


@pytest.mark.parametrize(
    "pack", ["dinkster-image", "dinkster-nodes-remote", "dinkster-vision-sam3"]
)
def test_official_artifacts_reject_static_errors_in_built_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pack: str
) -> None:
    def build_invalid_artifact(pack_dir: Path, output: Path) -> str:
        if output.name.startswith(pack.replace("-", "_") + "-"):
            (pack_dir / "src" / pack.replace("-", "_") / "runtime_import.py").write_text(
                "import dinkster_engine\n"
            )
        return build_pack_archive(pack_dir, output)

    monkeypatch.setattr("scripts.build_official_packs.build_pack_archive", build_invalid_artifact)
    with pytest.raises(ValueError, match=rf"{pack}.*static doctor"):
        build_official_packs(Path(__file__).resolve().parents[1], tmp_path / "output")
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
        "dinkster-text": "0.1.0",
        "dinkster-upscale": "0.2.1",
        "dinkster-nodes-unlisted-family": "4.2.1",
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
                    + f'[pack.entry]\nnodes = "{pack.replace("-", "_")}:NODES"\n'
                ),
                "pyproject.toml": f'[project]\nname = "{pack}"\nrequires-python = ">=3.12"\n',
                f"src/{pack.replace('-', '_')}/__init__.py": "NODES = []\n",
            }
            for name, text in files.items():
                data = text.encode()
                directory = (
                    pack.replace("dinkster-", "dinkster-nodes-", 1)
                    if pack in {"dinkster-text", "dinkster-upscale"}
                    else pack
                )
                entry = tarfile.TarInfo(f"packages/{directory}/{name}")
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
        assert not list(output.glob("*.whl"))
        assert not (output / "official-packs.lock").exists()
        return
    paths = build_official_packs(tmp_path, output, packs=requested)
    repeated = build_official_packs(tmp_path, tmp_path / "repeated", packs=requested)
    assert {path.name: path.read_bytes() for path in paths} == {
        path.name: path.read_bytes() for path in repeated
    }
    expected = versions if selection == "all" else {"dinkster-nodes-foundation": "2.7.1"}
    assert {path.name for path in paths} == {
        "official-packs.lock",
        *(
            f"{pack.replace('-', '_')}-{version}-py3-none-any.whl"
            for pack, version in expected.items()
        ),
    }
    lock = Lockfile.from_record_json(paths[-1].read_text())
    assert {pack.pack: pack.version for pack in lock.packs} == expected
    for pack in lock.packs:
        extracted = unpack_artifact(
            output / f"{pack.pack.replace('-', '_')}-{pack.version}-py3-none-any.whl",
            tmp_path / "unpacked" / pack.pack,
            pack.artifact_digest,
        ) / pack.pack.replace("-", "_")
        manifest = load_manifest(extracted / "dinkster-pack.toml")
        assert manifest.compatibility is not None
        assert manifest.compatibility.version == expected[pack.pack]
        assert manifest.compatibility.engine == ">=0.0.2,<0.1"


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "unmarked",
        "newly-marked",
        "requires-pack",
        "pack-requires-pack",
        "manifest",
        "library-rejected",
    ],
)
def test_official_library_selection_and_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str | None
) -> None:
    from dinkster_registry import Lockfile

    from scripts.select_official_packs import select_packs

    files = {
        "packages/helpers/pyproject.toml": (
            '[project]\nname = "dinkster-test-helpers"\nversion = "0.0.1"\n'
            'requires-python = ">=3.12"\n'
            + (
                'dependencies = ["dinkster-test-consumer>=0.0.1"]\n'
                if problem == "requires-pack"
                else ""
            )
            + '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
            '[tool.hatch.build.targets.wheel]\npackages = ["src/dinkster_test_helpers"]\n'
            + ("" if problem == "unmarked" else "[tool.dinkster]\nofficial-library = true\n")
        ),
        "packages/helpers/src/dinkster_test_helpers/__init__.py": "VALUE = 17\n",
        "packages/dinkster-nodes-test/pyproject.toml": (
            '[project]\nname = "dinkster-test-consumer"\nversion = "0.0.1"\n'
        ),
        "packages/dinkster-nodes-test/dinkster-pack.toml": (
            '[pack]\nname = "dinkster-test-consumer"\nversion = "0.0.1"\n'
            'publisher = "dinkster"\nengine = ">=0.0.2,<0.1"\n'
            'requires = ["dinkster-test-helpers>=0.0.1"]\n'
            '[pack.entry]\nnodes = "dinkster_test_consumer:NODES"\n'
        ),
        "packages/dinkster-nodes-test/src/dinkster_test_consumer/__init__.py": "NODES = []\n",
    }
    if problem == "pack-requires-pack":
        manifest_path = "packages/dinkster-nodes-test/dinkster-pack.toml"
        files["packages/dinkster-nodes-supplier/dinkster-pack.toml"] = (
            files[manifest_path]
            .replace("dinkster-test-consumer", "dinkster-test-supplier")
            .replace("dinkster_test_consumer", "dinkster_test_supplier")
        )
        files["packages/dinkster-nodes-supplier/pyproject.toml"] = (
            '[project]\nname = "dinkster-test-supplier"\nversion = "0.0.1"\n'
        )
        files["packages/dinkster-nodes-supplier/src/dinkster_test_supplier/__init__.py"] = (
            "NODES = []\n"
        )
        files[manifest_path] = files[manifest_path].replace(
            'requires = ["dinkster-test-helpers>=0.0.1"]',
            'requires = ["dinkster-test-supplier>=0.0.1"]',
        )
    if problem == "manifest":
        files["packages/helpers/dinkster-pack.toml"] = files[
            "packages/dinkster-nodes-test/dinkster-pack.toml"
        ]
    tracked = io.BytesIO()
    with tarfile.open(fileobj=tracked, mode="w") as archive:
        for name, text in files.items():
            data = text.encode()
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))

    def git(command, **kwargs):
        if "rev-parse" in command:
            return "a" * 40
        if "ls-tree" in command:
            return ("\0".join(files) + "\0").encode()
        if "show" in command:
            content = files[command[-1].split(":", 1)[1]]
            if problem == "newly-marked" and command[-1].startswith("b" * 40):
                content = content.replace("official-library = true", "official-library = false")
            return content.encode()
        if "diff" in command:
            return b"packages/helpers/src/dinkster_test_helpers/__init__.py\0"
        if "archive" in command:
            return tracked.getvalue()
        raise AssertionError(command)

    monkeypatch.setattr("scripts.build_official_packs.subprocess.check_output", git)
    if problem == "manifest":
        with pytest.raises(ValueError, match="node manifest"):
            build_official_packs(tmp_path, tmp_path / "built")
        with pytest.raises(ValueError, match="duplicates"):
            select_packs(tmp_path, before="0" * 40)
        return
    expected = (
        ("dinkster-test-consumer",)
        + (() if problem == "unmarked" else ("dinkster-test-helpers",))
        + (("dinkster-test-supplier",) if problem == "pack-requires-pack" else ())
    )
    assert select_packs(tmp_path, before="0" * 40) == expected
    assert select_packs(tmp_path, before="b" * 40) == (
        () if problem == "unmarked" else ("dinkster-test-helpers",)
    )
    assert select_packs(tmp_path, before="b" * 40, version_changes_only=True) == (
        ("dinkster-test-helpers",) if problem == "newly-marked" else ()
    )
    if problem in {"requires-pack", "pack-requires-pack"}:
        with pytest.raises(ValueError, match="wheel.requires-pack"):
            build_official_packs(tmp_path, tmp_path / "built")
        return
    paths = build_official_packs(tmp_path, tmp_path / "built")
    repeated = build_official_packs(tmp_path, tmp_path / "repeated")
    assert {path.name: path.read_bytes() for path in paths} == {
        path.name: path.read_bytes() for path in repeated
    }
    catalog = Lockfile.from_record_json(paths[-1].read_text())
    assert tuple(pack.pack for pack in catalog.packs) == expected
    if problem != "unmarked":
        library = next(pack for pack in catalog.packs if pack.pack == "dinkster-test-helpers")
        assert library.claims == ()
        with zipfile.ZipFile(
            tmp_path / "built/dinkster_test_helpers-0.0.1-py3-none-any.whl"
        ) as wheel:
            assert wheel.read("dinkster_test_helpers/__init__.py") == b"VALUE = 17\n"
            assert not any(name.endswith("dinkster-pack.toml") for name in wheel.namelist())
    if problem not in {None, "library-rejected"}:
        return

    import json

    from dinkster_workers.pack_lock import PackLock
    from test_installer import _serve_artifacts

    from dinkster.registries import NamedRegistry, RegistryPublishError
    from scripts.publish_official_packs import publish_official_packs

    constraints = tmp_path / "constraints.txt"
    constraints.write_text("")
    native = build_official_packs(
        tmp_path,
        tmp_path / "native",
        packs=("dinkster-test-consumer",),
        variant="linux-cpu",
        python=Path(sys.executable),
        constraints=constraints,
    )
    assert native[-1].read_bytes() == paths[-1].read_bytes()
    lock = PackLock.read(tmp_path / "native/dinkster-test-consumer-0.0.1-linux-cpu.pack-lock.json")
    assert [package.name for package in lock.packages] == ["dinkster-test-helpers"]
    library_bytes = (tmp_path / "built/dinkster_test_helpers-0.0.1-py3-none-any.whl").read_bytes()
    assert lock.packages[0].wheels[0].url == (
        "https://registry.dinkster.org/static/wheels/"
        + hashlib.sha256(library_bytes).hexdigest()
        + "/dinkster_test_helpers-0.0.1-py3-none-any.whl"
    )
    base = {
        "format": "dinkster.base/1",
        "variant": "linux-cpu",
        "platform": {"os": "linux", "arch": "x86_64"},
        "python": ".".join(map(str, sys.version_info[:3])),
        "interpreter": {},
        "wheels": [],
        "replaces": [],
        "workerProtocol": 11,
        "nvidia": None,
        "archive": {"format": "tar+zstd"},
    }
    identity = hashlib.sha256(
        json.dumps(base, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    base["id"] = "sha256:" + identity
    base["archive"]["parts"] = []
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/base_bundles.json").write_text(json.dumps({"linux-cpu": identity}))
    responses = {
        f"/v1/publish/{name}/versions/0.0.1": json.dumps(
            {"status": "queued", "candidateId": name}
        ).encode()
        for name in expected
    }
    server = _serve_artifacts(responses, [])
    respond = vars(server.RequestHandlerClass)["respond"]
    submitted = []

    def registry_response(handler):
        if handler.command == "POST":
            name = handler.path.split("/")[3]
            if name == "dinkster-test-consumer":
                assert submitted == ["dinkster-test-helpers"]
            submitted.append(name)
            if problem == "library-rejected":
                responses[f"/v1/publish/{name}"] = json.dumps(
                    {
                        "status": "rejected",
                        "reason": "library refused",
                        "findings": [{"code": "wheel.invalid", "message": "refused"}],
                    }
                ).encode()
            else:
                responses[f"/v1/packs/{name}/versions/0.0.1"] = json.dumps(
                    {
                        "packName": name,
                        "version": "0.0.1",
                        "state": "active",
                        "artifact": {
                            "digest": next(
                                pack.artifact_digest for pack in catalog.packs if pack.pack == name
                            )
                        },
                    }
                ).encode()
        respond(handler)

    monkeypatch.setattr(server.RequestHandlerClass, "respond", registry_response)
    try:
        if problem == "library-rejected":
            with pytest.raises(RegistryPublishError, match="activation refused.*wheel.invalid"):
                publish_official_packs(
                    tmp_path,
                    tmp_path / "built",
                    [base_path],
                    [path for path in native if path.name.endswith(".pack-lock.json")],
                    tmp_path / "submitted",
                    registry=NamedRegistry("", f"http://127.0.0.1:{server.server_port}"),
                )
            assert submitted == ["dinkster-test-helpers"]
            return
        result = publish_official_packs(
            tmp_path,
            tmp_path / "built",
            [base_path],
            [path for path in native if path.name.endswith(".pack-lock.json")],
            tmp_path / "submitted",
            registry=NamedRegistry("", f"http://127.0.0.1:{server.server_port}"),
        )
        assert (
            [row["pack"] for row in result]
            == submitted
            == ["dinkster-test-helpers", "dinkster-test-consumer"]
        )
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        (("src/dinkster/serve.py", "pyproject.toml"), ()),
        (("packages/dinkster-nodes-foundation/src/nodes.py",), ("foundation",)),
        (("packages/dinkster-nodes-image/deleted.py",), ("image",)),
        (("packages/dinkster-nodes-unlisted-family/src/nodes.py",), ("unlisted",)),
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
        "unlisted": "packages/dinkster-nodes-unlisted-family/dinkster-pack.toml",
    }
    committed = {
        path: (
            f'[pack]\nname = "{name}"\npublisher = "dinkster"\n'
            f'version = "{version}"\nengine = ">=0.2,<1"\n'
            f'[pack.entry]\nnodes = "vision.{name}:NODES"\n'
        ).encode()
        for (name, path), version in zip(
            paths.items(), ("1.0", "7.2", "3.4rc1", "0.5", "4.2.1"), strict=True
        )
    }

    def git(command, **kwargs):
        if "rev-parse" in command:
            return "a" * 40 + "\n"
        if "ls-tree" in command:
            manifests = [
                path for path in committed if "b" * 40 not in command or path != paths["first"]
            ]
            return "\0".join(
                [*manifests, "packages/dinkster-nodes-dev/dinkster-pack.toml", ""]
            ).encode()
        if "show" in command:
            data = committed[command[-1].split(":", 1)[1]]
            return (
                data.replace(b'version = "1.0"', b'version = "0.9"')
                if command[-1].startswith("b" * 40)
                else data
            )
        if "diff" in command:
            return ("\0".join(changed) + "\0").encode()
        raise AssertionError(command)

    monkeypatch.setattr("scripts.select_official_packs.subprocess.check_output", git)
    (tmp_path / "dinkster-pack.toml").write_text("invalid uncommitted data")
    assert select_packs(tmp_path, before="b" * 40) == expected
    assert select_packs(tmp_path, before="b" * 40, version_changes_only=True) == tuple(
        pack for pack in expected if pack in {"foundation", "first"}
    )
    assert select_packs(tmp_path, before="0" * 40) == (
        "first",
        "foundation",
        "image",
        "second",
        "unlisted",
    )
    assert select_packs(tmp_path, tag="pack/unlisted/v4.2.1") == ("unlisted",)
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


def test_default_admission_separates_expensive_children_without_changing_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, official_payloads: tuple[Path, ...]
) -> None:
    from dinkster_registry import Lockfile

    from scripts import check_default_pack_admission as admission

    catalog = Lockfile.from_record_json(official_payloads[-1].read_text())

    def registry_defaults(registry, minimums, artifacts, *, engine_version):
        for entry in catalog.packs:
            wheel = next(
                path
                for path in official_payloads
                if path.name == f"{entry.pack.replace('-', '_')}-{entry.version}-py3-none-any.whl"
            )
            shutil.copyfile(
                wheel, artifacts / f"{entry.artifact_digest.removeprefix('blake3:')}.zip"
            )
        return catalog, {}

    monkeypatch.setattr(admission, "prepare_registry_sources", registry_defaults)
    monkeypatch.setattr(admission.StaticPackIndex, "catalog", lambda *args, **kwargs: [])
    prepared = tmp_path / "prepared"
    result = admission.check_defaults(
        admission.NamedRegistry("", "https://registry.invalid"),
        engine_version="0.0.4",
        prepare_output=prepared,
    )
    groups = result["groups"]
    assert isinstance(groups, list)
    assert len(groups) == 13
    assert all(groups)
    assert sorted(name for group in groups for name in group) == sorted(
        entry.pack for entry in catalog.packs if entry.pack != "dinkster-pack-helpers"
    )
    assert ["dinkster-3d"] in groups
    assert ["dinkster-image"] in groups
    assert ["dinkster-video"] in groups
    assert ["dinkster-audio"] in groups
    assert Lockfile.from_record_json((prepared / "catalog.lock").read_text()) == catalog


@pytest.mark.parametrize(
    ("mode", "problem"),
    [
        ("candidate", None),
        ("candidate", "manifest"),
        ("candidate", "security"),
        ("prepared", None),
        ("prepared", "security"),
        ("prepared", "engine-version"),
        ("prepared", "incomplete-catalog"),
        ("prepared", "corrupt-artifact"),
        ("prepared", "unknown-pack"),
        ("locked-prepared", None),
        ("locked-prepared", "lock-base"),
        ("locked-prepared", "lock-artifact"),
        ("locked-prepared", "missing-lock"),
        ("locked-runtime", None),
        ("published", None),
        ("published", "promotion"),
        ("published", "timeout"),
        ("published", "digest"),
        ("published", "head"),
    ],
)
def test_default_admission_scans_unpublished_candidates_and_registry_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    problem: str | None,
) -> None:
    from dataclasses import replace

    from dinkster_registry import LockedPack, Lockfile, artifact_digest
    from dinkster_registry.archive_admission import StageResult
    from dinkster_store import pack_archive_path, pack_label
    from dinkster_workers import load_manifest
    from dinkster_workers.pack_lock import LockedDistribution, LockedWheel, PackLock
    from dinkster_workers.pack_wheel import wheel_filename

    from scripts import check_default_pack_admission as admission

    archives = {}
    entries = {}
    for name, version in (("candidate", "7.2"), ("unchanged", "1.3")):
        source = tmp_path / name
        source.mkdir()
        (source / "dinkster-pack.toml").write_text(
            f'[pack]\nname = "{name}"\npublisher = "dinkster"\n'
            f'version = "{version}"\nengine = ">=0.2,<1"\n'
            f'[pack.entry]\nnodes = "{name}:NODES"\n'
            '[pack.contracts]\nhost = "dinkster-pack-host/1"\napi = "dinkster-api/v1"\n'
        )
        if name == "candidate" and problem == "manifest":
            manifest = source / "dinkster-pack.toml"
            manifest.write_text(manifest.read_text().replace('version = "7.2"', 'version = "7.1"'))
        (source / name).mkdir()
        (source / name / "__init__.py").write_text("NODES = []\n")
        if name == "candidate" and problem == "security":
            (source / "bad.txt").write_text("dinkster-known-bad-test-payload")
        archive = tmp_path / wheel_filename(load_manifest(source / "dinkster-pack.toml"))
        build_pack_archive(source, archive)
        digest = artifact_digest(archive.read_bytes())
        archives[name] = archive.read_bytes()
        entries[name] = LockedPack(
            pack=name,
            version=version,
            artifact_digest=digest,
            publisher="dinkster",
            claims=(),
            source="git:Kosinkadink/Dinkster@" + "a" * 40 if name == "candidate" else "registry",
        )

    is_prepared = mode in {"prepared", "locked-prepared", "locked-runtime"}
    generation = tmp_path / "generation-python"
    generation.touch()
    constraints = tmp_path / "constraints.txt"
    constraints.touch()
    context = {
        "generation_python": generation,
        "base_id": "sha256:" + "f" * 64,
        "constraints": constraints,
    }
    if mode == "locked-runtime":
        from test_pack_environment import _generation, _wheel

        runtime = tmp_path / "runtime"
        wheelhouse = runtime / "wheelhouse"
        wheelhouse.mkdir(parents=True)
        _wheel(wheelhouse, "dinkster-api", "0.0.4.dev37", "dinkster_api", "")
        _wheel(wheelhouse, "locked-extra", "1.2+cpu", "locked_extra", "")
        python = _generation(tmp_path / "managed/python", [])
        constraints.write_text("dinkster-api==0.0.4.dev37\nlocked-extra==1.2+cpu\n")
        monkeypatch.setattr(
            admission,
            "read_runtime",
            lambda *args: (python, "sha256:" + "f" * 64, constraints, "linux-cpu"),
        )
        monkeypatch.setenv("UV_OFFLINE", "1")
        context.clear()
        context["runtime"] = runtime

    def dependency_lock(name):
        return PackLock(
            name,
            "linux-cpu",
            ">=3.12",
            (),
            (),
            (
                LockedDistribution(
                    name,
                    entries[name].version,
                    (
                        LockedWheel(
                            f"{name}-{entries[name].version}-py3-none-any.whl",
                            len(archives[name]),
                            hashlib.sha256(archives[name]).hexdigest(),
                            "https://registry.example/pack.whl",
                        ),
                    ),
                ),
            ),
        )

    def build_candidate(root, output, *, packs, **kwargs):
        assert packs == ("candidate",)
        output.mkdir()
        (output / "candidate-7.2-py3-none-any.whl").write_bytes(archives["candidate"])
        (output / "official-packs.lock").write_text(
            Lockfile.of([entries["candidate"]]).record_json()
        )
        lock = dependency_lock("candidate")
        if mode == "locked-runtime":
            from dinkster_workers.pack_lock import resolve_pack_lock

            resolved = resolve_pack_lock(
                pack="candidate",
                variant=kwargs["variant"],
                requirements=("dinkster-api>=0.0.2.dev0,<0.1", "locked-extra==1.2+cpu"),
                python=kwargs["python"],
                constraints=kwargs["constraints"],
                indexes=kwargs.get("indexes", ()),
                find_links=kwargs.get("find_links", ()),
                engine=">=0.0.2.dev0,<0.1",
            )
            lock = replace(resolved, packages=(*resolved.packages, *lock.packages))
        lock.write(output / "candidate-7.2-linux-cpu.pack-lock.json")

    def registry_defaults(registry, minimums, artifacts, *, engine_version):
        # The candidate has no registry release: requesting it would fail here.
        assert minimums == {"unchanged": "1.0"}
        entry = entries["unchanged"]
        archive = pack_archive_path(
            artifacts, entry.artifact_digest, label=pack_label(entry.pack, entry.version)
        )
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(archives["unchanged"])
        return Lockfile.of([entry]), {}

    observed = []

    def engine_tests(root, catalog, artifacts, *, generation_python, locks, variant, test_packs=()):
        assert root == Path(admission.__file__).resolve().parents[1]
        assert test_packs == (("candidate",) if is_prepared else ())
        assert generation_python.is_file()
        assert set(locks) == {pack.pack for pack in catalog.packs}
        assert variant == "linux-cpu"
        observed.extend((pack.pack, pack.version) for pack in catalog.packs)

    monkeypatch.setattr(
        admission, "default_pack_minimums", lambda: {"candidate": "1.0", "unchanged": "1.0"}
    )
    monkeypatch.setattr(admission, "select_packs", lambda *args, **kwargs: ("candidate",))
    monkeypatch.setattr(admission, "build", build_candidate)
    monkeypatch.setattr(admission, "prepare_registry_sources", registry_defaults)
    monkeypatch.setattr(admission, "run_executable_probe", lambda root: StageResult())
    monkeypatch.setattr(
        admission,
        "read_registry_pack_locks",
        lambda config, catalog, **kwargs: tuple(dependency_lock(p.pack) for p in catalog.packs),
    )
    monkeypatch.setattr(admission, "run_locked_probe", lambda root, lock, **kwargs: StageResult())
    monkeypatch.setattr(admission, "run_selected_tests", engine_tests)
    prepared = None
    if is_prepared:
        prepared = tmp_path / "prepared"
        output = admission.check_defaults(
            admission.NamedRegistry("", "https://registry.example"),
            engine_version="0.2",
            candidate_base="b" * 40,
            prepare_output=prepared,
            **context,
        )
        groups = output["groups"]
        assert isinstance(groups, list)
        if mode == "locked-runtime":
            lock = PackLock.read(prepared / "pack-locks/candidate.json")
            assert [(item.name, item.version) for item in lock.packages] == [
                ("locked-extra", "1.2+cpu"),
                ("candidate", "7.2"),
            ]
        assert sorted(name for group in groups for name in group) == [
            "candidate",
            "unchanged",
        ]
        assert not observed
        import yaml

        workflow = yaml.safe_load(
            (
                Path(admission.__file__).resolve().parents[1]
                / ".github/workflows/default-pack-admission.yml"
            ).read_text()
        )
        upload = next(
            step["with"]
            for step in workflow["jobs"]["prepare"]["steps"]
            if step.get("uses") == "actions/upload-artifact@v4"
        )
        transferred = tmp_path / "transferred"
        shutil.copytree(
            prepared,
            transferred,
            ignore=None
            if upload.get("include-hidden-files", False)
            else shutil.ignore_patterns(".*"),
        )
        prepared = transferred
        if problem == "engine-version":
            metadata = json.loads((prepared / "engine.json").read_text())
            metadata["engineVersion"] = "0.3"
            (prepared / "engine.json").write_text(json.dumps(metadata))
        elif problem == "incomplete-catalog":
            (prepared / "catalog.lock").write_text(
                Lockfile.of([entries["candidate"]]).record_json()
            )
        elif problem == "corrupt-artifact":
            digest = entries["candidate"].artifact_digest.removeprefix("blake3:")
            with zipfile.ZipFile(prepared / "artifacts" / f"{digest}.zip", "a") as wheel:
                wheel.writestr("changed.txt", "digest drift")
        elif problem == "lock-base":
            metadata = json.loads((prepared / "engine.json").read_text())
            metadata["baseId"] = "sha256:" + "e" * 64
            (prepared / "engine.json").write_text(json.dumps(metadata))
        elif problem == "lock-artifact":
            dependency_lock("unchanged").write(prepared / "pack-locks/candidate.json")
        elif problem == "missing-lock":
            (prepared / "pack-locks/candidate.json").unlink()
    publication = None
    if mode == "published":
        publication = tmp_path / "official-packs.lock"
        publication.write_text(Lockfile.of([entries["candidate"]]).record_json())
        monkeypatch.setattr(
            admission.subprocess,
            "check_output",
            lambda *args, **kwargs: "c" * 40 if problem == "head" else "a" * 40,
        )
        attempts = []

        def release(registry, pack, version, *, engine_version):
            assert (pack, version, engine_version) == ("candidate", "7.2", "0.2")
            attempts.append(pack)
            if problem == "timeout" or problem == "promotion" and len(attempts) == 1:
                raise admission.InstallError("not active yet")
            return replace(
                entries["candidate"],
                source="registry",
                artifact_digest=entries["unchanged"].artifact_digest
                if problem == "digest"
                else entries["candidate"].artifact_digest,
            )

        monkeypatch.setattr(admission, "resolve_release", release)
        monkeypatch.setattr(
            admission, "http_registry_fetcher", lambda endpoint: lambda entry: archives[entry.pack]
        )
        ticks = iter((0, 601)) if problem == "timeout" else iter(range(100))
        monkeypatch.setattr(admission.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(admission.time, "sleep", lambda seconds: None)
    expected_error = {
        "manifest": "candidate@7.2 admission refused",
        "security": "candidate@7.2 admission refused",
        "digest": "digest mismatch",
        "head": "do not match the engine checkout",
        "timeout": "not active yet",
        "engine-version": "do not match the engine checkout and version",
        "incomplete-catalog": "prepared sources omit defaults",
        "corrupt-artifact": "candidate: locked outer wheel differs from the archive",
        "unknown-pack": "requested admission packs are not in the catalog",
        "lock-base": "do not match the engine checkout and version",
        "lock-artifact": "dependency lock identity differs",
        "missing-lock": "candidate.json",
    }.get(problem or "")
    if expected_error:
        with pytest.raises((ValueError, OSError, admission.InstallError), match=expected_error):
            admission.check_defaults(
                admission.NamedRegistry("", "https://registry.example"),
                engine_version="0.2",
                candidate_base="b" * 40 if mode == "candidate" else None,
                published_sources=publication,
                prepared_sources=prepared,
                test_packs=("unknown",)
                if problem == "unknown-pack"
                else ("candidate",)
                if is_prepared
                else (),
                **context,
            )
        assert not observed
    else:
        result = admission.check_defaults(
            admission.NamedRegistry("", "https://registry.example"),
            engine_version="0.2",
            candidate_base="b" * 40 if mode == "candidate" else None,
            published_sources=publication,
            prepared_sources=prepared,
            test_packs=("candidate",) if is_prepared else (),
            **context,
        )
        assert sorted(observed) == [("candidate", "7.2"), ("unchanged", "1.3")]
        assert result == {
            "engineVersion": "0.2",
            "packs": [
                {
                    "pack": name,
                    "version": entries[name].version,
                    "artifactDigest": entries[name].artifact_digest,
                }
                for name in (("candidate",) if is_prepared else ("candidate", "unchanged"))
            ],
        }
    if problem in {"manifest", "security"}:
        stage = "manifest-admission" if problem == "manifest" else "fast-security"
        timings = [
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if line.startswith('{"admissionTiming":')
        ]
        assert any(
            record["pack"] == "candidate@7.2"
            and record["stage"] == stage
            and record.get("status") == "failed"
            for record in timings
        )


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "prepare-only",
        "published-identical",
        "check-published",
        "dependency-active",
        "dependency-levels",
        "dependency-levels-timeout",
        "dependency-external",
        "dependency-rejected",
        "dependency-timeout",
        "submitted-progress",
        "submitted-rejected",
        "submitted-timeout",
        "submitted-digest",
        "published-digest",
        "published-size",
        "published-lock",
        "published-lock-size",
        "published-base",
        "published-coverage",
        "published-publisher",
        "lookup-failed",
        "publish-conflict",
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
    tmp_path: Path, problem: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import json

    from dinkster_registry import (
        ArtifactError,
        LockedPack,
        Lockfile,
        artifact_digest,
        build_artifact,
    )
    from dinkster_workers.pack_lock import LockedDistribution, LockedWheel, PackLock
    from test_installer import _serve_artifacts

    from dinkster.registries import NamedRegistry, RegistryPublishError
    from scripts.publish_official_packs import publish_official_packs

    sources = tmp_path / "sources"
    sources.mkdir()
    engine_manifest = tmp_path / "release-manifest.json"
    engine_manifest.write_text(
        json.dumps(
            {
                "format": "dinkster.engine/1",
                "repository": "Kosinkadink/Dinkster",
                "commit": "c" * 40,
                "version": "0.3.1",
            }
        )
    )
    packs = []
    versions = [("first-pack", "1.2.3"), ("second-pack", "7.4.0rc1")]
    chained = problem in {"dependency-levels", "dependency-levels-timeout"}
    if chained:
        versions.extend([("third-pack", "2.0"), ("fourth-pack", "3.0")])
    for name, version in versions:
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
            + f'[pack.entry]\nnodes = "{name.replace("-", "_")}:NODES"\n'
            + (
                '[pack.dependencies]\nthird-pack = ">=2,<3"\n'
                if name == "first-pack" and problem == "dependency-external"
                else '[pack.dependencies]\nsecond-pack = ">=7,<8"\n'
                if name == "first-pack" and problem and problem.startswith("dependency-")
                else '[pack.dependencies]\nthird-pack = ">=2,<3"\n'
                if name == "second-pack" and chained
                else ""
            )
        )
        module = tree / name.replace("-", "_")
        module.mkdir()
        (module / "__init__.py").write_text("raise RuntimeError('must not import')\n")
        archive = sources / f"{module.name}-{version}-py3-none-any.whl"
        if declaration:
            built = tmp_path / f"{module.name}-{observed}-py3-none-any.whl"
            build_pack_archive(tree, built)
            built.rename(archive)
        else:
            build_artifact(tree, archive)
        digest = artifact_digest(archive.read_bytes())
        packs.append(LockedPack(name, version, digest, "dinkster", ()))
    (sources / "official-packs.lock").write_text(Lockfile.of(packs).record_json())
    if problem == "archive":
        (sources / "second_pack-7.4.0rc1-py3-none-any.whl").write_bytes(b"changed")
    bases, supported, locks = [], {}, []
    expected_locks = {}
    variants = {
        "linux-cpu": ("linux", "3.12.9"),
        "linux-nvidia": ("linux", "3.12.9"),
        "win-cpu": ("windows", "3.14.10"),
        "win-nvidia": ("windows", "3.14.10"),
        "win-nvidia-cu126": ("windows", "3.14.10"),
        "mac-arm64": ("macos", "3.12.9"),
    }
    for variant, (system, python) in variants.items():
        base = {
            "format": "dinkster.base/1",
            "variant": variant,
            "platform": {"os": system, "arch": "arm64" if system == "macos" else "x86_64"},
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
    files = {
        f"/v1/publish/{pack.pack}/versions/{pack.version}": json.dumps(
            {"status": "queued", "candidateId": pack.pack}
        ).encode()
        for pack in packs
    }
    if problem == "dependency-external":
        files["/v1/packs/third-pack/versions?limit=100"] = json.dumps(
            {"items": [{"version": "9.0", "state": "active"}], "nextCursor": "next"}
        ).encode()
        files["/v1/packs/third-pack/versions?limit=100&cursor=next"] = json.dumps(
            {"items": [{"version": "2.3", "state": "active"}], "nextCursor": None}
        ).encode()
        files["/v1/packs/third-pack/versions/2.3"] = json.dumps(
            {"packName": "third-pack", "version": "2.3", "state": "active"}
        ).encode()
    if problem and (problem.startswith("published-") or problem == "check-published"):
        for pack in packs:
            record = {
                "packName": pack.pack,
                "version": pack.version,
                "publishedBy": {"principalId": "dinkster"},
                "artifact": {
                    "digest": pack.artifact_digest,
                    "size": (
                        sources / f"{pack.pack.replace('-', '_')}-{pack.version}-py3-none-any.whl"
                    )
                    .stat()
                    .st_size,
                },
                "installLocks": {},
            }
            for variant in variants:
                raw = json.dumps(
                    expected_locks[pack.pack, variant], sort_keys=True, separators=(",", ":")
                ).encode("ascii")
                record["installLocks"][variant] = {
                    "baseId": "sha256:" + supported[variant],
                    "artifactDigest": pack.artifact_digest,
                    "digest": "blake3:" + blake3(raw).hexdigest(),
                    "size": len(raw),
                    "url": "/ignored/download/url",
                }
            if pack.pack == "second-pack":
                if problem == "published-digest":
                    record["artifact"]["digest"] = "blake3:" + "f" * 64
                elif problem == "published-size":
                    record["artifact"]["size"] += 1
                elif problem == "published-lock":
                    record["installLocks"]["win-nvidia"]["digest"] = "blake3:" + "e" * 64
                elif problem == "published-lock-size":
                    record["installLocks"]["win-nvidia"]["size"] += 1
                elif problem == "published-base":
                    record["installLocks"]["win-nvidia"]["baseId"] = "sha256:" + "e" * 64
                elif problem == "published-coverage":
                    del record["installLocks"]["win-nvidia"]
                elif problem == "published-publisher":
                    record["publishedBy"]["principalId"] = "other"
            files[f"/v1/packs/{pack.pack}/versions/{pack.version}"] = json.dumps(record).encode()
    server = _serve_artifacts(files, requests)
    respond = vars(server.RequestHandlerClass)["respond"]
    submitted = []
    clock = [0.0]
    polls = [0]
    activation_polls: dict[str, int] = {}
    if problem and problem.startswith(("dependency-", "submitted-")):
        monkeypatch.setenv("PUBLICATION_TEST_TOKEN", "test-token")
        monkeypatch.setattr("scripts.publish_official_packs.time.monotonic", lambda: clock[0])

        def advance(seconds):
            clock[0] += seconds

        monkeypatch.setattr("scripts.publish_official_packs.time.sleep", advance)

    def registry_response(handler):
        if handler.command == "POST":
            submitted.append(handler.path)
            name, version = handler.path.split("/")[3::2]
            if not (problem and problem.startswith("submitted-")) and not (
                problem
                and problem.startswith("dependency-")
                and problem != "dependency-external"
                and name in {"second-pack", "third-pack"}
            ):
                pack = next(pack for pack in packs if pack.pack == name)
                files[f"/v1/packs/{name}/versions/{version}"] = json.dumps(
                    {
                        "packName": name,
                        "version": version,
                        "state": "active",
                        "artifact": {"digest": pack.artifact_digest},
                    }
                ).encode()
        if (
            problem
            and problem.startswith("submitted-")
            and handler.path.startswith("/v1/publish/")
            and handler.command == "GET"
        ):
            name = handler.path.rsplit("/", 1)[1]
            pack = next(pack for pack in packs if pack.pack == name)
            count = activation_polls.get(name, 0)
            activation_polls[name] = count + 1
            state = ("queued", "scanning", "retrying", "accepted")[min(count, 3)]
            if problem == "submitted-rejected" and name == "second-pack" and count >= 2:
                state = "rejected"
            files[handler.path] = json.dumps(
                {"status": state, "reason": "probe.timeout" if state == "rejected" else None}
            ).encode()
            if count >= 3 and state == "accepted" and problem != "submitted-timeout":
                files[f"/v1/packs/{name}/versions/{pack.version}"] = json.dumps(
                    {
                        "packName": name,
                        "version": pack.version,
                        "state": "active",
                        "artifact": {
                            "digest": "blake3:" + "f" * 64
                            if problem == "submitted-digest" and name == "second-pack"
                            else pack.artifact_digest
                        },
                    }
                ).encode()
        if problem and problem.startswith("dependency-") and submitted:
            if handler.path in {"/v1/publish/second-pack", "/v1/publish/third-pack"}:
                assert handler.headers.get("Authorization") == "Bearer test-token"
                polls[0] += 1
                activation_polls[handler.path] = activation_polls.get(handler.path, 0) + 1
                files[handler.path] = json.dumps(
                    {
                        "status": "rejected" if problem == "dependency-rejected" else "scanning",
                        "reason": "scanner refused" if problem == "dependency-rejected" else None,
                        "findings": [{"code": "manifest.invalid", "message": "missing dependency"}],
                    }
                ).encode()
                if problem == "dependency-active" or (
                    chained
                    and activation_polls[handler.path] == 2
                    and not (
                        problem == "dependency-levels-timeout"
                        and handler.path == "/v1/publish/second-pack"
                    )
                ):
                    name = handler.path.rsplit("/", 1)[1]
                    version = dict(versions)[name]
                    files[f"/v1/packs/{name}/versions/{version}"] = json.dumps(
                        {
                            "packName": name,
                            "version": version,
                            "state": "active",
                            "artifact": {
                                "digest": next(
                                    pack.artifact_digest for pack in packs if pack.pack == name
                                )
                            },
                        }
                    ).encode()
            if handler.command == "POST" and "first-pack" in handler.path:
                assert polls[0] == (
                    4
                    if problem == "dependency-levels"
                    else 0
                    if problem == "dependency-external"
                    else 1
                )
        if handler.path.startswith("/v1/packs/"):
            assert handler.headers.get("Authorization") is None
        if (
            problem == "lookup-failed"
            or problem == "publish-conflict"
            and handler.command == "POST"
        ):
            requests.append((handler.path, handler.headers.get("Authorization")))
            handler.send_response(503 if problem == "lookup-failed" else 409)
            handler.end_headers()
            handler.wfile.write(b"release already has different archive or lock bindings")
        else:
            respond(handler)

    monkeypatch.setattr(server.RequestHandlerClass, "respond", registry_response)
    engine_bindings = []
    if problem is None:
        from email import policy
        from email.parser import BytesParser

        def receive(handler):
            body = handler.rfile.read(int(handler.headers["Content-Length"]))
            message = BytesParser(policy=policy.default).parsebytes(
                f"Content-Type: {handler.headers['Content-Type']}\r\n\r\n".encode() + body
            )
            parts = list(message.iter_parts())
            payload = parts[2].get_payload(decode=True)
            assert isinstance(payload, bytes)
            engine_bindings.append(json.loads(payload))
            handler.respond()

        monkeypatch.setattr(server.RequestHandlerClass, "do_POST", receive)
    output = tmp_path / "publication"
    before = {path: path.read_bytes() for path in [*sources.iterdir(), *locks, *bases]}
    try:
        registry = NamedRegistry(
            "", f"http://127.0.0.1:{server.server_port}", token_env="PUBLICATION_TEST_TOKEN"
        )
        lookup_paths = [f"/v1/packs/{pack.pack}/versions/{pack.version}" for pack in packs]
        if problem and problem.startswith("submitted-"):
            if problem == "submitted-progress":
                result = publish_official_packs(
                    tmp_path,
                    sources,
                    bases,
                    locks,
                    output,
                    registry=registry,
                    activation_timeout=20,
                )
                assert [row["state"] for row in result] == ["published", "published"]
                assert set(activation_polls) == {"first-pack", "second-pack"}
            else:
                with pytest.raises(RegistryPublishError) as failure:
                    publish_official_packs(
                        tmp_path,
                        sources,
                        bases,
                        locks,
                        output,
                        registry=registry,
                        activation_timeout=20,
                    )
                message = str(failure.value)
                assert "second-pack@7.4.0rc1" in message
                assert "candidate second-pack" in message
                if problem == "submitted-rejected":
                    assert "probe.timeout" in message
                elif problem == "submitted-timeout":
                    assert "first-pack@1.2.3" in message
                    assert "candidate first-pack" in message
                    assert "timed out" in message
                else:
                    assert "digest" in message
        elif problem and problem.startswith("dependency-"):
            if problem == "dependency-external":
                result = publish_official_packs(
                    tmp_path, sources, bases, locks, output, registry=registry, activation_timeout=5
                )
                assert submitted == [
                    "/v1/publish/first-pack/versions/1.2.3",
                    "/v1/publish/second-pack/versions/7.4.0rc1",
                ]
                assert [path for path, _ in requests][2:5] == [
                    "/v1/packs/third-pack/versions?limit=100",
                    "/v1/packs/third-pack/versions?limit=100&cursor=next",
                    "/v1/packs/third-pack/versions/2.3",
                ]
            elif problem in {"dependency-active", "dependency-levels"}:
                result = publish_official_packs(
                    tmp_path, sources, bases, locks, output, registry=registry, activation_timeout=5
                )
                assert [row["pack"] for row in result] == (
                    ["fourth-pack", "third-pack", "second-pack", "first-pack"]
                    if problem == "dependency-levels"
                    else ["second-pack", "first-pack"]
                )
                assert submitted == [
                    *(
                        [
                            "/v1/publish/fourth-pack/versions/3.0",
                            "/v1/publish/third-pack/versions/2.0",
                        ]
                        if problem == "dependency-levels"
                        else []
                    ),
                    "/v1/publish/second-pack/versions/7.4.0rc1",
                    "/v1/publish/first-pack/versions/1.2.3",
                ]
                if chained:
                    assert clock[0] == 8
            else:
                with pytest.raises(
                    RegistryPublishError,
                    match=(
                        "second-pack@7.4.0rc1.*manifest.invalid.*missing dependency"
                        if problem == "dependency-rejected"
                        else "second-pack@7.4.0rc1.*timed out"
                    ),
                ):
                    publish_official_packs(
                        tmp_path,
                        sources,
                        bases,
                        locks,
                        output,
                        registry=registry,
                        activation_timeout=5,
                    )
                assert submitted == [
                    *(
                        [
                            "/v1/publish/fourth-pack/versions/3.0",
                            "/v1/publish/third-pack/versions/2.0",
                        ]
                        if chained
                        else []
                    ),
                    "/v1/publish/second-pack/versions/7.4.0rc1",
                ]
                if chained:
                    assert clock[0] == 9
                else:
                    assert clock[0] <= 5
        elif problem in {"lookup-failed", "publish-conflict"}:
            with pytest.raises(
                RegistryPublishError,
                match="first-pack@1.2.3.*"
                + (
                    "lookup failed"
                    if problem == "lookup-failed"
                    else "not retried.*bump the pack version"
                ),
            ):
                publish_official_packs(tmp_path, sources, bases, locks, output, registry=registry)
            assert [path for path, _ in requests] == (
                lookup_paths[:1]
                if problem == "lookup-failed"
                else lookup_paths + ["/v1/publish/first-pack/versions/1.2.3"]
            )
        elif problem and problem.startswith("published-") and problem != "published-identical":
            with pytest.raises(
                RegistryPublishError, match="second-pack@7.4.0rc1.*bump the pack version"
            ):
                publish_official_packs(tmp_path, sources, bases, locks, output, registry=registry)
            assert [path for path, _ in requests] == lookup_paths
        elif problem not in {None, "prepare-only", "published-identical", "check-published"}:
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
                registry=registry if problem in {None, "published-identical"} else None,
                preflight_registry=registry.endpoint if problem == "check-published" else None,
                engine_manifest=engine_manifest if problem is None else None,
            )
            assert len(result) == 2
            assert [row["version"] for row in result] == ["1.2.3", "7.4.0rc1"]
            for pack, row in zip(packs, result, strict=True):
                lock_map = row["locks"]
                assert isinstance(lock_map, str)
                bindings = json.loads(Path(lock_map).read_bytes())
                assert set(bindings) == set(variants)
                for variant, identity in supported.items():
                    assert bindings[variant] == {
                        "baseId": "sha256:" + identity,
                        "artifactDigest": pack.artifact_digest,
                        "lock": expected_locks[pack.pack, variant],
                    }
                if problem is None:
                    assert row["state"] == "published" and row["candidateId"] == pack.pack
                elif problem == "published-identical":
                    assert row["state"] == "published" and "candidateId" not in row
            assert [path for path, _ in requests] == (
                lookup_paths
                + [f"/v1/publish/{pack.pack}/versions/{pack.version}" for pack in packs]
                + lookup_paths
                if problem is None
                else lookup_paths
                if problem in {"check-published", "published-identical"}
                else []
            )
            if problem is None:
                assert (
                    engine_bindings
                    == [
                        {
                            "commit": "c" * 40,
                            "manifestDigest": "sha256:"
                            + hashlib.sha256(engine_manifest.read_bytes()).hexdigest(),
                        }
                    ]
                    * 2
                )
        assert {path: path.read_bytes() for path in before} == before
        assert not list(tmp_path.glob("official-publication-*"))
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize(
    ("mode", "state"),
    [
        ("raw", "queued"),
        ("locks", "queued"),
        ("locks", "scanning"),
        ("locks", "retrying"),
        ("locks", "accepted"),
        ("engine", "queued"),
        ("missing", "queued"),
        ("refused", "queued"),
        ("invalid", "rejected"),
        ("invalid", "review"),
        ("invalid", "promoting"),
        ("invalid", "unknown"),
        ("invalid", None),
        ("invalid", ["queued"]),
    ],
)
def test_publication_client_preserves_archive_and_lock_bytes(
    tmp_path: Path, mode: str, state: object
) -> None:
    import json
    import threading
    from email import policy
    from email.parser import BytesParser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from dinkster.registries import NamedRegistry, RegistryPublishError, publish_release

    archive, locks = tmp_path / "demo-1.2.3-py3-none-any.whl", tmp_path / "locks.json"
    archive.write_bytes(b"exact archive\x00\xff\r\n")
    lock_bytes = b'{"win-cpu":{"baseId":"exact binding"}}\n'
    engine = {"commit": "a" * 40, "manifestDigest": "sha256:" + "b" * 64}
    if mode != "missing":
        locks.write_bytes(lock_bytes)
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(
                (
                    self.path,
                    self.headers["Content-Type"],
                    self.rfile.read(int(self.headers["Content-Length"])),
                )
            )
            self.send_response(400 if mode == "refused" else 202)
            self.end_headers()
            self.wfile.write(
                b'{"error":{"message":"lock binding refused"}}'
                if mode == "refused"
                else json.dumps({"status": state, "candidateId": "candidate-1"}).encode()
            )

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    registry = NamedRegistry("", f"http://127.0.0.1:{server.server_port}")
    try:
        if mode in {"missing", "refused", "invalid"}:
            with pytest.raises(
                RegistryPublishError,
                match={
                    "missing": "cannot read publication input",
                    "refused": "lock binding refused",
                    "invalid": "malformed receipt",
                }[mode],
            ):
                publish_release(registry, archive, "demo", "1.2.3", locks=locks)
        else:
            result = publish_release(
                registry,
                archive,
                "demo",
                "1.2.3",
                locks=locks if mode in {"locks", "engine"} else None,
                engine=engine if mode == "engine" else None,
            )
            assert result.state == state and result.candidate_id == "candidate-1"
        if mode == "missing":
            assert requests == []
            return
        assert len(requests) == 1
        path, content_type, body = requests[0]
        assert path == "/v1/publish/demo/versions/1.2.3"
        if mode == "raw":
            assert content_type == "application/octet-stream" and body == archive.read_bytes()
        else:
            message = BytesParser(policy=policy.default).parsebytes(
                f"Content-Type: {content_type}\r\n\r\n".encode() + body
            )
            parts = list(message.iter_parts())
            assert [part.get_param("name", header="content-disposition") for part in parts] == [
                "archive",
                "locks",
            ] + (["engine"] if mode == "engine" else [])
            assert parts[0].get_payload(decode=True) == archive.read_bytes()
            assert parts[1].get_payload(decode=True) == lock_bytes
            if mode == "engine":
                payload = parts[2].get_payload(decode=True)
                assert isinstance(payload, bytes)
                assert json.loads(payload) == engine
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_official_publication_declarations_match_archive_admission(
    tmp_path: Path, official_payloads: tuple[Path, ...]
) -> None:
    from dinkster_registry import Lockfile, unpack_artifact
    from dinkster_registry.archive_admission import run_ingest_integrity, run_manifest_admission

    from scripts.build_official_packs import library_metadata

    first = official_payloads
    catalog = Lockfile.from_record_json(first[-1].read_text())
    assert len(catalog.packs) == 37
    for pack in catalog.packs:
        package_name = pack.pack.replace("-", "_")
        archive = first[-1].parent / f"{package_name}-{pack.version}-py3-none-any.whl"
        data = archive.read_bytes()
        if pack.pack == "dinkster-pack-helpers":
            assert library_metadata(archive) == (
                "dinkster-pack-helpers",
                "0.0.4",
                (
                    "dinkster-api<0.1,>=0.0.2.dev0",
                    "dinkster-inference",
                    "numpy>=1.26",
                    "pillow>=10",
                    "torch",
                ),
            )
            assert f"blake3:{blake3(data).hexdigest()}" == pack.artifact_digest
            continue
        assert not run_ingest_integrity(data, len(data), pack.artifact_digest).blocked
        source = (
            unpack_artifact(archive, tmp_path / "sources" / pack.pack, pack.artifact_digest)
            / package_name
        )
        accepted = run_manifest_admission(
            source, pack.pack, pack.version, publisher="dinkster", engine_version="0.0.4.dev123"
        )
        assert not accepted.blocked, accepted.findings
        for engine in ("0.0.1", "0.1"):
            refused = run_manifest_admission(
                source, pack.pack, pack.version, publisher="dinkster", engine_version=engine
            )
            assert refused.blocked
            assert refused.findings[0].code == "manifest.invalid"


def test_official_sources_prepare_catalogs_and_execute_existing_node_ids(
    tmp_path: Path, official_payloads: tuple[Path, ...]
) -> None:
    import json
    from dataclasses import replace

    from dinkster_graph import graph_from_wire
    from dinkster_registry import Lockfile, artifact_digest, unpack_artifact
    from dinkster_workers import load_manifest
    from dinkster_workers.catalog import read_catalog
    from dinkster_workers.doctor import prepare_catalog
    from dinkster_workers.pack_wheel import pack_manifest_path

    from dinkster.compose import CompositionError, PackSpec, ServingComposer
    from dinkster.packs import pack_info_from_manifest

    saved = tmp_path / "workflow.json"
    original = (
        b'{"nodes":{"length":{"nodeType":"dinkster.text.string_length",'
        b'"inputs":{"string":"source packs"}},'
        b'"image":{"nodeType":"dinkster.image.empty_image",'
        b'"inputs":{"width":4,"height":5,"batch_size":2}},'
        b'"size":{"nodeType":"dinkster.image.get_image_size",'
        b'"inputs":{"image":{"$link":{"node":"image","output":"IMAGE"}}}}}}'
    )
    saved.write_bytes(original)
    payloads = official_payloads
    lock = Lockfile.from_record_json(payloads[-1].read_text())
    specs = []
    for entry in lock.packs:
        if entry.pack == "dinkster-pack-helpers":
            assert entry.claims == ()
            continue
        filename = f"{entry.pack.replace('-', '_')}-{entry.version}-py3-none-any.whl"
        source = unpack_artifact(
            payloads[-1].parent / filename,
            tmp_path / "sources" / entry.pack,
            entry.artifact_digest,
        )
        manifest = load_manifest(pack_manifest_path(source))
        assert read_catalog(manifest) is None
        report = prepare_catalog(manifest.path, environment={"PYTHONPATH": str(source)})
        assert report.ok, report
        assert read_catalog(manifest) is not None
        repacked = tmp_path / "repacked" / filename
        build_pack_archive(manifest.root, repacked)
        assert artifact_digest(repacked.read_bytes()) == entry.artifact_digest
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
            assert composition.node_packs["dinkster.text.string_length"] == "dinkster-text"
            assert composition.node_packs["dinkster.image.empty_image"] == "dinkster-image"
            assert composition.packs["dinkster-text"].version == "0.0.3"
            assert composition.packs["dinkster-image"].version == "0.0.9"
            result = await composition.make_engine(lambda _event: None).run(
                graph_from_wire(json.loads(saved.read_bytes())), ["length", "size"]
            )
            assert result.diagnostics == ()
            assert result.outputs["length"]["length"].resolve() == 12
            assert result.outputs["size"]["width"].resolve() == 4
            assert result.outputs["size"]["height"].resolve() == 5
            assert result.outputs["size"]["batch_size"].resolve() == 2
            assert saved.read_bytes() == original
            source = tmp_path / "sources" / "dinkster-text" / "dinkster_text"
            (source / "changed.txt").write_text("source drift")
            with pytest.raises(CompositionError, match="missing or stale"):
                composer.validate_catalogs(ordered)
        finally:
            await composer.close()

    run_scenario(execute())


@pytest.mark.parametrize("problem", [None, "path", "case", "compression", "order", "junk"])
def test_registry_archive_admission_requires_verified_wheel(
    tmp_path: Path, problem: str | None
) -> None:
    from dinkster_registry import artifact_digest
    from dinkster_registry.archive_admission import extracted_archive, run_ingest_integrity

    root = Path(__file__).parent / "fixtures/registry-admission/color-tools"
    wheel = tmp_path / "color_tools-1.2.3-py3-none-any.whl"
    build_pack_archive(root, wheel)
    with zipfile.ZipFile(wheel) as archive:
        files = [(name, archive.read(name)) for name in archive.namelist()]
    if problem == "path":
        files.append(("../escape.py", b"escape"))
    elif problem == "case":
        files.append(("color_tools/__INIT__.py", b"collision"))
    elif problem == "order":
        files.reverse()
    elif problem == "junk":
        files.append(("build/output.py", b"ignored"))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, payload in files:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o644 << 16
            if problem == "compression":
                info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, payload)
    data = output.getvalue()
    digest = artifact_digest(data)
    result = run_ingest_integrity(data, len(data), digest)
    assert result.blocked is (problem in {"path", "case", "junk"})
    if problem in {None, "compression", "order"}:
        assert result.data == {"artifactDigest": digest, "size": len(data)}
        with extracted_archive(data) as extracted:
            assert (extracted / "__init__.py").read_bytes() == (
                root / "color_tools/__init__.py"
            ).read_bytes()
        assert not extracted.exists()
        assert run_ingest_integrity(data, len(data) + 1, digest).findings[0].code == (
            "integrity.size-mismatch"
        )
        assert run_ingest_integrity(data, len(data), artifact_digest(b"other")).findings[
            0
        ].code == ("integrity.digest-mismatch")
    else:
        assert result.findings[0].code == "integrity.invalid-archive"


@pytest.mark.parametrize(
    ("filename", "payload", "code", "effect"),
    [
        ("clean.py", b"NODES = []", None, None),
        ("secret.txt", b"-----BEGIN PRIVATE KEY-----", "security.confirmed-secret", "block"),
        ("bad.txt", b"dinkster-known-bad-test-payload", "security.known-bad-file", "block"),
        ("native.bin", b"MZbinary", "security.native-binary", "review"),
        ("nested.bin", b"\x1f\x8bgzip", "security.nested-archive", "review"),
        ("nodes.pyc", b"bytecode", "security.obfuscated-code", "review"),
    ],
)
def test_registry_archive_security_preserves_block_and_review(
    tmp_path: Path, filename: str, payload: bytes, code: str | None, effect: str | None
) -> None:
    from dinkster_registry.archive_admission import run_fast_security

    root = tmp_path / "pack"
    shutil.copytree(Path(__file__).parent / "fixtures/registry-admission/color-tools", root)
    (root / "color_tools" / filename).write_bytes(payload)
    archive = tmp_path / "color_tools-1.2.3-py3-none-any.whl"
    build_pack_archive(root, archive)
    result = run_fast_security(archive.read_bytes())
    assert [(finding.code, finding.effect) for finding in result.findings] == (
        [] if code is None else [(code, effect)]
    )
    assert result.blocked is (effect == "block")
    assert result.review is (effect == "review")
    assert result.data == {
        "status": "failed" if effect == "block" else "warning" if effect else "passed",
        "policyVersion": "registry-scan/1",
    }


@pytest.mark.parametrize("engine", ["0.8.9", "0.9.0", "0.9.7", "1.0.0"])
def test_registry_manifest_admission_preserves_engine_range(tmp_path: Path, engine: str) -> None:
    from dinkster_registry.archive_admission import run_manifest_admission

    root = write_pack(tmp_path / "pack")
    manifest = root / "dinkster-pack.toml"
    manifest.write_text(
        MANIFEST.replace('version = "1.0"', 'version = "2.3.0"')
        + '\n[pack.contracts]\nhost = "dinkster-pack-host/1"\napi = "dinkster-api/v1"\n'
    )
    result = run_manifest_admission(root, "archive-test", "2.3.0", engine_version=engine)
    assert result.blocked is (engine in {"0.8.9", "1.0.0"})
    if result.blocked:
        message = result.findings[0].message
        assert "archive-test@2.3.0" in message and ">=0.9,<1" in message and engine in message
    else:
        assert result.data is not None and result.data["dinkster"] == ">=0.9,<1"
    mismatch = run_manifest_admission(root, "archive-test", "2.3.1")
    assert mismatch.blocked and "does not match submission" in mismatch.findings[0].message


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        ('license = "LicenseRef-Private"', "requires a root LICENSE"),
        ('license = "not-a-license"', "Unknown license"),
        ('repository = "http://example.org/pack"', "must be an HTTPS URL"),
        ('dinkster = "nonsense"', "Invalid specifier"),
    ],
)
def test_registry_manifest_admission_refuses_invalid_metadata(
    tmp_path: Path, declaration: str, message: str
) -> None:
    from dinkster_registry.archive_admission import run_manifest_admission

    root = write_pack(tmp_path / "pack")
    (root / "dinkster-pack.toml").write_text(
        MANIFEST.replace('version = "1.0"\nengine = ">=0.9,<1"\n', "").replace(
            "[pack]\n", f"[pack]\n{declaration}\n"
        )
        + '\n[pack.contracts]\nhost = "dinkster-pack-host/1"\napi = "dinkster-api/v1"\n'
    )
    result = run_manifest_admission(root, "archive-test", "1.0.0")
    assert result.blocked and result.findings[0].code == "manifest.invalid"
    assert message in result.findings[0].message


@pytest.mark.skipif(sys.platform == "win32", reason="the admission probe requires POSIX limits")
@pytest.mark.parametrize("problem", ["empty", "remote", "exception", "output", "timeout"])
def test_registry_executable_probe_bounds_and_sanitizes(tmp_path: Path, problem: str) -> None:
    from dinkster_registry.archive_admission import ProbeSettings, run_executable_probe

    root = write_pack(tmp_path / "pack")
    if problem == "remote":
        manifest = root / "dinkster-pack.toml"
        manifest.write_text(
            MANIFEST.replace("[pack]\n", '[pack]\nnode-source = "remote-gateway-catalog"\n')
        )
    elif problem == "exception":
        (root / "archive_test/__init__.py").write_text(
            'raise ValueError("private operator detail")\n'
        )
    elif problem == "output":
        (root / "archive_test/__init__.py").write_text(
            'import os\nos.write(1, b"x" * 100000)\nNODES = []\n'
        )
    elif problem == "timeout":
        (root / "archive_test/__init__.py").write_text("import time\ntime.sleep(2)\nNODES = []\n")
    result = run_executable_probe(
        root,
        settings=ProbeSettings(
            probe_timeout_seconds=1 if problem == "timeout" else 30,
            probe_output_limit_bytes=4096,
            probe_operator_tail_bytes=512,
        ),
        require_isolation=os.environ.get("DINKSTER_TEST_FULL_PROBE") == "1",
    )
    codes = {finding.code for finding in result.findings}
    if problem == "remote":
        assert not result.blocked and result.data == {"schemas": []}
    else:
        assert result.blocked
        expected = {
            "empty": "probe.no-schemas",
            "exception": "entry.unresolvable",
            "output": "probe.output-limit",
            "timeout": "probe.timeout",
        }[problem]
        assert expected in codes
    assert all("private operator detail" not in finding.message for finding in result.findings)
    assert result.operator_details is not None
    assert len(str(result.operator_details["stdoutTail"]).encode()) <= 512


def test_registry_probe_refuses_degraded_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_registry.archive_admission import run_executable_probe
    from dinkster_workers.sandbox import ProbeJail

    root = write_pack(tmp_path / "pack")
    monkeypatch.setattr(
        "dinkster_registry.archive_admission.detect_probe_jail",
        lambda: ProbeJail(bwrap="/usr/bin/true", user_namespaces=False),
    )
    result = run_executable_probe(root)
    assert result.blocked and result.findings[0].code == "probe.isolation-unavailable"
    assert result.operator_details == {"jail": "degraded"}


@pytest.mark.skipif(sys.platform != "linux", reason="engine pack admission runs on Linux")
@pytest.mark.parametrize(
    "problem",
    [
        None,
        "broken",
        "missing",
        "skipped",
        "unpublished-only",
        "mixed-class",
        "shadowed",
        "autouse",
        "xunit",
        "class-fixture",
        "parameter-shadow",
        "aliased-fixture",
        "unittest",
        "class-shadow",
        "collection",
        "dependency",
        "teardown",
        "library-lock-mismatch",
    ],
)
def test_selected_pack_tests_cannot_fall_back_to_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    problem: str | None,
    capfd: pytest.CaptureFixture[str],
    selected_pack_runtime,
) -> None:
    from dinkster_registry import LockedPack, Lockfile, artifact_digest
    from dinkster_workers.pack_lock import LockedDistribution
    from test_pack_environment import _wheel

    from scripts.run_selected_pack_tests import run_selected_tests

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "inherited-device")
    monkeypatch.setenv("DINKSTER_ACCELERATOR", "cuda")
    monkeypatch.setenv("DINKSTER_TEST_HISTORY_ROOT", str(tmp_path / "wrong-history"))
    engine = tmp_path / "engine"
    tests = engine / "tests"
    tests.mkdir(parents=True)
    (engine / "pyproject.toml").write_text("")
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "src/dinkster",
        engine / "src/dinkster",
        ignore=shutil.ignore_patterns("__pycache__", "official_packs"),
    )
    checkout_module = engine / "packages/dinkster-nodes-image/src/dinkster_image"
    checkout_module.mkdir(parents=True)
    (checkout_module.parent.parent / "pyproject.toml").write_text(
        '[project]\nname = "dinkster-image"\n'
    )
    (checkout_module / "__init__.py").write_text("VALUE = 47\n")
    dev_module = engine / "packages/dinkster-nodes-dev/src/dinkster_nodes_dev"
    dev_module.mkdir(parents=True)
    (dev_module / "__init__.py").write_text("VALUE = 73\n")
    (tests / "test_image_nodes.py").write_text(
        "import importlib.metadata\nimport os\nimport subprocess\nimport sys\nimport pytest\n"
        "import tomllib\n"
        "from pathlib import Path\nimport dinkster_image\nimport dinkster_nodes_dev\n"
        "from dinkster_image import VALUE, NEW_NODE as Unpublished\n"
        + ("pytest.skip('unavailable', allow_module_level=True)\n" if problem == "skipped" else "")
        + (
            "def selected_value():\n"
            if problem
            in {
                "unpublished-only",
                "mixed-class",
                "shadowed",
                "class-fixture",
                "aliased-fixture",
                "xunit",
                "unittest",
            }
            else "def test_selected_value():\n"
        )
        + "    assert VALUE == 47\n"
        "    assert dinkster_nodes_dev.VALUE == 73\n"
        "    assert Path(dinkster_nodes_dev.__file__).resolve().is_relative_to(\n"
        "        Path.cwd() / 'packages/dinkster-nodes-dev/src')\n"
        "    from locked_only_dependency import VALUE as LOCKED_VALUE\n"
        "    assert LOCKED_VALUE == 61\n"
        "    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''\n"
        "    assert os.environ['DINKSTER_ACCELERATOR'] == 'cpu'\n"
        "    history = Path(os.environ['DINKSTER_TEST_HISTORY_ROOT'])\n"
        "    assert not (Path.cwd() / '.git').exists()\n"
        "    assert str(history) not in sys.path\n"
        "    assert subprocess.check_output(['git', 'show', ':pyproject.toml'], "
        "cwd=history) == b''\n"
        "    pack_root = Path(__file__).parents[1] / 'packages/dinkster-nodes-image'\n"
        "    assert tomllib.loads((pack_root / 'pyproject.toml').read_text()) "
        "== {'project': {'name': 'dinkster-image'}}\n"
        "    assert (pack_root / 'locales/en.json').read_text() == '{}'\n"
        "    assert not (pack_root / 'locales').is_symlink()\n"
        "    subprocess.run([sys.executable, '-c', 'from types import MappingProxyType'], "
        "cwd=pack_root, check=True)\n"
        "    from dinkster.compose import default_pack_spec\n"
        "    info = default_pack_spec('dinkster-image').packs['dinkster-image']\n"
        "    assert info.version == '1.2.3'\n"
        "    assert info.source == f'local:{pack_root}'\n"
        "    assert Path(dinkster_image.__file__).resolve()"
        ".is_relative_to(pack_root / 'src')\n"
        "    distribution = importlib.metadata.distribution('dinkster-image')\n"
        "    assert distribution.version == '1.2.3'\n"
        "    assert Path(distribution.locate_file('dinkster_image/__init__.py')).resolve() "
        "== Path(dinkster_image.__file__).resolve()\n"
        "    subprocess.run([sys.executable, '-I', '-c', "
        '"from pathlib import Path; import sys; import dinkster_image as pack; '
        "expected = Path(pack.__file__).resolve(); "
        "sys.path.insert(0, str(expected.parents[2])); "
        "del sys.modules['dinkster_image']; import dinkster_image as pack; "
        "assert Path(pack.__file__).resolve() == expected; assert pack.VALUE == 47; "
        "sys.path.insert(0, str(expected.parents[4] / 'tests')); "
        'import test_image_nodes"], '
        "check=True)\n"
        "    with pytest.raises(importlib.metadata.PackageNotFoundError):\n"
        "        importlib.metadata.distribution('dinkster-nodes-image')\n"
        "def future_helper():\n"
        "    return Unpublished.execute()\n"
        "@pytest.fixture\n"
        "def future_value():\n"
        "    return future_helper()\n"
        "def test_unpublished_fixture(future_value):\n"
        "    assert future_value == 17\n"
        "class TestUnpublished:\n"
        "    def test_node(self):\n"
        "        assert Unpublished.execute() == 17\n"
        + (
            "    def test_shipped(self):\n        assert VALUE == 47\n"
            if problem in {None, "mixed-class"}
            else ""
        )
        + (
            "def test_shadowed():\n    Unpublished = VALUE\n    assert Unpublished == 47\n"
            if problem == "shadowed"
            else ""
        )
        + (
            "@pytest.fixture(autouse=True)\ndef future_setup():\n    return Unpublished.execute()\n"
            if problem == "autouse"
            else ""
        )
        + (
            "class TestSetup:\n    def setup_method(self):\n        Unpublished.execute()\n"
            "    def test_shipped(self):\n        assert True\n"
            if problem == "xunit"
            else ""
        )
        + (
            "@pytest.fixture\ndef value():\n    return 47\n"
            "class TestFixture:\n    @pytest.fixture\n    def value(self):\n"
            "        return Unpublished.execute()\n"
            "    @pytest.fixture(autouse=True)\n    def configure(self, value):\n"
            "        assert value == 47\n    def test_shipped(self):\n        assert True\n"
            if problem == "class-fixture"
            else ""
        )
        + (
            "@pytest.mark.parametrize('Unpublished', [13])\n"
            "def test_parameter_shadow(Unpublished):\n    assert Unpublished == 47\n"
            if problem == "parameter-shadow"
            else ""
        )
        + (
            "from pytest import fixture as fx\n@fx\ndef value():\n    return 47\n"
            "class TestAlias:\n    @fx\n    def value(self):\n"
            "        return Unpublished.execute()\n"
            "    @fx(autouse=True)\n    def configure(self, value):\n        assert value == 47\n"
            "    def test_shipped(self):\n        assert True\n"
            if problem == "aliased-fixture"
            else ""
        )
        + (
            "import unittest\nclass TestUnit(unittest.TestCase):\n"
            "    def setUp(self):\n        Unpublished.execute()\n"
            "    def test_shipped(self):\n        self.assertTrue(True)\n"
            if problem == "unittest"
            else ""
        )
        + (
            "class TestClassShadow:\n    Unpublished = VALUE\n"
            "    @pytest.mark.parametrize('value', [Unpublished])\n"
            "    def test_shipped(self, value):\n        assert value == 47\n"
            if problem == "class-shadow"
            else ""
        )
        + (
            "def test_control():\n    assert True\n"
            if problem in {"mixed-class", "shadowed", "parameter-shadow"}
            else ""
        )
        + ("COLLECTION_VALUE = Unpublished.execute()\n" if problem == "collection" else "")
        + ("def teardown_function():\n    Unpublished.execute()\n" if problem == "teardown" else "")
    )
    if problem is None:
        package_tests = engine / "packages/dinkster-nodes-image/tests"
        package_tests.mkdir(parents=True)
        (tests / "selected_root_helper.py").write_text("VALUE = 17\n")
        (package_tests / "selected_package_helper.py").write_text("VALUE = 29\n")
        (package_tests / "test_image_nodes.py").write_text(
            "from dinkster_image import VALUE\n"
            "from selected_root_helper import VALUE as ROOT_VALUE\n"
            "from selected_package_helper import VALUE as PACKAGE_VALUE\n"
            "def test_package_selected_value():\n"
            "    assert (VALUE, ROOT_VALUE, PACKAGE_VALUE) == (47, 17, 29)\n"
        )
        monkeypatch.setattr(
            "scripts.run_selected_pack_tests._SUITES",
            {
                "dinkster-image": (
                    "tests/test_image_nodes.py",
                    "packages/dinkster-nodes-image/tests/test_image_nodes.py",
                )
            },
        )
    subprocess.run(["git", "init", "-q", str(engine)], check=True)
    subprocess.run(["git", "-C", str(engine), "add", "."], check=True)
    source = tmp_path / "selected"
    source.mkdir()
    (source / "pyproject.toml").write_text('[project]\nname = "dinkster-image"\n')
    (source / "dinkster-pack.toml").write_text(
        '[pack]\nname = "dinkster-image"\npublisher = "dinkster"\n'
        'version = "1.2.3"\nengine = ">=0.0.2.dev0,<2"\n'
        'requires = ["locked-only-dependency==1.7"]\n'
        'namespaces = ["image-fixture"]\n'
        '[pack.entry]\nnodes = "dinkster_image:NODES"\n'
    )
    module = source / "dinkster_image"
    module.mkdir()
    (module / "types.py").write_text("import json\n")
    (module / "locales").mkdir()
    (module / "locales/en.json").write_text("{}")
    value = 13 if problem in {"broken", "mixed-class", "shadowed", "class-shadow"} else 47
    (module / "__init__.py").write_text(
        ("import missing_selected_internal_dependency\n" if problem == "dependency" else "")
        + f"from locked_only_dependency import VALUE as DEPENDENCY_VALUE\n"
        f"assert DEPENDENCY_VALUE == 61\nVALUE = {value}\nNODES = []\n"
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    archive = artifacts / "dinkster_image-1.2.3-py3-none-any.whl"
    build_pack_archive(source, archive)
    if problem == "missing":
        with zipfile.ZipFile(archive) as wheel:
            files = {name: wheel.read(name) for name in wheel.namelist()}
        del files["dinkster_image/__init__.py"]
        with zipfile.ZipFile(archive, "w") as wheel:
            for name, data in files.items():
                wheel.writestr(name, data)
    digest = artifact_digest(archive.read_bytes())
    if problem == "broken":
        archive.rename(artifacts / f"{digest.removeprefix('blake3:')}.zip")
    else:
        from dinkster_store import pack_archive_path

        selected = pack_archive_path(artifacts, digest, label="dinkster-image/2026-11-03-1.2.3")
        selected.parent.mkdir(parents=True)
        archive.rename(selected)
    catalog = Lockfile.of(
        [
            LockedPack(
                pack="dinkster-image",
                version="1.2.3",
                artifact_digest=digest,
                publisher="dinkster",
                claims=("image-fixture",),
                source="registry:https://registry.invalid",
            )
        ]
    )
    python, locks = selected_pack_runtime(catalog, artifacts)
    from dataclasses import replace

    if problem in {None, "library-lock-mismatch"}:
        helper = _wheel(tmp_path, "selected-helper", "2.4", "selected_helper", "VALUE=83")
        helper_archive = tmp_path / helper.filename
        helper_digest = artifact_digest(helper_archive.read_bytes())
        from dinkster_store import pack_archive_path

        pack_archive_path(artifacts, helper_digest).write_bytes(helper_archive.read_bytes())
        catalog = Lockfile.of(
            [*catalog.packs, LockedPack("selected-helper", "2.4", helper_digest, "dinkster", ())]
        )
        python, locks = selected_pack_runtime(catalog, artifacts)
        for name, lock in locks.items():
            locks[name] = replace(
                lock,
                packages=tuple(
                    replace(
                        package,
                        wheels=tuple(
                            replace(
                                wheel,
                                url="http://127.0.0.1:1/unpublished-helper.whl",
                                sha256=(
                                    "f" * 64
                                    if problem == "library-lock-mismatch"
                                    and name == "dinkster-image"
                                    else wheel.sha256
                                ),
                            )
                            for wheel in package.wheels
                        ),
                    )
                    if package.name == "selected-helper"
                    else package
                    for package in lock.packages
                ),
            )
        test_file = engine / "tests/test_image_nodes.py"
        test_file.write_text(
            test_file.read_text()
            + "\nfrom selected_helper import VALUE as HELPER_VALUE\nassert HELPER_VALUE == 83\n"
        )
        subprocess.run(["git", "-C", str(engine), "add", "."], check=True)

    dependency = _wheel(
        tmp_path, "locked-only-dependency", "1.7", "locked_only_dependency", "VALUE=61"
    )
    locks["dinkster-image"] = replace(
        locks["dinkster-image"],
        requirements=("locked-only-dependency==1.7",),
        packages=(
            *locks["dinkster-image"].packages,
            LockedDistribution("locked-only-dependency", "1.7", (dependency,)),
        ),
    )
    (tmp_path / "locked_only_dependency.py").write_text("VALUE=13\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_INDEX_URL", "http://127.0.0.1:1/forbidden")
    context = {"generation_python": python, "locks": locks, "variant": "linux-cpu"}
    if problem is None:
        run_selected_tests(engine, catalog, artifacts, **context)
        output = capfd.readouterr().out
        assert "3 passed" in output
        assert "2 skipped" in output
        assert '"admissionTests": "collected"' in output
        assert '"pack": "dinkster-image@1.2.3"' in output
        assert '"nodeid": "tests/test_image_nodes.py::test_selected_value"' in output
        assert '"outcome": "passed"' in output
        assert '"outcome": "skipped"' in output
        assert "Selected test skipped tests/test_image_nodes.py::test_unpublished_fixture" in output
        assert (
            "Selected test skipped tests/test_image_nodes.py::TestUnpublished::test_node" in output
        )
        assert "dinkster_image:NEW_NODE" in output
    elif problem == "missing":
        with pytest.raises(PackArchiveError, match="wheel.import-name"):
            run_selected_tests(engine, catalog, artifacts, **context)
    elif problem == "library-lock-mismatch":
        from dinkster_workers.provision import ProvisionError

        with pytest.raises(
            ProvisionError, match="unpublished-helper.whl: download/verification failed"
        ):
            run_selected_tests(engine, catalog, artifacts, **context)
    else:
        with pytest.raises(subprocess.CalledProcessError):
            run_selected_tests(engine, catalog, artifacts, **context)


def test_selected_pack_tests_refuse_incompatible_base_before_test_tool_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_registry import LockedPack, Lockfile, artifact_digest
    from dinkster_workers.pack_lock import LockedDistribution, PackLock
    from dinkster_workers.provision import ProvisionError
    from test_pack_environment import _generation, _wheel

    from scripts.run_selected_pack_tests import run_selected_tests

    source = tmp_path / "selected"
    source.mkdir()
    (source / "dinkster_text").mkdir()
    (source / "dinkster_text/__init__.py").write_text("NODES = []\n")
    (source / "dinkster-pack.toml").write_text(
        '[pack]\nname = "dinkster-text"\npublisher = "dinkster"\n'
        'version = "1.2.3"\nengine = ">=0.0.2.dev0,<2"\n'
        '[pack.entry]\nnodes = "dinkster_text:NODES"\n'
    )
    archive = tmp_path / "dinkster_text-1.2.3-py3-none-any.whl"
    build_pack_archive(source, archive)
    digest = artifact_digest(archive.read_bytes())
    archive.rename(tmp_path / f"{digest.removeprefix('blake3:')}.zip")
    catalog = Lockfile.of([LockedPack("dinkster-text", "1.2.3", digest, "dinkster", ())])
    base = _wheel(tmp_path, "simpleeval", "1.0.7", "simpleeval", "")
    required = _wheel(tmp_path, "simpleeval", "1.0.8", "simpleeval", "")
    python = _generation(tmp_path / "generation/python", [base])
    lock = PackLock(
        "dinkster-text",
        "linux-cpu",
        ">=3.12",
        ("simpleeval==1.0.8",),
        (),
        (LockedDistribution("simpleeval", "1.0.8", (required,)),),
    )
    monkeypatch.setenv("UV_OFFLINE", "1")
    monkeypatch.setenv("UV_INDEX_URL", "http://127.0.0.1:1/forbidden")
    with (
        pytest.raises(
            RuntimeError, match="selected pack admission tests require the Linux CI environment"
        )
        if sys.platform != "linux"
        else pytest.raises(
            ProvisionError, match="simpleeval: generation has 1.0.7, lock needs 1.0.8"
        )
    ):
        run_selected_tests(
            tmp_path,
            catalog,
            tmp_path,
            generation_python=python,
            locks={"dinkster-text": lock},
            variant="linux-cpu",
        )


@pytest.mark.parametrize("selected_value", [17, 29])
@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.skipif(sys.platform != "linux", reason="engine pack admission runs on Linux")
def test_selected_pack_runtime_id_uses_distribution_layout(
    tmp_path: Path, selected_value: int, vision: bool, selected_pack_runtime
) -> None:
    from dinkster_registry import LockedPack, Lockfile, artifact_digest

    from scripts.run_selected_pack_tests import run_selected_tests

    engine = tmp_path / "engine"
    distribution = "dinkster-nodes-vision-sam3" if vision else "dinkster-nodes-text"
    pack_id = "dinkster-vision-sam3" if vision else "dinkster-text"
    entry = "dinkster_vision_sam3" if vision else "dinkster_text"
    tests = engine / "tests"
    tests.mkdir(parents=True)
    (engine / "pyproject.toml").write_text("")
    shutil.copytree(
        Path(__file__).resolve().parents[1] / "src/dinkster",
        engine / "src/dinkster",
        ignore=shutil.ignore_patterns("__pycache__", "official_packs"),
    )
    checkout_module = engine / "packages" / distribution / "src" / entry.replace(".", "/")
    checkout_module.mkdir(parents=True)
    (checkout_module.parent.parent / "pyproject.toml").write_text(
        f'[project]\nname = "{pack_id}"\n'
    )
    (checkout_module / "__init__.py").write_text("VALUE = 17\n")
    if vision:
        tests = engine / "packages" / distribution / "tests"
        tests.mkdir()
    (tests / ("test_sam3_torch.py" if vision else "test_comfy_text_nodes.py")).write_text(
        f"from pathlib import Path\nimport {entry} as pack\n"
        "def test_selected_archive():\n"
        "    assert pack.VALUE == 17\n"
        + (
            "    root = Path(__file__).parents[1]\n"
            if vision
            else "    root = Path(__file__).parents[1] / 'packages/dinkster-nodes-text'\n"
        )
        + "    assert Path(pack.__file__).resolve().is_relative_to(root / 'src')\n"
        "    from dinkster.compose import default_pack_spec\n"
        f"    info = default_pack_spec('{pack_id}').packs['{pack_id}']\n"
        "    assert info.source == f'local:{root}'\n"
    )
    subprocess.run(["git", "init", "-q", str(engine)], check=True)
    subprocess.run(["git", "-C", str(engine), "add", "."], check=True)
    source = tmp_path / "selected"
    module = source / entry.replace(".", "/")
    module.mkdir(parents=True)
    (module / "__init__.py").write_text(f"VALUE = {selected_value}\nNODES = []\n")
    (source / "pyproject.toml").write_text(f'[project]\nname = "{pack_id}"\n')
    (source / "dinkster-pack.toml").write_text(
        f'[pack]\nname = "{pack_id}"\npublisher = "dinkster"\n'
        'version = "1.2.3"\nengine = ">=0.0.2.dev0,<2"\n'
        f'namespaces = ["{pack_id}"]\n'
        f'[pack.entry]\nnodes = "{entry}:NODES"\n'
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    archive = artifacts / f"{entry}-1.2.3-py3-none-any.whl"
    build_pack_archive(source, archive)
    digest = artifact_digest(archive.read_bytes())
    archive.rename(artifacts / f"{digest.removeprefix('blake3:')}.zip")
    catalog = Lockfile.of([LockedPack(pack_id, "1.2.3", digest, "dinkster", (pack_id,))])
    python, locks = selected_pack_runtime(catalog, artifacts)
    context = {"generation_python": python, "locks": locks, "variant": "linux-cpu"}
    if selected_value == 17:
        run_selected_tests(engine, catalog, artifacts, **context)
    else:
        with pytest.raises(subprocess.CalledProcessError):
            run_selected_tests(engine, catalog, artifacts, **context)
