"""Generation construction uses real offline wheel installation and interpreter trees."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import sysconfig
import zipfile
from functools import partial
from pathlib import Path, PureWindowsPath

import pytest
from dinkster_supervisor.environment_generations import (
    build_generation,
    pack_operation,
    prepare_engine_interpreter,
    prepare_registry_inputs,
    stage_pack_sources,
)
from dinkster_supervisor.environment_state import (
    EnvironmentError,
    interpreter,
    read_current,
    set_current,
)


def _wheel(directory: Path, name: str, version: str, value: int) -> tuple[Path, str]:
    path = directory / f"{name}-{version}-py3-none-any.whl"
    metadata = f"{name}-{version}.dist-info"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr(f"{name}.py", f"VALUE = {value}\n")
        wheel.writestr(
            f"{metadata}/METADATA",
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
        )
        wheel.writestr(
            f"{metadata}/WHEEL",
            "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        wheel.writestr(f"{metadata}/RECORD", "")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mode", ["locks", "empty", "refuse", "escape", "explicit"])
def test_registry_preparation_allows_custom_locks_from_selected_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from tests.fixtures.environments.build import wheel

    store = tmp_path / "store"
    _standalone_tree(store / "bases" / ("a" * 64) / "python")
    wheels = store / "wheels"
    wheels.mkdir()
    archive = wheel(wheels, "fixture_engine", "1.0", 73, engine=True)
    constraints = tmp_path / "constraints.txt"
    constraints.write_text(
        f"fixture-engine==1.0 --hash=sha256:{hashlib.sha256(archive.read_bytes()).hexdigest()}\n"
    )
    monkeypatch.setenv("DINKSTER_FIXTURE_REGISTRY_PROBLEM", "path")
    explicit = tmp_path / "provided.pack-lock.json"
    explicit.write_text("explicit input bytes")
    captured: list[Path] = []

    def custom(engine: Path, catalog: Path, artifacts: Path, output: Path) -> tuple[Path, ...]:
        captured.append(engine.parent)
        assert (engine / "constraints.txt").read_bytes() == constraints.read_bytes()
        assert json.loads(catalog.read_text())["packs"] == []
        assert artifacts.is_dir()
        result = subprocess.check_output(
            [
                str(interpreter(engine)),
                "-I",
                "-B",
                "-c",
                "import fixture_engine; print(fixture_engine.VALUE)",
            ]
        )
        assert result.strip() == b"73"
        if mode in {"refuse", "explicit"}:
            raise EnvironmentError("custom lock preparation refused")
        output.mkdir()
        if mode == "empty":
            return ()
        if mode == "escape":
            return (explicit,)
        path = output / "custom.pack-lock.json"
        path.write_text(
            '{"format":"dinkster.pack-lock/1","pack":"custom","variant":"unpublished-custom","requiresPython":">=3.12","requirements":[],"indexes":[],"packages":[]}'
        )
        return (path,)

    def prepare() -> None:
        with prepare_registry_inputs(
            store=store,
            base="sha256:" + "a" * 64,
            constraints=constraints,
            variant="unpublished-custom",
            engine_version="1.0",
            project=tmp_path / "project",
            prepare_locks=custom,
            pack_locks=(explicit,) if mode == "explicit" else (),
        ) as (catalog, artifacts, locks, unavailable):
            assert catalog.is_file() and artifacts.is_dir()
            assert unavailable == {"fixture-pack": "registry offline"}
            if mode == "explicit":
                assert locks == (explicit,)
            elif mode == "empty":
                assert locks == ()
            else:
                assert len(locks) == 1
                assert json.loads(locks[0].read_text())["variant"] == "unpublished-custom"
                assert locks[0].parent.parent == catalog.parent == artifacts.parent

    if mode in {"refuse", "escape"}:
        with pytest.raises(
            EnvironmentError,
            match="custom lock preparation refused" if mode == "refuse" else "invalid lock paths",
        ):
            prepare()
    else:
        prepare()
    assert all(not path.exists() for path in captured)
    assert not list(store.glob(".registry-*"))
    assert not (tmp_path / "project").exists()
    assert explicit.read_text() == "explicit input bytes"


def _standalone_tree(root: Path) -> None:
    source = Path(sys.base_prefix)
    if os.name == "nt":
        root.mkdir(parents=True)
        for path in source.iterdir():
            if path.suffix in {".exe", ".dll"}:
                shutil.copy2(path, root / path.name)
        shutil.copytree(source / "DLLs", root / "DLLs")
        shutil.copytree(
            source / "Lib", root / "Lib", ignore=shutil.ignore_patterns("site-packages")
        )
    else:
        (root / "bin").mkdir(parents=True)
        shutil.copy2(Path(sys.executable).resolve(), root / "bin/python3")
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        shutil.copytree(
            Path(sysconfig.get_path("stdlib")),
            root / "lib" / version,
            ignore=shutil.ignore_patterns("site-packages", "__pycache__"),
        )
        library = sysconfig.get_config_var("LDLIBRARY")
        if library:
            path = Path(sysconfig.get_config_var("LIBDIR")) / library
            if path.is_file():
                shutil.copy2(path.resolve(), root / "lib" / path.name)


@pytest.mark.parametrize("cached", ["exact", "absent", "corrupt", "directory", "changed-copy"])
def test_generation_retains_verified_engine_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached: str
) -> None:
    root, store = tmp_path / "project", tmp_path / "store"
    base_id = "sha256:" + "a" * 64
    _standalone_tree(store / "bases" / ("a" * 64) / "python")
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("")
    raw = b'{  "version": "0.5", "unknown": [3, 1] }\r\n'
    release = hashlib.sha256(raw).hexdigest()
    cached_manifest = store / "engines" / release / "release-manifest.json"
    if cached != "absent":
        cached_manifest.parent.mkdir(parents=True)
        if cached == "directory":
            cached_manifest.mkdir()
        else:
            cached_manifest.write_bytes(raw if cached != "corrupt" else b"corrupt")
    if cached == "changed-copy":
        copyfile = shutil.copyfile

        def change_copy(src, dst, **kwargs):
            result = copyfile(src, dst, **kwargs)
            if Path(dst).name == "engine-manifest.json":
                Path(dst).write_bytes(b"changed after copying")
            return result

        monkeypatch.setattr(shutil, "copyfile", change_copy)
    callbacks: list[Path] = []

    def prepare(target: Path) -> None:
        callbacks.append(target)
        if cached == "exact":
            assert (target / "engine-manifest.json").read_bytes() == raw

    def build():
        return build_generation(
            root=root,
            number=1,
            store=store,
            base=base_id,
            engine_version="0.5",
            engine_release="sha256:" + release,
            constraints=constraints,
            variant="win-cpu" if os.name == "nt" else "linux-cpu",
            worker_protocol=1,
            prepare_packs=prepare,
        )

    target = root / "generations" / "1"
    if cached in {"corrupt", "directory", "changed-copy"}:
        with pytest.raises((EnvironmentError, OSError)):
            build()
        assert not target.exists()
        assert callbacks == []
    else:
        generation = build()
        assert callbacks == [target]
        assert generation.drift(target) == ()
        if cached == "exact":
            assert (target / "engine-manifest.json").read_bytes() == raw
            assert dict(generation.files)["engine-manifest.json"] == "sha256:" + release
        else:
            assert not (target / "engine-manifest.json").exists()
    assert read_current(root) is None


def _assert_shared(first: Path, second: Path) -> None:
    if first.samefile(second):
        return
    if sys.platform == "linux":
        import fcntl

        extents = []
        for path in (first, second):
            # FIEMAP with SYNC: one extent suffices to distinguish clones from plain copies.
            mapping = bytearray(struct.pack("=QQIIII", 0, 2**64 - 1, 1, 0, 1, 0) + bytes(56))
            with path.open("rb") as handle:
                fcntl.ioctl(handle.fileno(), 0xC020660B, mapping, True)
            assert struct.unpack_from("=I", mapping, 20)[0] == 1
            physical = struct.unpack_from("=Q", mapping, 40)[0]
            flags = struct.unpack_from("=I", mapping, 72)[0]
            assert physical > 0 and flags & 0x2000, "expected a shared CoW extent"
            extents.append(physical)
        assert extents[0] == extents[1]
    else:
        pytest.fail("expected shared inode or Linux CoW extents")


@pytest.mark.parametrize("directories,root_length", [(154, 117), (500, 200)])
def test_generation_sync_command_does_not_expand_wheel_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, directories: int, root_length: int
) -> None:
    deep = tmp_path / ("x" * max(1, root_length - len(str(tmp_path)) - 1))
    root, store = deep / "project", deep / "store"
    root.mkdir(parents=True)
    base = store / "bases" / ("a" * 64) / "python"
    python = base / ("python.exe" if os.name == "nt" else "bin/python3")
    python.parent.mkdir(parents=True)
    python.write_bytes(b"interpreter fixture")
    constraints = deep / "constraints.txt"
    rows = []
    for number in range(directories):
        digest = f"{number:064x}"
        wheel = store / "wheels" / digest / f"fixture_{number}-1.0-py3-none-any.whl"
        wheel.parent.mkdir(parents=True)
        wheel.write_bytes(b"wheel fixture")
        rows.append(f"fixture-{number}==1.0 --hash=sha256:{digest}")
    constraints.write_text("\n".join(rows) + "\n")

    windows_root = PureWindowsPath("C:/") / ("r" * (root_length - 3))
    previous = [
        "uv",
        "pip",
        "sync",
        "--python",
        str(windows_root / "project/generations/1/python/python.exe"),
        "--break-system-packages",
        "--offline",
        "--no-index",
        "--require-hashes",
        "--only-binary",
        ":all:",
        "--link-mode",
        "hardlink",
    ]
    for directory in [
        windows_root / "store/wheels",
        *(windows_root / "store/wheels" / f"{number:064x}" for number in range(directories)),
    ]:
        previous.extend(["--find-links", str(directory)])
    previous.append(str(windows_root / "project/generations/1/constraints.txt"))
    previous_units = len(subprocess.list2cmdline(previous).encode("utf-16-le")) // 2 + 1
    assert previous_units > 32767
    if directories == 154:
        assert previous_units == 32768

    def sync(command, **kwargs):
        assert len(subprocess.list2cmdline(command).encode("utf-16-le")) // 2 < 2048
        assert command[1:3] == ["pip", "sync"]
        assert "--offline" in command and "--no-index" in command
        assert "--require-hashes" in command and "--only-binary" in command
        assert command[command.index("--link-mode") + 1] in {"hardlink", "clone"}
        assert kwargs["input"].count("--find-links ") == directories + 1
        assert kwargs["input"].endswith(constraints.read_text())
        raise RuntimeError("sync inspected")

    monkeypatch.setattr(subprocess, "run", sync)
    with pytest.raises(RuntimeError, match="sync inspected"):
        build_generation(
            root=root,
            number=1,
            store=store,
            base="sha256:" + "a" * 64,
            engine_version="1.0",
            engine_release="sha256:" + "b" * 64,
            constraints=constraints,
            variant="win-cpu" if os.name == "nt" else "linux-cpu",
            worker_protocol=1,
        )
    assert not (root / "generations" / "1").exists()


def test_checkout_pack_commands_do_not_expand_wheel_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_supervisor.environment_local import prepare_local_packs
    from dinkster_workers.pack_lock import resolve_pack_lock

    store = tmp_path / ("x" * max(1, 200 - len(str(tmp_path)) - 1))
    generation = tmp_path / "generation"
    generation.mkdir()
    constraints = generation / "constraints.txt"
    constraints.write_text(
        "\n".join(f"fixture-{number}==1.0 --hash=sha256:{number:064x}" for number in range(500))
        + "\n"
    )
    links = [store / "wheels" / f"{number:064x}" for number in range(500)]

    def captured(command, **kwargs):
        assert len(subprocess.list2cmdline(command).encode("utf-16-le")) // 2 < 2048
        if "--find-links-stdin" in command:
            assert json.loads(kwargs["input"]) == [str(path) for path in links]
        else:
            source = Path(command[3]).read_text()
            assert all(f"--find-links {path.as_uri()}\n" in source for path in links)
            assert "--no-index" in command
        raise RuntimeError("command inspected")

    monkeypatch.setattr(subprocess, "run", captured)
    with pytest.raises(RuntimeError, match="command inspected"):
        prepare_local_packs(
            generation,
            source=tmp_path / "source",
            store=store,
            release="sha256:" + "a" * 64,
            selection={"version": "1.0", "variant": "win-cpu", "workerProtocol": 1},
        )
    with pytest.raises(RuntimeError, match="command inspected"):
        resolve_pack_lock(
            pack="fixture",
            variant="win-cpu",
            requirements=["fixture-0==1.0"],
            python=interpreter(generation),
            constraints=constraints,
            find_links=links,
        )


@pytest.mark.parametrize("provisioner", ["current", "cached-without-acquire", "without-acquire"])
def test_public_setup_acquires_pack_wheel_absent_from_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provisioner: str
) -> None:
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    from dinkster_workers import pack_environment
    from dinkster_workers.boundary import PROTOCOL_VERSION
    from dinkster_workers.pack_environment import venv_python
    from dinkster_workers.pack_lock import LockedDistribution, LockedWheel, PackLock
    from test_environment_cli import _run
    from test_environment_runtime import free_port

    from tests.fixtures.environments import build as publisher

    original = publisher.standalone_tree
    encode = publisher.canonical_json

    def canonical_json(document: dict) -> bytes:
        if "workerProtocol" in document:
            document["workerProtocol"] = PROTOCOL_VERSION
        return encode(document)

    def standalone(root: Path) -> None:
        original(root)
        python = root / ("python.exe" if os.name == "nt" else "bin/python3")
        site = Path(
            subprocess.check_output(
                [str(python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
                text=True,
            ).strip()
        )
        site.mkdir(parents=True, exist_ok=True)
        (site / "fixture.pth").write_text(
            "\n".join(str(Path(path).resolve()) for path in sys.path if path) + "\n"
        )
        if provisioner != "current":
            target = site / "dinkster_workers"
            shutil.copytree(
                Path(pack_environment.__file__).parent,
                target,
                ignore=shutil.ignore_patterns("__pycache__"),
            )
            module = target / "pack_environment.py"
            text = module.read_text()
            text = text.replace(
                'choices=("preflight", "acquire", "rebuild", "doctor", "terminal", "references")',
                'choices=("preflight", "rebuild", "doctor", "terminal", "references")',
            )
            module.write_text(text)

    monkeypatch.setattr(publisher, "standalone_tree", standalone)
    monkeypatch.setattr(publisher, "canonical_json", canonical_json)
    for variable in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.setenv(variable, str(tmp_path / "user-data"))
    fixture = publisher.build(tmp_path / "release", "fixture-cpu", free_port())
    mirror = Path(fixture["mirror"])
    index = tmp_path / "pypi"
    index.mkdir()
    wheel, sha = _wheel(index, "fixture_flatbuffers", "25.12.19", 73)
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(index))
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root, store = Path(fixture["root"]), Path(fixture["store"])
    if provisioner == "cached-without-acquire":
        cache = store / "wheels" / sha / wheel.name
        cache.parent.mkdir(parents=True)
        shutil.copyfile(wheel, cache)
    try:
        channel_path = mirror / "channels/main.json"
        channel = json.loads(channel_path.read_bytes())
        release = mirror / "engine" / channel["engine"]["release"][7:]
        lock_path = release / "example-1.0-fixture-cpu.pack-lock.json"
        PackLock(
            "example",
            "fixture-cpu",
            ">=3.12",
            ("fixture-flatbuffers==25.12.19",),
            (),
            (
                LockedDistribution(
                    "fixture-flatbuffers",
                    "25.12.19",
                    (
                        LockedWheel(
                            wheel.name,
                            wheel.stat().st_size,
                            sha,
                            f"http://127.0.0.1:{server.server_port}/{wheel.name}",
                        ),
                    ),
                ),
            ),
        ).write(lock_path)
        manifest = json.loads((release / "release-manifest.json").read_bytes())
        manifest["artifacts"].append(publisher.artifact(lock_path.name, lock_path.read_bytes()))
        assert wheel.name not in {row["name"] for row in manifest["artifacts"]}
        data = publisher.canonical_json(manifest) + b"\n"
        (release / "release-manifest.json").write_bytes(data)
        identity = hashlib.sha256(data).hexdigest()
        release.rename(release.parent / identity)
        channel["engine"]["release"] = "sha256:" + identity
        channel["engine"]["locations"][0]["path"] = f"engine/{identity}/release-manifest.json"
        channel_path.write_bytes(publisher.canonical_json(channel))
        code, result = _run(root, *fixture["commands"]["setup"][5:], "--yes")
        if provisioner == "without-acquire":
            assert code == 1, result
            assert "selected engine cannot acquire missing pack wheels" in result["error"]
            assert result["current"] is None
            assert not (root / "generations/1").exists()
            return
        assert code == 0, result
        assert result["current"] == 1
        assert (store / "wheels" / sha / wheel.name).read_bytes() == wheel.read_bytes()
        python = venv_python(root / "generations/1/packs/example")
        assert (
            subprocess.check_output(
                [str(python), "-c", "import fixture_flatbuffers; print(fixture_flatbuffers.VALUE)"],
                text=True,
            ).strip()
            == "73"
        )
    finally:
        _run(root, "stop", "--yes")
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "problem",
    [
        None,
        "update",
        "extra",
        "missing",
        "variant",
        "wheel",
        "repair",
        "repair-extra",
        "repair-missing",
        "move",
        "move-interrupted",
        "move-running-interrupted",
        "move-cross-volume-plain",
        "move-cross-volume-aliases",
        "move-cross-volume-cached-aliases",
        "move-cross-volume-missing-resource",
        "move-cross-volume-corrupt-wheel-cache",
    ],
)
def test_generation_requires_dependency_locks_for_the_complete_admitted_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
    problem: str | None,
) -> None:
    from dinkster_registry import LockedPack, Lockfile, build_artifact
    from dinkster_workers import load_manifest
    from dinkster_workers.boundary import PROTOCOL_VERSION
    from dinkster_workers.catalog import read_catalog
    from dinkster_workers.pack_environment import venv_python
    from dinkster_workers.pack_lock import LockedDistribution, LockedWheel, PackLock
    from test_doctor import HEALTHY_MANIFEST, HEALTHY_NODES

    root, store = tmp_path / "project", tmp_path / "store"
    root.mkdir()
    base = store / "bases" / ("a" * 64) / "python"
    _standalone_tree(base)
    cross_volume = problem is not None and problem.startswith("move-cross-volume-")
    aliases = problem in {"move-cross-volume-aliases", "move-cross-volume-cached-aliases"}
    executable = Path("python.exe") if os.name == "nt" else Path("bin/python3")
    if cross_volume:
        request.addfinalizer(partial(shutil.rmtree, tmp_path, ignore_errors=True))
        resources = base / "share" / "fixture"
        resources.mkdir(parents=True)
        (resources / "resource.txt").write_bytes(b"sealed non-startup resource")
    if aliases:
        try:
            (base / "executable-alias").symlink_to(executable)
            (base / "resource-alias").symlink_to(Path("share/fixture/resource.txt"))
            (base / "directory-alias").symlink_to(Path("share/fixture"), target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"internal interpreter aliases require symlink support: {exc}")
    site = Path(
        subprocess.check_output(
            [
                str(interpreter(base.parent)),
                "-B",
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            text=True,
        ).strip()
    )
    site.mkdir(parents=True, exist_ok=True)
    (site / "fixture.pth").write_text(
        "\n".join(str(Path(path).resolve()) for path in sys.path if path) + "\n"
    )
    wheels = store / "wheels"
    wheels.mkdir()
    _, sha = _wheel(wheels, "stable_fixture", "1.0", 40)
    constraints = tmp_path / "constraints.txt"
    constraints.write_text(f"stable-fixture==1.0 --hash=sha256:{sha}\n")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    entries, locks = [], []
    wheel_sha: str | None = None
    variant = "win-cpu" if os.name == "nt" else "linux-cpu"
    for name in ("healthy", "other"):
        source = tmp_path / name
        source.mkdir()
        (source / "dinkster-pack.toml").write_text(HEALTHY_MANIFEST.replace("healthy", name))
        (source / f"{name}_nodes.py").write_text(HEALTHY_NODES.replace("healthy", name))
        archive = artifacts / "staging.zip"
        identity = build_artifact(source, archive)
        archive.rename(artifacts / f"{identity.removeprefix('blake3:')}.zip")
        entries.append(LockedPack(name, "1.0.0", identity, "local", (name,)))
        if name == "healthy" or problem != "missing":
            path = tmp_path / f"{name}-1.0.0-{variant}.pack-lock.json"
            locked_variant = (
                "wrong-variant" if problem == "variant" and name == "other" else variant
            )
            packages = ()
            if problem in {"wheel", "extra"} and name == "other":
                wheel, wheel_sha = _wheel(
                    wheels if problem == "extra" else tmp_path, "unavailable_extra", "1.0", 73
                )
                packages = (
                    LockedDistribution(
                        "unavailable-extra",
                        "1.0",
                        (LockedWheel(wheel.name, wheel.stat().st_size, wheel_sha, wheel.as_uri()),),
                    ),
                )
            PackLock(name, locked_variant, ">=3.12", (), (), packages).write(path)
            locks.append(path)
    catalog = tmp_path / "official-packs.lock"
    catalog.write_text(Lockfile.of(entries).record_json())
    previous = None
    number = 1
    if problem == "update":
        previous = root / "generations/1"
        build_generation(
            root=root,
            number=1,
            store=store,
            base="sha256:" + "a" * 64,
            engine_version="1.0.0",
            engine_release="sha256:" + "b" * 64,
            constraints=constraints,
            variant=variant,
            worker_protocol=PROTOCOL_VERSION,
        )
        assert not (previous / "packs").exists()
        set_current(root, 1)
        number = 2
    invoke = partial(
        build_generation,
        root=root,
        number=number,
        store=store,
        base="sha256:" + "a" * 64,
        engine_version="1.0.0",
        engine_release="sha256:" + "b" * 64,
        constraints=constraints,
        variant=variant,
        worker_protocol=PROTOCOL_VERSION,
        in_process_packs=("healthy",),
        pack_locks=tuple(locks),
        own_environment_size=1000,
        catalog_lock=catalog,
        artifacts=artifacts,
        source_generation=previous,
    )
    repairs = {"repair", "repair-extra", "repair-missing"}
    moves = {
        "move",
        "move-interrupted",
        "move-running-interrupted",
        "move-cross-volume-plain",
        "move-cross-volume-aliases",
        "move-cross-volume-cached-aliases",
        "move-cross-volume-missing-resource",
        "move-cross-volume-corrupt-wheel-cache",
    }
    if problem is not None and problem not in {"update", "extra"} | repairs | moves:
        message = {
            "missing": "selected packs have no dependency lock: other",
            "variant": "variant wrong-variant does not match",
            "wheel": "offline pack wheel missing or corrupt",
        }[problem]
        with pytest.raises(EnvironmentError, match=message):
            invoke()
        assert not (root / "generations/1").exists()
    else:
        generation = invoke()
        target = root / "generations" / str(number)
        assert generation.drift(target) == ()
        assert not (target / "packs/healthy/bin").exists()
        assert venv_python(target / "packs/other").is_file()
        for name in ("healthy", "other"):
            assert f"packs/{name}/pack-lock.json" in dict(generation.files)
            recorded = PackLock.read(target / "packs" / name / "pack-lock.json")
            assert tuple(package.name for package in recorded.packages) == (
                ("unavailable-extra",) if problem == "extra" and name == "other" else ()
            )
            manifest = load_manifest(target / "pack-sources" / name / "dinkster-pack.toml")
            prepared = read_catalog(manifest)
            assert prepared is not None
            assert set(prepared.schemas) == {f"{name}.doubler", f"{name}.tagger"}
        assert sum("dinkster-schema-catalog" in name for name, _ in generation.files) == 2
        if problem in moves:
            from dinkster_supervisor import environment_move, environment_stores
            from dinkster_supervisor.environment_runtime import (
                engine_assignments,
                start_engine,
                stop_engine,
            )
            from dinkster_supervisor.environment_state import (
                Generation,
                Journal,
                content_digest,
                write_document,
            )
            from test_environment_cli import _run

            monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user-data"))
            monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "user-data"))
            set_current(root, 1)
            (root / "store").write_text(str(store) + "\n")
            write_document(
                root / "project.json",
                {
                    "format": "dinkster.project/1",
                    "channel": "main",
                    "variant": variant,
                    "mirrors": [str(tmp_path / "unavailable-mirror")],
                    "port": 3639,
                },
            )
            write_document(
                store / "projects.json", {"format": "dinkster.store/1", "projects": [str(root)]}
            )
            for cached in wheels.glob("*.whl"):
                directory = wheels / hashlib.sha256(cached.read_bytes()).hexdigest()
                directory.mkdir()
                cached.rename(directory / cached.name)
            shutil.copytree(artifacts, store / "engines" / ("b" * 64) / "artifacts")
            (root / "library").mkdir()
            (root / "library/asset").write_bytes(b"preserve this project asset")
            destination = tmp_path / "moved project"
            again = destination.parent / "moved again"
            target_store = store
            missing = Path("python/share/fixture/resource.txt")

            def move(source: Path, target: Path) -> None:
                if cross_volume:
                    plan = environment_move.plan_move(source, target, target_store)
                    assert plan.cross_volume == (source == root)
                    environment_move.execute_move(plan)
                else:
                    code, result = _run(source, "move", str(target), "--yes")
                    assert code == 0 and result["status"] == "applied", result

            if cross_volume:
                # Exercise volume-dependent branches without requiring another filesystem.
                def volume(path: Path) -> str:
                    return (
                        "1"
                        if path.resolve().is_relative_to(root)
                        or path.resolve().is_relative_to(store)
                        else "2"
                    )

                monkeypatch.setattr(environment_move, "volume", volume)
                monkeypatch.setattr(environment_stores, "volume", volume)
                target_store = tmp_path / "destination-store"
                if problem in {
                    "move-cross-volume-cached-aliases",
                    "move-cross-volume-missing-resource",
                    "move-cross-volume-corrupt-wheel-cache",
                }:
                    for category in ("bases", "engines", "wheels"):
                        environment_move.copy_tree(store / category, target_store / category)
                if problem == "move-cross-volume-corrupt-wheel-cache":
                    warm = tmp_path / "warm-cache-project"
                    warm.mkdir()
                    invoke(root=warm, store=target_store)
                    cached = next((target_store / "uv-cache").rglob("stable_fixture.py"))
                    cached.write_text("VALUE = 999\n")
                    assert cached.read_text() == "VALUE = 999\n"
                    assert (
                        next((root / "generations/1").rglob("stable_fixture.py")).read_text()
                        == "VALUE = 40\n"
                    )
                if problem == "move-cross-volume-missing-resource":
                    assert missing.as_posix() in dict(generation.files)
                    (target_store / "bases" / ("a" * 64) / missing).unlink()

                if problem == "move-cross-volume-missing-resource":
                    with pytest.raises(EnvironmentError):
                        move(root, destination)
                    assert read_current(root) == 1
                    assert generation.drift(root / "generations/1") == ()
                    assert (root / "library/asset").read_bytes() == b"preserve this project asset"
                    assert (
                        root / "generations/1" / missing
                    ).read_bytes() == b"sealed non-startup resource"
                    assert not destination.exists()
                    assert all(
                        not Journal.read(path).incomplete
                        for path in (root / "journal").glob("*.json")
                    )
                    return
            assignments = None
            if problem == "move-running-interrupted":
                from test_environment_runtime import free_port

                for location in (root, destination, again):
                    for name in ("main", "spare"):
                        request.addfinalizer(partial(stop_engine, location, name, generation=1))
                start_engine(root, 1, "main", free_port())
                start_engine(root, 1, "spare", free_port())
                stop_engine(root, "spare")
                assignments = engine_assignments(root, 1)
                assert {row["name"]: row["running"] for row in assignments} == {
                    "main": True,
                    "spare": False,
                }
            before = {
                p.relative_to(root): content_digest(p)
                for p in root.rglob("*")
                if p.is_file()
                and not (p.is_relative_to(root / "runtime") and p.name.endswith((".log", ".lock")))
            }
            for flags in ((), ("--plan",)):
                if cross_volume:
                    assert environment_move.plan_move(root, destination, target_store).cross_volume
                else:
                    code, planned = _run(root, "move", str(destination), *flags)
                    assert code == 0 and planned["status"] == "planned", planned
                assert not destination.exists()
                assert before == {
                    p.relative_to(root): content_digest(p)
                    for p in root.rglob("*")
                    if p.is_file()
                    and not (
                        p.is_relative_to(root / "runtime") and p.name.endswith((".log", ".lock"))
                    )
                }
            if problem in {"move-interrupted", "move-running-interrupted"}:
                script = """
import os, sys
from dinkster_supervisor import environment, environment_move
hook = sys.argv[3]
original = getattr(environment_move, hook)
def interrupt(*args):
    original(*args)
    os._exit(73)
setattr(environment_move, hook, interrupt)
sys.argv = ['dinkster-env', '--root', sys.argv[1], 'move', sys.argv[2], '--yes', '--json']
environment.main()
"""
                child = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        script,
                        str(root),
                        str(destination),
                        "_resume" if assignments is not None else "_prepare",
                    ],
                    timeout=60,
                )
                assert child.returncode == 73
                assert (root / "library/asset").read_bytes() == b"preserve this project asset"
                assert _run(root, "stop", "--engine", "spare", "--yes")[0] == 0
                assert not destination.exists()
                assert read_current(root) == 1
                if assignments is not None:
                    assert engine_assignments(root, 1) == assignments
                assert all(
                    not Journal.read(p).incomplete for p in (root / "journal").glob("*.json")
                )
            move(root, destination)
            assert read_current(destination) == 1 and read_current(root) is None
            if assignments is not None:
                assert engine_assignments(destination, 1) == assignments
            assert (destination / "library/asset").read_bytes() == b"preserve this project asset"
            moved_store = Path((destination / "store").read_text().strip())
            assert json.loads((moved_store / "projects.json").read_bytes())["projects"] == [
                str(destination)
            ]
            if moved_store != store:
                assert json.loads((store / "projects.json").read_bytes())["projects"] == []
            assert _run(destination, "doctor")[1]["status"] == "clean"
            assert "project moved to" in _run(root, "start", "--yes")[1]["error"]
            moved_tree = destination / "generations/1"
            assert Generation.read(moved_tree / "generation.json").drift(moved_tree) == ()
            if cross_volume:
                moved_generation = Generation.read(moved_tree / "generation.json")
                assert dict(moved_generation.files).keys() == dict(generation.files).keys()
                assert (
                    moved_tree / "python/share/fixture/resource.txt"
                ).read_bytes() == b"sealed non-startup resource"
            if aliases:
                for alias, original in (
                    ("executable-alias", executable.as_posix()),
                    ("resource-alias", "share/fixture/resource.txt"),
                    ("directory-alias/resource.txt", "share/fixture/resource.txt"),
                ):
                    retained = moved_tree / "python" / alias
                    assert not retained.is_symlink()
                    assert retained.read_bytes() == (moved_tree / "python" / original).read_bytes()
            assert (
                subprocess.check_output(
                    [
                        str(venv_python(moved_tree / "packs/other")),
                        "-c",
                        "import stable_fixture,sys; print(stable_fixture.VALUE); print(sys.prefix)",
                    ],
                    text=True,
                ).strip()
                == f"40\n{moved_tree / 'packs/other'}"
            )
            assert (
                f"home = {moved_tree / 'python' / ('' if os.name == 'nt' else 'bin')}"
                in (moved_tree / "packs/other/pyvenv.cfg").read_text()
            )
            _assert_shared(
                next(moved_tree.rglob("stable_fixture.py")),
                next((moved_store / "uv-cache").rglob("stable_fixture.py")),
            )
            move(destination, again)
            if assignments is not None:
                assert engine_assignments(again, 1) == assignments
            assert _run(again, "stop", "--yes")[0] == 0
            assert _run(again, "doctor")[1]["status"] == "clean"
            return
        if problem in repairs:
            from dataclasses import replace

            from dinkster_supervisor.environment_state import write_document
            from test_environment_cli import _run

            monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "user-data"))
            monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "user-data"))
            legacy_files = {
                name: sha for name, sha in generation.files if not name.startswith("packs/")
            }
            if problem == "repair-extra":
                del legacy_files["pack-sources/other/other_nodes.py"]
            elif problem == "repair-missing":
                (target / "recorded.txt").write_bytes(b"retained input")
                from dinkster_supervisor.environment_state import content_digest

                legacy_files["recorded.txt"] = content_digest(target / "recorded.txt")
            replace(generation, files=tuple(sorted(legacy_files.items()))).write(
                target / "generation.json"
            )
            set_current(root, number)
            (root / "store").write_text(str(store) + "\n")
            write_document(
                root / "project.json",
                {
                    "format": "dinkster.project/1",
                    "channel": "main",
                    "variant": variant,
                    "mirrors": [str(tmp_path / "unavailable-mirror")],
                    "port": 3639,
                },
            )
            release = store / "engines" / ("b" * 64)
            shutil.copytree(artifacts, release / "artifacts")
            code, result = _run(root, "doctor", "--repair", "--yes")
            if problem == "repair":
                assert code == 0 and result["current"] == 2, result
                from dinkster_supervisor.environment_state import Generation

                repaired = Generation.read(root / "generations/2/generation.json")
                assert repaired.drift(root / "generations/2") == ()
                assert all(
                    f"packs/{name}/pack-lock.json" in dict(repaired.files)
                    for name in ("healthy", "other")
                )
                assert (
                    subprocess.check_output(
                        [
                            str(venv_python(root / "generations/2/packs/other")),
                            "-c",
                            "import stable_fixture, sys; "
                            "print(stable_fixture.VALUE); print(sys.prefix)",
                        ],
                        text=True,
                    ).strip()
                    == f"40\n{root / 'generations/2/packs/other'}"
                )
                for name in ("healthy", "other"):
                    manifest = load_manifest(
                        root / "generations/2/pack-sources" / name / "dinkster-pack.toml"
                    )
                    prepared = read_catalog(manifest)
                    assert prepared is not None
                    assert set(prepared.schemas) == {f"{name}.doubler", f"{name}.tagger"}
            else:
                assert code == 1 and "repaired file inventory differs" in result["error"], result
                assert read_current(root) == 1
                assert not (root / "generations/2").exists()
            return
        if problem == "extra":
            from dinkster_supervisor.environment_collection import store_collection_plan
            from dinkster_supervisor.environment_state import write_document

            (root / "store").write_text(str(store) + "\n")
            for cached in wheels.glob("*.whl"):
                directory = wheels / hashlib.sha256(cached.read_bytes()).hexdigest()
                directory.mkdir(exist_ok=True)
                cached.rename(directory / cached.name)
            write_document(
                store / "projects.json", {"format": "dinkster.store/1", "projects": [str(root)]}
            )
            plan = store_collection_plan(root, store, ())
            assert wheel_sha is not None
            assert f"wheels/{wheel_sha}" not in plan
            pack_tree = target / "packs/other"
            parked = tmp_path / "parked-pack"
            pack_tree.rename(parked)
            pack_tree.write_text("not a readable pack reference")
            with pytest.raises(EnvironmentError, match="recorded dependency locks"):
                store_collection_plan(root, store, ())
            pack_tree.unlink()
            with pytest.raises(EnvironmentError, match="recorded dependency locks"):
                store_collection_plan(root, store, ())
            parked.rename(pack_tree)
            assert (
                subprocess.check_output(
                    [
                        str(venv_python(target / "packs/other")),
                        "-c",
                        "import unavailable_extra; print(unavailable_extra.VALUE)",
                    ]
                ).strip()
                == b"73"
            )
    assert read_current(root) == (1 if problem == "update" else None)


def test_offline_generations_share_unchanged_wheels_and_failed_sync_preserves_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, store = tmp_path / "project", tmp_path / "data with space # %"
    root.mkdir()
    base = store / "bases" / ("a" * 64) / "python"
    _standalone_tree(base)
    wheels = store / "wheels"
    wheels.mkdir()
    _, stable_hash = _wheel(wheels, "stable_fixture", "1.0", 40)
    _, first_hash = _wheel(wheels, "changed_fixture", "1.0", 2)
    _, second_hash = _wheel(wheels, "changed_fixture", "2.0", 7)
    for path in wheels.glob("*.whl"):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        destination = wheels / digest / path.name
        destination.parent.mkdir()
        path.rename(destination)
    other_build = tmp_path / "other-build"
    other_build.mkdir()
    stale, stale_hash = _wheel(other_build, "changed_fixture", "2.0", 999)
    (wheels / stale_hash).mkdir()
    stale.rename(wheels / stale_hash / stale.name)
    decoy, decoy_hash = _wheel(other_build, "changed_fixture", "3.0", 101)
    (wheels / ("d" * 64)).mkdir()
    shutil.copyfile(decoy, wheels / ("d" * 64) / decoy.name)
    (wheels / ("f" * 64)).mkdir()
    decoy.rename(wheels / ("f" * 64) / decoy.name)
    constraints = tmp_path / "constraints.txt"
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("UV_INDEX_URL", "http://127.0.0.1:1/unreachable")

    def build(
        number: int,
        version: str,
        sha: str,
        *,
        reject_packs: bool = False,
        source_inputs: bool = False,
    ):
        constraints.write_text(
            f"stable-fixture==1.0 --hash=sha256:{stable_hash}\n"
            f"changed-fixture=={version} --hash=sha256:{sha}\n"
        )

        def prepare_packs(target: Path) -> None:
            site = next(target.rglob("stable_fixture.py")).parent
            value = 97 if reject_packs else 11
            (site / "in_process_fixture.py").write_text(f"VALUE = {value}\n")
            catalog = target / "pack-sources/fixture/__pycache__/dinkster-schema-catalog/pack.json"
            catalog.parent.mkdir(parents=True)
            catalog.write_text('{"prepared": true}\n')
            if reject_packs:
                staged = subprocess.check_output(
                    [
                        str(interpreter(target)),
                        "-B",
                        "-c",
                        "import in_process_fixture; print(in_process_fixture.VALUE)",
                    ]
                )
                assert staged.strip() == b"97"
                raise EnvironmentError("later pack refused after interpreter staging")

        return build_generation(
            root=root,
            number=number,
            store=store,
            base="sha256:" + "a" * 64,
            engine_version=version,
            engine_release="sha256:" + str(number) * 64,
            constraints=constraints,
            variant="win-cpu" if os.name == "nt" else "linux-cpu",
            worker_protocol=1,
            catalog_lock=tmp_path / "catalog.lock" if source_inputs else None,
            artifacts=tmp_path / "artifacts" if source_inputs else None,
            prepare_packs=prepare_packs,
            in_process_packs=("fixture",),
        )

    first = build(1, "1.0", first_hash)
    assert first.in_process_packs == ("fixture",)
    first_dir = root / "generations" / "1"
    set_current(root, 1)
    constraints.write_text(
        f"stable-fixture==1.0 --hash=sha256:{stable_hash}\n"
        f"changed-fixture==2.0 --hash=sha256:{second_hash}\n"
    )
    scratch = store / "preflight"
    target = prepare_engine_interpreter(
        scratch, store=store, base="sha256:" + "a" * 64, constraints=constraints
    )
    assert (
        subprocess.check_output(
            [
                str(target),
                "-B",
                "-c",
                "import stable_fixture,changed_fixture; "
                "print(stable_fixture.VALUE+changed_fixture.VALUE)",
            ]
        ).strip()
        == b"47"
    )
    assert {path.name for path in scratch.iterdir()} == {"python", "constraints.txt"}
    assert read_current(root) == 1
    assert first.drift(first_dir) == ()
    assert not list(base.rglob("stable_fixture.py"))
    assert not list(base.rglob("changed_fixture.py"))
    with pytest.raises(EnvironmentError, match="destination already exists"):
        prepare_engine_interpreter(
            scratch, store=store, base="sha256:" + "a" * 64, constraints=constraints
        )
    assert target.is_file()
    shutil.rmtree(scratch)
    second = build(2, "2.0", second_hash)
    second_dir = root / "generations" / "2"
    assert read_current(root) == 1
    script = (
        "import stable_fixture,changed_fixture,in_process_fixture; "
        "print(stable_fixture.VALUE+changed_fixture.VALUE+in_process_fixture.VALUE)"
    )
    for directory, expected in [(first_dir, "53"), (second_dir, "58")]:
        result = subprocess.run(
            [str(interpreter(directory)), "-B", "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == expected
    stable_relative = next(name for name, _ in first.files if name.endswith("stable_fixture.py"))
    _assert_shared(first_dir / stable_relative, second_dir / stable_relative)
    _assert_shared(interpreter(first_dir), interpreter(second_dir))
    assert first.drift(first_dir) == ()
    assert second.drift(second_dir) == ()
    assert any(name.endswith("in_process_fixture.py") for name, _ in first.files)
    catalog_name = "pack-sources/fixture/__pycache__/dinkster-schema-catalog/pack.json"
    assert catalog_name in dict(first.files)
    catalog = first_dir / catalog_name
    catalog.unlink()
    assert first.drift(first_dir) == (f"missing: {catalog_name}",)
    catalog.write_text('{"prepared": true}\n')
    set_current(root, 2)
    set_current(root, 1)
    assert read_current(root) == 1

    for digest in (decoy_hash, "f" * 64):
        with pytest.raises(EnvironmentError, match="offline generation sync failed"):
            build(3, "3.0", digest)
    assert read_current(root) == 1
    assert not (root / "generations" / "3").exists()
    with pytest.raises(EnvironmentError, match="later pack refused after interpreter staging"):
        build(4, "2.0", second_hash, reject_packs=True)
    assert read_current(root) == 1
    assert not (root / "generations" / "4").exists()
    run = subprocess.run

    def refused_sources(command, *args, **kwargs):
        if command[1:4] == ["-B", "-m", "dinkster.generation_packs"]:
            target = Path(command[command.index("--generation") + 1])
            sources = target / "pack-sources"
            sources.mkdir()
            (sources / "partial-source.py").write_text("partial source admission\n")
            return subprocess.CompletedProcess(
                command, 0, '{"packs": [], "errors": ["source hash mismatch"]}', ""
            )
        return run(command, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "run", refused_sources)
        with pytest.raises(EnvironmentError, match="source hash mismatch"):
            build(5, "2.0", second_hash, source_inputs=True)
    assert read_current(root) == 1
    assert not (root / "generations" / "5").exists()

    def refused_catalog(command, *args, **kwargs):
        if command[1:4] == ["-B", "-m", "dinkster.generation_packs"]:
            target = Path(command[command.index("--generation") + 1])
            (target / "dinkster.lock").write_text("admitted source inventory\n")
            return subprocess.CompletedProcess(command, 0, '{"packs": [], "errors": []}', "")
        if command[1:4] == ["-B", "-m", "dinkster.manager"]:
            return subprocess.CompletedProcess(command, 1, "", "pack import failed")
        return run(command, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "run", refused_catalog)
        with pytest.raises(EnvironmentError, match="pack catalog preparation failed: pack import"):
            build(6, "2.0", second_hash, source_inputs=True)
    assert read_current(root) == 1
    assert not (root / "generations" / "6").exists()

    def missing_lock(command, *args, **kwargs):
        if command[1:4] == ["-B", "-m", "dinkster.generation_packs"]:
            target = Path(command[command.index("--generation") + 1])
            (target / "dinkster.lock").write_text("admitted source inventory\n")
            return subprocess.CompletedProcess(
                command, 0, '{"packs": ["example"], "errors": []}', ""
            )
        return run(command, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "run", missing_lock)
        with pytest.raises(
            EnvironmentError, match="selected packs have no dependency lock: example"
        ):
            build(7, "2.0", second_hash, source_inputs=True)
    assert read_current(root) == 1
    assert not (root / "generations" / "7").exists()
    assert first.drift(first_dir) == ()
    assert (
        subprocess.check_output([str(interpreter(first_dir)), "-B", "-c", script]).strip() == b"53"
    )
    shutil.rmtree(base.parent)
    shutil.rmtree(second_dir)
    assert (
        subprocess.check_output([str(interpreter(first_dir)), "-B", "-c", script]).strip() == b"53"
    )


@pytest.mark.parametrize(
    ("operation", "output", "code", "message"),
    [
        ("preflight", '{"packs": [], "changes": []}', 0, None),
        ("preflight", "not-json", 0, "invalid provisioner result"),
        ("preflight", '{"packs": [], "changes": []}\n{}', 0, "invalid provisioner result"),
        ("preflight", '{"packs": [], "changes": ["missing lock"]}', 1, "missing lock"),
        ("doctor", '{"packs": [], "changes": ["example: missing venv"]}', 0, "missing venv"),
        (
            "preflight",
            json.dumps(
                {
                    "packs": [
                        {
                            "pack": "example",
                            "conflicts": ["torch version mismatch"],
                            "downloadBytes": 12,
                            "ownEnvironmentBytes": 34,
                        }
                    ],
                    "changes": [],
                }
            ),
            0,
            "torch version mismatch",
        ),
        (
            "preflight",
            '{"packs": [{"pack": "example", "conflicts": [], '
            '"downloadBytes": true, "ownEnvironmentBytes": 34}], "changes": []}',
            0,
            "nonnegative integer",
        ),
    ],
)
def test_pack_result_must_be_one_valid_object_without_conflicts_or_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    output: str,
    code: int,
    message: str | None,
) -> None:
    # Treat the provisioner's stdout as an external wire boundary, including incorrect exit codes.
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], code, output, "")
    )
    invoke = partial(
        pack_operation,
        operation,
        generation=tmp_path / "generations/2",
        store=tmp_path / "store",
        variant="linux-cpu",
        worker_protocol=11,
        own_environment_size=34,
    )
    if message is None:
        assert invoke() == {"packs": [], "changes": []}
    else:
        with pytest.raises(EnvironmentError, match=message):
            invoke()


@pytest.mark.parametrize(
    ("output", "code", "message"),
    [
        ('{"packs": ["example"], "errors": []}', 0, None),
        ('{"packs": [], "errors": ["source hash mismatch"]}', 1, "source hash mismatch"),
        ('{"packs": [], "errors": ["source hash mismatch"]}', 0, "source hash mismatch"),
        ('{"packs": [], "errors": []}', 1, "admission failed"),
        ('{"packs": ["example", "example"], "errors": []}', 0, "repeat an id"),
        ('{"packs": [], "errors": [false]}', 0, "errors must contain strings"),
        ('{"packs": [], "errors": []}\n{}', 0, "invalid admission result"),
    ],
)
def test_source_admission_requires_a_single_valid_success_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output: str,
    code: int,
    message: str | None,
) -> None:
    (tmp_path / "dinkster.lock").write_text("fixture installed source inventory\n")
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], code, output, "")
    )
    invoke = partial(
        stage_pack_sources,
        tmp_path,
        catalog_lock=tmp_path / "catalog.lock",
        artifacts=tmp_path / "artifacts",
    )
    if message is None:
        invoke()
        (tmp_path / "dinkster.lock").unlink()
        with pytest.raises(EnvironmentError, match="did not write the installed source lock"):
            invoke()
    else:
        with pytest.raises(EnvironmentError, match=message):
            invoke()
