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
from pathlib import Path

import pytest
from dinkster_supervisor.environment_generations import (
    build_generation,
    pack_operation,
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


def _assert_shared(first: Path, second: Path) -> None:
    if first.samefile(second):
        return
    assert sys.platform.startswith("linux"), "expected shared inode or Linux CoW extents"
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


def test_offline_generations_share_unchanged_wheels_and_failed_sync_preserves_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, store = tmp_path / "project", tmp_path / "data"
    root.mkdir()
    base = store / "bases" / ("a" * 64) / "python"
    _standalone_tree(base)
    wheels = store / "wheels"
    wheels.mkdir()
    _, stable_hash = _wheel(wheels, "stable_fixture", "1.0", 40)
    _, first_hash = _wheel(wheels, "changed_fixture", "1.0", 2)
    _, second_hash = _wheel(wheels, "changed_fixture", "2.0", 7)
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
        )

    first = build(1, "1.0", first_hash)
    first_dir = root / "generations" / "1"
    set_current(root, 1)
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

    with pytest.raises(EnvironmentError, match="offline generation sync failed"):
        build(3, "3.0", "f" * 64)
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
