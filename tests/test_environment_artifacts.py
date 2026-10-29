"""Discovery and acquisition consume the publishers' existing manifest formats."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import py7zr
import pytest
from dinkster_registry import LockedPack, Lockfile, build_artifact
from dinkster_supervisor.environment_artifacts import acquire, acquire_sources, discover
from dinkster_supervisor.environment_downloads import DownloadError
from dinkster_supervisor.environment_state import EnvironmentError

from dinkster.generation_packs import stage_generation_packs


def _json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _artifact(name: str, data: bytes, *, key: str = "name") -> dict:
    return {key: name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _mirror(
    root: Path, *, protocol: int = 11, damaged_archive: bool = False, variant: str = "fixture-cpu"
) -> tuple[dict, dict, dict]:
    root.mkdir(parents=True)
    python = "python/python.exe" if os.name == "nt" else "python/bin/python3"
    base_wheel = b"base wheel bytes; execution is covered by generation tests"
    wheel = {
        **_artifact("fixture_base-1.0-py3-none-any.whl", base_wheel, key="filename"),
        "name": "fixture-base",
        "version": "1.0",
        "lockedVersion": "1.0",
        "index": "https://pypi.org/simple",
        "url": "https://example.org/fixture.whl",
        "verification": "lock",
    }
    interpreter = {
        **_artifact("interpreter.tar.gz", b"unused source archive", key="filename"),
        "release": "fixture",
        "url": "https://example.org/interpreter.tar.gz",
        "executable": python,
    }
    base = {
        "format": "dinkster.base/1",
        "variant": variant,
        "platform": {"os": "windows" if os.name == "nt" else "linux", "arch": "x86_64"},
        "python": "3.14.10",
        "interpreter": interpreter,
        "wheels": [wheel],
        "replaces": ["fixture-base", "nvidia-cublas"],
        "workerProtocol": protocol,
        "nvidia": None,
    }
    base_id = hashlib.sha256(_json(base)).hexdigest()
    archive = root / "fixture.7z"
    with py7zr.SevenZipFile(archive, "w") as output:
        output.writestr(b"interpreter fixture, not executable", python)
        output.writestr(b"wrong" if damaged_archive else base_wheel, "wheels/" + wheel["filename"])
    data = archive.read_bytes()
    archive.unlink()
    base_path = root / f"base/{variant}/{base_id}"
    base_path.mkdir(parents=True)
    parts = (data[: len(data) // 2], data[len(data) // 2 :])
    base["parts"] = []
    for index, part in enumerate(parts, 1):
        name = f"base.7z.{index:03}"
        (base_path / name).write_bytes(part)
        base["parts"].append(_artifact(name, part, key="filename"))
    base["id"], base["signature"] = "sha256:" + base_id, None
    (base_path / "manifest.json").write_bytes(_json(base) + b"\n")
    engine_wheel = b"engine wheel bytes"
    constraints = (
        "fixture-base==1.0 --hash=sha256:" + "a" * 64 + "\n"
        "nvidia-cublas==13.0 --hash=sha256:" + "c" * 64 + "\n"
        "fixture-engine==2.0 --hash=sha256:" + hashlib.sha256(engine_wheel).hexdigest() + "\n"
    ).encode("ascii")
    engine = {
        "format": "dinkster.engine/1",
        "repository": "Kosinkadink/Dinkster",
        "tag": "main-" + "d" * 40,
        "commit": "d" * 40,
        "version": "0.4.0.dev1",
        "workerProtocol": 11,
        "artifacts": [
            _artifact("constraints.txt", constraints),
            _artifact("fixture_engine-2.0-py3-none-any.whl", engine_wheel),
        ],
    }
    engine_data = _json(engine) + b"\n"
    release = hashlib.sha256(engine_data).hexdigest()
    engine_path = root / f"engine/{release}"
    engine_path.mkdir(parents=True)
    (engine_path / "release-manifest.json").write_bytes(engine_data)
    (engine_path / "constraints.txt").write_bytes(constraints)
    (engine_path / "fixture_engine-2.0-py3-none-any.whl").write_bytes(engine_wheel)
    channel = {
        "format": "dinkster.channel/1",
        "engine": {
            "version": engine["version"],
            "release": "sha256:" + release,
            "locations": [{"type": "mirror", "path": f"engine/{release}/release-manifest.json"}],
        },
        "base": {
            variant: {
                "id": base["id"],
                "locations": [
                    {"type": "mirror", "path": f"base/{variant}/{base_id}/manifest.json"}
                ],
            }
        },
        "minimumDesktopVersion": "0.2.0",
        "signature": None,
    }
    (root / "channels").mkdir()
    for name in ("stable", "main"):
        (root / f"channels/{name}.json").write_bytes(_json(channel))
    return base, engine, channel


def _release_sources(mirror: Path) -> tuple[Path, dict[str, bytes]]:
    channel_path = mirror / "channels/main.json"
    channel = json.loads(channel_path.read_bytes())
    previous = mirror / "engine" / channel["engine"]["release"][7:]
    engine = json.loads((previous / "release-manifest.json").read_bytes())
    entries, payloads = [], {}
    for name in ("first-pack", "second-pack"):
        source = mirror / name
        source.mkdir()
        (source / "dinkster-pack.toml").write_text(
            f'[pack]\nname = "{name}"\npublisher = "dinkster"\n'
            f'namespaces = ["{name}"]\n[pack.entry]\nnodes = "nodes:NODES"\n'
        )
        (source / "nodes.py").write_text(f'VALUE = "{name}"\n')
        archive = mirror / f"{name}-1.0.zip"
        identity = build_artifact(source, archive)
        entries.append(LockedPack(name, "1.0", identity, "dinkster", (name,)))
        payloads[archive.name] = archive.read_bytes()
    payloads["official-packs.lock"] = (Lockfile.of(entries).record_json() + "\n").encode()
    engine["artifacts"].extend(_artifact(name, data) for name, data in payloads.items())
    data = _json(engine) + b"\n"
    identity = hashlib.sha256(data).hexdigest()
    release = mirror / "engine" / identity
    shutil.copytree(previous, release)
    (release / "release-manifest.json").write_bytes(data)
    for name, payload in payloads.items():
        (release / name).write_bytes(payload)
    channel["engine"]["release"] = "sha256:" + identity
    channel["engine"]["locations"] = [
        {"type": "mirror", "path": f"engine/{identity}/release-manifest.json"}
    ]
    channel_path.write_bytes(_json(channel))
    return release, payloads


def test_acquires_release_sources_into_the_existing_artifact_identity_cache(tmp_path: Path):
    mirror, store = tmp_path / "mirror", tmp_path / "store"
    _mirror(mirror)
    release, payloads = _release_sources(mirror)
    catalog = discover((str(mirror),), "main")
    with pytest.raises(EnvironmentError, match="intact cached release source"):
        acquire_sources(catalog, store, download=False)
    assert not store.exists()
    acquired = acquire_sources(catalog, store)
    assert acquired is not None
    lock, artifacts = acquired
    assert lock.read_bytes() == payloads["official-packs.lock"]
    entries = Lockfile.from_record_json(lock.read_text()).packs
    assert len(entries) == 2
    assert acquire_sources(catalog, store, download=False) == acquired
    for entry in entries:
        cached = artifacts / f"{entry.artifact_digest.removeprefix('blake3:')}.zip"
        assert cached.read_bytes() == payloads[f"{entry.pack}-1.0.zip"]
        cached.write_bytes(b"damaged cache")
    with pytest.raises(EnvironmentError, match="intact cached release source"):
        acquire_sources(catalog, store, download=False)
    for entry in entries:
        cached = artifacts / f"{entry.artifact_digest.removeprefix('blake3:')}.zip"
        assert cached.read_bytes() == b"damaged cache"
    acquire_sources(catalog, store)
    for entry in entries:
        cached = artifacts / f"{entry.artifact_digest.removeprefix('blake3:')}.zip"
        assert cached.read_bytes() == payloads[f"{entry.pack}-1.0.zip"]
    target = tmp_path / "target"
    target.mkdir()
    stage_generation_packs(
        target,
        Lockfile.from_record_json(lock.read_text()),
        {
            entry.pack: artifacts / f"{entry.artifact_digest.removeprefix('blake3:')}.zip"
            for entry in entries
        },
    )
    assert (target / "dinkster.lock").read_bytes() == payloads["official-packs.lock"]
    for entry in entries:
        assert (
            target / "pack-sources" / entry.pack / "nodes.py"
        ).read_text() == f'VALUE = "{entry.pack}"\n'
    (release / "first-pack-1.0.zip").write_bytes(b"damaged release")
    with pytest.raises(DownloadError, match="could not be downloaded"):
        acquire_sources(catalog, tmp_path / "fresh-store")
    assert not list((tmp_path / "fresh-store").rglob("*.zip"))


@pytest.mark.parametrize("channel_name", ["stable", "main"])
def test_discovers_intact_manifests_without_creating_cache_then_acquires_exact_bytes(
    tmp_path: Path,
    channel_name: str,
):
    mirror, store = tmp_path / "mirror", tmp_path / "data"
    base, engine, channel = _mirror(mirror)
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    catalog = discover((str(mirror),), channel_name)
    assert catalog.document == channel
    assert catalog.engine == engine
    assert catalog.variants() == [
        {
            "variant": "fixture-cpu",
            "base": base["id"],
            "manifest": base,
            "downloadBytes": sum(row["size"] for row in base["parts"]),
        }
    ]
    assert sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*")) == before
    constraints = acquire(catalog, "fixture-cpu", store)
    assert "nvidia-cublas" not in constraints.read_text()
    assert (
        f"fixture-base==1.0 --hash=sha256:{base['wheels'][0]['sha256']}" in constraints.read_text()
    )
    acquired = sorted(store.glob("wheels/*/*.whl"))
    assert sorted(path.read_bytes() for path in acquired) == sorted(
        [b"base wheel bytes; execution is covered by generation tests", b"engine wheel bytes"]
    )
    for path in acquired:
        assert path.parent.name == hashlib.sha256(path.read_bytes()).hexdigest()


def test_corrupt_manifest_and_wheel_fall_back_across_configured_mirrors(tmp_path: Path):
    first, second = tmp_path / "first", tmp_path / "second"
    _, engine, channel = _mirror(first)
    _mirror(second)
    # Reuse the exact same base archive/manifest since archive part identity is not deterministic.
    import shutil

    shutil.rmtree(second)
    shutil.copytree(first, second)
    release = channel["engine"]["release"][7:]
    (first / f"engine/{release}/release-manifest.json").write_bytes(b"corrupt")
    (first / f"engine/{release}/fixture_engine-2.0-py3-none-any.whl").write_bytes(b"short")
    catalog = discover((str(first), str(second)), "main")
    assert catalog.engine == engine
    acquire(catalog, "fixture-cpu", tmp_path / "store")
    assert (
        next((tmp_path / "store/wheels").glob("*/fixture_engine*.whl")).read_bytes()
        == b"engine wheel bytes"
    )


def test_refuses_protocol_mismatch_before_creating_a_store(tmp_path: Path):
    _mirror(tmp_path / "mirror", protocol=12)
    with pytest.raises(EnvironmentError, match="workerProtocol mismatch"):
        discover((str(tmp_path / "mirror"),), "main")


def test_base_identity_cannot_ignore_changed_replacement_membership(tmp_path: Path):
    base, _, _ = _mirror(tmp_path / "mirror")
    path = tmp_path / "mirror/base/fixture-cpu" / base["id"][7:] / "manifest.json"
    base["replaces"] = ["fixture-base"]
    path.write_bytes(_json(base))
    with pytest.raises(EnvironmentError, match="intact expected artifact manifest"):
        discover((str(tmp_path / "mirror"),), "main")


def test_unsigned_external_channel_is_refused_before_network(monkeypatch: pytest.MonkeyPatch):
    def fail(*args, **kwargs):
        pytest.fail("untrusted channel attempted network access")

    monkeypatch.setattr("dinkster_supervisor.environment_artifacts.read_metadata", fail)
    with pytest.raises(EnvironmentError, match="valid trusted channel"):
        discover(("https://not-dinkster.example",), "stable")


def test_part_hashes_do_not_substitute_for_verifying_extracted_wheel_hashes(tmp_path: Path):
    base, _, _ = _mirror(tmp_path / "mirror", damaged_archive=True)
    catalog = discover((str(tmp_path / "mirror"),), "main")
    store = tmp_path / "store"
    with pytest.raises(EnvironmentError, match="extracted base wheel differs"):
        acquire(catalog, "fixture-cpu", store)
    assert not (store / "bases" / base["id"][7:]).exists()
    assert not list(store.glob("wheels/*/*.whl"))
