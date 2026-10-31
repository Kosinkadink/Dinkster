"""Discovery and acquisition consume the publishers' existing manifest formats."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from platform import machine
from types import SimpleNamespace

import pytest
import zstandard
from dinkster_registry import LockedPack, Lockfile, build_artifact
from dinkster_supervisor import environment_artifacts
from dinkster_supervisor.environment_artifacts import (
    acquire,
    acquire_pack_locks,
    acquire_sources,
    discover,
)
from dinkster_supervisor.environment_channels import write_channel
from dinkster_supervisor.environment_downloads import DownloadError
from dinkster_supervisor.environment_state import EnvironmentError
from test_environment_generations import _standalone_tree

from dinkster.generation_packs import stage_generation_packs


def _json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def _artifact(name: str, data: bytes, *, key: str = "name") -> dict:
    return {key: name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _mirror(
    root: Path,
    *,
    protocol: int = 11,
    damaged_archive: bool = False,
    variant: str = "fixture-cpu",
    target: tuple[str, str] | None = None,
) -> tuple[dict, dict, dict]:
    root.mkdir(parents=True)
    system, arch = target or (
        "windows" if os.name == "nt" else "macos" if sys.platform == "darwin" else "linux",
        {"amd64": "x86_64", "aarch64": "arm64"}.get(machine().lower(), machine().lower()),
    )
    python = "python/python.exe" if system == "windows" else "python/bin/python3"
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
        "platform": {"os": system, "arch": arch},
        "python": "3.14.10",
        "interpreter": interpreter,
        "wheels": [wheel],
        "replaces": ["fixture-base", "nvidia-cublas"],
        "workerProtocol": protocol,
        "nvidia": None,
        "archive": {"format": "tar+zstd"},
    }
    base_id = hashlib.sha256(_json(base)).hexdigest()
    tree = root / "fixture-tree"
    _standalone_tree(tree / "python")
    (tree / "wheels").mkdir()
    (tree / "wheels" / wheel["filename"]).write_bytes(b"wrong" if damaged_archive else base_wheel)
    compressed = io.BytesIO()
    with zstandard.ZstdCompressor().stream_writer(compressed, closefd=False) as encoded:
        with tarfile.open(fileobj=encoded, mode="w|", dereference=False) as archive:
            archive.add(tree / "python", "python")
            archive.add(tree / "wheels", "wheels")
    data = compressed.getvalue()
    shutil.rmtree(tree)
    base_path = root / f"base/{variant}/{base_id}"
    base_path.mkdir(parents=True)
    parts = (data[: len(data) // 2], data[len(data) // 2 :])
    base["archive"]["parts"] = []
    for index, part in enumerate(parts, 1):
        name = f"base.tar.zst.{index:03}"
        (base_path / name).write_bytes(part)
        base["archive"]["parts"].append(_artifact(name, part, key="filename"))
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


def _release_sources(
    mirror: Path, *, dependency_locks: dict[str, bytes] | None = None
) -> tuple[Path, dict[str, bytes]]:
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
    payloads.update(dependency_locks or {})
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


def test_acquires_only_selected_variant_dependency_locks(tmp_path: Path):
    from dinkster_workers.pack_lock import PackLock

    mirror, store = tmp_path / "mirror", tmp_path / "store"
    _mirror(mirror)
    payloads = {}
    for pack, variant in (
        ("first-pack", "fixture-cpu"),
        ("second-pack", "fixture-cpu"),
        ("first-pack", "win-cpu"),
    ):
        path = tmp_path / f"{pack}-0.4.0.dev1-{variant}.pack-lock.json"
        PackLock(pack, variant, ">=3.12", (), (), ()).write(path)
        payloads[path.name] = path.read_bytes()
    release, _ = _release_sources(mirror, dependency_locks=payloads)
    (release / "first-pack-0.4.0.dev1-win-cpu.pack-lock.json").write_bytes(b"unselected corruption")
    catalog = discover((str(mirror),), "main")
    acquired = acquire_pack_locks(catalog, "fixture-cpu", store)
    assert {path.name for path in acquired} == {
        "first-pack-0.4.0.dev1-fixture-cpu.pack-lock.json",
        "second-pack-0.4.0.dev1-fixture-cpu.pack-lock.json",
    }
    for path in acquired:
        assert path.read_bytes() == payloads[path.name]
        assert PackLock.read(path).packages == ()
    assert not list(store.rglob("*win-cpu.pack-lock.json"))
    acquired[0].write_bytes(b"damaged cache")
    acquire_pack_locks(catalog, "fixture-cpu", store)
    assert acquired[0].read_bytes() == payloads[acquired[0].name]
    (release / acquired[0].name).write_bytes(b"damaged selected release")
    with pytest.raises(DownloadError):
        acquire_pack_locks(catalog, "fixture-cpu", tmp_path / "fresh-store")


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
            "downloadBytes": sum(row["size"] for row in base["archive"]["parts"]),
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


@pytest.mark.parametrize(
    ("target", "host", "architecture", "accepted"),
    [
        (("macos", "arm64"), "darwin", "arm64", True),
        (("macos", "arm64"), "darwin", "aarch64", True),
        (("macos", "arm64"), "darwin", "x86_64", False),
        (("macos", "arm64"), "win32", "AMD64", False),
        (("macos", "arm64"), "win32", "ARM64", False),
        (("macos", "arm64"), "linux", "x86_64", False),
        (("macos", "arm64"), "linux", "aarch64", False),
        (("macos", "x86_64"), "darwin", "arm64", False),
        (("windows", "x86_64"), "darwin", "arm64", False),
        (("linux", "x86_64"), "darwin", "arm64", False),
    ],
)
def test_acquisition_requires_manifest_os_and_architecture_to_match_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: tuple[str, str],
    host: str,
    architecture: str,
    accepted: bool,
):
    mirror, store = tmp_path / "mirror", tmp_path / "store"
    base, _, _ = _mirror(
        mirror, variant="mac-arm64" if target[0] == "macos" else "fixture-cpu", target=target
    )
    catalog = discover((str(mirror),), "main")
    monkeypatch.setattr(environment_artifacts, "sys", SimpleNamespace(platform=host))
    monkeypatch.setattr(environment_artifacts, "machine", lambda: architecture)
    variant = next(iter(catalog.bases))
    if not accepted:
        with pytest.raises(EnvironmentError, match="operating system|architecture"):
            acquire(catalog, variant, store)
        assert not store.exists()
        return
    # Exercise host admission without trying to execute a foreign interpreter on the test host.
    cached_wheels = store / "bases" / base["id"][7:] / "wheels"
    cached_wheels.mkdir(parents=True)
    (cached_wheels / base["wheels"][0]["filename"]).write_bytes(
        b"base wheel bytes; execution is covered by generation tests"
    )
    constraints = acquire(catalog, variant, store)
    assert "fixture-base==1.0 --hash=sha256:" in constraints.read_text()
    assert "nvidia-cublas" not in constraints.read_text()
    assert next(store.glob("wheels/*/fixture_base*.whl")).read_bytes() == (
        b"base wheel bytes; execution is covered by generation tests"
    )


@pytest.mark.parametrize("target", [("windows", "arm64"), ("linux", "arm64"), ("macos", "riscv64")])
def test_discovery_rejects_unsupported_manifest_architectures(
    tmp_path: Path, target: tuple[str, str]
):
    mirror = tmp_path / "mirror"
    _mirror(mirror, target=target)
    with pytest.raises(EnvironmentError, match="intact expected artifact manifest"):
        discover((str(mirror),), "main")


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


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "base",
        "engine",
        "github",
        "head",
        "config",
        "event",
        "superseded",
        "identical",
        "absent",
        "diverged",
        "compare",
        "unknown",
        "channel",
        "identical-bytes",
        "upload-recover",
        "upload-exhaust",
        "upload-forbidden",
        "upload-corrupt",
        "channel-recover",
        "collision",
        "partial",
    ],
)
@pytest.mark.parametrize("labelled", [False, True])
def test_channel_publication_exposes_only_verified_payloads(
    tmp_path: Path, monkeypatch, request, capsys, failure, labelled
):
    from boto3.exceptions import S3UploadFailedError
    from botocore.exceptions import ClientError

    from scripts import publish_environment_channel as publisher

    source, public = tmp_path / "source", tmp_path / "public"
    base, engine, channel = _mirror(source)
    release = source / "engine" / channel["engine"]["release"][7:]
    (release / "SHA256SUMS").write_bytes(b"fixture checksums")
    base_path = source / "base/fixture-cpu" / base["id"][7:] / "manifest.json"
    if labelled:
        engine.pop("tag")
        engine.update(label="2026-10-31-0.4.0.dev1", builtAt="2026-10-31T12:00:00Z")
        (release / "release-manifest.json").write_bytes(_json(engine) + b"\n")
        base["label"] = f"2026-10-30-py3.14.10-torch2.14.1-{base['id'][7:15]}"
        base_path.write_bytes(_json(base))
        target = base_path.parent.with_name(base["label"])
        base_path.parent.rename(target)
        base_path = target / "manifest.json"
    shutil.copytree(source / "base", public / "base")
    (public / "channels").mkdir()
    stable = b"stable channel must not change"
    (public / "channels/stable.json").write_bytes(stable)
    write_channel(
        release, [base_path], channel="main", output_root=public, minimum_desktop_version="0.2.0"
    )
    published_commit = engine["commit"]
    if failure not in {"identical", "identical-bytes"}:
        published_engine = {**engine, "commit": "e" * 40}
        published_commit = published_engine["commit"]
        data = _json(published_engine)
        identity = hashlib.sha256(data).hexdigest()
        name = f"{engine['label']}-{identity[:8]}" if labelled else identity
        target = public / f"engine/{name}/release-manifest.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(data)
        document = json.loads((public / "channels/main.json").read_bytes())
        document["engine"]["release"] = "sha256:" + identity
        document["engine"]["locations"] = [
            {"type": "mirror", "path": f"engine/{name}/release-manifest.json"}
        ]
        (public / "channels/main.json").write_bytes(_json(document))
    elif failure == "identical-bytes":
        document = json.loads((public / "channels/main.json").read_bytes())
        document["minimumDesktopVersion"] = "0.1.0"
        (public / "channels/main.json").write_bytes(_json(document))
    previous = (public / "channels/main.json").read_bytes()
    if failure == "absent":
        (public / "channels/main.json").unlink()
    elif failure == "collision":
        short = hashlib.sha256((release / "release-manifest.json").read_bytes()).hexdigest()[:8]
        orphan = public / f"engine/2026-10-29-0.4.0-{short}/constraints.txt"
        orphan.parent.mkdir(parents=True)
        orphan.write_bytes(b"interrupted unknown publication")
    elif failure == "partial":
        sha = hashlib.sha256((release / "release-manifest.json").read_bytes()).hexdigest()
        name = f"{engine['label']}-{sha[:8]}" if labelled else sha
        unfinished = public / f"engine/{name}/release-manifest.json"
        unfinished.parent.mkdir(parents=True)
        shutil.copyfile(release / unfinished.name, unfinished)
        (public / "engine/index.json").write_bytes(b"interrupted stale index")

    class MirrorHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if failure == "channel" and self.path == "/channels/main.json":
                self.send_error(403)
                return
            if self.headers.get("User-Agent", "").startswith("Python-urllib/"):
                self.send_error(403)
                return
            super().do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(MirrorHandler, directory=str(public)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def cleanup():
        server.shutdown()
        thread.join()
        server.server_close()

    request.addfinalizer(cleanup)
    for name, value in {
        "GITHUB_REPOSITORY": "Kosinkadink/Dinkster",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_SHA": engine["commit"],
        "R2_ACCESS_KEY_ID": "test",
        "R2_SECRET_ACCESS_KEY": "test",
        "R2_ACCOUNT_ID": "test",
        "R2_BUCKET": "test",
        "R2_PUBLIC_BASE_URL": f"http://127.0.0.1:{server.server_port}",
    }.items():
        monkeypatch.setenv(name, value)
    if failure == "config":
        monkeypatch.delenv("R2_SECRET_ACCESS_KEY")
    elif failure == "event":
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    elif failure == "base":
        (public / base_path.relative_to(source)).write_bytes(b"wrong base manifest")
    original_urlopen = publisher.urlopen

    def published_input(url, **kwargs):
        if url.startswith("https://api.github.com/"):
            assert url == (
                "https://api.github.com/repos/Kosinkadink/Dinkster/compare/"
                f"{published_commit}...{engine['commit']}"
            )
            if failure == "compare":
                raise ValueError("compare unavailable")
            status = (
                "identical"
                if published_commit == engine["commit"]
                else {"superseded": "behind", "diverged": "diverged", "unknown": "unknown"}.get(
                    failure, "ahead"
                )
            )
            return io.BytesIO(json.dumps({"status": status}).encode())
        if url.startswith("https://github.com/"):
            return io.BytesIO(
                b"wrong"
                if failure == "github"
                else (release / "release-manifest.json").read_bytes()
            )
        return original_urlopen(url, **kwargs)

    attempts = {}
    active = set()
    sleeps = []

    def backoff(delay):
        assert not active, "application retry overlapped an unfinished object transfer"
        sleeps.append(delay)

    monkeypatch.setattr(publisher.time, "sleep", backoff)

    class ObjectStore:
        def get_paginator(self, operation):
            assert operation == "list_objects_v2"
            return self

        def paginate(self, *, Bucket, Prefix):
            rows = [
                {"Key": path.relative_to(public).as_posix(), "Size": path.stat().st_size}
                for path in sorted((public / "engine").rglob("*"))
                if path.is_file()
            ]
            yield {"Contents": rows[:1]}
            yield {"Contents": rows[1:]}

        def get_object(self, *, Bucket, Key):
            return {"Body": io.BytesIO((public / Key).read_bytes())}

        def head_object(self, *, Bucket, Key):
            if failure == "head":
                raise ClientError({"ResponseMetadata": {"HTTPStatusCode": 403}}, "HeadObject")
            if not (public / Key).exists():
                raise ClientError({"ResponseMetadata": {"HTTPStatusCode": 404}}, "HeadObject")

        def upload_file(self, filename, bucket, key, **kwargs):
            if failure in {"identical", "identical-bytes"} and key == "channels/main.json":
                pytest.fail("identical engine must not upload the channel")
            assert key not in active
            active.add(key)
            try:
                attempts[key] = attempts.get(key, 0) + 1
                config = kwargs["Config"]
                assert config.max_concurrency == 1 and config.use_threads is False
                affected = (
                    failure
                    in {"upload-recover", "upload-exhaust", "upload-forbidden", "upload-corrupt"}
                    and key.endswith("/constraints.txt")
                ) or (failure == "channel-recover" and key == "channels/main.json")
                if affected and (
                    attempts[key] < 3 or failure in {"upload-exhaust", "upload-forbidden"}
                ):
                    error = ClientError(
                        {
                            "ResponseMetadata": {
                                "HTTPStatusCode": 403 if failure == "upload-forbidden" else 503
                            },
                            "Error": {
                                "Code": "AccessDenied"
                                if failure == "upload-forbidden"
                                else "ServiceUnavailable",
                                "Message": (
                                    "Reduce your concurrent request rate for the same object."
                                ),
                            },
                        },
                        "PutObject",
                    )
                    raise S3UploadFailedError("fixture upload failed") from error
            finally:
                active.remove(key)
            target = public / key
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(filename, target)
            if failure in {"engine", "upload-corrupt"} and target.name == "constraints.txt":
                target.write_bytes(b"corrupted uploaded engine bytes")

    monkeypatch.setattr(publisher, "urlopen", published_input)
    monkeypatch.setattr(publisher.boto3, "client", lambda *args, **kwargs: ObjectStore())
    if failure in {
        "base",
        "engine",
        "github",
        "head",
        "event",
        "compare",
        "unknown",
        "channel",
        "identical-bytes",
        "upload-exhaust",
        "upload-forbidden",
        "upload-corrupt",
        "collision",
    }:
        from urllib.error import HTTPError

        with pytest.raises((ValueError, ClientError, HTTPError, S3UploadFailedError)):
            publisher.publish(release, [base_path], "main", "0.2.0")
    else:
        publisher.publish(release, [base_path], "main", "0.2.0")
    assert (public / "channels/stable.json").read_bytes() == stable
    if failure == "collision":
        assert attempts == {}
    elif failure == "partial":
        assert not any(key.endswith("/release-manifest.json") for key in attempts)
    if failure in {
        "upload-recover",
        "upload-exhaust",
        "upload-forbidden",
        "upload-corrupt",
        "channel-recover",
    }:
        key = (
            "channels/main.json"
            if failure == "channel-recover"
            else next(key for key in attempts if key.endswith("/constraints.txt"))
        )
        assert attempts[key] == {"upload-exhaust": 4, "upload-forbidden": 1}.get(failure, 3)
        if failure == "upload-forbidden":
            assert sleeps == []
        else:
            assert sleeps[:2] == [1, 2]
        if failure == "upload-exhaust":
            assert sleeps == [1, 2, 4]
        if failure in {"upload-exhaust", "upload-forbidden", "upload-corrupt"}:
            assert "channels/main.json" not in attempts
    if failure not in {None, "identical", "absent", "upload-recover", "channel-recover", "partial"}:
        assert (public / "channels/main.json").read_bytes() == previous
        if failure in {"superseded", "diverged"}:
            output = capsys.readouterr().out
            assert published_commit in output
            assert engine["commit"] in output
    else:
        selected = discover((str(public),), "main")
        assert selected.engine == engine
        assert selected.bases == {"fixture-cpu": base}
        index = json.loads((public / "engine/index.json").read_bytes())
        full_id = hashlib.sha256((release / "release-manifest.json").read_bytes()).hexdigest()
        assert len(index) == 1
        assert index[0]["sha256"] == full_id
        assert index[0]["label"] == engine.get("label")
        assert index[0]["size"] == sum(path.stat().st_size for path in release.iterdir())
        before = {path: path.read_bytes() for path in public.rglob("*") if path.is_file()}
        published_commit = engine["commit"]
        publisher.publish(release, [base_path], "main", "0.2.0")
        assert {path: path.read_bytes() for path in before} == before


def test_engine_inventory_rebuilds_complete_index_and_refuses_real_hash8_collision(
    tmp_path: Path, monkeypatch
):
    from scripts import publish_environment_channel as publisher

    monkeypatch.setenv("R2_BUCKET", "fixture")
    objects: dict[str, bytes] = {}

    class Bucket:
        def get_paginator(self, operation):
            assert operation == "list_objects_v2"
            return self

        def paginate(self, **kwargs):
            rows = [{"Key": key, "Size": len(value)} for key, value in objects.items()]
            for row in rows:
                yield {"Contents": [row]}

        def get_object(self, *, Bucket, Key):
            return {"Body": io.BytesIO(objects[Key])}

        def upload_file(self, *args, **kwargs):
            pytest.fail("inventory and collision preflight must never upload")

    def manifest(commit: int, *, date: str = "2026-10-31", legacy: bool = False):
        engine = {
            "format": "dinkster.engine/1",
            "repository": "Kosinkadink/Dinkster",
            "label": f"{date}-0.4.2",
            "builtAt": f"{date}T12:00:00Z",
            "commit": f"{commit:040x}",
            "version": "0.4.2",
            "workerProtocol": 11,
            "artifacts": [_artifact("constraints.txt", b"")],
        }
        if legacy:
            del engine["label"], engine["builtAt"]
            engine["tag"] = f"main-{commit:040x}"
        data = _json(engine)
        digest = hashlib.sha256(data).hexdigest()
        name = digest if legacy else f"{engine['label']}-{digest[:8]}"
        return engine, data, digest, f"engine/{name}"

    first, first_data, first_id, first_path = manifest(42806)
    _, _, colliding_id, _ = manifest(54162)
    assert first_id != colliding_id and first_id[:8] == colliding_id[:8] == "49f9b0cc"
    objects[f"{first_path}/release-manifest.json"] = first_data
    bucket = Bucket()
    publisher.check_collision(bucket, "sha256:" + first_id)
    with pytest.raises(ValueError, match="hash8 collision"):
        publisher.check_collision(bucket, "sha256:" + colliding_id)
    del objects[f"{first_path}/release-manifest.json"]
    objects[f"{first_path}/constraints.txt"] = b""
    with pytest.raises(ValueError, match="unidentifiable directory"):
        publisher.check_collision(bucket, "sha256:" + first_id)
    objects[f"{first_path}/release-manifest.json"] = first_data
    objects[f"{first_path}/SHA256SUMS"] = b"checksums"
    _, old_data, old_id, old_path = manifest(1, legacy=True)
    objects.update(
        {
            f"{old_path}/release-manifest.json": old_data,
            f"{old_path}/constraints.txt": b"",
            f"{old_path}/SHA256SUMS": b"old",
        }
    )
    newer, new_data, new_id, new_path = manifest(2, date="2026-11-01")
    objects[f"{new_path}/release-manifest.json"] = new_data
    index = tmp_path / "index.json"
    index.write_text("stale or interrupted index")
    publisher.write_index(bucket, index)
    rows = json.loads(index.read_bytes())
    assert [row["sha256"] for row in rows] == [first_id, old_id]
    assert rows[0] == {
        "label": first["label"],
        "sha256": first_id,
        "commit": first["commit"],
        "version": "0.4.2",
        "builtAt": "2026-10-31T12:00:00Z",
        "size": len(first_data) + len(b"checksums"),
        "path": first_path,
    }
    assert rows[1]["label"] is None and rows[1]["builtAt"] is None
    objects[f"{new_path}/constraints.txt"] = b""
    objects[f"{new_path}/SHA256SUMS"] = b"new"
    publisher.write_index(bucket, index)
    rows = json.loads(index.read_bytes())
    assert [row["sha256"] for row in rows] == [new_id, first_id, old_id]
    assert rows[0]["label"] == newer["label"]
    before = index.read_bytes()
    publisher.write_index(bucket, index)
    assert index.read_bytes() == before


@pytest.mark.parametrize("labelled", [False, True])
def test_channel_builder_round_trips_both_channels_without_changing_publisher_inputs(
    tmp_path: Path,
    labelled: bool,
):
    mirror = tmp_path / "mirror"
    base, engine, channel = _mirror(mirror, variant="linux-cpu")
    second = tmp_path / "second"
    extra, _, _ = _mirror(second, variant="win-cpu")
    shutil.copytree(second / "base/win-cpu", mirror / "base/win-cpu")
    release = mirror / "engine" / channel["engine"]["release"][7:]
    bases = [
        mirror / "base" / variant / manifest["id"][7:] / "manifest.json"
        for variant, manifest in (("linux-cpu", base), ("win-cpu", extra))
    ]
    if labelled:
        engine.pop("tag")
        engine.update(label="2026-10-31-0.4.0.dev1", builtAt="2026-10-31T12:00:00Z")
        data = _json(engine) + b"\n"
        (release / "release-manifest.json").write_bytes(data)
        name = f"{engine['label']}-{hashlib.sha256(data).hexdigest()[:8]}"
        target = release.with_name(name)
        release.rename(target)
        release = target
    before = {
        path: path.read_bytes()
        for tree in (mirror / "engine", mirror / "base")
        for path in tree.rglob("*")
        if path.is_file()
    }
    expected_id = (
        "sha256:" + hashlib.sha256((release / "release-manifest.json").read_bytes()).hexdigest()
    )
    for name in ("main", "stable"):
        command = [
            sys.executable,
            "-m",
            "dinkster_supervisor.environment_channels",
            "--release",
            str(release),
            "--channel",
            name,
            "--output-root",
            str(mirror),
            "--minimum-desktop-version",
            "0.7.2",
        ]
        for path in reversed(bases):
            command.extend(["--base-manifest", str(path)])
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        generated = json.loads((mirror / "channels" / f"{name}.json").read_bytes())
        assert generated["engine"]["release"] == expected_id
        assert generated["engine"]["tag"] == (
            (f"v{engine['version']}" if name == "stable" else f"main-{release.name}")
            if labelled
            else engine["tag"]
        )
        assert generated["minimumDesktopVersion"] == "0.7.2"
        selected = discover((str(mirror),), name)
        assert selected.engine == engine
        assert selected.bases == {"linux-cpu": base, "win-cpu": extra}
        assert selected.channel.base_for("linux-cpu") == base["id"]
        assert selected.channel.base_for("win-cpu") == extra["id"]
        assert {path: path.read_bytes() for path in before} == before
    if not labelled:
        assert (mirror / "channels/main.json").read_bytes() == (
            mirror / "channels/stable.json"
        ).read_bytes()


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        ("constraints", "artifact constraints.txt differs"),
        ("wheel", "artifact fixture_engine.* differs"),
        ("base-id", "base manifest digest mismatch"),
        ("protocol", "workerProtocol mismatch"),
        ("version", "disagrees with engine lock"),
        ("duplicate", "duplicate selected base variant"),
        ("alias", "alias tags"),
        ("empty", "at least one base"),
    ],
)
def test_channel_builder_refuses_invalid_inputs_without_replacing_channels(
    tmp_path: Path, damage: str, reason: str
):
    mirror = tmp_path / "mirror"
    base, engine, channel = _mirror(mirror)
    release = mirror / "engine" / channel["engine"]["release"][7:]
    base_path = mirror / "base/fixture-cpu" / base["id"][7:] / "manifest.json"
    selected = [base_path]
    if damage == "constraints":
        (release / "constraints.txt").write_bytes(b"tampered constraints")
    elif damage == "wheel":
        wheel = release / "fixture_engine-2.0-py3-none-any.whl"
        wheel.write_bytes(b"x" * wheel.stat().st_size)
    elif damage == "base-id":
        base["python"] = "3.14.99"
        base_path.write_bytes(_json(base))
    elif damage == "protocol":
        engine["workerProtocol"] += 1
        (release / "release-manifest.json").write_bytes(_json(engine))
    elif damage == "version":
        constraints = (
            (release / "constraints.txt")
            .read_bytes()
            .replace(b"fixture-base==1.0", b"fixture-base==9.0")
        )
        (release / "constraints.txt").write_bytes(constraints)
        engine["artifacts"][0] = _artifact("constraints.txt", constraints)
        (release / "release-manifest.json").write_bytes(_json(engine))
    elif damage == "duplicate":
        selected.append(base_path)
    elif damage == "alias":
        engine["tag"] = "latest"
        (release / "release-manifest.json").write_bytes(_json(engine))
    else:
        selected.clear()
    before = {path: path.read_bytes() for path in (mirror / "channels").iterdir()}
    for output in (mirror, tmp_path / "absent"):
        with pytest.raises(EnvironmentError, match=reason):
            write_channel(
                release,
                selected,
                channel="main",
                output_root=output,
                minimum_desktop_version="0.2.0",
            )
        assert {path: path.read_bytes() for path in before} == before
        assert not (tmp_path / "absent").exists()
