"""Signed notices preserve authenticated blocks and exact decisions across restarts."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from uuid import UUID

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from dinkster_registry import InstallError, LockedPack, Lockfile, build_artifact
from dinkster_security.documents import Release, SecurityError
from dinkster_security.host import SecurityHost
from dinkster_security.policy import Artifact, assess
from dinkster_security.store import SecurityStore

REGISTRY = "https://registry.example.test"
NOTICE = "5b2fa364-f808-4dd2-b58e-e02187f9a2cc"
VERIFIED_AT = datetime(2026, 11, 10, 12, tzinfo=UTC)


def _key(key: Ed25519PrivateKey) -> dict[str, str]:
    public = key.public_key().public_bytes_raw()
    return {
        "keyId": "sha256:" + hashlib.sha256(public).hexdigest(),
        "publicKey": base64.b64encode(public).decode(),
    }


def _trust(key: Ed25519PrivateKey, revision: int = 1) -> dict[str, Any]:
    return {
        "format": "dinkster.security-trust/1",
        "registry": REGISTRY,
        "revision": revision,
        "issuedAt": "2026-11-01T00:00:00Z",
        "expiresAt": "2026-12-01T00:00:00Z",
        "keys": [_key(key)],
    }


def _feed(state: str = "vulnerable", revision: int = 1) -> dict[str, Any]:
    return {
        "format": "dinkster.security/1",
        "registry": REGISTRY,
        "revision": revision,
        "issuedAt": "2026-11-01T00:00:00Z",
        "expiresAt": "2026-11-02T00:00:00Z",
        "notices": [
            {
                "id": NOTICE,
                "revision": 1,
                "previousRevision": None,
                "release": {
                    "pack": "test-pack",
                    "version": "1.2.3",
                    "artifactDigest": "sha256:" + "a" * 64,
                },
                "state": state,
                "status": "active",
                "actor": "test-security-role",
                "authority": "security",
                "securityConfirmed": True,
                "changedAt": "2026-11-01T00:00:00Z",
                "reason": "test classification",
                "message": "benign signed fixture",
                "incident": "test-incident",
                "fixedRelease": None,
                "fixedLockDigest": None,
                "causes": [],
            }
        ],
        "ownershipChanges": [],
    }


def _signed(payload: dict[str, Any], key: Ed25519PrivateKey) -> bytes:
    # This fixture signs raw wire fields, independently of consumer serialization.
    message = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return json.dumps(
        {
            "signed": payload,
            "signatures": [
                {
                    "keyId": _key(key)["keyId"],
                    "signature": base64.b64encode(key.sign(message.encode("ascii"))).decode(),
                }
            ],
        }
    ).encode()


def _store(tmp_path: Path, key: Ed25519PrivateKey) -> SecurityStore:
    store = SecurityStore(tmp_path / "security.sqlite3")
    store.provision_trust(json.dumps(_trust(key)).encode())
    return store


@pytest.fixture
def security_feed() -> Iterator[tuple[str, dict[str, Any]]]:
    response: dict[str, Any] = {"status": 404, "body": b"", "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            response["requests"].append((self.path, self.headers.get("Authorization")))
            self.send_response(response["status"])
            self.end_headers()
            self.wfile.write(response["body"])

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", response
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_host_refresh_uses_independent_issuer_and_preserves_blocks_on_bad_feed(
    tmp_path: Path, security_feed: tuple[str, dict[str, Any]]
):
    url, response = security_feed
    key = Ed25519PrivateKey.generate()
    store = _store(tmp_path, key)
    (tmp_path / "security.toml").write_text(
        f'[feed_urls]\n"{REGISTRY}" = "{url}/security/index.json"\n'
    )
    host = SecurityHost(tmp_path)
    response.update(status=200, body=_signed(_feed("malicious"), key))
    host.refresh()
    document, verified = store.cached(REGISTRY)
    assert document is not None and verified is not None
    bad = Artifact(REGISTRY, document.notices[0].release)
    control = Artifact(
        REGISTRY, bad.release.model_copy(update={"artifact_digest": "sha256:" + "b" * 64})
    )
    assert not host.assess((bad,)).allowed
    assert host.assess((control,)).allowed
    assert response["requests"] == [("/security/index.json", None)]
    wrong_issuer = _feed("malicious")
    wrong_issuer["registry"] = "https://another.example.test"
    for payload in (
        b"{}",
        _signed(_feed(), Ed25519PrivateKey.generate()),
        _signed(wrong_issuer, key),
    ):
        response["body"] = payload
        host.refresh()
        assert store.cached(REGISTRY) == (document, verified)
        assert not host.assess((bad,)).allowed
        assert host.assess((control,)).allowed
    response["status"] = 503
    host.refresh()
    assert store.cached(REGISTRY) == (document, verified)
    assert not host.assess((bad,)).allowed
    assert host.assess((control,)).allowed
    lifted = _feed("malicious", revision=2)
    lifted["notices"][0].update(revision=2, previousRevision=1, status="cleared")
    response.update(status=200, body=_signed(lifted, key))
    host.refresh()
    assert host.assess((bad,)).allowed
    assert host.refresh_errors == {}
    response["body"] = _signed(_feed("malicious"), key)
    host.refresh()
    assert host.assess((bad,)).allowed
    current, _ = store.cached(REGISTRY)
    assert current is not None and current.revision == 2


def test_host_default_outage_continues_and_local_policy_is_explicit(
    tmp_path: Path, security_feed: tuple[str, dict[str, Any]]
):
    url, response = security_feed
    key = Ed25519PrivateKey.generate()
    store = SecurityStore(tmp_path / "security.sqlite3")
    trust = _trust(key)
    trust["registry"] = url
    store.provision_trust(json.dumps(trust).encode())
    host = SecurityHost(tmp_path)
    release = Artifact(url, Release.model_validate(_feed()["notices"][0]["release"]))
    host.refresh()
    assert host.assess((release,)).allowed
    assert not host.assess((release,)).findings
    assert response["requests"] == [("/v1/security/blocks", None)]
    feed = _feed()
    feed["registry"] = url
    response.update(status=200, body=_signed(feed, key))
    host.refresh()
    assert host.assess((release,)).allowed
    assert not host.assess((release,), installing=True).allowed
    config = tmp_path / "security.toml"
    config.write_text("refuse_vulnerable = true\n")
    host.refresh()
    assert not host.assess((release,)).allowed
    config.write_text("maximum_feed_age_seconds = 3600\n")
    host.refresh()
    assert host.assess((release,)).stale_registries == (url,)
    response["status"] = 404
    config.write_text("")
    host.refresh()
    assert host.assess((release,)).allowed
    assert host.assess((release,)).findings[0].decision == "pending"
    store.answer(url, UUID(NOTICE), 1, approved=False)
    assert not host.assess((release,)).allowed


@pytest.mark.parametrize(
    "config",
    [
        'refuse_vulnerable = "false"\n',
        "maximum_feed_age_seconds = 0\n",
        "request_timeout_seconds = inf\n",
        "misspelled_policy = true\n",
        '[feed_urls]\n"untrusted" = "http://127.0.0.1:1/security/index.json"\n',
    ],
)
def test_host_refuses_invalid_local_policy_before_transport(tmp_path: Path, config: str):
    (tmp_path / "security.toml").write_text(config)
    host = SecurityHost(tmp_path)
    with pytest.raises(ValueError):
        host.refresh()
    assert host.store.registries() == ()


@pytest.mark.parametrize("stop_reason", ["malicious", "denied"])
@pytest.mark.parametrize("shared", [False, True])
def test_composer_blocks_import_and_stops_pinned_execution_with_unaffected_control(
    tmp_path: Path, stop_reason: str, shared: bool
):
    from dinkster_protocol import Invocation
    from dinkster_workers.doctor import prepare_catalog

    from dinkster.compose import PackSpec, ServingComposer

    key = Ed25519PrivateKey.generate()
    store = _store(tmp_path, key)
    host = SecurityHost(tmp_path)
    specs = []
    roots = []
    names = (
        ("test-pack", "shared-pack", "control-pack") if shared else ("test-pack", "control-pack")
    )
    for name in names:
        root = tmp_path / name
        root.mkdir()
        marker = tmp_path / f"{name}-markers"
        marker.mkdir()
        roots.append(marker)
        module = "security_marker_" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
        manifest = root / "dinkster-pack.toml"
        manifest.write_text(f'[pack]\nname = "{name}"\n[pack.entry]\nnodes = "{module}:NODES"\n')
        (root / f"{module}.py").write_text(
            "import asyncio\nfrom pathlib import Path\n"
            "from dinkster_schema import Node, NodeSchema, OutputSpec, TypeExpr\n"
            "from dinkster_values import CORE_STRING\n"
            f"ROOT = Path({str(marker)!r})\n"
            "(ROOT / 'imported').write_text('imported')\n"
            "class Marker(Node):\n"
            "    @classmethod\n"
            "    def define_schema(cls):\n"
            f"        return NodeSchema(node_type='{name}.run', display_name='Marker', "
            "category='test', inputs=(), outputs=(OutputSpec('value', "
            "TypeExpr.concrete(CORE_STRING)),), io_bound=True)\n"
            "    @classmethod\n"
            "    async def execute(cls):\n"
            "        (ROOT / 'started').write_text('started')\n"
            "        while (ROOT / 'hold').exists() and not (ROOT / 'release').exists():\n"
            "            await asyncio.sleep(0.01)\n"
            "        with (ROOT / 'finished').open('a') as stream:\n"
            "            stream.write('finished\\n')\n"
            f"        return cls.outputs(value='{name}')\n"
            "NODES = [Marker]\n"
        )
        report = prepare_catalog(manifest, environment={"PYTHONPATH": str(root)})
        assert report.ok, report
        (marker / "imported").unlink()
        digest = build_artifact(root, tmp_path / f"{name}.zip")
        artifact = Artifact(
            REGISTRY,
            Release.model_validate(
                {
                    "pack": name,
                    "version": "1.2.3",
                    "artifactDigest": digest,
                }
            ),
        )
        specs.append(
            PackSpec(
                manifest,
                env={"PYTHONPATH": str(root)},
                require_catalog=True,
                security_artifacts=(artifact,),
            )
        )
    if shared:
        manifests = tuple(Path(spec.manifest) for spec in specs[:2])
        environment = {"PYTHONPATH": os.pathsep.join(str(path.parent) for path in manifests)}
        specs[:2] = [
            replace(spec, env=environment, worker_group="shared", group_manifests=manifests)
            for spec in specs[:2]
        ]
    feed = _feed("malicious")
    feed["notices"][0]["release"]["artifactDigest"] = (
        specs[0].security_artifacts[0].release.artifact_digest
    )
    store.import_feed(REGISTRY, _signed(feed, key))

    async def scenario() -> None:
        composer = ServingComposer(security=host)
        try:
            with pytest.raises(SecurityError, match="security malicious"):
                await composer.add_pack(specs[0])
            assert not (roots[0] / "imported").exists()
            await composer.add_pack(specs[-1])
            feed["revision"] = 2
            feed["notices"][0].update(revision=2, previousRevision=1, status="cleared")
            store.import_feed(REGISTRY, _signed(feed, key))
            await composer.add_pack(specs[0])
            if shared:
                await composer.add_pack(specs[1])
            assert not (roots[0] / "imported").exists()
            pinned = composer.composition._pin_execution()

            async def invoke(name: str):
                return await pinned.worker.invoke(
                    Invocation(
                        invocation_id=name,
                        node_id=name,
                        node_type=f"{name}.run",
                        inputs={},
                        effective_schema=composer.composition.schemas[f"{name}.run"],
                        executor=name,
                    )
                )

            (roots[0] / "hold").touch()
            running = asyncio.create_task(invoke("test-pack"))
            async with asyncio.timeout(15):
                while not (roots[0] / "started").exists():
                    await asyncio.sleep(0.01)
            sibling = None
            if shared:
                (roots[1] / "hold").touch()
                sibling = asyncio.create_task(invoke("shared-pack"))
                async with asyncio.timeout(15):
                    while not (roots[1] / "started").exists():
                        await asyncio.sleep(0.01)
            feed["revision"] = 3
            feed["notices"][0].update(
                revision=3, previousRevision=2, status="active", state="vulnerable"
            )
            store.import_feed(REGISTRY, _signed(feed, key))
            assert not await composer.enforce_security()
            assert not running.done()
            findings = composer.security_status()["findings"]
            assert isinstance(findings, list)
            assert findings[0]["decision"] == "pending"
            control = await invoke("control-pack")
            assert control.error is None and control.outputs is not None
            assert control.outputs["value"].resolve() == "control-pack"
            if stop_reason == "denied":
                store.answer(REGISTRY, UUID(NOTICE), 3, approved=False)
            else:
                feed["revision"] = 4
                feed["notices"][0].update(revision=4, previousRevision=3, state="malicious")
                store.import_feed(REGISTRY, _signed(feed, key))
            assert not await composer.enforce_security()
            await asyncio.wait_for(asyncio.gather(running, return_exceptions=True), 10)
            if sibling is not None:
                await asyncio.wait_for(asyncio.gather(sibling, return_exceptions=True), 10)
                assert not (roots[1] / "finished").exists()
            assert not (roots[0] / "finished").exists()
            with pytest.raises(SecurityError):
                await invoke("test-pack")
            control = await invoke("control-pack")
            assert control.error is None and control.outputs is not None
            assert control.outputs["value"].resolve() == "control-pack"
            assert (roots[-1] / "finished").read_text() == "finished\nfinished\n"
            feed["revision"] = 5
            previous = feed["notices"][0]["revision"]
            feed["notices"][0].update(revision=5, previousRevision=previous, status="cleared")
            store.import_feed(REGISTRY, _signed(feed, key))
            assert not composer.security_status()["findings"]
            assert not await composer.enforce_security()
            with pytest.raises(SecurityError, match="explicit restart"):
                await invoke("test-pack")
            assert not (roots[0] / "finished").exists()
        finally:
            await composer.close()

    asyncio.run(scenario())


@pytest.fixture
def execution_artifacts(
    tmp_path: Path,
) -> tuple[list[LockedPack], dict[str, Path], dict[str, Path]]:
    from dinkster_workers.doctor import prepare_catalog

    entries = []
    archives = {}
    markers = {}
    for name in ("test-pack", "control-pack"):
        source = tmp_path / name
        source.mkdir()
        marker = tmp_path / f"{name}-markers"
        marker.mkdir()
        markers[name] = marker
        module = name.replace("-", "_") + "_security_nodes"
        manifest = source / "dinkster-pack.toml"
        manifest.write_text(f'[pack]\nname = "{name}"\n[pack.entry]\nnodes = "{module}:NODES"\n')
        (source / f"{module}.py").write_text(
            "import asyncio\nfrom pathlib import Path\n"
            "from dinkster_schema import Node, NodeSchema, InputSpec, OutputSpec, TypeExpr\n"
            "from dinkster_values import CORE_STRING\n"
            f"ROOT = Path({str(marker)!r})\n"
            "(ROOT / 'imported').touch()\n"
            "class Marker(Node):\n"
            "    @classmethod\n"
            "    def define_schema(cls):\n"
            f"        return NodeSchema(node_type='{name}.run', display_name='Marker', "
            "category='test', inputs=(InputSpec('nonce', TypeExpr.concrete(CORE_STRING)),), "
            "outputs=(OutputSpec('value', "
            "TypeExpr.concrete(CORE_STRING)),), io_bound=True)\n"
            "    @classmethod\n"
            "    async def execute(cls, nonce):\n"
            "        (ROOT / 'started').touch()\n"
            "        while (ROOT / 'hold').exists():\n"
            "            await asyncio.sleep(0.01)\n"
            "        with (ROOT / 'finished').open('a') as stream:\n"
            "            stream.write('finished\\n')\n"
            f"        return cls.outputs(value='{name}')\n"
            "NODES = [Marker]\n"
        )
        assert prepare_catalog(manifest, environment={"PYTHONPATH": str(source)}).ok
        (marker / "imported").unlink()
        archive = tmp_path / f"{name}.zip"
        digest = build_artifact(source, archive)
        entries.append(
            LockedPack(name, "1.2.3", digest, "publisher", (name,), "registry:" + REGISTRY)
        )
        archives[digest] = archive
    return entries, archives, markers


def _execution_generation(
    root: Path,
    entries: list[LockedPack],
    archives: dict[str, Path],
    in_process: tuple[str, ...],
) -> Path:
    from dinkster_supervisor.environment_state import interpreter
    from test_environment_generations import _workspace_tree

    from dinkster.generation_packs import stage_generation_packs

    generation = root / "generations/1"
    generation.mkdir(parents=True)
    _workspace_tree(generation / "python")
    stage_generation_packs(
        generation,
        Lockfile.of(entries),
        {entry.pack: archives[entry.artifact_digest] for entry in entries},
    )
    receipt = json.dumps({"format": "dinkster.installed-wheels/1", "wheels": [], "inherited": []})
    receipts = ["installed-wheels.json"]
    for entry in entries:
        pack = generation / "packs" / entry.pack
        if entry.pack not in in_process:
            subprocess.run(
                [
                    str(interpreter(generation)),
                    "-m",
                    "venv",
                    "--system-site-packages",
                    "--without-pip",
                    str(pack),
                ],
                check=True,
                capture_output=True,
            )
        receipts.append(f"packs/{entry.pack}/installed-wheels.json")
    for name in receipts:
        path = generation / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(receipt)
    (generation / "constraints.txt").write_text("")
    files = {
        name: "sha256:" + hashlib.sha256((generation / name).read_bytes()).hexdigest()
        for name in ["constraints.txt", *receipts]
    }
    if set(in_process) != {entry.pack for entry in entries}:
        from dinkster_supervisor.environment_generations import prepare_pack_catalogs

        prepare_pack_catalogs(generation, in_process)
    (generation / "generation.json").write_text(
        json.dumps(
            {
                "format": "dinkster.generation/1",
                "generation": 1,
                "base": "sha256:" + "a" * 64,
                "engine": {"version": "test", "release": "sha256:" + "b" * 64},
                "constraints": files["constraints.txt"],
                "variant": "linux-cpu",
                "workerProtocol": 1,
                "inProcessPacks": list(in_process),
                "files": files,
            }
        )
    )
    (root / "current").write_text("1\n")
    return generation


@pytest.mark.parametrize(
    ("stop_reason", "in_process"), [("malicious", False), ("denied", False), ("malicious", True)]
)
def test_supervised_host_refreshes_pending_restart_and_stops_running_work(
    tmp_path: Path,
    security_feed: tuple[str, dict[str, Any]],
    execution_artifacts: tuple[list[LockedPack], dict[str, Path], dict[str, Path]],
    stop_reason: str,
    in_process: bool,
    capfd: pytest.CaptureFixture[str],
):
    import aiohttp
    from dinkster_graph import Graph, GraphNode, graph_to_wire
    from dinkster_supervisor.supervisor import EngineLink, EngineProcess

    entries, archives, markers = execution_artifacts
    root = tmp_path / "managed"
    generation = _execution_generation(
        root, entries, archives, ("test-pack",) if in_process else ()
    )
    key = Ed25519PrivateKey.generate()
    store = _store(root, key)
    url, response = security_feed
    (root / "security.toml").write_text(
        "refresh_interval_seconds = 0.05\n[feed_urls]\n"
        f'"{REGISTRY}" = "{url}/security/index.json"\n'
    )
    feed = _feed()
    feed["notices"][0]["release"]["artifactDigest"] = entries[0].artifact_digest
    response.update(status=200, body=_signed(feed, key))

    async def scenario() -> None:
        link = EngineLink()
        engine = EngineProcess(
            [
                sys.executable,
                "-m",
                "dinkster.serve",
                "--no-default-packs",
                "--library-root",
                "",
                "--generation",
                str(generation),
            ],
            link,
            probe_interval=0.05,
        )
        async with aiohttp.ClientSession() as client:

            async def ready() -> None:
                async with asyncio.timeout(60):
                    while True:
                        assert link.state != "failed", link.detail
                        if link.state == "ready":
                            async with client.get(link.base_url + "/api/nodes") as result:
                                nodes = await result.json()
                            if all(f"{name}.run" in nodes["nodes"] for name in markers):
                                return
                        await asyncio.sleep(0.02)

            async def submit(name: str, job: str) -> None:
                graph = Graph(nodes={"marker": GraphNode(f"{name}.run", {"nonce": job})})
                async with client.post(
                    link.base_url + "/api/jobs",
                    json={
                        "clientId": "security",
                        "jobId": job,
                        "graph": graph_to_wire(graph),
                        "targets": ["marker"],
                    },
                ) as result:
                    assert result.status == 202, await result.text()

            async def completed(job: str) -> None:
                async with asyncio.timeout(15):
                    while True:
                        async with client.get(
                            link.base_url + f"/api/jobs/security/{job}"
                        ) as result:
                            state = await result.json()
                        if state["state"] in {"completed", "failed", "cancelled"}:
                            assert state["state"] == "completed", state
                            return
                        await asyncio.sleep(0.02)

            try:
                await engine.start()
                await ready()
                async with client.get(link.base_url + "/api/security") as result:
                    status = await result.json()
                assert status["findings"][0]["decision"] == "pending"
                await submit("test-pack", "before-restart")
                await completed("before-restart")
                assert (markers["test-pack"] / "finished").read_text() == "finished\n"
                await engine.restart()
                await ready()
                await submit("test-pack", "after-restart")
                await completed("after-restart")
                assert (markers["test-pack"] / "finished").read_text() == "finished\nfinished\n"
                assert "decision pending, execution allowed" in capfd.readouterr().err
                await submit("control-pack", "before-block")
                await completed("before-block")
                assert (markers["control-pack"] / "finished").read_text() == "finished\n"
                (markers["test-pack"] / "started").unlink()
                (markers["test-pack"] / "hold").touch()
                await submit("test-pack", "running")
                async with asyncio.timeout(15):
                    while not (markers["test-pack"] / "started").exists():
                        await asyncio.sleep(0.02)
                if stop_reason == "denied":
                    store.answer(REGISTRY, UUID(NOTICE), 1, approved=False)
                else:
                    feed["revision"] = 2
                    feed["notices"][0].update(revision=2, previousRevision=1, state="malicious")
                    response["body"] = _signed(feed, key)
                if in_process:
                    async with asyncio.timeout(15):
                        while link.state != "failed":
                            await asyncio.sleep(0.02)
                    assert link.exit_code not in (None, 0)
                    assert (markers["test-pack"] / "finished").read_text() == "finished\nfinished\n"
                    assert (markers["control-pack"] / "finished").read_text() == "finished\n"
                    (markers["test-pack"] / "hold").unlink()
                    feed["revision"] = 3
                    feed["notices"][0].update(revision=3, previousRevision=2, status="cleared")
                    response["body"] = _signed(feed, key)
                    assert link.state == "failed"
                    await engine.restart()
                    await ready()
                    async with client.get(link.base_url + "/api/security") as result:
                        assert not (await result.json())["findings"]
                    await submit("test-pack", "explicit-restart")
                    await completed("explicit-restart")
                    assert (
                        markers["test-pack"] / "finished"
                    ).read_text() == "finished\nfinished\nfinished\n"
                    return
                async with asyncio.timeout(15):
                    while True:
                        async with client.get(link.base_url + "/api/security") as result:
                            status = await result.json()
                        if status["findings"] and not status["findings"][0]["allowed"]:
                            break
                        await asyncio.sleep(0.02)
                await submit("control-pack", "control")
                await completed("control")
                assert (markers["control-pack"] / "finished").read_text() == "finished\nfinished\n"
                async with asyncio.timeout(15):
                    while True:
                        async with client.get(
                            link.base_url + "/api/jobs/security/running"
                        ) as result:
                            state = await result.json()
                        if state["state"] in {"failed", "cancelled"}:
                            break
                        await asyncio.sleep(0.02)
                (markers["test-pack"] / "hold").unlink()
                feed["revision"] = 3
                previous = feed["notices"][0]["revision"]
                feed["notices"][0].update(revision=3, previousRevision=previous, status="cleared")
                response["body"] = _signed(feed, key)
                async with asyncio.timeout(15):
                    while True:
                        async with client.get(link.base_url + "/api/security") as result:
                            status = await result.json()
                        if not status["findings"]:
                            break
                        await asyncio.sleep(0.02)
                assert (markers["test-pack"] / "finished").read_text() == "finished\nfinished\n"
                await submit("test-pack", "after-lift")
                async with asyncio.timeout(15):
                    while True:
                        async with client.get(
                            link.base_url + "/api/jobs/security/after-lift"
                        ) as result:
                            state = await result.json()
                        if state["state"] in {"completed", "failed", "cancelled"}:
                            assert state["state"] == "failed", state
                            break
                        await asyncio.sleep(0.02)
                assert (markers["test-pack"] / "finished").read_text() == "finished\nfinished\n"
            finally:
                await engine.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("policy", ["pending", "outage", "malicious", "denied", "refuse", "age"])
def test_managed_generation_start_enforces_cached_and_refreshed_policy(
    tmp_path: Path,
    security_feed: tuple[str, dict[str, Any]],
    execution_artifacts: tuple[list[LockedPack], dict[str, Path], dict[str, Path]],
    policy: str,
):
    from dinkster_graph import Graph, GraphNode, graph_to_wire
    from dinkster_supervisor.environment_runtime import start_engine, stop_engine
    from dinkster_supervisor.environment_state import EnvironmentError
    from test_environment_runtime import free_port

    entries, archives, markers = execution_artifacts
    root = tmp_path / "managed"
    _execution_generation(root, entries, archives, tuple(entry.pack for entry in entries))
    key = Ed25519PrivateKey.generate()
    store = _store(root, key)
    url, response = security_feed
    config = (
        "refuse_vulnerable = true\n"
        if policy == "refuse"
        else "maximum_feed_age_seconds = 1\n"
        if policy == "age"
        else ""
    )
    (root / "security.toml").write_text(
        config + f'[feed_urls]\n"{REGISTRY}" = "{url}/security/index.json"\n'
    )
    feed = _feed("malicious" if policy == "malicious" else "vulnerable")
    feed["notices"][0]["release"]["artifactDigest"] = entries[0].artifact_digest
    if policy in {"outage", "denied"}:
        store.import_feed(REGISTRY, _signed(feed, key))
    if policy == "denied":
        store.answer(REGISTRY, UUID(NOTICE), 1, approved=False)
    response.update(status=503 if policy == "outage" else 200, body=_signed(feed, key))
    port = free_port()
    try:
        if policy not in {"pending", "outage"}:
            expected = (
                "maximum age"
                if policy == "age"
                else "security " + ("malicious" if policy == "malicious" else "vulnerable")
            )
            with pytest.raises(EnvironmentError, match=expected):
                start_engine(root, 1, "main", port)
            assert all(not (marker / "imported").exists() for marker in markers.values())
            assert store.cached(REGISTRY)[0] is not None
            return
        for attempt in range(2):
            row = start_engine(root, 1, "main", port)
            assert row["state"] == "ready"
            base = f"http://127.0.0.1:{port}"
            for name in markers:
                job = f"{name}-{attempt}"
                request = Request(
                    base + "/api/jobs",
                    data=json.dumps(
                        {
                            "clientId": "security",
                            "jobId": job,
                            "graph": graph_to_wire(
                                Graph(nodes={"marker": GraphNode(f"{name}.run", {"nonce": job})})
                            ),
                            "targets": ["marker"],
                        }
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(request, timeout=5) as result:  # noqa: S310
                    assert result.status == 202
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    with urlopen(base + f"/api/jobs/security/{job}", timeout=5) as result:  # noqa: S310
                        state = json.load(result)
                    if state["state"] in {"completed", "failed", "cancelled"}:
                        assert state["state"] == "completed", state
                        break
                    time.sleep(0.02)
                else:
                    pytest.fail("managed generation job did not finish")
                assert (markers[name] / "finished").read_text() == "finished\n" * (attempt + 1)
            with urlopen(base + "/api/security", timeout=5) as result:  # noqa: S310
                status = json.load(result)
            assert status["findings"][0]["decision"] == "pending"
            assert bool(status["refreshErrors"]) is (policy == "outage")
            stop_engine(root, "main")
    finally:
        stop_engine(root, "main")
    assert "decision pending, execution allowed" in (root / "runtime/1/main.log").read_text()


@pytest.mark.parametrize("state", ["malicious", "denied"])
def test_generation_catalog_preparation_refuses_signed_blocks_before_import(
    tmp_path: Path, security_feed: tuple[str, dict[str, Any]], state: str
):
    from dinkster_store import pack_archive_path
    from dinkster_supervisor.environment_generations import (
        prepare_pack_catalogs,
        restore_pack_sources,
    )
    from dinkster_supervisor.environment_state import EnvironmentError
    from test_doctor import HEALTHY_MANIFEST, HEALTHY_NODES
    from test_environment_generations import _workspace_tree

    from dinkster.generation_packs import stage_generation_packs

    generation = tmp_path / "generations/1"
    generation.mkdir(parents=True)
    _workspace_tree(generation / "python")
    source = tmp_path / "source"
    source.mkdir()
    marker = tmp_path / "imported"
    (source / "dinkster-pack.toml").write_text(HEALTHY_MANIFEST)
    (source / "healthy_nodes.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n" + HEALTHY_NODES
    )
    archive = tmp_path / "healthy.zip"
    identity = build_artifact(source, archive)
    entry = LockedPack(
        "healthy-nodes", "1.2.3", identity, "dinkster", ("healthy",), "registry:" + REGISTRY
    )
    stage_generation_packs(generation, Lockfile.of([entry]), {entry.pack: archive})
    receipt = json.dumps({"format": "dinkster.installed-wheels/1", "wheels": [], "inherited": []})
    (generation / "installed-wheels.json").write_text(receipt)
    pack = generation / "packs" / entry.pack
    pack.mkdir(parents=True)
    (pack / "installed-wheels.json").write_text(receipt)
    key = Ed25519PrivateKey.generate()
    store = _store(tmp_path, key)
    url, response = security_feed
    (tmp_path / "security.toml").write_text(
        f'[feed_urls]\n"{REGISTRY}" = "{url}/security/index.json"\n'
    )
    feed = _feed("malicious" if state == "malicious" else "vulnerable")
    feed["notices"][0]["release"].update(pack=entry.pack, artifactDigest=identity)
    if state == "denied":
        store.import_feed(REGISTRY, _signed(feed, key))
        store.answer(REGISTRY, UUID(NOTICE), 1, approved=False)
    response.update(status=200, body=_signed(feed, key))
    artifacts = tmp_path / "artifacts"
    retained = pack_archive_path(artifacts, identity)
    retained.parent.mkdir(parents=True)
    retained.write_bytes(archive.read_bytes())
    restore_pack_sources(
        generation,
        catalog_lock=generation / "dinkster.lock",
        artifacts=artifacts,
        project=tmp_path,
        store=tmp_path / "store",
    )
    assert not marker.exists()
    with pytest.raises(
        EnvironmentError,
        match="security " + ("malicious" if state == "malicious" else "vulnerable"),
    ):
        prepare_pack_catalogs(generation, (entry.pack,))
    assert not marker.exists()
    assert store.cached(REGISTRY)[0] is not None
    assert archive.is_file()
    assert not (generation / "generation.json").exists()
    feed["revision"] = 2
    feed["notices"][0].update(revision=2, previousRevision=1, status="cleared")
    response["body"] = _signed(feed, key)
    prepare_pack_catalogs(generation, (entry.pack,))
    assert marker.exists()


@pytest.mark.parametrize("previous_publisher", [None, "original-publisher"])
def test_signed_ownership_change_requires_explicit_previous_publisher(
    tmp_path: Path, previous_publisher: str | None
):
    key = Ed25519PrivateKey.generate()
    store = _store(tmp_path, key)
    original = _feed("malicious")
    store.import_feed(REGISTRY, _signed(original, key), now=VERIFIED_AT)
    change = {
        "kind": "ownership_transferred",
        "id": "636d54c0-ef6b-44cb-9ee3-bf7381bf0ed8",
        "revision": 2,
        "packId": "9aa02405-70c5-4648-9bf8-01341233759a",
        "pack": "test-pack",
        "previousPublisher": previous_publisher,
        "publisher": "new-publisher",
        "actor": "test-security-role",
        "changedAt": "2026-11-10T12:00:00Z",
        "reason": "continue maintained releases",
    }
    replacement = copy.deepcopy(original)
    replacement.update(revision=2, ownershipChanges=[change])
    invalid = [
        {name: value for name, value in change.items() if name != "previousPublisher"},
        {**change, "publisher": ""},
        {**change, "publisher": None},
        {**change, "previousPublisher": "new-publisher"},
    ]
    for candidate in invalid:
        replacement["ownershipChanges"] = [candidate]
        with pytest.raises(SecurityError):
            store.import_feed(REGISTRY, _signed(replacement, key))
        cached, verified = SecurityStore(store.path).cached(REGISTRY)
        assert cached is not None and cached.revision == 1
        assert cached.notices[0].state == "malicious"
        assert verified == VERIFIED_AT
    replacement["ownershipChanges"] = [change]
    store.import_feed(REGISTRY, _signed(replacement, key), now=VERIFIED_AT)
    cached, verified = SecurityStore(store.path).cached(REGISTRY)
    assert cached is not None and cached.revision == 2
    assert cached.ownership_changes[0].previous_publisher == previous_publisher
    assert cached.ownership_changes[0].publisher == "new-publisher"
    assert cached.notices[0].state == "malicious"
    assert verified == VERIFIED_AT


def test_cached_notice_and_changeable_exact_decision_survive_restart_and_lift(tmp_path: Path):
    key = Ed25519PrivateKey.generate()
    store = _store(tmp_path, key)
    original = _feed()
    store.import_feed(REGISTRY, _signed(original, key), now=VERIFIED_AT)
    notice = UUID(NOTICE)
    assert store.decision(REGISTRY, notice, 1) == "pending"
    store.answer(REGISTRY, notice, 1, approved=True)
    restarted = SecurityStore(store.path)
    assert restarted.decision(REGISTRY, notice, 1) == "approved"
    restarted.answer(REGISTRY, notice, 1, approved=False)
    assert store.decision(REGISTRY, notice, 1) == "denied"
    newer = copy.deepcopy(original)
    newer["revision"] = 2
    newer["notices"][0].update(revision=2, previousRevision=1, message="new finding")
    store.import_feed(REGISTRY, _signed(newer, key), now=VERIFIED_AT)
    assert store.decision(REGISTRY, notice, 2) == "pending"
    with pytest.raises(SecurityError, match="changed"):
        store.answer(REGISTRY, notice, 1, approved=True)
    newer["revision"] = 3
    newer["notices"][0].update(revision=3, previousRevision=2, status="cleared")
    store.import_feed(REGISTRY, _signed(newer, key), now=VERIFIED_AT)
    cached, last = SecurityStore(store.path).cached(REGISTRY)
    assert cached is not None and cached.notices[0].status == "cleared"
    assert last == VERIFIED_AT
    assert store.decision(REGISTRY, notice, 1) == "denied"
    with pytest.raises(SecurityError, match="active vulnerable"):
        store.answer(REGISTRY, notice, 3, approved=True)


@pytest.mark.parametrize(
    "attack",
    [
        "forged",
        "registry",
        "older",
        "dropped",
        "identity",
        "same-revision",
        "duplicate",
        "unknown",
        "bool",
    ],
)
def test_rejected_feed_preserves_cached_block_and_verification_time(tmp_path: Path, attack: str):
    key = Ed25519PrivateKey.generate()
    store = _store(tmp_path, key)
    original = _feed("malicious", revision=4)
    accepted = store.import_feed(REGISTRY, _signed(original, key), now=VERIFIED_AT)
    candidate = copy.deepcopy(original)
    candidate["revision"] = 5
    signer = key
    if attack == "forged":
        signer = Ed25519PrivateKey.generate()
    elif attack == "registry":
        candidate["registry"] = "https://other.example.test"
    elif attack == "older":
        candidate["revision"] = 3
    elif attack == "dropped":
        candidate["notices"] = []
    elif attack == "identity":
        candidate["notices"][0]["release"]["artifactDigest"] = "sha256:" + "b" * 64
    elif attack == "same-revision":
        candidate["revision"] = 4
        del candidate["notices"][0]["fixedRelease"]
    elif attack == "unknown":
        candidate["notices"][0]["continueMalicious"] = True
    elif attack == "bool":
        candidate["revision"] = True
    data = _signed(candidate, signer)
    if attack == "duplicate":
        data = data.replace(b'"revision": 5', b'"revision": 5, "revision": 6', 1)
    with pytest.raises(SecurityError):
        store.import_feed(REGISTRY, data)
    cached, last = SecurityStore(store.path).cached(REGISTRY)
    assert cached == accepted and last == VERIFIED_AT
    with pytest.raises(SecurityError, match="active vulnerable"):
        store.answer(REGISTRY, UUID(NOTICE), 1, approved=True)


def test_rotation_and_independent_recovery_preserve_blocks_decisions_and_checkpoint(tmp_path: Path):
    key, replacement, recovery = [Ed25519PrivateKey.generate() for _ in range(3)]
    store = _store(tmp_path, key)
    document = _feed()
    store.import_feed(REGISTRY, _signed(document, key), now=VERIFIED_AT)
    store.answer(REGISTRY, UUID(NOTICE), 1, approved=False)
    with pytest.raises(SecurityError):
        store.rotate_trust(REGISTRY, _signed(_trust(replacement, 2), replacement))
    store.rotate_trust(REGISTRY, _signed(_trust(replacement, 2), key))
    with pytest.raises(SecurityError):
        store.import_feed(REGISTRY, _signed(document, key))
    store.import_feed(REGISTRY, _signed(document, replacement), now=VERIFIED_AT)
    with pytest.raises(SecurityError):
        store.rotate_trust(REGISTRY, _signed(_trust(key, 1), replacement))
    store.provision_trust(json.dumps(_trust(recovery, 3)).encode(), recovery=True)
    assert store.decision(REGISTRY, UUID(NOTICE), 1) == "denied"
    cached, last = store.cached(REGISTRY)
    assert cached is not None and cached.notices[0].state == "vulnerable"
    assert last == VERIFIED_AT
    document["revision"] = 2
    store.import_feed(REGISTRY, _signed(document, recovery), now=VERIFIED_AT)
    with pytest.raises(SecurityError, match="rolled back"):
        store.import_feed(REGISTRY, _signed(_feed(), recovery))


def test_vulnerable_execution_continues_but_new_install_requires_exact_approval(tmp_path: Path):
    key = Ed25519PrivateKey.generate()
    store = _store(tmp_path, key)
    document = store.import_feed(REGISTRY, _signed(_feed(), key), now=VERIFIED_AT)
    inventory = (Artifact(REGISTRY, document.notices[0].release),)
    assert assess(store, inventory).allowed
    assert not assess(store, inventory, installing=True).allowed
    assert not assess(store, inventory, refuse_vulnerable=True).allowed
    store.answer(REGISTRY, UUID(NOTICE), 1, approved=True)
    assert assess(SecurityStore(store.path), inventory, installing=True).allowed
    assert not assess(store, inventory, refuse_vulnerable=True).allowed
    assert assess(store, inventory, maximum_feed_age=timedelta(days=10), now=VERIFIED_AT).allowed
    assert not assess(store, inventory, maximum_feed_age=timedelta(days=8), now=VERIFIED_AT).allowed
    store.answer(REGISTRY, UUID(NOTICE), 1, approved=False)
    assert not assess(SecurityStore(store.path), inventory).allowed
    other = Artifact("https://another.example.test", document.notices[0].release)
    assert assess(store, (other,)).allowed
    later = VERIFIED_AT + timedelta(days=3)
    assert assess(store, (other,), now=later).allowed
    assert not assess(store, inventory, maximum_feed_age=timedelta(days=1), now=later).allowed
    assert not assess(store, (other,), maximum_feed_age=timedelta(days=1), now=later).allowed


def test_malicious_bytes_block_rehosting_and_dependencies_without_misclassifying_consumers(
    tmp_path: Path,
):
    key = Ed25519PrivateKey.generate()
    store = _store(tmp_path, key)
    feed = _feed("malicious")
    document = store.import_feed(REGISTRY, _signed(feed, key), now=VERIFIED_AT)
    bad = document.notices[0].release
    rehosted = bad.model_copy(update={"pack": "another-pack", "version": "9.8.7"})
    good = bad.model_copy(update={"artifact_digest": "sha256:" + "b" * 64})
    inventory = (
        Artifact(REGISTRY, bad),
        Artifact("https://another.example.test", rehosted),
        Artifact(REGISTRY, good, (bad.artifact_digest,)),
        Artifact(REGISTRY, good),
    )
    result = assess(SecurityStore(store.path), inventory)
    assert not result.allowed
    assert [(item.artifact, item.state, item.decision) for item in result.findings] == [
        (inventory[0], "malicious", "blocked"),
        (inventory[1], "malicious", "blocked"),
        (inventory[2], "affected", "blocked"),
    ]
    assert assess(store, (inventory[3],)).allowed
    feed["revision"] = 2
    feed["notices"][0].update(revision=2, previousRevision=1, status="cleared")
    store.import_feed(REGISTRY, _signed(feed, key), now=VERIFIED_AT)
    assert assess(store, inventory).allowed


def test_installer_cached_blocks_cover_new_install_serving_and_retained_rollback(
    tmp_path: Path, security_feed: tuple[str, dict[str, Any]]
):
    from dinkster.installer import Installer

    installer = Installer(tmp_path / "installed", accelerator="cpu", runtime_probe=lambda _: ())
    source = tmp_path / "pack"
    source.mkdir()
    (source / "dinkster-pack.toml").write_text(
        '[pack]\nname = "test-pack"\n[pack.entry]\nnodes = "test_pack:NODES"\n[pack.sandbox]\n'
    )
    (source / "test_pack").mkdir()
    targets = []
    archives = {}
    for version in ("1.2.3", "2.0.0", "3.0.0"):
        (source / "test_pack/__init__.py").write_text(f'NODES = []\nVERSION = "{version}"\n')
        archive = tmp_path / f"{version}.zip"
        digest = build_artifact(source, archive)
        entry = LockedPack(
            "test-pack", version, digest, "local", ("test-pack",), "local:" + str(source)
        )
        targets.append(Lockfile.of([entry]))
        archives[version] = archive
    installer.apply(
        targets[0], {targets[0].packs[0].artifact_digest: archives["1.2.3"]}, venvs=False
    )
    installer.apply(
        targets[1], {targets[1].packs[0].artifact_digest: archives["2.0.0"]}, venvs=False
    )
    key = Ed25519PrivateKey.generate()
    store = SecurityStore(installer.root / "security.sqlite3")
    store.provision_trust(json.dumps(_trust(key)).encode())
    url, response = security_feed
    (installer.root / "security.toml").write_text(
        f'[feed_urls]\n"{REGISTRY}" = "{url}/security/index.json"\n'
    )
    feed = _feed("malicious")
    feed["notices"][0]["release"]["artifactDigest"] = targets[0].packs[0].artifact_digest
    response.update(status=200, body=_signed(feed, key))
    assert store.cached(REGISTRY)[0] is None
    current = installer.current_number()
    with pytest.raises(InstallError, match="security malicious"):
        installer.rollback(venvs=False)
    assert store.cached(REGISTRY)[0] is not None
    assert response["requests"] == [("/security/index.json", None)]
    response["status"] = 404
    assert installer.current_number() == current
    assert archives["1.2.3"].is_file()
    spec = installer.packs_for_serving()[0]
    assert spec.packs is not None
    assert spec.packs["test-pack"].artifact_digest == targets[1].packs[0].artifact_digest
    previous, _ = store.cached(REGISTRY)
    assert previous is not None
    feed["revision"] = 2
    feed["notices"][0]["release"]["artifactDigest"] = targets[1].packs[0].artifact_digest
    feed["notices"][0].update(id="636d54c0-ef6b-44cb-9ee3-bf7381bf0ed8", revision=2)
    feed["notices"].append(previous.notices[0].model_dump(mode="json", by_alias=True))
    store.import_feed(REGISTRY, _signed(feed, key), now=VERIFIED_AT)
    with pytest.raises(InstallError, match="security malicious"):
        installer.packs_for_serving()
    with pytest.raises(InstallError, match="security malicious"):
        installer.apply(targets[1], venvs=False)
    assert installer.current_number() == current

    acquisition = Installer(
        tmp_path / "acquisition",
        accelerator="cpu",
        registry_fetch=lambda entry: archives[entry.version].read_bytes(),
    )
    store = SecurityStore(acquisition.root / "security.sqlite3")
    store.provision_trust(json.dumps(_trust(key)).encode())
    feed = _feed("vulnerable")
    feed["notices"][0]["release"]["artifactDigest"] = targets[0].packs[0].artifact_digest
    published = replace(targets[0].packs[0], source="registry:" + REGISTRY)
    store.import_feed(REGISTRY, _signed(feed, key), now=VERIFIED_AT)
    with pytest.raises(InstallError, match="decision pending"):
        acquisition.acquire(published)
    assert not acquisition.artifact_available(published)
    store.answer(REGISTRY, UUID(NOTICE), 1, approved=True)
    acquisition.acquire(published)
    assert acquisition.artifact_available(published)
    assert acquisition.current_number() is None
    store.answer(REGISTRY, UUID(NOTICE), 1, approved=False)
    with pytest.raises(InstallError, match="decision denied"):
        acquisition.acquire(published)
    feed["revision"] = 2
    feed["notices"][0].update(revision=2, previousRevision=1, state="vulnerable")
    store.import_feed(REGISTRY, _signed(feed, key), now=VERIFIED_AT)
    with pytest.raises(InstallError, match="decision pending"):
        acquisition.acquire(published)
    feed["revision"] = 3
    feed["notices"][0].update(id="636d54c0-ef6b-44cb-9ee3-bf7381bf0ed8", revision=3)
    feed["notices"][0]["release"].update(
        version="3.0.0", artifactDigest=targets[2].packs[0].artifact_digest
    )
    # New notices are added; authenticated updates cannot drop the original block record.
    previous, _ = store.cached(REGISTRY)
    assert previous is not None
    feed["notices"].append(previous.notices[0].model_dump(mode="json", by_alias=True))
    store.import_feed(REGISTRY, _signed(feed, key), now=VERIFIED_AT)
    published = replace(targets[2].packs[0], source="registry:" + REGISTRY)
    with pytest.raises(InstallError, match="decision pending"):
        acquisition.acquire(published)
    store.answer(REGISTRY, UUID("636d54c0-ef6b-44cb-9ee3-bf7381bf0ed8"), 3, approved=True)
    acquisition.acquire(published)
    assert acquisition.artifact_available(published)
    assert acquisition.current_number() is None
    transport = f'[feed_urls]\n"{REGISTRY}" = "{url}/security/index.json"\n'
    config = acquisition.root / "security.toml"
    config.write_text("maximum_feed_age_seconds = 1\n" + transport)
    with pytest.raises(InstallError, match="configured maximum age"):
        acquisition.acquire(published)
    config.write_text("refuse_vulnerable = true\n" + transport)
    with pytest.raises(InstallError, match="security vulnerable"):
        acquisition.acquire(published)
    config.write_text(transport)
    acquisition.acquire(published)
    assert acquisition.artifact_available(published)
    assert acquisition.current_number() is None


@pytest.mark.parametrize("cause", ["root", "shared", "isolated", "source", "unused"])
def test_generation_selection_uses_sealed_installed_dependency_bytes(tmp_path: Path, cause: str):
    from dinkster.compose import CompositionError
    from dinkster.generation_packs import load_generation_packs, stage_generation_packs

    generation = tmp_path / "generations/1"
    generation.mkdir(parents=True)
    entries = []
    archives = {}
    for name in ("alpha-consumer", "z-provider"):
        source = tmp_path / name
        source.mkdir()
        (source / "dinkster-pack.toml").write_text(
            f'[pack]\nname="{name}"\n[pack.entry]\nnodes="nodes:NODES"\n'
            + ('[pack.dependencies]\nz-provider=">=1"\n' if name == "alpha-consumer" else "")
        )
        (source / "nodes.py").write_text("NODES = []\n")
        archive = tmp_path / f"{name}.zip"
        identity = build_artifact(source, archive)
        entries.append(
            LockedPack(name, "1.2.3", identity, "publisher", (name,), "registry:" + REGISTRY)
        )
        archives[name] = archive
    stage_generation_packs(generation, Lockfile.of(entries), archives)
    python = (
        generation
        / "packs/alpha-consumer"
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    python.parent.mkdir(parents=True)
    python.write_bytes(b"interpreter not launched by selection")
    bad = {"filename": "bad_library-1.0-py3-none-any.whl", "size": 7, "sha256": "a" * 64}
    receipts = {
        "installed-wheels.json": {
            "format": "dinkster.installed-wheels/1",
            "wheels": [],
            "inherited": [],
        },
        "packs/alpha-consumer/installed-wheels.json": {
            "format": "dinkster.installed-wheels/1",
            "wheels": [],
            "inherited": [],
        },
        "packs/z-provider/installed-wheels.json": {
            "format": "dinkster.installed-wheels/1",
            "wheels": [],
            "inherited": [],
        },
    }
    if cause in {"root", "shared", "isolated"}:
        name = {
            "root": "installed-wheels.json",
            "shared": "packs/z-provider/installed-wheels.json",
            "isolated": "packs/alpha-consumer/installed-wheels.json",
        }[cause]
        receipts[name]["wheels"] = [bad]
    for name, receipt in receipts.items():
        path = generation / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(receipt))
    unused = generation / "packs/alpha-consumer/pack-lock.json"
    unused.write_text(json.dumps({"unusedCandidate": bad}))
    files = {
        path.relative_to(generation).as_posix(): "sha256:"
        + hashlib.sha256(path.read_bytes()).hexdigest()
        for path in generation.rglob("*")
        if path.is_file()
    }
    (generation / "generation.json").write_text(
        json.dumps(
            {
                "format": "dinkster.generation/1",
                "variant": "linux-cpu",
                "inProcessPacks": ["z-provider"],
                "files": files,
            }
        )
    )
    key = Ed25519PrivateKey.generate()
    store = _store(tmp_path, key)
    assert len(load_generation_packs(generation)[1]) == 2
    feed = _feed("malicious")
    if cause == "source":
        feed["notices"][0]["release"].update(
            pack="z-provider", artifactDigest=entries[1].artifact_digest
        )
    store.import_feed(REGISTRY, _signed(feed, key), now=VERIFIED_AT)
    if cause == "unused":
        assert len(load_generation_packs(generation)[1]) == 2
    else:
        with pytest.raises(CompositionError, match="security affected: alpha-consumer"):
            load_generation_packs(generation)
    receipt_path = generation / "packs/alpha-consumer/installed-wheels.json"
    receipt_path.write_text("{}\n")
    with pytest.raises(SecurityError, match="sealed provenance"):
        load_generation_packs(generation)
