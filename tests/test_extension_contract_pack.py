from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import aiohttp
import pytest
from dinkster_graph import Graph, GraphNode, Link, graph_to_wire
from dinkster_workers.host import load_pack
from dinkster_workers.manifest import ManifestError, load_manifest

from dinkster.compose import PackSpec, ServingComposer

ROOT = Path(__file__).parent.parent
PACK = ROOT / "tests" / "fixtures" / "extension-contract-pack" / "dinkster-pack.toml"
PACK_ID = "dinkster-extension-contract-fixture"
EVENT = "fixture.extension-contract.executed"
ROUTE = f"/api/extensions/{PACK_ID}/routes/extension-contract"


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def fixture_graph() -> Graph:
    return Graph(
        nodes={
            "value": GraphNode("fixture.extension.value", {"width": 13, "height": 7}),
            "proof": GraphNode(
                "fixture.extension.contract",
                {"sample": Link("value", "sample")},
            ),
        }
    )


@pytest.mark.parametrize("_serving_process", range(2))
def test_ordinary_pack_exercises_the_extension_contract(
    tmp_path: Path, _serving_process: int
) -> None:
    port = free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "dinkster.serve",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--library-root",
            str(tmp_path / "library"),
            "--no-default-packs",
            "--disable-p2p",
            "--pack",
            str(PACK),
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "DINKSTER_SERVING_PYTHON": sys.executable,
            "PYTHONPATH": os.pathsep.join(
                filter(
                    None,
                    (str(PACK.parent), os.environ.get("PYTHONPATH")),
                )
            ),
        },
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    async def scenario() -> None:
        base = f"http://127.0.0.1:{port}"
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with asyncio.timeout(60):
                while True:
                    assert process.poll() is None, "dinkster-serve exited before composition"
                    try:
                        async with session.get(base + "/api/nodes?wire=43") as response:
                            catalog = await response.json()
                        if (
                            "fixture.extension.contract" in catalog.get("nodes", {})
                            and "fixture.extension.value" in catalog.get("nodes", {})
                            and "composing" not in catalog
                        ):
                            break
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(0.05)

            async with session.get(base + ROUTE) as response:
                assert response.status == 200
                assert await response.json() == {"message": "Third-party pack route ready"}

            async with session.get(base + "/api/extensions/snapshot") as response:
                snapshot_bytes = await response.read()
            snapshot_digest = "sha256:" + hashlib.sha256(snapshot_bytes).hexdigest()
            assert snapshot_digest == catalog["extensionSnapshotDigest"]
            extension = next(
                item for item in json.loads(snapshot_bytes)["extensions"] if item["id"] == PACK_ID
            )
            assert len(extension["events"]) == 1
            assert extension["events"][0]["name"] == EVENT
            assert extension["events"][0]["payload"] == {
                "height": "integer",
                "mean": "number",
                "width": "integer",
            }
            module = extension["frontend"][0]
            assert module["contributions"] == [
                {
                    "event": EVENT,
                    "id": f"{PACK_ID}.event",
                    "kind": "eventConsumer",
                },
                {"id": f"{PACK_ID}.status", "kind": "hostUi"},
                {"id": f"{PACK_ID}.canvas", "kind": "canvasLayer"},
            ]
            async with session.get(base + module["moduleUrl"]) as response:
                assert response.status == 200
                assert (
                    "sha256:" + hashlib.sha256(await response.read()).hexdigest()
                    == module["moduleDigest"]
                )

            async with session.ws_connect(base + "/api/events?clientId=extension-contract") as ws:
                async with session.post(
                    base + "/api/jobs",
                    json={
                        "clientId": "extension-contract",
                        "jobId": "ordinary-pack-proof",
                        "graph": graph_to_wire(fixture_graph()),
                        "targets": ["proof"],
                    },
                ) as response:
                    assert response.status == 202, await response.text()
                received = None
                async with asyncio.timeout(30):
                    while True:
                        message = await ws.receive_json()
                        if message.get("event") == EVENT:
                            received = message
                        if message.get("type") == "job_state" and message.get("state") in {
                            "completed",
                            "failed",
                            "cancelled",
                        }:
                            assert message["state"] == "completed", message
                            break
                assert received is not None
                assert received["data"] == {
                    "height": 7,
                    "mean": pytest.approx(0.5),
                    "width": 13,
                }
                assert received["pack"] == PACK_ID
                assert received["nodeId"] == "proof"
                assert received["extensionSnapshotDigest"] == snapshot_digest

    try:
        asyncio.run(scenario())
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)


def test_type_registration_failure_names_pack_type_and_stage(tmp_path: Path) -> None:
    pack_root = tmp_path / "duplicate-type-pack"
    shutil.copytree(PACK.parent, pack_root)
    module = pack_root / "duplicate_contract_pack.py"
    module.write_text(
        (pack_root / "extension_contract_pack.py").read_text(encoding="utf-8")
        + "\n\ndef register_duplicate_types(registry):\n"
        "    registry.register(VALUE_TYPE)\n"
        "    registry.register(VALUE_TYPE)\n",
        encoding="utf-8",
    )
    manifest_path = pack_root / "dinkster-pack.toml"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            "extension_contract_pack:", "duplicate_contract_pack:"
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            "duplicate_contract_pack:register_types",
            "duplicate_contract_pack:register_duplicate_types",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError) as raised:
        load_pack(load_manifest(manifest_path), import_from_pack_root=True)

    message = str(raised.value)
    assert PACK_ID in message
    assert "fixture.extension.value" in message
    assert "registry stage" in message


def test_type_registration_requires_a_manifest_entry(tmp_path: Path) -> None:
    pack_root = tmp_path / "source-only-pack"
    pack_root.mkdir()
    (pack_root / "source_only_pack.py").write_text(
        'VALUE_TYPE = "fixture.source-only.value"\n'
        "NODES = ()\n\n"
        "def register_types(registry):\n"
        "    registry.register(VALUE_TYPE)\n",
        encoding="utf-8",
    )
    manifest_path = pack_root / "dinkster-pack.toml"
    manifest_path.write_text(
        '[pack]\nname = "fixture-source-only-pack"\n'
        'namespaces = ["fixture"]\n\n'
        '[pack.entry]\nnodes = "source_only_pack:NODES"\n',
        encoding="utf-8",
    )

    worker, registry, _nodes, _arms = load_pack(
        load_manifest(manifest_path), import_from_pack_root=True
    )
    try:
        assert "fixture.source-only.value" not in registry
    finally:
        asyncio.run(worker.close())


def test_full_manifest_executes_after_pack_reload() -> None:
    async def scenario() -> None:
        composer = ServingComposer(worker_env={"PYTHONPATH": str(PACK.parent)})
        try:
            await composer.add_pack(PackSpec(PACK))
            for reload_pack in (False, True):
                if reload_pack:
                    await composer.reload_pack(PACK_ID)
                result = await composer.composition.make_engine(lambda _event: None).run(
                    fixture_graph(), ["proof"]
                )
                assert result.outputs["proof"]["mean"].resolve() == pytest.approx(0.5)
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_pack_composes_model_family_into_sampling_worker(tmp_path: Path) -> None:
    host = tmp_path / "host" / "dinkster-pack.toml"
    host.parent.mkdir()
    host.write_text(
        '[pack]\nname = "sampling-host"\nnamespaces = ["dinkster", "comfy"]\n\n'
        '[pack.arms]\nnative = ["dinkster.ksampler"]\n\n'
        '[pack.entry]\nnodes = "s1_sampler_host:NODES"\n'
        'arm_nodes = "s1_sampler_host:ARM_NODES"\n'
        'choices = "s1_sampler_host:choices"\n',
        encoding="utf-8",
    )

    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(ROOT / "tests" / "fixtures" / "attention_provider"),
                        str(PACK.parent),
                        str(ROOT / "tests"),
                    )
                )
            }
        )
        try:
            await composer.add_pack(PackSpec(host, trust_reserved=True))
            await composer.add_pack(PackSpec(PACK))
            extension = next(
                item
                for item in composer._runtime_seat.pin().extension_snapshot.extensions
                if item.id == PACK_ID
            )
            assert [(item.surface_id, item.id) for item in extension.keyed_contributions] == [
                ("inference.families", "fixture.toy-image"),
                ("inference.components", "fixture.toy-image"),
                ("inference.assemblies", "fixture.toy-image"),
            ]
        finally:
            await composer.close()

    asyncio.run(scenario())
