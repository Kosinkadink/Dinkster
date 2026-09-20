from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import aiohttp
import pytest
from dinkster_graph import Graph, GraphNode, graph_to_wire

ROOT = Path(__file__).parent.parent
PACK = ROOT / "tests" / "fixtures" / "extension-contract-pack" / "dinkster-pack.toml"
PACK_ID = "dinkster-extension-contract-fixture"
EVENT = "fixture.extension-contract.executed"
ROUTE = f"/api/extensions/{PACK_ID}/routes/extension-contract"


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def test_ordinary_pack_exercises_the_extension_contract(tmp_path: Path) -> None:
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
                filter(None, (str(PACK.parent), os.environ.get("PYTHONPATH")))
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
            async with session.get(base + module["moduleUrl"]) as response:
                assert response.status == 200
                assert (
                    "sha256:" + hashlib.sha256(await response.read()).hexdigest()
                    == module["moduleDigest"]
                )

            async with session.ws_connect(base + "/api/events?clientId=extension-contract") as ws:
                graph = Graph(
                    nodes={
                        "proof": GraphNode(
                            "fixture.extension.contract", {"width": 13, "height": 7}
                        ),
                    }
                )
                async with session.post(
                    base + "/api/jobs",
                    json={
                        "clientId": "extension-contract",
                        "jobId": "ordinary-pack-proof",
                        "graph": graph_to_wire(graph),
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
