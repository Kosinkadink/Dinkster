from __future__ import annotations

import asyncio
import base64
import io
import os
import threading
from collections.abc import Sequence
from pathlib import Path

import av
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_api.v1 import publish_preview_artifact
from dinkster_assets import AssetError, AssetVault, digest_bytes
from dinkster_assets.vault import VaultWriter
from dinkster_engine import ExecutionError
from dinkster_graph import Graph, GraphNode, graph_to_wire
from dinkster_schema.reporting import EncodedPreviewArtifact, capture_preview_artifacts
from dinkster_server import create_app
from dinkster_workers.boundary import BoundaryError, _decode_preview_artifacts
from dinkster_workers.preview_artifacts import PreviewArtifactPublisher

from dinkster.compose import PackSpec, ServingComposer


def _tiny_mp4(color: int = 0) -> bytes:
    encoded = io.BytesIO()
    with av.open(encoded, "w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=1)
        stream.width = 2
        stream.height = 2
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame.from_ndarray(
            np.full((2, 2, 3), color, dtype=np.uint8), format="rgb24"
        )
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return encoded.getvalue()


@pytest.fixture
def mp4() -> bytes:
    return _tiny_mp4(23)


def _write_preview_pack(tmp_path: Path, data: bytes) -> Path:
    payload = base64.b64encode(data).decode("ascii")
    failed_payload = base64.b64encode(_tiny_mp4(99)).decode("ascii")
    (tmp_path / "preview_fixture.py").write_text(
        "from __future__ import annotations\n"
        "import base64\n"
        "from collections.abc import Mapping\n"
        "from dinkster_api.v1 import (CORE_STRING, InputSpec, Node, NodeSchema, "
        "OutputSpec, TypeExpr, publish_preview_artifact)\n"
        f"DATA = base64.b64decode({payload!r})\n"
        f"FAILED_DATA = base64.b64decode({failed_payload!r})\n"
        "class Preview(Node):\n"
        " @classmethod\n"
        " def define_schema(cls):\n"
        "  return NodeSchema(node_type='preview-fixture.emit', inputs=(InputSpec('mode', "
        "TypeExpr.concrete(CORE_STRING)),), outputs=(OutputSpec('out', "
        "TypeExpr.concrete(CORE_STRING)),))\n"
        " @classmethod\n"
        " def execute(cls, *, mode: str) -> Mapping[str, object]:\n"
        "  data = FAILED_DATA if mode == 'fail' else DATA\n"
        "  publish_preview_artifact(data, name='preview.mp4', media_type='video/mp4')\n"
        "  if mode == 'fail': raise RuntimeError('fixture failure')\n"
        "  return cls.outputs(out='ok')\n"
        "NODES = (Preview,)\n"
        "def register_types(registry): pass\n",
        encoding="ascii",
    )
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        "[pack]\n"
        "name = 'preview-fixture'\n"
        "publisher = 'dinkster'\n"
        "namespaces = ['preview-fixture']\n"
        "[pack.entry]\n"
        "nodes = 'preview_fixture:NODES'\n"
        "types = 'preview_fixture:register_types'\n",
        encoding="ascii",
    )
    return manifest


@pytest.mark.parametrize("in_process", (True, False))
def test_composed_hosts_publish_encoded_preview_to_result_and_asset_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool, mp4: bytes
) -> None:
    manifest = _write_preview_pack(tmp_path, mp4)
    vault = AssetVault(tmp_path / "vault")
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
    worker_env = {"PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path.cwd())))}

    async def scenario() -> None:
        composer = ServingComposer(worker_env=worker_env)
        try:
            await composer.add_pack(PackSpec(manifest, in_process=in_process, env=worker_env))
            app = create_app(
                lambda on_event: composer.composition.make_engine(on_event),
                composer.composition.schemas,
                asset_export=vault,
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                graph = Graph(nodes={"preview": GraphNode("preview-fixture.emit", {"mode": "ok"})})
                response = await client.post(
                    "/api/jobs",
                    json={
                        "clientId": "preview-client",
                        "jobId": "preview-job",
                        "graph": graph_to_wire(graph),
                        "targets": ["preview"],
                    },
                )
                assert response.status == 202
                async with asyncio.timeout(5):
                    while True:
                        status = await (
                            await client.get("/api/jobs/preview-client/preview-job")
                        ).json()
                        if status["state"] != "pending" and status["state"] != "running":
                            break
                        await asyncio.sleep(0)
                assert status["state"] == "completed"
                artifact = status["artifacts"][0]
                assert artifact["nodeId"] == "preview"
                assert artifact["digest"] == digest_bytes(mp4)
                assert artifact["name"] == "preview.mp4"
                assert artifact["mediaType"] == "video/mp4"
                assert artifact["size"] == len(mp4)
                asset = await client.get(f"/assets/{artifact['digest']}")
                assert asset.status == 200
                assert await asset.read() == mp4

                engine = composer.composition.make_engine(lambda _event: None)
                with pytest.raises(ExecutionError, match="fixture failure"):
                    await engine.run(
                        Graph(
                            nodes={"failed": GraphNode("preview-fixture.emit", {"mode": "fail"})}
                        ),
                        ["failed"],
                    )
                assert vault.digests() == [digest_bytes(mp4)]
            finally:
                await client.close()
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_preview_api_requires_an_open_invocation_context(mp4: bytes) -> None:
    with pytest.raises(RuntimeError, match="active invocation"):
        publish_preview_artifact(mp4, name="preview.mp4", media_type="video/mp4")


def test_copied_preview_context_rejects_publication_after_closing(mp4: bytes) -> None:
    async def scenario() -> None:
        release = asyncio.Event()

        async def late_publish() -> None:
            await release.wait()
            publish_preview_artifact(mp4, name="preview.mp4", media_type="video/mp4")

        with capture_preview_artifacts():
            task = asyncio.create_task(late_publish())
        release.set()
        with pytest.raises(RuntimeError, match="closed"):
            await task

    asyncio.run(scenario())
    with capture_preview_artifacts() as artifacts:
        publish_preview_artifact(mp4, name="preview.mp4", media_type="video/mp4")
    assert artifacts == [EncodedPreviewArtifact(mp4, "preview.mp4", "video/mp4")]
    with pytest.raises(RuntimeError, match="active invocation"):
        publish_preview_artifact(mp4, name="preview.mp4", media_type="video/mp4")


@pytest.mark.parametrize(
    "claim,match",
    (
        (EncodedPreviewArtifact(b"not media", "preview.mp4", "video/mp4"), "malformed media"),
        (EncodedPreviewArtifact(_tiny_mp4(), "../preview.mp4", "video/mp4"), "safe basename"),
        (EncodedPreviewArtifact(_tiny_mp4(), "preview.mp4", "image/png"), "media_type"),
    ),
)
def test_preview_publisher_rejects_independent_untrusted_claims(
    tmp_path: Path, claim: EncodedPreviewArtifact, match: str
) -> None:
    async def scenario() -> None:
        publisher = PreviewArtifactPublisher(AssetVault(tmp_path / "vault"))
        with pytest.raises((AssetError, ValueError), match=match):
            await publisher("authoritative-node", (claim,))

    asyncio.run(scenario())


def test_preview_publisher_requires_a_host_vault(mp4: bytes) -> None:
    async def scenario() -> None:
        with pytest.raises(AssetError, match="host asset vault"):
            await PreviewArtifactPublisher(None)(
                "node", (EncodedPreviewArtifact(mp4, "preview.mp4", "video/mp4"),)
            )

    asyncio.run(scenario())


def test_preview_boundary_rejects_independent_blob_claims(mp4: bytes) -> None:
    with pytest.raises(BoundaryError, match="fields"):
        _decode_preview_artifacts(
            {"previewArtifacts": [{"name": "preview.mp4", "mediaType": "video/mp4"}]}, [mp4]
        )
    with pytest.raises(BoundaryError, match="blob index"):
        _decode_preview_artifacts(
            {"previewArtifacts": [{"name": "preview.mp4", "mediaType": "video/mp4", "blob": 1}]},
            [mp4],
        )


def test_failed_batch_removes_only_newly_committed_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mp4: bytes
) -> None:
    vault = AssetVault(tmp_path / "vault")
    publisher = PreviewArtifactPublisher(vault)
    second = _tiny_mp4(99)
    existing = digest_bytes(mp4)
    with vault.writer(existing) as writer:
        writer.write(mp4)
        writer.commit()
    calls = 0
    original = type(vault.writer(digest_bytes(mp4))).commit_with_result

    def fail_second(self: VaultWriter) -> tuple[Path, bool]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise AssetError("commit failed")
        return original(self)

    monkeypatch.setattr(type(vault.writer(digest_bytes(mp4))), "commit_with_result", fail_second)

    async def scenario() -> None:
        with pytest.raises(AssetError, match="commit failed"):
            await publisher(
                "node",
                (
                    EncodedPreviewArtifact(mp4, "first.mp4", "video/mp4"),
                    EncodedPreviewArtifact(second, "second.mp4", "video/mp4"),
                ),
            )
        assert vault.digests() == [existing]

    asyncio.run(scenario())


def test_concurrent_identical_previews_do_not_delete_the_other_result(
    tmp_path: Path, mp4: bytes
) -> None:
    vault = AssetVault(tmp_path / "vault")
    publisher = PreviewArtifactPublisher(vault)
    claim = EncodedPreviewArtifact(mp4, "preview.mp4", "video/mp4")

    async def scenario() -> None:
        first, second = await asyncio.gather(publisher("one", (claim,)), publisher("two", (claim,)))
        assert first[0].digest == second[0].digest == digest_bytes(mp4)
        stored = vault.resolve(digest_bytes(mp4))
        assert stored is not None
        assert stored.read_bytes() == mp4

    asyncio.run(scenario())


def test_cancellation_during_staging_discards_before_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mp4: bytes
) -> None:
    vault = AssetVault(tmp_path / "vault")
    publisher = PreviewArtifactPublisher(vault)
    entered = threading.Event()
    release = threading.Event()
    original = publisher._stage

    def blocked_stage(node_id: str, claims: Sequence[EncodedPreviewArtifact]) -> object:
        entered.set()
        release.wait()
        return original(node_id, claims)

    monkeypatch.setattr(publisher, "_stage", blocked_stage)

    async def scenario() -> None:
        task = asyncio.create_task(
            publisher("node", (EncodedPreviewArtifact(mp4, "preview.mp4", "video/mp4"),))
        )
        await asyncio.to_thread(entered.wait)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert vault.digests() == []

    asyncio.run(scenario())
