from __future__ import annotations

import asyncio
import base64
import faulthandler
import io
import os
import shutil
import sqlite3
import sys
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import av
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer
from dinkster_api.v1 import publish_preview_artifact
from dinkster_assets import (
    AssetError,
    AssetVault,
    AssetWriter,
    LibraryStore,
    MountDef,
    MountTable,
    digest_bytes,
)
from dinkster_assets.produced import ProducedAssetInventory
from dinkster_assets.scratch import (
    ProducedScratchState,
    claim_inactive_produced_vault,
    produced_job_scratch_root,
)
from dinkster_assets.vault import VaultWriter
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener, ExecutionError, RunResult
from dinkster_graph import Graph, GraphNode, graph_to_wire
from dinkster_protocol import AssetFolderContext
from dinkster_schema.reporting import EncodedPreviewArtifact, capture_preview_artifacts
from dinkster_server import STATE_KEY, ServerLibrary, create_app
from dinkster_server.library import recover_asset_observations
from dinkster_server.queue import Job, JobKey
from dinkster_workers.preview_artifacts import PreviewArtifactPublisher

from dinkster.compose import PackSpec, ServingComposer
from tests.async_scenarios import run_scenario


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


def _write_preview_pack(tmp_path: Path, data: bytes, *, document: bytes | None = None) -> Path:
    payload = base64.b64encode(data).decode("ascii")
    failed_payload = base64.b64encode(_tiny_mp4(99)).decode("ascii")
    module = "preview_fixture_" + digest_bytes(str(tmp_path).encode())[7:19]
    (tmp_path / f"{module}.py").write_text(
        "from __future__ import annotations\n"
        "import base64\n"
        "import os\n"
        "from collections.abc import Mapping\n"
        "from pathlib import Path\n"
        "from dinkster_api.v1 import (CORE_STRING, ExpansionBuilder, InputSpec, Node, NodeSchema, "
        "OutputSpec, TypeExpr, publish_preview_artifact, publish_saved_file_preview, "
        "attach_saved_artifact, AssetWriter, MountSnapshotWriter, SaveTarget)\n"
        "from dinkster_schema.reporting import publish_image_document\n"
        "from dinkster_image_document import ImageDocument\n"
        "from dinkster_image_document.document import (empty_document, document_meta, "
        "validate_image_document_encoded)\n"
        f"DATA = base64.b64decode({payload!r})\n"
        f"FAILED_DATA = base64.b64decode({failed_payload!r})\n"
        f"DOCUMENT = {document!r}\n"
        "LAST_REF = None\n"
        "class Preview(Node):\n"
        " @classmethod\n"
        " def define_schema(cls):\n"
        "  return NodeSchema(node_type='preview-fixture.emit', inputs=(InputSpec('mode', "
        "TypeExpr.concrete(CORE_STRING)),), outputs=(OutputSpec('out', "
        "TypeExpr.concrete(CORE_STRING)),))\n"
        " @classmethod\n"
        " def execute(cls, *, mode: str) -> Mapping[str, object]:\n"
        "  global LAST_REF\n"
        "  data = FAILED_DATA if mode == 'fail' else DATA\n"
        "  if mode.startswith('document'):\n"
        "   publish_image_document(DOCUMENT)\n"
        "   publish_image_document(empty_document(7, 3).data)\n"
        "   if mode != 'document-only':\n"
        "    publish_preview_artifact(DATA, name='preview.mp4', media_type='video/mp4')\n"
        "   if mode == 'document-fail': raise ValueError('document fixture failed')\n"
        "   return cls.outputs(out='document')\n"
        "  if mode == 'mounted-stale':\n"
        "   attach_saved_artifact(LAST_REF)\n"
        "   publish_saved_file_preview(LAST_REF, lambda: DATA, name='preview.mp4', "
        "media_type='video/mp4')\n"
        "   return cls.outputs(out='stale')\n"
        "  if mode.startswith('mounted'):\n"
        "   writer = MountSnapshotWriter(os.environ['DINKSTER_MOUNTS_SNAPSHOT'])\n"
        "   for index in range(65 if mode == 'mounted-batch' else 2):\n"
        "    ref = AssetWriter(writer).save_bytes(SaveTarget('renders', f'saved-{index}'), "
        "b'duplicate saved bytes', suffix='.exr', media_type='image/x-exr')\n"
        "    claim = ref.to_wire()\n"
        "    LAST_REF = claim\n"
        "    if mode == 'mounted-forged': claim['digest'] = 'blake3:' + 'a'*64\n"
        "    attach_saved_artifact(claim)\n"
        "    if mode == 'mounted-mismatch': claim = {**claim, 'size': ref.size + 1}\n"
        "    def encode():\n"
        "     if mode == 'mounted-encode-fail': raise RuntimeError('encoder failed')\n"
        "     if mode == 'mounted-oversized': return b'x' * (16 * 1024 * 1024 + 1)\n"
        "     return DATA\n"
        "    publish_saved_file_preview(claim, encode, name='preview.mp4', "
        "media_type='video/mp4')\n"
        "   return cls.outputs(out='saved')\n"
        "  if mode == 'saved':\n"
        "   ref = dict(digest='blake3:' + 'a'*64, name='saved.exr', size=7, "
        "mediaType='image/x-exr', virtualPath='mounts/output/saved.exr')\n"
        "   attach_saved_artifact(ref)\n"
        "   publish_saved_file_preview(ref, lambda: DATA, name='preview.mp4', "
        "media_type='video/mp4')\n"
        "   return cls.outputs(out='saved')\n"
        "  staged = Path(os.environ['DINKSTER_PACK_SCRATCH']) / ('encoded-' + mode + '.mp4')\n"
        "  staged.write_bytes(data)\n"
        "  publish_preview_artifact(staged.read_bytes(), "
        "name='preview.mp4', media_type='video/mp4')\n"
        "  if mode == 'fail': raise RuntimeError('fixture failure')\n"
        "  return cls.outputs(out='ok')\n"
        "class Expand(Preview):\n"
        " @classmethod\n"
        " def define_schema(cls):\n"
        "  return NodeSchema(node_type='preview-fixture.expand', inputs=(InputSpec('mode', "
        "TypeExpr.concrete(CORE_STRING)),), outputs=(OutputSpec('out', "
        "TypeExpr.concrete(CORE_STRING)),), may_expand_graph=True)\n"
        " @classmethod\n"
        " def execute(cls, *, mode: str):\n"
        "  graph = ExpansionBuilder()\n"
        "  child = graph.node('preview-fixture.expand' if mode == 'nested' else "
        "'preview-fixture.emit', 'inner' if mode == 'nested' else 'emit', mode='ok')\n"
        "  return graph.finalize(out=child.out('out'))\n"
        "NODES = (Preview, Expand)\n"
        "def register_types(registry):\n"
        " if DOCUMENT is not None:\n"
        "  registry.register('dinkster.layers', coerce=lambda obj: obj, "
        "encode=lambda obj: obj.data, decode=ImageDocument, meta=document_meta, "
        "validate_encoded=validate_image_document_encoded, asset_references_complete=True)\n",
        encoding="ascii",
    )
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        "[pack]\n"
        "name = 'preview-fixture'\n"
        "publisher = 'dinkster'\n"
        "namespaces = ['preview-fixture']\n"
        "type-namespaces = ['dinkster.layers']\n"
        "[pack.entry]\n"
        f"nodes = '{module}:NODES'\n"
        f"types = '{module}:register_types'\n",
        encoding="ascii",
    )
    return manifest


@pytest.mark.parametrize("in_process", (True, False))
@pytest.mark.parametrize("retained", (True, False))
@pytest.mark.parametrize("nested", (False, True))
def test_composed_hosts_publish_encoded_preview_to_result_and_asset_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    in_process: bool,
    retained: bool,
    nested: bool,
    mp4: bytes,
) -> None:
    manifest = _write_preview_pack(tmp_path, mp4)
    vault = AssetVault(tmp_path / "vault")
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
    scratch_root = tmp_path / "scratch"
    scratch = scratch_root / "packs" / "preview-fixture"
    scratch.mkdir(parents=True)
    monkeypatch.setenv("DINKSTER_PACK_SCRATCH", str(scratch))
    worker_env = {"PYTHONPATH": os.pathsep.join((str(tmp_path), str(Path.cwd())))}
    output = tmp_path / "selected-output"
    output.mkdir()
    mounts = MountTable()
    mounts.add(MountDef("renders", output, "readwrite"))
    mounts.scan("renders")
    (output / "preview.mp4").write_bytes(b"existing file must not be overwritten")
    library = (
        ServerLibrary(
            vault=vault,
            store=LibraryStore(tmp_path / "library.sqlite"),
            resolver=mounts,
            local_resolver=mounts,
            upload_writer=AssetWriter(mounts),
            project_folders_for=lambda _project: AssetFolderContext(("renders",), "renders"),
        )
        if retained
        else None
    )

    async def scenario() -> None:
        composer = ServingComposer(worker_env=worker_env, pack_scratch_root=scratch_root)
        engines: list[Engine] = []

        def make_engine(on_event: EventListener) -> Engine:
            engine = composer.composition.make_engine(on_event)
            engines.append(engine)
            return engine

        try:
            await composer.add_pack(PackSpec(manifest, in_process=in_process, env=worker_env))
            app = create_app(
                make_engine,
                composer.composition.schemas,
                asset_export=vault,
                library=library,
            )
            client = TestClient(TestServer(app))
            await client.start_server()
            try:
                graph = Graph(
                    nodes={
                        "preview": GraphNode(
                            "preview-fixture.expand" if nested else "preview-fixture.emit",
                            {"mode": "nested" if nested else "ok"},
                        )
                    }
                )
                response = await client.post(
                    "/api/jobs",
                    json={
                        "clientId": "preview-client",
                        "jobId": "preview-job",
                        "project": "film",
                        "graph": graph_to_wire(graph),
                        "targets": ["preview"],
                    },
                )
                assert response.status == 202
                try:
                    async with asyncio.timeout(5):
                        while True:
                            status = await (
                                await client.get("/api/jobs/preview-client/preview-job")
                            ).json()
                            if status["state"] != "pending" and status["state"] != "running":
                                break
                            await asyncio.sleep(0)
                except TimeoutError:
                    state = app[STATE_KEY]
                    job = state.queue.get("preview-client", "preview-job")
                    print(
                        "Preview status timeout before cleanup:",
                        {"in_process": in_process, "retained": retained, "nested": nested},
                        None
                        if job is None
                        else {
                            "state": job.state,
                            "jobRef": job.run_id,
                            "attempt": job.attempt,
                            "nodes": state.node_states(job.run_id),
                        },
                        file=sys.stderr,
                    )
                    for task in asyncio.all_tasks():
                        task.print_stack(file=sys.stderr)
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                    raise
                assert status["state"] == "completed"
                artifact = status["artifacts"][0]
                producing_node = "preview.0.0.inner.0.0.emit" if nested else "preview"
                assert artifact["nodeId"] == producing_node
                assert artifact["digest"] == digest_bytes(mp4)
                assert artifact["name"] == ("preview_00001.mp4" if retained else "preview.mp4")
                assert artifact["mediaType"] == "video/mp4"
                assert artifact["size"] == len(mp4)
                if retained:
                    assert artifact["virtualPath"] == "mounts/renders/preview_00001.mp4"
                    assert (output / "preview_00001.mp4").read_bytes() == mp4
                    assert (
                        output / "preview.mp4"
                    ).read_bytes() == b"existing file must not be overwritten"
                    catalog = await (
                        await client.get("/api/assets/catalog?scope=local&project=film")
                    ).json()
                    assert len(catalog["assets"]) == 1
                    assert library is not None
                    observations = library.store.query_assets("local", project="film")[0]
                    assert {record.asset.virtual_path for record in observations} == {
                        "",
                        "mounts/renders/preview_00001.mp4",
                    }
                    assert all(record.origin["node"] == producing_node for record in observations)
                    assert library.store.query_assets("local", project="another") == []
                    (offer,) = status["artifactOffers"]
                    assert offer == {
                        "token": offer["token"],
                        "scope": "local",
                        "jobRef": status["jobRef"],
                        "attemptId": 1,
                        "nodeId": producing_node,
                        "publication": 1,
                        "artifact": {
                            "nodeId": producing_node,
                            "digest": digest_bytes(mp4),
                            "name": "preview_00001.mp4",
                            "size": len(mp4),
                            "mediaType": "video/mp4",
                            "virtualPath": "mounts/renders/preview_00001.mp4",
                            "index": 0,
                        },
                        "digests": [digest_bytes(mp4)],
                        "complete": True,
                    }
                    source = status["resultOffers"][producing_node]["out"]
                    assert source["nodeId"] == producing_node
                    assert source["outputId"] == "out"
                    assert source["digests"] == [] and source["complete"] is True
                    response = await client.get(
                        f"/api/library/terminal-results/{status['jobRef']}?scope=local"
                    )
                    assert response.status == 200
                    terminal = await response.json()
                    assert terminal["complete"] is True
                    assert terminal["artifacts"] == [offer]
                    assert terminal["resultOffers"][producing_node]["out"] == source
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
                assert (scratch / "encoded-ok.mp4").read_bytes() == mp4
                assert engines
                for completed_engine in engines:
                    assert isinstance(completed_engine.cache, MemoryLRUCache)
                    assert completed_engine.cache.clear() > 0
                    assert len(completed_engine.cache) == 0
            finally:
                await client.close()
        finally:
            await composer.close()

        shutil.rmtree(scratch_root)
        assert not scratch.exists()
        if retained:
            assert library is not None
            library.store.close()
            reopened_mounts = MountTable()
            reopened_mounts.add(MountDef("renders", output, "readwrite"))
            reopened_mounts.scan("renders")
            reopened = ServerLibrary(
                vault=AssetVault(vault.root),
                store=LibraryStore(tmp_path / "library.sqlite"),
                resolver=reopened_mounts,
                local_resolver=reopened_mounts,
                upload_writer=AssetWriter(reopened_mounts),
                project_folders_for=lambda _project: AssetFolderContext(("renders",), "renders"),
            )
            restarted = ServingComposer(worker_env=worker_env)
            try:
                await restarted.add_pack(PackSpec(manifest, in_process=in_process, env=worker_env))
                restarted_client = TestClient(
                    TestServer(
                        create_app(
                            restarted.composition.make_engine,
                            restarted.composition.schemas,
                            library=reopened,
                            asset_export=reopened.vault,
                        )
                    )
                )
                await restarted_client.start_server()
                try:
                    catalog = await (
                        await restarted_client.get(
                            "/api/assets/catalog?scope=local&project=film&kind=media/video"
                        )
                    ).json()
                    assert len(catalog["assets"]) == 1
                    assert (output / "preview_00001.mp4").read_bytes() == mp4
                    asset = await restarted_client.get(f"/assets/{digest_bytes(mp4)}")
                    assert asset.status == 200
                    assert await asset.read() == mp4
                    assert reopened.store.query_assets("local", project="another") == []
                finally:
                    await restarted_client.close()
            finally:
                await restarted.close()
                reopened.store.close()

    run_scenario(scenario())


@pytest.mark.parametrize("in_process", [True, False])
@pytest.mark.parametrize("enabled", [True, False])
def test_optional_previews_associate_only_with_authorized_files_and_survive_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mp4: bytes,
    in_process: bool,
    enabled: bool,
) -> None:
    import json

    from dinkster_caches import DiskCacheStore
    from dinkster_protocol import CachedResult
    from dinkster_server.app import job_to_wire

    manifest = _write_preview_pack(tmp_path, mp4)
    vault = AssetVault(tmp_path / "vault")
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
    output = tmp_path / "output"
    output.mkdir()
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps({"mounts": [{"id": "renders", "root": str(output), "mode": "readwrite"}]}),
        "ascii",
    )
    monkeypatch.setenv("DINKSTER_MOUNTS_SNAPSHOT", str(snapshot))
    env = {"PYTHONPATH": str(tmp_path), "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot)}

    async def scenario() -> None:
        composer = ServingComposer(worker_env=env, saved_file_previews_enabled=enabled)
        try:
            await composer.add_pack(PackSpec(manifest, in_process=in_process, env=env))
            engine = composer.composition.make_engine(lambda _event: None)
            for mode in (
                "mounted",
                "mounted-batch",
                "mounted-mismatch",
                "mounted-encode-fail",
                "mounted-oversized",
                "mounted-publish-fail",
            ):
                graph = Graph(nodes={"save": GraphNode("preview-fixture.emit", {"mode": mode})})
                with monkeypatch.context() as publication:
                    if mode == "mounted-publish-fail":

                        def fail_publication(*_args: object) -> object:
                            raise AssetError("optional publisher unavailable")

                        publication.setattr(PreviewArtifactPublisher, "_stage", fail_publication)
                    result = await engine.run(graph, ["save"])
                count = 65 if mode == "mounted-batch" else 2
                assert len(result.artifacts) == count
                assert len({item.virtual_path for item in result.artifacts}) == count
                assert {item.digest for item in result.artifacts} == {
                    digest_bytes(b"duplicate saved bytes")
                }
                expected = enabled and mode in ("mounted", "mounted-batch")
                assert all((item.preview is not None) == expected for item in result.artifacts)
                cached = await engine.run(graph, ["save"])
                assert cached.cached == ("save",)
                assert cached.artifacts == result.artifacts
                disk = DiskCacheStore(tmp_path / ("cache-" + mode), composer.composition._registry)
                await disk.put(
                    "saved-result", CachedResult(result.outputs["save"], artifacts=result.artifacts)
                )
                restored = await disk.get("saved-result")
                assert restored is not None and restored.artifacts == result.artifacts
                job = Job(JobKey("client", mode), graph, ("save",), 0, mode, "fixture")
                job.result = result
                serialized = job_to_wire(job, None, composer.composition._registry)
                rows = serialized["artifacts"]
                assert isinstance(rows, list)
                assert all(("preview" in row) == expected for row in rows)
                mounts = MountTable()
                mounts.add(MountDef("renders", output, "readwrite"))
                mounts.scan("renders")
                library = ServerLibrary(
                    vault=vault,
                    store=LibraryStore(tmp_path / (mode + ".sqlite")),
                    resolver=mounts,
                    local_resolver=mounts,
                    upload_writer=AssetWriter(mounts),
                    project_folders_for=lambda _project: AssetFolderContext(
                        ("renders",), "renders"
                    ),
                )
                try:
                    retained = library.record_saved_outputs(job, result)
                    assert retained.artifacts == result.artifacts
                    if expected:
                        assert vault.resolve(digest_bytes(mp4)) is not None
                        with sqlite3.connect(tmp_path / (mode + ".sqlite")) as connection:
                            connection.execute(
                                "DELETE FROM asset_observations WHERE digest = ?",
                                (digest_bytes(mp4),),
                            )
                            connection.execute(
                                "CREATE TRIGGER fail_optional BEFORE INSERT ON asset_observations "
                                f"WHEN NEW.digest = '{digest_bytes(mp4)}' "
                                "BEGIN SELECT RAISE(ABORT, 'optional catalog offline'); END"
                            )
                        without_preview = library.record_saved_outputs(job, result)
                        assert all(item.preview is None for item in without_preview.artifacts)
                        assert [item.virtual_path for item in without_preview.artifacts] == [
                            item.virtual_path for item in result.artifacts
                        ]
                        assert not list(vault.root.glob(".upload-observations/*.json"))
                finally:
                    library.store.close()
            for mode in ("mounted-forged", "mounted-stale"):
                with pytest.raises(ExecutionError):
                    await engine.run(
                        Graph(nodes={"save": GraphNode("preview-fixture.emit", {"mode": mode})}),
                        ["save"],
                    )
        finally:
            await composer.close()

    run_scenario(scenario())


@pytest.mark.parametrize("existing", ["absent", "pending", "untracked"])
def test_saved_file_preview_transport_adopts_bytes_before_worker_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mp4: bytes, existing: str
) -> None:
    from dinkster_protocol import Invocation
    from dinkster_values import CORE_STRING, TypeRegistry, register_core_types
    from dinkster_workers import IsolatedWorker

    manifest = _write_preview_pack(tmp_path, mp4)
    vault = AssetVault(tmp_path / "host-vault")
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
    digest = digest_bytes(mp4)
    if existing == "pending":
        ProducedAssetInventory(vault.root).pending(digest, len(mp4))
    if existing != "absent":
        with vault.writer(digest) as writer:
            writer.write(mp4)
            writer.commit()
    scratch = tmp_path / "worker-scratch"
    scratch.mkdir()

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        worker = IsolatedWorker(
            manifest,
            registry,
            extra_env={"PYTHONPATH": str(tmp_path), "DINKSTER_PACK_SCRATCH": str(scratch)},
        )
        try:
            await worker.start()
            schema = worker.schemas["preview-fixture.emit"]
            result = await worker.invoke(
                Invocation(
                    "transport",
                    "save",
                    schema.node_type,
                    {"mode": registry.wrap(CORE_STRING, "saved")},
                    schema,
                    saved_file_previews_enabled=True,
                )
            )
            assert result.error is None
            assert len(result.saved_file_previews) == 1
            assert len(result.saved_artifact_claims) == 1
            assert result.preview_artifacts == ()
            state = ProducedScratchState(produced_job_scratch_root(scratch, "transport", 1))
            assert state.publications() == ((digest, len(mp4), True),)
            if existing == "untracked":
                assert not (vault.root / ".produced-assets.sqlite").exists()
            else:
                entries = ProducedAssetInventory(vault.root).entries()
                assert len(entries) == 1
                assert (entries[0].digest, entries[0].size) == (digest, len(mp4))
                assert entries[0].adopted_at_ns is not None
        finally:
            await worker.close()
        shutil.rmtree(scratch)
        path = vault.resolve(digest_bytes(mp4))
        assert path is not None and path.read_bytes() == mp4

    run_scenario(scenario())


@pytest.mark.parametrize("count,enabled", [(65, True), (4096, True), (65, False)])
def test_saved_file_previews_are_lazy_file_backed_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, enabled: bool, mp4: bytes
) -> None:
    import tracemalloc

    from dinkster_api.v1 import (
        CORE_STRING,
        Node,
        NodeSchema,
        OutputSpec,
        TypeExpr,
        attach_saved_artifact,
        publish_saved_file_preview,
    )
    from dinkster_protocol import Invocation
    from dinkster_values import TypeRegistry, register_core_types
    from dinkster_workers import InProcessWorker
    from dinkster_workers.boundary import ValueCodec, decode_result, encode_result

    monkeypatch.setenv("DINKSTER_PACK_SCRATCH", str(tmp_path / "scratch"))
    calls = 0
    measure_memory = count == 65 and enabled
    payload = mp4 + b"\0" * (1024 * 1024) if measure_memory else mp4

    def encode() -> bytes:
        nonlocal calls
        calls += 1
        return bytes(bytearray(payload))

    class Save(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="preview-fixture.saved",
                outputs=(OutputSpec("out", TypeExpr.concrete(CORE_STRING)),),
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            for index in range(count):
                ref = {
                    "digest": digest_bytes(b"duplicate saved bytes"),
                    "name": f"saved-{index}.exr",
                    "size": 21,
                    "mediaType": "image/x-exr",
                    "virtualPath": f"mounts/output/saved-{index}.exr",
                }
                attach_saved_artifact(ref)
                assert (
                    publish_saved_file_preview(
                        ref, encode, name="preview.mp4", media_type="video/mp4"
                    )
                    is enabled
                )
            return cls.outputs(out="saved")

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        schema = Save.schema()
        worker = InProcessWorker({schema.node_type: Save}, registry)
        invocation = Invocation(
            "saved-preview",
            "save",
            schema.node_type,
            {},
            schema,
            saved_file_previews_enabled=enabled,
        )
        if measure_memory:
            tracemalloc.start()
        try:
            result = await worker.invoke(invocation)
        finally:
            if measure_memory:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                # Retaining 65 distinct 1 MiB encoded objects would exceed this.
                assert peak < 16 * 1024 * 1024, f"encoded staging peak: {peak} bytes"
        assert result.error is None
        assert len(result.saved_artifact_claims) == count
        assert len(result.saved_file_previews) == (count if enabled else 0)
        assert result.preview_artifacts == ()
        codec = ValueCodec(registry)
        header, blobs, segments = encode_result(codec, result, "saved-preview", 0)
        assert not segments
        assert mp4 not in blobs
        decoded, _ = decode_result(codec, header, blobs, [])
        assert decoded.saved_file_previews == result.saved_file_previews
        if enabled:
            assert len({str(item["saved"]) for item in decoded.saved_file_previews}) == count
            scratch = produced_job_scratch_root(
                tmp_path / "scratch", invocation.invocation_id, invocation.attempt_id
            )
            vault = AssetVault(scratch)
            assert vault.digests() == [digest_bytes(payload)]
            path = vault.resolve(digest_bytes(payload))
            assert path is not None and path.read_bytes() == payload
            assert ProducedScratchState(scratch).publications() == (
                (digest_bytes(payload), len(payload), True),
            )

    run_scenario(scenario())
    assert calls == (count if enabled else 0)


def test_cancelled_async_node_closes_a_threaded_optional_encoder_without_waiting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mp4: bytes
) -> None:
    from dinkster_api.v1 import (
        CORE_STRING,
        Node,
        NodeSchema,
        OutputSpec,
        TypeExpr,
        publish_saved_file_preview,
    )
    from dinkster_protocol import Invocation
    from dinkster_values import TypeRegistry, register_core_types
    from dinkster_workers import InProcessWorker

    monkeypatch.setenv("DINKSTER_PACK_SCRATCH", str(tmp_path / "scratch"))
    release = threading.Event()
    finished = threading.Event()
    background: list[asyncio.Task[bool]] = []
    calls = 0

    async def scenario() -> None:
        entered = asyncio.Event()
        loop = asyncio.get_running_loop()

        def encode() -> bytes:
            nonlocal calls
            calls += 1
            loop.call_soon_threadsafe(entered.set)
            try:
                assert release.wait(10), "test did not release encoder"
                return mp4
            finally:
                finished.set()

        class Save(Node):
            @classmethod
            def define_schema(cls) -> NodeSchema:
                return NodeSchema(
                    node_type="preview-fixture.cancel",
                    outputs=(OutputSpec("out", TypeExpr.concrete(CORE_STRING)),),
                )

            @classmethod
            async def execute(cls) -> Mapping[str, object]:
                task = asyncio.create_task(
                    asyncio.to_thread(
                        publish_saved_file_preview,
                        {
                            "digest": digest_bytes(b"saved"),
                            "name": "saved.exr",
                            "size": 5,
                            "mediaType": "image/x-exr",
                            "virtualPath": "mounts/output/saved.exr",
                        },
                        encode,
                        name="preview.mp4",
                        media_type="video/mp4",
                    )
                )
                background.append(task)
                await asyncio.shield(task)
                return cls.outputs(out="saved")

        registry = TypeRegistry()
        register_core_types(registry)
        schema = Save.schema()
        worker = InProcessWorker({schema.node_type: Save}, registry)
        invocation = asyncio.create_task(
            worker.invoke(
                Invocation(
                    "cancelled",
                    "save",
                    schema.node_type,
                    {},
                    schema,
                    saved_file_previews_enabled=True,
                )
            )
        )
        try:
            async with asyncio.timeout(10):
                await entered.wait()
                invocation.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await invocation
                assert not finished.is_set()
        finally:
            release.set()
            if background:
                assert await background[0] is False
        assert calls == 1
        vault = AssetVault(tmp_path / "scratch" / "produced-assets")
        assert vault.digests() == []
        assert not list(vault.root.glob(".ingest-*"))

    run_scenario(scenario())


@pytest.mark.parametrize("failure", ("", "revoked", "offline", "unselected", "corrupt", "catalog"))
def test_encoded_retention_requires_live_output_and_recovers_catalog(
    tmp_path: Path, mp4: bytes, failure: str
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    mounts = MountTable()
    mounts.add(MountDef("renders", output, "readwrite"))
    mounts.scan("renders")
    vault = AssetVault(tmp_path / "vault")
    database = tmp_path / "library.sqlite"
    library = ServerLibrary(
        vault=vault,
        store=LibraryStore(database),
        resolver=mounts,
        upload_writer=AssetWriter(mounts),
        project_folders_for=lambda _project: AssetFolderContext(
            ("renders",), None if failure == "unselected" else "renders"
        ),
    )
    artifacts = run_scenario(
        PreviewArtifactPublisher(vault)(
            "nested/encode", (EncodedPreviewArtifact(mp4, "movie.mp4", "video/mp4"),)
        )
    )
    job = Job(
        JobKey("client", "job"), Graph(nodes={}), (), 0, "run-7", "fingerprint", project="film"
    )
    result = RunResult("run-7", {}, (), (), (), artifacts=artifacts)
    if failure == "revoked":
        mounts.remove("renders")
    elif failure == "offline":
        output.rename(tmp_path / "disconnected-output")
    elif failure == "corrupt":
        path = vault.resolve(digest_bytes(mp4))
        assert path is not None
        path.write_bytes(b"corrupt")
    elif failure == "catalog":
        with sqlite3.connect(database) as connection:
            connection.execute(
                "CREATE TRIGGER fail_retention BEFORE INSERT ON asset_observations "
                "BEGIN SELECT RAISE(ABORT, 'catalog offline'); END"
            )
    if failure:
        with pytest.raises((AssetError, OSError)):
            library.record_saved_outputs(job, result)
        assert library.store.query_assets("local", project="film") == []
        assert list(output.glob("*.mp4")) == (
            [output / "movie.mp4"] if failure == "catalog" else []
        )
        if failure != "catalog":
            assert not list(vault.root.glob(".upload-observations/*.json"))
            return
        with sqlite3.connect(database) as connection:
            connection.execute("DROP TRIGGER fail_retention")
    else:
        with ThreadPoolExecutor(max_workers=3) as pool:
            completed = list(
                pool.map(lambda _index: library.record_saved_outputs(job, result), range(3))
            )
        assert {item.artifacts[0].virtual_path for item in completed} == {
            "mounts/renders/movie.mp4",
            "mounts/renders/movie_00001.mp4",
            "mounts/renders/movie_00002.mp4",
        }
        assert all(path.read_bytes() == mp4 for path in output.glob("*.mp4"))
    library.store.close()
    restarted = ServerLibrary(
        vault=AssetVault(vault.root),
        store=LibraryStore(database),
        resolver=mounts,
        upload_writer=AssetWriter(mounts),
        project_folders_for=library.project_folders_for,
    )
    recover_asset_observations(restarted)
    observations = restarted.store.query_assets("local", project="film")[0]
    assert any(record.asset.virtual_path == "mounts/renders/movie.mp4" for record in observations)
    assert all(
        dict(record.origin) == {"kind": "produced", "job": "run-7", "node": "nested/encode"}
        for record in observations
    )
    assert not list(vault.root.glob(".upload-observations/*.json"))
    # A kept file has its own mounted bytes, not just a cache or vault residency.
    path = vault.resolve(digest_bytes(mp4))
    assert path is not None
    path.unlink()
    assert (output / "movie.mp4").read_bytes() == mp4
    assert mounts.resolve(digest_bytes(mp4)) is not None


@pytest.mark.parametrize(
    "sources", ("none", "available", "missing", "corrupt", "size-mismatch", "noncanonical")
)
@pytest.mark.parametrize("fail", (False, True))
@pytest.mark.parametrize("remote", (False, True), ids=("isolated", "remote"))
def test_image_document_attachments_adopt_sources_before_worker_teardown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mp4: bytes,
    fail: bool,
    sources: str,
    remote: bool,
) -> None:
    from dinkster_assets import AssetResolver
    from dinkster_caches import BudgetedDiskCAS
    from dinkster_image_document import ImageDocument
    from dinkster_image_document.document import empty_document
    from dinkster_protocol import Invocation
    from dinkster_values import CORE_STRING, TypeRegistry, Value, register_core_types
    from dinkster_workers import IsolatedWorker, RemoteWorker
    from test_image_document_rendering import _png, _resource
    from test_remote import TOKEN, start_service, stop_service

    source = _png([[(73, 21, 190, 255)]])
    digest = digest_bytes(source)
    producer = AssetVault(tmp_path / "producer")
    engine = AssetVault(tmp_path / "engine")
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(engine.root))
    record = empty_document(3, 2).to_record()
    if sources != "none":
        record["allocation"]["nextOrdinal"] = 1
        record["resources"] = {"r0": _resource("r0", source, 1, 1)}
        if sources == "size-mismatch":
            record["resources"]["r0"]["byteSize"] = len(source) + 1
        if sources != "missing":
            with producer.writer(digest) as writer:
                writer.write(source)
                writer.commit()
            if sources == "corrupt":
                path = producer.resolve(digest)
                assert path is not None
                path.write_bytes(b"not the declared image")
    original = ImageDocument.from_record(record).data
    if sources == "noncanonical":
        original = original.replace(b'{"', b'{ "', 1)
    manifest = _write_preview_pack(tmp_path, mp4, document=original)

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        process = None
        transfer = BudgetedDiskCAS(tmp_path / "transfer", max_bytes=1)
        worker: IsolatedWorker | RemoteWorker
        if remote:
            process, host, port = await start_service(
                manifest,
                tmp_path,
                "--asset-vault",
                str(producer.root),
                "--value-store",
                str(tmp_path / "daemon-transfer"),
                pythonpath=(tmp_path,),
            )
            worker = RemoteWorker(
                host, port, TOKEN, registry, name="documents", value_store=transfer
            )
        else:
            worker = IsolatedWorker(
                manifest,
                registry,
                extra_env={
                    "PYTHONPATH": str(tmp_path),
                    "DINKSTER_ASSET_VAULT": str(producer.root),
                },
            )
        try:
            await worker.start()
            if remote:
                capture = worker._session._produced_assets.capture

                def pressure_during_adoption(
                    outputs: Mapping[str, Value],
                    transferred: AssetResolver | None = None,
                    inputs: Mapping[str, Value] | None = None,
                ) -> None:
                    transfer.put(b"another result arrived before document adoption")
                    capture(outputs, transferred, inputs)

                monkeypatch.setattr(
                    worker._session._produced_assets, "capture", pressure_during_adoption
                )
            schema = worker.schemas["preview-fixture.emit"]
            result = await worker.invoke(
                Invocation(
                    "document",
                    "node",
                    schema.node_type,
                    {"mode": registry.wrap(CORE_STRING, "document-fail" if fail else "document")},
                    schema,
                )
            )
        finally:
            await worker.close()
            if process is not None:
                await stop_service(process)
        assert not worker._session._received_asset_pins
        shutil.rmtree(transfer.root)
        if fail or sources in ("missing", "corrupt", "size-mismatch", "noncanonical"):
            assert result.error is not None
            assert result.image_documents == result.preview_artifacts == ()
            assert engine.resolve(digest) is None
        else:
            assert result.error is None
            assert [item.data for item in result.image_documents] == [
                original,
                empty_document(7, 3).data,
            ]
            assert [item.data for item in result.preview_artifacts] == [mp4]
            if sources == "available":
                path = producer.resolve(digest)
                assert path is not None
                path.unlink()
                adopted = engine.resolve(digest)
                assert adopted is not None and adopted.read_bytes() == source
                inventory = ProducedAssetInventory(engine.root).entries()
                assert any(item.digest == digest and item.adopted_at_ns for item in inventory)

    run_scenario(scenario())


@pytest.mark.parametrize("fail", (False, True))
def test_image_documents_cross_successful_invocations_without_becoming_previews(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool, mp4: bytes
) -> None:
    from dinkster_api.v1 import CORE_STRING, Node, NodeSchema, OutputSpec, TypeExpr
    from dinkster_assets.declared import produced_output_scratch
    from dinkster_image.types import register_image_types
    from dinkster_image_document import ImageDocument
    from dinkster_image_document.document import empty_document
    from dinkster_protocol import Invocation
    from dinkster_schema.reporting import publish_image_document
    from dinkster_values import TypeRegistry, register_core_types
    from dinkster_workers import InProcessWorker
    from dinkster_workers.boundary import ValueCodec, decode_result, encode_result
    from test_image_document_rendering import _png, _resource

    source = _png([[(22, 48, 7, 255)]])
    digest = digest_bytes(source)
    record = empty_document(3, 2).to_record()
    record["allocation"]["nextOrdinal"] = 1
    record["resources"] = {"r0": _resource("r0", source, 1, 1)}
    original = ImageDocument.from_record(record).data
    edited = empty_document(7, 3).data
    vault = AssetVault(tmp_path / "host-vault")
    scratch = tmp_path / "pack-scratch"
    scratch.mkdir()
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
    monkeypatch.setenv("DINKSTER_PACK_SCRATCH", str(scratch))

    class Document(Node):
        @classmethod
        def define_schema(cls) -> NodeSchema:
            return NodeSchema(
                node_type="document-fixture.emit",
                outputs=(OutputSpec("out", TypeExpr.concrete(CORE_STRING)),),
            )

        @classmethod
        def execute(cls) -> Mapping[str, object]:
            produced = produced_output_scratch()
            assert produced is not None
            with produced.writer(digest) as writer:
                writer.write(source)
                writer.commit()
            publish_image_document(original)
            publish_image_document(edited)
            publish_preview_artifact(mp4, name="preview.mp4", media_type="video/mp4")
            if fail:
                raise ValueError("document fixture failed")
            return cls.outputs(out="unchanged output")

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_image_types(registry)
        schema = Document.schema()
        worker = InProcessWorker({schema.node_type: Document}, registry)
        result = await worker.invoke(Invocation("document", "node", schema.node_type, {}, schema))
        if fail:
            assert result.error is not None
            assert result.image_documents == result.preview_artifacts == ()
            header, blobs, segments = encode_result(ValueCodec(registry), result, "document", 0)
            assert "imageDocuments" not in header and blobs == [] and segments == []
            assert vault.resolve(digest) is None
            return
        assert result.error is None
        shutil.rmtree(scratch)
        adopted = vault.resolve(digest)
        assert adopted is not None and adopted.read_bytes() == source
        assert [item.data for item in result.image_documents] == [original, edited]
        codec = ValueCodec(registry)
        header, blobs, segments = encode_result(codec, result, "document", 0)
        assert not segments
        decoded, _ = decode_result(codec, header, blobs, [])
        assert [item.data for item in decoded.image_documents] == [original, edited]
        assert decoded.preview_artifacts == result.preview_artifacts
        assert decoded.outputs is not None
        assert decoded.outputs["out"].payload.load() == "unchanged output"

    run_scenario(scenario())


def test_image_document_attachment_rejects_closed_copied_context() -> None:
    from contextvars import copy_context

    from dinkster_schema.reporting import capture_image_documents, publish_image_document

    with capture_image_documents() as documents:
        copied = copy_context()
        publish_image_document(b"original")
    with pytest.raises(RuntimeError, match="closed"):
        copied.run(publish_image_document, b"late")
    assert [item.data for item in documents] == [b"original"]


@pytest.mark.parametrize("in_process", (True, False))
@pytest.mark.parametrize("publication", ("available", "missing", "failed"))
def test_original_documents_publish_without_output_folder_and_survive_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mp4: bytes,
    in_process: bool,
    publication: str,
) -> None:
    from dinkster_image_document import ImageDocument
    from dinkster_image_document.document import empty_document
    from test_image_document_rendering import _png, _resource

    source = _png([[(73, 21, 190, 255)]])
    record = empty_document(3, 2).to_record()
    record["allocation"]["nextOrdinal"] = 1
    record["resources"] = {"r0": _resource("r0", source, 1, 1)}
    original = ImageDocument.from_record(record).data
    expected = {digest_bytes(original): original}
    second = empty_document(7, 3).data
    expected[digest_bytes(second)] = second
    manifest = _write_preview_pack(tmp_path, mp4, document=original)
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest_bytes(source)) as writer:
        writer.write(source)
        writer.commit()
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
    library = ServerLibrary(vault=vault, store=LibraryStore(tmp_path / "library.sqlite"))

    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={"PYTHONPATH": str(tmp_path)},
            cache_mode="layered",
            cache_dir=tmp_path / "cache",
        )
        try:
            await composer.add_pack(PackSpec(manifest, in_process=in_process))
            graph = Graph(
                nodes={"original": GraphNode("preview-fixture.emit", {"mode": "document-only"})}
            )
            direct = composer.composition.make_engine(lambda _event: None)
            result = await direct.run(graph, ["original"])
            assert result.diagnostics == ()
            assert result.artifacts == ()
            assert all(vault.resolve(digest) is None for digest in expected)
            if publication == "failed":

                def fail_commit(_writer: VaultWriter) -> tuple[Path, bool]:
                    raise AssetError("document storage unavailable")

                monkeypatch.setattr(VaultWriter, "commit_with_result", fail_commit)
            client = TestClient(
                TestServer(
                    create_app(
                        composer.composition.make_engine,
                        composer.composition.schemas,
                        asset_export=vault,
                        library=library if publication != "missing" else None,
                    )
                )
            )
            await client.start_server()
            try:
                artifacts = None
                for job_id in ("first", "cached"):
                    response = await client.post(
                        "/api/jobs",
                        json={
                            "clientId": "documents",
                            "jobId": job_id,
                            "project": "film",
                            "graph": graph_to_wire(graph),
                            "targets": ["original"],
                        },
                    )
                    assert response.status == 202
                    async with asyncio.timeout(10):
                        while True:
                            status = await (
                                await client.get(f"/api/jobs/documents/{job_id}")
                            ).json()
                            if status["state"] not in ("pending", "running"):
                                break
                            await asyncio.sleep(0)
                    if publication != "available":
                        assert status["state"] == "failed", status
                        assert not status.get("outputs")
                        assert not status.get("artifacts")
                        assert all(vault.resolve(digest) is None for digest in expected)
                        assert library.store.query_assets("local", project="film") == []
                        continue
                    assert status["state"] == "completed", status
                    result = status
                    assert {item["digest"] for item in result["artifacts"]} == set(expected)
                    if artifacts is not None:
                        assert result["artifacts"] == artifacts
                        assert result["cached"] == ["original"]
                    artifacts = result["artifacts"]
                    for digest, data in expected.items():
                        asset = await client.get(f"/assets/{digest}")
                        assert asset.status == 200
                        assert await asset.read() == data
                if publication != "available":
                    return
                restarted = composer.composition.make_engine(lambda _event: None)
                cached = await restarted.run(
                    graph,
                    ["original"],
                    execution=replace(
                        restarted.pin_execution(),
                        publish_image_documents=library.publish_image_documents,
                        require_image_document_publication=True,
                    ),
                )
                assert cached.cached == ("original",)
                assert {item.digest for item in cached.artifacts} == set(expected)
                observations = library.store.query_assets("local", project="film")
                assert {item.asset.digest for group in observations for item in group} == set(
                    expected
                )
                assert list(library.vault.root.glob(".upload-observations/*.json")) == []
                assert library.store.get_dependencies(digest_bytes(original)) == [
                    {
                        "resourceId": "r0",
                        "digest": digest_bytes(source),
                        "byteSize": len(source),
                        "mediaType": "image/png",
                        "width": 1,
                        "height": 1,
                        "colorSpace": "srgb",
                        "channelDepth": 8,
                        "alphaMode": "straight",
                    }
                ]
            finally:
                await client.close()
        finally:
            await composer.close()
            library.store.close()

    run_scenario(scenario())


@pytest.mark.parametrize(
    "invalid",
    (
        "json",
        "noncanonical",
        "missing",
        "size",
        "dimensions",
        "corrupt",
        "inline-dimensions",
        "inline-decode",
    ),
)
def test_original_document_publisher_validates_entire_batch_before_writing(
    tmp_path: Path, invalid: str
) -> None:
    from dinkster_image_document import ImageDocument, InvalidDocument
    from dinkster_image_document.document import empty_document
    from dinkster_schema.reporting import ImageDocumentArtifact
    from test_image_document_rendering import _png, _resource

    source = _png([[(73, 21, 190, 255)]])
    record = empty_document(3, 2).to_record()
    record["allocation"]["nextOrdinal"] = 1
    resource = _resource("r0", source, 1, 1)
    record["resources"] = {"r0": resource}
    vault = AssetVault(tmp_path / "vault")
    if invalid != "missing":
        with vault.writer(digest_bytes(source)) as writer:
            writer.write(source)
            writer.commit()
    if invalid == "size":
        resource["byteSize"] = len(source) + 1
    if invalid == "dimensions":
        resource["width"] = 2
    if invalid.startswith("inline-"):
        record["formatVersion"] = 2
        if invalid == "inline-dimensions":
            resource["width"] = 2
        else:
            source = b"not a raster"
            resource["digest"] = digest_bytes(source)
            resource["byteSize"] = len(source)
        resource["inline"] = base64.b64encode(source).decode("ascii")
    if invalid == "corrupt":
        path = vault.resolve(digest_bytes(source))
        assert path is not None
        path.write_bytes(b"not the declared image")
    data = ImageDocument.from_record(record).data
    if invalid == "json":
        data = b"not JSON"
    if invalid == "noncanonical":
        data = data.replace(b'{"', b'{ "', 1)
    library = ServerLibrary(vault=vault, store=LibraryStore(tmp_path / "library.sqlite"))
    valid = empty_document(8, 5).data

    async def scenario() -> None:
        try:
            with pytest.raises((InvalidDocument, AssetError)):
                await library.publish_image_documents(
                    "node",
                    (
                        ImageDocumentArtifact(valid),
                        ImageDocumentArtifact(data),
                    ),
                )
            assert vault.resolve(digest_bytes(valid)) is None
            assert vault.resolve(digest_bytes(data)) is None
            assert library.store.get_dependencies(digest_bytes(valid)) is None
            assert list(vault.root.glob(".ingest-*")) == []
        finally:
            library.store.close()

    run_scenario(scenario())


def test_original_document_cancellation_discards_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_image_document.document import empty_document
    from dinkster_schema.reporting import ImageDocumentArtifact

    vault = AssetVault(tmp_path / "vault")
    library = ServerLibrary(vault=vault, store=LibraryStore(tmp_path / "library.sqlite"))
    data = empty_document(8, 5).data
    entered = threading.Event()
    release = threading.Event()
    write = VaultWriter.write

    def blocked_write(self: VaultWriter, data: bytes) -> None:
        write(self, data)
        entered.set()
        assert release.wait(10)

    monkeypatch.setattr(VaultWriter, "write", blocked_write)

    async def scenario() -> None:
        task = asyncio.create_task(
            library.publish_image_documents("node", (ImageDocumentArtifact(data),))
        )
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            with claim_inactive_produced_vault(vault.root) as inactive:
                assert not inactive
            task.cancel()
        finally:
            release.set()
        try:
            with pytest.raises(asyncio.CancelledError):
                await task
            assert vault.digests() == []
            assert library.store.get_dependencies(digest_bytes(data)) is None
            assert list(vault.root.glob(".ingest-*")) == []
            with claim_inactive_produced_vault(vault.root) as inactive:
                assert inactive
        finally:
            library.store.close()

    run_scenario(scenario())


def test_original_document_commit_failure_retains_shared_bytes_and_can_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dinkster_image_document.document import empty_document
    from dinkster_schema.reporting import ImageDocumentArtifact

    vault = AssetVault(tmp_path / "vault")
    library = ServerLibrary(vault=vault, store=LibraryStore(tmp_path / "library.sqlite"))
    first, second = empty_document(3, 2).data, empty_document(8, 5).data
    calls = 0
    commit = VaultWriter.commit_with_result

    def fail_commit(self: VaultWriter) -> tuple[Path, bool]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise AssetError("commit failed")
        result = commit(self)
        with vault.writer(digest_bytes(first)) as independent:
            independent.write(first)
            commit(independent)
        return result

    async def scenario() -> None:
        claims = (ImageDocumentArtifact(first), ImageDocumentArtifact(second))
        try:
            with monkeypatch.context() as patch:
                patch.setattr(VaultWriter, "commit_with_result", fail_commit)
                with pytest.raises(AssetError, match="commit failed"):
                    await library.publish_image_documents("node", claims)
            retained = vault.resolve(digest_bytes(first))
            assert retained is not None and retained.read_bytes() == first
            assert list(vault.root.glob(".ingest-*")) == []
            with claim_inactive_produced_vault(vault.root) as inactive:
                assert inactive
            artifacts = await library.publish_image_documents("node", claims)
            assert [item.digest for item in artifacts] == [
                digest_bytes(first),
                digest_bytes(second),
            ]
            for data in (first, second):
                retained = vault.resolve(digest_bytes(data))
                assert retained is not None and retained.read_bytes() == data
        finally:
            library.store.close()

    run_scenario(scenario())


@pytest.mark.parametrize("table", ("asset_dependencies", "asset_observations"))
def test_original_document_catalog_failure_recovers_after_restart(
    tmp_path: Path, table: str
) -> None:
    from dinkster_image_document import ImageDocument
    from dinkster_image_document.document import empty_document
    from dinkster_schema.reporting import ImageDocumentArtifact
    from test_image_document_rendering import _png, _resource

    source = _png([[(73, 21, 190, 255)]])
    record = empty_document(3, 2).to_record()
    record["allocation"]["nextOrdinal"] = 1
    record["resources"] = {"r0": _resource("r0", source, 1, 1)}
    data = ImageDocument.from_record(record).data
    vault = AssetVault(tmp_path / "vault")
    with vault.writer(digest_bytes(source)) as writer:
        writer.write(source)
        writer.commit()
    database = tmp_path / "library.sqlite"
    library = ServerLibrary(vault=vault, store=LibraryStore(database))
    try:
        with sqlite3.connect(database) as connection:
            connection.execute(
                f"CREATE TRIGGER fail_document BEFORE INSERT ON {table} "
                "BEGIN SELECT RAISE(ABORT, 'catalog offline'); END"
            )
        artifacts = run_scenario(
            library.publish_image_documents("create", (ImageDocumentArtifact(data),))
        )
        job = Job(JobKey("client", "job"), Graph(nodes={}), (), 0, "run", "fp", project="film")
        with pytest.raises(AssetError, match="catalog could not be committed"):
            library.record_saved_outputs(job, RunResult("run", {}, (), (), (), artifacts=artifacts))
        assert len(list(vault.root.glob(".upload-observations/*.json"))) == 1
        assert library.store.query_assets("local", project="film") == []
        with sqlite3.connect(database) as connection:
            connection.execute("DROP TRIGGER fail_document")
    finally:
        library.store.close()
    restarted = ServerLibrary(vault=vault, store=LibraryStore(database))
    try:
        recover_asset_observations(restarted)
        assert list(vault.root.glob(".upload-observations/*.json")) == []
        dependencies = restarted.store.get_dependencies(digest_bytes(data))
        assert dependencies is not None and len(dependencies) == 1
        assert dependencies[0]["resourceId"] == "r0"
        assert dependencies[0]["digest"] == digest_bytes(source)
        assert dependencies[0]["byteSize"] == len(source)
        observations = restarted.store.query_assets("local", project="film")
        assert [item.asset.digest for group in observations for item in group] == [
            digest_bytes(data)
        ]
        retained = vault.resolve(digest_bytes(data))
        assert retained is not None and retained.read_bytes() == data
    finally:
        restarted.store.close()


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

    run_scenario(scenario())
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

    run_scenario(scenario())


@pytest.mark.parametrize("pre_existing", (False, True))
def test_failed_batch_preserves_content_adopted_by_an_independent_ingest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mp4: bytes, pre_existing: bool
) -> None:
    vault = AssetVault(tmp_path / "vault")
    publisher = PreviewArtifactPublisher(vault)
    second = _tiny_mp4(99)
    existing = digest_bytes(mp4)
    if pre_existing:
        with vault.writer(existing) as writer:
            writer.write(mp4)
            writer.commit()
    calls = 0
    original = VaultWriter.commit_with_result

    def fail_second(self: VaultWriter) -> tuple[Path, bool]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise AssetError("commit failed")
        result = original(self)
        with vault.writer(existing) as independent:
            independent.write(mp4)
            adopted_path, created = original(independent)
            assert not created
            assert adopted_path.read_bytes() == mp4
        return result

    monkeypatch.setattr(VaultWriter, "commit_with_result", fail_second)

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
        stored = vault.resolve(existing)
        assert stored is not None and stored.read_bytes() == mp4
        assert list(vault.root.glob(".ingest-*")) == []

    run_scenario(scenario())


@pytest.mark.parametrize("existing", ["absent", "pending", "untracked"])
def test_concurrent_identical_previews_do_not_delete_the_other_result(
    tmp_path: Path, mp4: bytes, existing: str
) -> None:
    vault = AssetVault(tmp_path / "vault")
    publisher = PreviewArtifactPublisher(vault)
    claim = EncodedPreviewArtifact(mp4, "preview.mp4", "video/mp4")
    digest = digest_bytes(mp4)
    if existing == "pending":
        ProducedAssetInventory(vault.root).pending(digest, len(mp4))
    if existing != "absent":
        with vault.writer(digest) as writer:
            writer.write(mp4)
            writer.commit()

    async def scenario() -> None:
        first, second = await asyncio.gather(publisher("one", (claim,)), publisher("two", (claim,)))
        assert first[0].digest == second[0].digest == digest_bytes(mp4)
        stored = vault.resolve(digest_bytes(mp4))
        assert stored is not None
        assert stored.read_bytes() == mp4
        if existing == "untracked":
            assert not (vault.root / ".produced-assets.sqlite").exists()
        else:
            entries = ProducedAssetInventory(vault.root).entries()
            assert len(entries) == 1
            assert (entries[0].digest, entries[0].size) == (digest, len(mp4))
            assert entries[0].adopted_at_ns is not None
            await publisher("again", (claim,))
            assert ProducedAssetInventory(vault.root).entries() == entries

    run_scenario(scenario())


def test_cancellation_during_staging_discards_before_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mp4: bytes
) -> None:
    vault = AssetVault(tmp_path / "vault")
    publisher = PreviewArtifactPublisher(vault)
    entered = threading.Event()
    release = threading.Event()
    original = publisher._stage

    def blocked_stage(node_id: str, claims: Sequence[EncodedPreviewArtifact]) -> object:
        staged = original(node_id, claims)
        entered.set()
        assert release.wait(10)
        return staged

    monkeypatch.setattr(publisher, "_stage", blocked_stage)

    async def scenario() -> None:
        task = asyncio.create_task(
            publisher("node", (EncodedPreviewArtifact(mp4, "preview.mp4", "video/mp4"),))
        )
        assert await asyncio.to_thread(entered.wait, 10)
        with claim_inactive_produced_vault(vault.root) as inactive:
            assert not inactive
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert vault.digests() == []
        entries = ProducedAssetInventory(vault.root).entries()
        assert len(entries) == 1 and entries[0].adopted_at_ns is None
        with claim_inactive_produced_vault(vault.root) as inactive:
            assert inactive

    run_scenario(scenario())
