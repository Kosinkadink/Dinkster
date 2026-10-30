from __future__ import annotations

import asyncio
import base64
import io
import os
import shutil
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
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
from dinkster_assets.vault import VaultWriter
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EventListener, ExecutionError, RunResult
from dinkster_graph import Graph, GraphNode, graph_to_wire
from dinkster_protocol import AssetFolderContext
from dinkster_schema.reporting import EncodedPreviewArtifact, capture_preview_artifacts
from dinkster_server import ServerLibrary, create_app
from dinkster_server.library import recover_asset_observations
from dinkster_server.queue import Job, JobKey
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
        "  if mode.startswith('mounted'):\n"
        "   writer = MountSnapshotWriter(os.environ['DINKSTER_MOUNTS_SNAPSHOT'])\n"
        "   for index in range(65 if mode == 'mounted-batch' else 2):\n"
        "    ref = AssetWriter(writer).save_bytes(SaveTarget('renders', f'saved-{index}'), "
        "b'duplicate saved bytes', suffix='.exr', media_type='image/x-exr')\n"
        "    claim = ref.to_wire()\n"
        "    if mode == 'mounted-forged': claim['digest'] = 'blake3:' + 'a'*64\n"
        "    attach_saved_artifact(claim)\n"
        "    if mode == 'mounted-mismatch': claim = {**claim, 'size': ref.size + 1}\n"
        "    def encode():\n"
        "     if mode == 'mounted-encode-fail': raise RuntimeError('encoder failed')\n"
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

    asyncio.run(scenario())


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
            for mode in ("mounted", "mounted-batch", "mounted-mismatch", "mounted-encode-fail"):
                graph = Graph(nodes={"save": GraphNode("preview-fixture.emit", {"mode": mode})})
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
                finally:
                    library.store.close()
            with pytest.raises(ExecutionError):
                await engine.run(
                    Graph(
                        nodes={
                            "save": GraphNode("preview-fixture.emit", {"mode": "mounted-forged"})
                        }
                    ),
                    ["save"],
                )
        finally:
            await composer.close()

    asyncio.run(scenario())


def test_saved_file_preview_transport_adopts_bytes_before_worker_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mp4: bytes
) -> None:
    from dinkster_protocol import Invocation
    from dinkster_values import CORE_STRING, TypeRegistry, register_core_types
    from dinkster_workers import IsolatedWorker

    manifest = _write_preview_pack(tmp_path, mp4)
    vault = AssetVault(tmp_path / "host-vault")
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
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
        finally:
            await worker.close()
        shutil.rmtree(scratch)
        path = vault.resolve(digest_bytes(mp4))
        assert path is not None and path.read_bytes() == mp4

    asyncio.run(scenario())


@pytest.mark.parametrize("count,enabled", [(65, True), (4096, True), (65, False)])
def test_saved_file_previews_are_lazy_file_backed_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, enabled: bool, mp4: bytes
) -> None:
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

    def encode() -> bytes:
        nonlocal calls
        calls += 1
        return mp4

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
        result = await worker.invoke(
            Invocation(
                "saved-preview",
                "save",
                schema.node_type,
                {},
                schema,
                saved_file_previews_enabled=enabled,
            )
        )
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
            assert AssetVault(tmp_path / "scratch" / "produced-assets").digests() == [
                digest_bytes(mp4)
            ]

    asyncio.run(scenario())
    assert calls == (count if enabled else 0)


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
    artifacts = asyncio.run(
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
