"""Saved VIDEO_EDIT workflows produce the same lazy VIDEO in every worker lane."""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import pytest
from aiohttp import web
from dinkster_assets import AssetRef, AssetVault, digest_bytes
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, graph_from_wire, graph_to_wire
from dinkster_nodes_media_io import MEDIA_IO_NODES, CropVideo, TrimVideo, register_media_types
from dinkster_schema import build_node_types
from dinkster_values import (
    TypeRegistry,
    decode_video,
    encode_video,
    register_core_types,
    validate_video_encoded,
    video_from_source,
    video_meta,
)
from dinkster_video import disassemble_video
from dinkster_workers import BoundaryDiagnostic, InProcessWorker, IsolatedWorker, RemoteWorker
from test_remote import TOKEN, start_service, stop_service

MEDIA_MANIFEST = (
    Path(__file__).resolve().parents[1] / "packages/dinkster-nodes-media-io/dinkster-pack.toml"
)
ASSET_TOKEN = "video-editor-asset-token-0123456789"


def _source(*, width: int = 64, height: int = 32, frame_count: int = 20) -> bytes:
    output = io.BytesIO()
    with av.open(output, "w", format="mp4") as opened:
        mux = cast(Any, opened)
        stream = mux.add_stream("libx264", rate=10)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = 1
        stream.codec_context.max_b_frames = 0
        stream.gop_size = 5
        stream.options = {"crf": "0", "x264-params": "scenecut=0"}
        for index in range(frame_count):
            pixels = np.empty((height, width, 3), np.uint8)
            pixels[..., 0] = index * 10
            pixels[..., 1] = np.arange(width, dtype=np.uint8)
            pixels[..., 2] = np.arange(height, dtype=np.uint8)[:, None]
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 10)
            for packet in stream.encode(frame):
                mux.mux(packet)
        for packet in stream.encode():
            mux.mux(packet)
    return output.getvalue()


def _registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    register_media_types(registry)
    return registry


def _saved_graph(
    ref: AssetRef, edit: Mapping[str, object], *, strict_duration: bool = False
) -> tuple[str, Graph]:
    graph = Graph(
        {
            "load": GraphNode("dinkster.load_video_value", {"video": ref.to_wire()}),
            "trim": GraphNode(
                "dinkster.video.trim",
                {
                    "video": Link("load", "video"),
                    "video_edit": dict(edit),
                    "strict_duration": strict_duration,
                },
            ),
            "crop": GraphNode(
                "dinkster.video.crop",
                {"video": Link("trim", "video"), "video_edit": dict(edit)},
            ),
            "frames": GraphNode("dinkster.video.disassemble", {"video": Link("crop", "video")}),
        }
    )
    document = json.dumps(graph_to_wire(graph), sort_keys=True, separators=(",", ":"))
    return document, graph_from_wire(json.loads(document))


def _resolved_outputs(result: object) -> tuple[bytes, np.ndarray]:
    outputs = cast(Any, result).outputs
    video = outputs["crop"]["video"].resolve()
    frames = cast(np.ndarray, outputs["frames"]["images"].resolve())
    return encode_video(video), frames


class _AssetHost:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.digest = digest_bytes(data)
        self.requests = 0
        self.endpoint = ""
        self._runner: web.AppRunner | None = None

    async def _get(self, request: web.Request) -> web.Response:
        if request.headers.get("Authorization") != f"Bearer {ASSET_TOKEN}":
            return web.Response(status=401)
        if request.match_info["digest"] != self.digest:
            return web.Response(status=404)
        self.requests += 1
        return web.Response(body=self.data)

    async def __aenter__(self) -> _AssetHost:
        app = web.Application()
        app.router.add_get("/assets/{digest}", self._get)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        host, port = self._runner.addresses[0]
        self.endpoint = f"http://{host}:{port}"
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        assert self._runner is not None
        await self._runner.cleanup()


@pytest.mark.parametrize(
    ("widget", "expected"),
    [
        ({}, (64, 32, [2, 1])),
        ({"trim": {}}, (64, 32, [2, 1])),
        ({"crop": {}}, (64, 32, [2, 1])),
        ({"trim": {"start_time": -0.5, "duration": 0}}, (64, 32, [1, 2])),
        ({"crop": {"x": 7, "y": 3, "width": 31, "height": 17}}, (30, 16, [2, 1])),
        ({"crop": {"x": 9, "y": 9, "width": 0, "height": -1}}, (64, 32, [2, 1])),
        ({"crop": {"x": 9, "y": 9, "width": 0, "height": 17}}, (64, 32, [2, 1])),
        ({"crop": {"x": 9, "y": 9, "width": 31, "height": 0}}, (64, 32, [2, 1])),
        ({"crop": {"x": 63, "y": 31, "width": 50, "height": 50}}, (2, 2, [2, 1])),
    ],
)
def test_video_edit_sections_sentinels_and_clamping(
    widget: dict[str, object], expected: tuple[int, int, list[int]]
) -> None:
    raw = json.dumps(widget, sort_keys=True)
    source = video_from_source(_source())
    trimmed = TrimVideo.execute(video=source, video_edit=widget)["video"]
    result = CropVideo.execute(video=trimmed, video_edit=widget)["video"]
    effective = cast(Mapping[str, object], video_meta(result)["effective"])
    assert (effective["width"], effective["height"], effective["duration"]) == expected
    assert json.dumps(widget, sort_keys=True) == raw


def test_raw_editor_document_survives_save_reload_and_import_unchanged() -> None:
    before = {
        "vendor": {"future": [1, "keep"]},
        "trim": {"start_time": 1.25, "duration": 3.5, "extra": True},
        "crop": {"x": 100, "y": 40, "width": 1280, "height": 720, "extra": "keep"},
    }
    trim_after = json.loads(json.dumps(before))
    trim_after["trim"].update({"start_time": 2, "duration": 0})
    crop_after = json.loads(json.dumps(before))
    crop_after["crop"].update({"x": 13, "y": 5, "width": 201, "height": 101})
    source = _source()
    ref = AssetRef(digest_bytes(source), "source.mp4", len(source))

    for after in (trim_after, crop_after):
        saved, reloaded = _saved_graph(ref, after, strict_duration=True)
        imported = graph_from_wire(json.loads(json.dumps(graph_to_wire(reloaded))))
        for graph in (reloaded, imported):
            trim_inputs = cast(GraphNode, graph.nodes["trim"]).inputs
            crop_inputs = cast(GraphNode, graph.nodes["crop"]).inputs
            assert trim_inputs["video_edit"] == after
            assert crop_inputs["video_edit"] == after
            assert trim_inputs["strict_duration"] is True
            assert "strict_duration" not in cast(Mapping[str, object], trim_inputs["video_edit"])
            assert "features" not in trim_inputs
            assert "features" not in crop_inputs
        assert json.loads(saved)["nodes"]["trim"]["inputs"]["video_edit"] == after
    assert before == {
        "vendor": {"future": [1, "keep"]},
        "trim": {"start_time": 1.25, "duration": 3.5, "extra": True},
        "crop": {"x": 100, "y": 40, "width": 1280, "height": 720, "extra": "keep"},
    }


def test_raw_crop_persists_while_runtime_uses_even_aligned_values() -> None:
    raw_crop = {"x": 13, "y": 5, "width": 201, "height": 101}
    source = video_from_source(_source(width=256, height=128, frame_count=3))
    raw = CropVideo.execute(video=source, video_edit={"crop": raw_crop})["video"]
    normalized = CropVideo.execute(video=source, x=12, y=4, width=200, height=100)["video"]

    assert decode_video(encode_video(raw))["edits"] == [{"crop": raw_crop}]
    assert cast(Mapping[str, object], video_meta(raw)["effective"])["width"] == 200
    assert cast(Mapping[str, object], video_meta(raw)["effective"])["height"] == 100
    np.testing.assert_array_equal(
        disassemble_video(raw)["images"], disassemble_video(normalized)["images"]
    )


def test_widget_sections_are_authoritative_and_scalars_remain_linkable() -> None:
    source = video_from_source(_source())
    unchanged = TrimVideo.execute(
        video=source,
        video_edit={"crop": {"width": 20}},
        start_time=0.5,
        duration=0.5,
    )["video"]
    assert encode_video(unchanged) == encode_video(source)

    linked = Graph(
        {
            "load": GraphNode("dinkster.load_video_value", {"video": {}}),
            "info": GraphNode("dinkster.video.info", {"video": Link("load", "video")}),
            "trim": GraphNode(
                "dinkster.video.trim",
                {"video": Link("load", "video"), "duration": Link("info", "duration")},
            ),
            "crop": GraphNode(
                "dinkster.video.crop",
                {
                    "video": Link("trim", "video"),
                    "width": Link("info", "width"),
                    "height": Link("info", "height"),
                },
            ),
        }
    )
    restored = graph_from_wire(json.loads(json.dumps(graph_to_wire(linked))))
    assert restored == linked


def test_strict_duration_is_separate_from_video_edit() -> None:
    source = video_from_source(_source())
    widget = {"trim": {"start_time": 1.5, "duration": 1.0}}
    with pytest.raises(ValueError, match="exceeds available duration"):
        TrimVideo.execute(video=source, video_edit=widget, strict_duration=True)
    clamped = TrimVideo.execute(video=source, video_edit=widget, strict_duration=False)["video"]
    assert cast(Mapping[str, object], video_meta(clamped)["effective"])["duration"] == [1, 2]
    assert "strict_duration" not in widget


def test_video_validation_accepts_remote_qualified_ram_cost() -> None:
    value = video_from_source(_source())
    encoded = encode_video(value)
    metadata = dict(video_meta(value))
    cost = cast(Mapping[str, int], metadata["cost"])["ram"]
    metadata["cost"] = {"ram@remote-video": cost}
    validate_video_encoded(encoded, metadata)
    metadata["cost"] = {"ram@remote-video": cost + 1}
    with pytest.raises(ValueError, match="cost does not match"):
        validate_video_encoded(encoded, metadata)


def test_same_saved_editor_graph_in_process_shm_and_loopback_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        data = _source()
        producer_vault = AssetVault(tmp_path / "producer-vault")
        digest = digest_bytes(data)
        with producer_vault.writer(digest) as writer:
            writer.write(data)
            writer.commit()
        monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(producer_vault.root))
        ref = AssetRef(digest, "source.mp4", len(data), resolver=producer_vault)
        widget = {
            "trim": {"start_time": 0.5, "duration": 1.0},
            "crop": {"x": 3, "y": 3, "width": 33, "height": 17},
        }
        document, graph = _saved_graph(ref, widget)
        assert json.loads(document)["nodes"]["trim"]["inputs"]["video_edit"] == widget
        assert json.loads(document)["nodes"]["crop"]["inputs"]["video_edit"] == widget

        registry = _registry()
        local_worker = InProcessWorker(build_node_types(MEDIA_IO_NODES), registry)
        local = await Engine(
            schemas=dict(local_worker.schemas),
            registry=registry,
            worker=local_worker,
            cache=MemoryLRUCache(),
        ).run(graph, ["crop", "frames"])
        expected = _resolved_outputs(local)

        diagnostics: list[BoundaryDiagnostic] = []
        isolated = IsolatedWorker(
            MEDIA_MANIFEST, registry, shm_threshold=64, on_diagnostic=diagnostics.append
        )
        await isolated.start()
        try:
            actual = await Engine(
                schemas=dict(isolated.schemas),
                registry=registry,
                worker=isolated,
                cache=MemoryLRUCache(),
            ).run(graph_from_wire(json.loads(document)), ["crop", "frames"])
            isolated_result = _resolved_outputs(actual)
        finally:
            await isolated.close()
        assert isolated_result[0] == expected[0]
        np.testing.assert_array_equal(isolated_result[1], expected[1])
        assert any(
            crossing.type_id == "comfy.VIDEO" and crossing.transport == "shm"
            for diagnostic in diagnostics
            for crossing in (*diagnostic.inputs, *diagnostic.outputs)
        )

        daemon_vault = tmp_path / "daemon-vault"
        proc, host, port = await start_service(
            MEDIA_MANIFEST, tmp_path, "--asset-vault", str(daemon_vault)
        )
        try:
            async with _AssetHost(data) as assets:
                remote = RemoteWorker(
                    host,
                    port,
                    TOKEN,
                    registry,
                    name="loopback-video",
                    connect_timeout=10,
                    asset_endpoint=assets.endpoint,
                    asset_endpoint_token=ASSET_TOKEN,
                )
                await remote.start()
                try:
                    original_open = av.open
                    source_host_opens = 0

                    def source_host_open(*args: object, **kwargs: object) -> object:
                        nonlocal source_host_opens
                        source_host_opens += 1
                        return cast(Any, original_open)(*args, **kwargs)

                    monkeypatch.setattr(av, "open", source_host_open)
                    remote_result = await Engine(
                        schemas=dict(remote.schemas),
                        registry=registry,
                        worker=remote,
                        cache=MemoryLRUCache(),
                    ).run(graph_from_wire(json.loads(document)), ["crop", "frames"])
                    monkeypatch.setattr(av, "open", original_open)
                    assert source_host_opens == 0
                    actual = _resolved_outputs(remote_result)
                    assert assets.requests == 1
                finally:
                    await remote.close()
        finally:
            await stop_service(proc)
        assert actual[0] == expected[0]
        np.testing.assert_array_equal(actual[1], expected[1])

    asyncio.run(scenario())
