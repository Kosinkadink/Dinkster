"""Run bounded VIDEO editor and codec conformance against a remote worker."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
from aiohttp import web
from dinkster_api.v1 import Node, NodeSchema, OutputSpec, TypeExpr
from dinkster_assets import AssetRef, AssetVault, digest_bytes
from dinkster_caches import BudgetedDiskCAS, MemoryLRUCache
from dinkster_engine import Engine
from dinkster_graph import Graph, GraphNode, Link, graph_from_wire, graph_to_wire
from dinkster_nodes_media_io import register_media_types
from dinkster_protocol import Invocation
from dinkster_values import (
    TypeRegistry,
    edit_video,
    encode_video,
    register_core_types,
    video_from_source,
    video_meta,
)
from dinkster_video import assemble_video, disassemble_video, save_video_stream
from dinkster_workers import RemoteWorker
from dinkster_workers.service import READY_LINE_PREFIX

ROOT = Path(__file__).resolve().parents[1]
MEDIA_MANIFEST = ROOT / "packages/dinkster-nodes-media-io/dinkster-pack.toml"
DEFAULT_EDIT = {
    "trim": {"start_time": 0.5, "duration": 1.0},
    "crop": {"x": 2, "y": 2, "width": 32, "height": 16},
}


class _ProduceConformanceVideo(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="conformance.produce_video",
            display_name="Produce Conformance Video",
            category="test",
            inputs=(),
            outputs=(OutputSpec("video", TypeExpr.concrete("comfy.VIDEO")),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(video=video_from_source(_produced_source()))


PRODUCER_NODES = (_ProduceConformanceVideo,)


def _registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_core_types(registry)
    register_media_types(registry)
    return registry


def _frames(count: int = 20) -> np.ndarray:
    result = np.empty((count, 32, 64, 3), np.float32)
    for index in range(count):
        result[index, ..., 0] = index / count
        result[index, ..., 1] = np.arange(64, dtype=np.float32) / 64
        result[index, ..., 2] = np.arange(32, dtype=np.float32)[:, None] / 32
    return result


def _source() -> bytes:
    output = io.BytesIO()
    with av.open(output, "w", format="mp4") as opened:
        mux = cast(Any, opened)
        stream = mux.add_stream("libx264", rate=10)
        stream.width, stream.height = 64, 32
        stream.pix_fmt = "yuv420p"
        stream.codec_context.thread_count = 1
        stream.codec_context.max_b_frames = 0
        stream.gop_size = 5
        stream.options = {"crf": "0", "x264-params": "scenecut=0"}
        for index, pixels in enumerate((_frames() * 255).astype(np.uint8)):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 10)
            for packet in stream.encode(frame):
                mux.mux(packet)
        for packet in stream.encode():
            mux.mux(packet)
    return output.getvalue()


def _produced_source() -> bytes:
    output = io.BytesIO()
    rng = np.random.default_rng(1250)
    with av.open(output, "w", format="matroska") as opened:
        mux = cast(Any, opened)
        stream = mux.add_stream("ffv1", rate=10)
        stream.width = stream.height = 64
        stream.pix_fmt = "bgra"
        stream.codec_context.thread_count = 1
        for index in range(30):
            frame = av.VideoFrame.from_ndarray(
                rng.integers(0, 256, (64, 64, 4), dtype=np.uint8), format="rgba"
            )
            frame.pts = index
            for packet in stream.encode(frame):
                mux.mux(packet)
        for packet in stream.encode():
            mux.mux(packet)
    return output.getvalue()


def _probe(data: bytes) -> dict[str, object]:
    probe = cast(dict[str, object], video_meta(video_from_source(data))["probe"])
    return {
        key: probe[key]
        for key in (
            "container",
            "video_codec",
            "pix_fmt",
            "bit_depth",
            "color_space",
            "primaries",
            "transfer",
            "matrix",
            "range",
            "alpha",
            "width",
            "height",
            "frame_count",
            "duration",
        )
    }


def _save(
    value: object, *, container: str = "auto", codec: str = "auto", crf: int | None = None
) -> bytes:
    output = io.BytesIO()
    save_video_stream(value, output, container=container, codec=codec, crf=crf)
    return output.getvalue()


def _decoded_alpha(data: bytes) -> list[np.ndarray[Any, Any]]:
    opened = cast(Any, av.open(io.BytesIO(data), "r"))
    try:
        return [frame.to_ndarray(format="rgba")[..., 3].copy() for frame in opened.decode(video=0)]
    finally:
        opened.close()


def _codec_evidence() -> dict[str, object]:
    images = _frames(6)
    hlg_source = _save(assemble_video(images, fps=3, bit_depth="10", color_space="HDR"), crf=0)
    hlg_trim = _save(
        edit_video(video_from_source(hlg_source), {"trim": {"start_time": 0, "duration": 1}})
    )
    hlg_probe = _probe(hlg_trim)

    pq_source = _save(assemble_video(images, fps=3, bit_depth="10", color_space="HDR PQ"), crf=0)
    components = disassemble_video(video_from_source(pq_source))
    pq_roundtrip = _save(
        assemble_video(
            components["images"],
            fps=cast(float, components["fps"]),
            bit_depth=str(components["bit_depth"]),
            color_space=cast(str, components["color_space"]),
        ),
        crf=0,
    )
    pq_probe = _probe(pq_roundtrip)

    av1_source = _save(assemble_video(images[:3], fps=3), codec="av1", crf=20)
    av1_trim = _save(
        edit_video(video_from_source(av1_source), {"trim": {"start_time": 0.4, "duration": 0.4}})
    )

    alpha_output = io.BytesIO()
    with av.open(alpha_output, "w", format="matroska") as opened:
        mux = cast(Any, opened)
        stream = mux.add_stream("ffv1", rate=1)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "bgra"
        rgba = np.zeros((3, 32, 64, 4), np.uint8)
        for index in range(3):
            rgba[index, ..., 3] = ((np.arange(64, dtype=np.uint16) * 4 + index * 17) % 256).astype(
                np.uint8
            )
        for index, pixels in enumerate(rgba):
            frame = av.VideoFrame.from_ndarray(pixels, format="rgba")
            frame.pts = index
            for packet in stream.encode(frame):
                mux.mux(packet)
        for packet in stream.encode():
            mux.mux(packet)
    alpha_source = alpha_output.getvalue()
    alpha_saved = _save(
        edit_video(video_from_source(alpha_source), {"trim": {"start_time": 1.0, "duration": 1.0}})
    )
    alpha_source_samples = _decoded_alpha(alpha_source)
    alpha_saved_samples = _decoded_alpha(alpha_saved)
    alpha_probe = _probe(alpha_saved)

    if [hlg_probe[key] for key in ("bit_depth", "primaries", "transfer", "matrix")] != [
        10,
        9,
        18,
        9,
    ]:
        raise RuntimeError("10-bit HLG trim did not preserve remux tags")
    if [pq_probe[key] for key in ("bit_depth", "primaries", "transfer", "matrix")] != [
        10,
        9,
        16,
        9,
    ]:
        raise RuntimeError("PQ disassemble/assemble did not preserve tags")
    av1_probe = _probe(av1_trim)
    if av1_probe["video_codec"] != "av1":
        raise RuntimeError("AV1 trim changed codec")
    if alpha_probe["alpha"] is not True:
        raise RuntimeError("supported FFV1 alpha was not detected")
    if len(alpha_saved_samples) != 1 or not np.array_equal(
        alpha_source_samples[1], alpha_saved_samples[0]
    ):
        raise RuntimeError("Dinkster FFV1 edit/save did not preserve decoded alpha samples")
    return {
        "ten_bit_hlg_trim": {"sha256": hashlib.sha256(hlg_trim).hexdigest(), "probe": hlg_probe},
        "pq_disassemble_assemble": {
            "sha256": hashlib.sha256(pq_roundtrip).hexdigest(),
            "probe": pq_probe,
        },
        "av1_trim": {"sha256": hashlib.sha256(av1_trim).hexdigest(), "probe": av1_probe},
        "supported_alpha": {
            "comparison": "source frame 1 versus saved frame 0, exact decoded uint8 alpha",
            "source_sha256": hashlib.sha256(alpha_source).hexdigest(),
            "saved_sha256": hashlib.sha256(alpha_saved).hexdigest(),
            "saved_probe": alpha_probe,
        },
    }


class _AssetServer:
    def __init__(self, data: bytes, listen: str, advertised: str, token: str) -> None:
        self.data = data
        self.digest = digest_bytes(data)
        self.host, self.port = _endpoint(listen)
        self.endpoint = advertised.rstrip("/")
        self.token = token
        self.requests = 0
        self._runner: web.AppRunner | None = None

    async def _get(self, request: web.Request) -> web.Response:
        if request.headers.get("Authorization") != f"Bearer {self.token}":
            return web.Response(status=401)
        if request.match_info["digest"] != self.digest:
            return web.Response(status=404)
        self.requests += 1
        return web.Response(body=self.data)

    async def __aenter__(self) -> _AssetServer:
        app = web.Application()
        app.router.add_get("/assets/{digest}", self._get)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        if self.port == 0:
            host, port = self._runner.addresses[0]
            self.endpoint = f"http://{host}:{port}"
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        assert self._runner is not None
        await self._runner.cleanup()


def _endpoint(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator or not host:
        raise ValueError(f"endpoint must be HOST:PORT, got {value!r}")
    return host, int(port)


def _token(environment_name: str) -> str:
    value = os.environ.get(environment_name, "")
    if len(value) < 16:
        raise ValueError(f"${environment_name} must contain at least 16 characters")
    return value


def _saved_graph(ref: AssetRef) -> tuple[str, Graph]:
    graph = Graph(
        {
            "load": GraphNode("dinkster.load_video_value", {"video": ref.to_wire()}),
            "trim": GraphNode(
                "dinkster.video.trim",
                {"video": Link("load", "video"), "video_edit": DEFAULT_EDIT},
            ),
            "crop": GraphNode(
                "dinkster.video.crop",
                {"video": Link("trim", "video"), "video_edit": DEFAULT_EDIT},
            ),
            "save": GraphNode("dinkster.save_video", {"video": Link("crop", "video")}),
        }
    )
    saved = json.dumps(graph_to_wire(graph), sort_keys=True, separators=(",", ":"))
    return saved, graph_from_wire(json.loads(saved))


async def _run_remote(
    worker_endpoint: str,
    worker_token: str,
    asset_listen: str,
    asset_endpoint: str,
    asset_token: str,
    root: Path,
    tls_ca_file: Path | None,
) -> dict[str, object]:
    source = _source()
    vault = AssetVault(root / "producer-vault")
    digest = digest_bytes(source)
    with vault.writer(digest) as writer:
        writer.write(source)
        writer.commit()
    os.environ["DINKSTER_ASSET_VAULT"] = str(vault.root)
    ref = AssetRef(digest, "source.mp4", len(source), resolver=vault)
    saved_graph, graph = _saved_graph(ref)
    host, port = _endpoint(worker_endpoint)
    registry = _registry()
    transfer_store = BudgetedDiskCAS(root / "receiver-transfer-store")
    events: list[Any] = []
    async with _AssetServer(source, asset_listen, asset_endpoint, asset_token) as assets:
        worker = RemoteWorker(
            host,
            port,
            worker_token,
            registry,
            name="video-conformance",
            connect_timeout=30,
            asset_endpoint=assets.endpoint,
            asset_endpoint_token=asset_token,
            value_store=transfer_store,
            tls_ca_file=tls_ca_file,
        )
        await worker.start()
        try:
            source_host_opens = 0
            original_open = av.open

            def counted_open(*args: object, **kwargs: object) -> object:
                nonlocal source_host_opens
                source_host_opens += 1
                return cast(Any, original_open)(*args, **kwargs)

            av.open = counted_open
            try:
                result = await Engine(
                    schemas=dict(worker.schemas),
                    registry=registry,
                    worker=worker,
                    cache=MemoryLRUCache(),
                    on_event=events.append,
                ).run(graph, ["crop", "save"])
            finally:
                av.open = original_open
            value = result.outputs["crop"]["video"].resolve()
            asset = result.outputs["save"]["asset"].resolve()
            if not isinstance(asset, AssetRef):
                raise RuntimeError("remote Save Video did not return an asset")
        finally:
            await worker.close()
    if source_host_opens != 0:
        raise RuntimeError(
            f"source host opened media {source_host_opens} times during remote execution"
        )
    if assets.requests != 1:
        raise RuntimeError(f"receiver fetched the source {assets.requests} times, expected once")
    encoded = encode_video(value)
    saved_output_buffer = io.BytesIO()
    save_video_stream(value, saved_output_buffer)
    saved_output = saved_output_buffer.getvalue()
    if digest_bytes(saved_output) != asset.digest:
        raise RuntimeError("receiver saved output differs from the returned edited VIDEO")
    counts = {
        kind: sum(event.kind == kind for event in events)
        for kind in ("node_finished", "node_cached", "node_skipped")
    }
    expected_counts = {"node_finished": 4, "node_cached": 0, "node_skipped": 0}
    if counts != expected_counts:
        raise RuntimeError(f"remote editor node counters {counts} do not match {expected_counts}")
    return {
        "saved_graph_sha256": hashlib.sha256(saved_graph.encode()).hexdigest(),
        "serialized_edit": DEFAULT_EDIT,
        "source_host_pyav_open_count": source_host_opens,
        "receiver_source_fetch_count": assets.requests,
        "remote_node_counts": counts,
        "save_count": 1,
        "output_digest": hashlib.sha256(encoded).hexdigest(),
        "output_probe": dict(cast(Mapping[str, object], video_meta(value)["effective"])),
        "saved_output": {
            "sha256": hashlib.sha256(saved_output).hexdigest(),
            "probe": _probe(saved_output),
            "receiver_asset": asset.to_wire(),
        },
        "source": {
            "bytes": len(source),
            "blake3": digest,
            "sha256": hashlib.sha256(source).hexdigest(),
        },
    }


async def _loopback(args: argparse.Namespace, root: Path) -> dict[str, object]:
    token = "loopback-video-worker-token-0123456789"
    asset_token = "loopback-video-asset-token-0123456789"
    token_file = root / "worker-token"
    token_file.write_text(token, encoding="utf-8")
    token_file.chmod(0o600)
    daemon_vault = root / "daemon-vault"
    daemon_store = root / "daemon-transfer-store"
    output = root / "output"
    output.mkdir()
    index = output / ".dinkster-asset-index.json"
    index.write_text("{}", encoding="utf-8")
    snapshot = root / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {
                        "id": "comfy-output",
                        "root": str(output),
                        "index": str(index),
                        "mode": "readwrite",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["DINKSTER_MOUNTS_SNAPSHOT"] = str(snapshot)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "dinkster_workers.service",
        "--listen",
        "127.0.0.1:0",
        "--manifest",
        str(MEDIA_MANIFEST),
        "--token-file",
        str(token_file),
        "--asset-vault",
        str(daemon_vault),
        "--value-store",
        str(daemon_store),
        stdout=asyncio.subprocess.PIPE,
        env=environment,
    )
    assert process.stdout is not None
    try:
        line = (await asyncio.wait_for(process.stdout.readline(), 60)).decode()
        if not line.startswith(READY_LINE_PREFIX):
            raise RuntimeError(f"unexpected loopback daemon ready line: {line!r}")
        worker_endpoint = line[len(READY_LINE_PREFIX) :].strip().removeprefix("tcp:")
        result = await _run_remote(
            worker_endpoint,
            token,
            "127.0.0.1:0",
            "http://127.0.0.1:0",
            asset_token,
            root,
            None,
        )
        result["lane"] = "loopback"
        return result
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 10)
            except TimeoutError:
                process.kill()
                await process.wait()


async def _produced_source_lifetime(root: Path) -> dict[str, object]:
    root.mkdir(parents=True)
    token = "loopback-video-producer-token-0123456789"
    token_file = root / "worker-token"
    token_file.write_text(token, encoding="utf-8")
    token_file.chmod(0o600)
    producer_vault = root / "producer-vault"
    engine_vault = AssetVault(root / "engine-vault")
    transfer_store = BudgetedDiskCAS(root / "transfer-store")
    daemon_transfer_store = root / "daemon-transfer-store"
    os.environ["DINKSTER_ASSET_VAULT"] = str(engine_vault.root)
    registry = _registry()
    manifest = root / "producer-pack.toml"
    manifest.write_text(
        '[pack]\nname = "video-conformance-producer"\n'
        'namespaces = ["conformance"]\n'
        '[pack.entry]\nnodes = "video_lanes:PRODUCER_NODES"\n'
        'types = "video_lanes:register_media_types"\n',
        encoding="utf-8",
    )
    python_path = str(ROOT / "tools")
    if inherited := os.environ.get("PYTHONPATH"):
        python_path += os.pathsep + inherited
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "dinkster_workers.service",
        "--listen",
        "127.0.0.1:0",
        "--manifest",
        str(manifest),
        "--token-file",
        str(token_file),
        "--asset-vault",
        str(producer_vault),
        "--value-store",
        str(daemon_transfer_store),
        stdout=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": python_path},
    )
    assert process.stdout is not None
    value = None
    try:
        line = (await asyncio.wait_for(process.stdout.readline(), 60)).decode()
        if not line.startswith(READY_LINE_PREFIX):
            raise RuntimeError(f"unexpected producer ready line: {line!r}")
        endpoint = line[len(READY_LINE_PREFIX) :].strip().removeprefix("tcp:")
        host, port = _endpoint(endpoint)
        worker = RemoteWorker(
            host,
            port,
            token,
            registry,
            name="video-producer",
            value_store=transfer_store,
        )
        await worker.start()
        try:
            result = await worker.invoke(
                Invocation(
                    invocation_id="produce-video",
                    node_id="produce-video",
                    node_type="conformance.produce_video",
                    inputs={},
                    effective_schema=worker.schemas["conformance.produce_video"],
                )
            )
            if result.error is not None or result.outputs is None:
                raise RuntimeError(f"remote video production failed: {result.error}")
            value = result.outputs["video"]
        finally:
            await worker.close()
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 10)
            except TimeoutError:
                process.kill()
                await process.wait()
    assert value is not None
    shutil.rmtree(producer_vault)
    shutil.rmtree(transfer_store.root)
    shutil.rmtree(daemon_transfer_store)
    refs = cast(list[dict[str, object]], value.meta.get("asset_refs"))
    if len(refs) != 1:
        raise RuntimeError("produced VIDEO did not declare exactly one source asset")
    ref = AssetRef.from_wire(refs[0], engine_vault)
    source = ref.read_bytes()
    if digest_bytes(source) != ref.digest:
        raise RuntimeError("adopted produced VIDEO source digest does not match")
    return {
        "verified": True,
        "lane": "loopback",
        "producer_process_stopped": process.returncode is not None,
        "producer_vault_removed": not producer_vault.exists(),
        "transfer_store_removed": not transfer_store.root.exists(),
        "daemon_transfer_store_removed": not daemon_transfer_store.exists(),
        "source_digest": ref.digest,
        "source_bytes": len(source),
        "source_probe": _probe(source),
    }


def _write_evidence(path: Path, evidence: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


async def _main(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix="dinkster-video-lanes-") as directory:
        root = Path(directory)
        if args.loopback:
            remote = await _loopback(args, root)
        else:
            remote = await _run_remote(
                args.worker,
                _token(args.worker_token_env),
                args.asset_listen,
                args.asset_endpoint,
                _token(args.asset_token_env),
                root,
                args.tls_ca_file,
            )
            remote["lane"] = "external"
        lifetime_failure: Exception | None = None
        try:
            lifetime = await _produced_source_lifetime(root / "produced-source-lifetime")
        except Exception as error:
            lifetime_failure = error
            lifetime = {
                "verified": False,
                "lane": "loopback",
                "failure": {"type": type(error).__name__, "message": str(error)},
                "reproduce": (
                    ".venv/bin/python tools/video_lanes.py --loopback "
                    "--evidence evidence/video-lanes-loopback.json"
                ),
            }
        evidence = {
            "format": "dinkster-video-lanes/1",
            "remote_editor": remote,
            "codec_checks": _codec_evidence(),
            "producer_source_lifetime": lifetime,
            "single_job_multi_gpu": {
                "applicable": False,
                "reason": (
                    "The exercised PyAV probe, edit, remux, and encode nodes execute on CPU and "
                    "declare no distributed producer or GPU workgroup to shard."
                ),
            },
        }
        _write_evidence(args.evidence, evidence)
        print(json.dumps({"evidence": str(args.evidence), "lane": remote["lane"]}, sort_keys=True))
        if lifetime_failure is not None:
            raise RuntimeError(
                f"produced-source lifetime failed; details persisted to {args.evidence}"
            ) from lifetime_failure


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--loopback", action="store_true")
    parser.add_argument("--worker", help="remote worker HOST:PORT")
    parser.add_argument("--worker-token-env", default="DINKSTER_VIDEO_WORKER_TOKEN")
    parser.add_argument("--asset-listen", default="0.0.0.0:8766")
    parser.add_argument("--asset-endpoint", help="advertised HTTP asset base URL")
    parser.add_argument("--asset-token-env", default="DINKSTER_VIDEO_ASSET_TOKEN")
    parser.add_argument("--tls-ca-file", type=Path, help="PEM CA/certificate for the worker")
    args = parser.parse_args(argv)
    if args.loopback:
        if args.worker or args.asset_endpoint:
            parser.error("--loopback cannot be combined with external endpoints")
    elif not args.worker or not args.asset_endpoint:
        parser.error("external execution requires --worker and --asset-endpoint")
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
