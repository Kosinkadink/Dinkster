"""Pinned ComfyUI video workflows through the isolated compat boundary."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import pytest
from dinkster_assets import (
    AssetRef,
    LocalAssetLibrary,
    MediaClassification,
    MountSnapshotResolver,
    classify_media_file,
)
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, Invocation, InvocationResult, OnInvocationEvent, Worker
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_media_io.video import LoadVideo
from dinkster_protocol import MediaSourceAuthority, SavedArtifact
from dinkster_values import TypeRegistry, register_core_types
from dinkster_workers import IsolatedWorker

from dinkster.comfy_compose import register_comfy_host_types

COMFY_ROOT = os.environ.get("DINKSTER_COMFYUI_ROOT", "")
PINNED_COMFY_COMMIT = "15eb748b3ec5f8a0a2d470b7fb280e2d7579f916"
VIDEO_NODE_IDS = (
    "LoadVideo",
    "Video Slice",
    "CreateVideo",
    "GetVideoComponents",
    "SaveVideo",
    "SaveWEBM",
)
REQUIRED_ARM_NODE_IDS = (
    "CreateHookLora",
    "CreateHookKeyframe",
    "SetHookKeyframes",
    "ConditioningTimestepsRange",
    "ConditioningSetPropertiesAndCombine",
    "PairConditioningSetProperties",
)

pytestmark = pytest.mark.skipif(
    not COMFY_ROOT, reason="DINKSTER_COMFYUI_ROOT not set (live ComfyUI video test)"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPAT_MANIFEST = REPO_ROOT / "packages" / "dinkster-compat-comfy" / "dinkster-pack.toml"


def _comfy_python() -> str:
    explicit = os.environ.get("DINKSTER_COMFYUI_PYTHON", "")
    if explicit:
        return explicit
    candidate = Path(COMFY_ROOT) / "venv" / "bin" / "python"
    return str(candidate) if candidate.is_file() else sys.executable


def _dinkster_pythonpath() -> str:
    return os.pathsep.join(str(path) for path in sorted((REPO_ROOT / "packages").glob("*/src")))


def _write_source_video(path: Path) -> None:
    with av.open(path, mode="w", format="mp4") as container:
        video = container.add_stream("libx264", rate=4)
        video.width = 64
        video.height = 64
        video.pix_fmt = "yuv420p"
        video.options = {"crf": "0"}
        audio = container.add_stream("aac", rate=48_000)
        audio.layout = "mono"
        for index in range(4):
            pixels = np.zeros((64, 64, 3), dtype=np.uint8)
            pixels[..., 0] = index * 64
            pixels[..., 1] = 32
            pixels[..., 2] = 192
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 4)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        waveform = np.sin(
            2 * np.pi * 440 * np.arange(48_000, dtype=np.float32) / 48_000,
            dtype=np.float32,
        )[None, :]
        for offset in range(0, waveform.shape[1], 1024):
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(waveform[:, offset : offset + 1024]),
                format="fltp",
                layout="mono",
            )
            frame.sample_rate = 48_000
            frame.pts = offset
            frame.time_base = Fraction(1, 48_000)
            for packet in audio.encode(frame):
                container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)


def _mount_snapshot(tmp_path: Path) -> tuple[Path, AssetRef, Path]:
    media_root = tmp_path / "media"
    output_root = tmp_path / "output"
    media_root.mkdir()
    output_root.mkdir()
    source_path = media_root / "source.mp4"
    _write_source_video(source_path)
    media = LocalAssetLibrary(media_root, namespace="mounts/media")
    media.scan()
    source_ref = media.ref("mounts/media/source.mp4")
    output_index = output_root / ".dinkster-asset-index.json"
    output_index.write_text("{}", "utf-8")
    snapshot = tmp_path / "mounts.json"
    snapshot.write_text(
        json.dumps(
            {
                "mounts": [
                    {
                        "id": "media",
                        "root": str(media_root),
                        "index": str(media.index_path),
                        "mode": "readonly",
                        "kind": "media/video",
                        "priority": 0,
                    },
                    {
                        "id": "comfy-output",
                        "root": str(output_root),
                        "index": str(output_index),
                        "mode": "readwrite",
                        "kind": "media/video",
                        "priority": 0,
                    },
                ]
            }
        ),
        "utf-8",
    )
    return snapshot, source_ref, output_root


def _authority(ref: AssetRef, classification: MediaClassification) -> MediaSourceAuthority:
    return MediaSourceAuthority(
        ref.digest,
        classification.kind,
        classification.media_type,
        classification.extension,
        ref.size,
    )


def _artifact_ref(artifact: SavedArtifact, snapshot: Path) -> AssetRef:
    return AssetRef(
        digest=artifact.digest,
        name=artifact.name,
        size=artifact.size,
        media_type=artifact.media_type,
        virtual_path=artifact.virtual_path,
        resolver=MountSnapshotResolver(snapshot),
    )


class _AuthenticatedWorker:
    def __init__(self, inner: IsolatedWorker) -> None:
        self.inner = inner

    async def prepare(self, node_types: Sequence[str]) -> None:
        await self.inner.prepare(node_types)

    async def invoke(
        self,
        invocation: Invocation,
        on_event: OnInvocationEvent | None = None,
    ) -> InvocationResult:
        route = cast(Any, self.inner.attention_route_token)
        if route is not None:
            invocation = replace(
                invocation,
                attention_policy=route.requested_policy,
                attention_route_token=route,
            )
        return await self.inner.invoke(invocation, on_event)


def test_pinned_core_video_workflows_cross_the_compat_boundary(tmp_path: Path) -> None:
    commit = subprocess.run(
        ["git", "-C", COMFY_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit == PINNED_COMFY_COMMIT
    snapshot, source_ref, output_root = _mount_snapshot(tmp_path)
    source_classification = classify_media_file(output_root.parent / "media" / "source.mp4")
    input_root = tmp_path / "input"
    temp_root = tmp_path / "temp"
    user_root = tmp_path / "user"
    for path in (input_root, temp_root, user_root):
        path.mkdir()

    async def scenario() -> None:
        registry = TypeRegistry()
        register_core_types(registry)
        register_comfy_host_types(registry)
        worker = IsolatedWorker(
            COMPAT_MANIFEST,
            registry,
            python=_comfy_python(),
            extra_env={
                "DINKSTER_COMFYUI_ROOT": COMFY_ROOT,
                "DINKSTER_COMFY_NODES": ",".join((*VIDEO_NODE_IDS, *REQUIRED_ARM_NODE_IDS)),
                "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
                "PYTHONPATH": _dinkster_pythonpath(),
            },
            comfy_args=(
                "--cpu",
                "--input-directory",
                str(input_root),
                "--output-directory",
                str(output_root),
                "--temp-directory",
                str(temp_root),
                "--user-directory",
                str(user_root),
                "--disable-metadata",
            ),
            start_timeout=180.0,
        )
        await worker.start()
        try:
            schemas = dict(worker.schemas)
            assert set(f"comfy.{node_id}" for node_id in VIDEO_NODE_IDS) <= set(schemas)
            for node_id in VIDEO_NODE_IDS:
                translated = schemas[f"comfy.{node_id}"]
                assert translated.node_type == f"comfy.{node_id}"
                assert translated.aliases == (node_id,)
            assert schemas["comfy.LoadVideo"].inputs[0].type.types == ("dinkster.asset",)
            assert schemas["comfy.LoadVideo"].outputs[0].type.types == ("comfy.VIDEO",)
            assert schemas["comfy.CreateVideo"].outputs[0].type.types == ("comfy.VIDEO",)
            assert schemas["comfy.SaveVideo"].output_node is True
            assert schemas["comfy.SaveWEBM"].output_node is True

            engine = Engine(
                schemas=schemas,
                registry=registry,
                worker=cast(Worker, _AuthenticatedWorker(worker)),
                cache=MemoryLRUCache(),
            )
            authority = _authority(source_ref, source_classification)
            graph = Graph(
                nodes={
                    "load": GraphNode("comfy.LoadVideo", {"file": source_ref.to_wire()}),
                    "slice": GraphNode(
                        "comfy.Video Slice",
                        {
                            "video": Link("load", "VIDEO"),
                            "start_time": 0.0,
                            "duration": 0.75,
                            "strict_duration": True,
                        },
                    ),
                    "components": GraphNode(
                        "comfy.GetVideoComponents", {"video": Link("slice", "VIDEO")}
                    ),
                    "create": GraphNode(
                        "comfy.CreateVideo",
                        {
                            "images": Link("components", "images"),
                            "audio": Link("components", "audio"),
                            "fps": Link("components", "fps"),
                            "bit_depth": "10",
                        },
                    ),
                    "save": GraphNode(
                        "comfy.SaveVideo",
                        {
                            "video": Link("create", "VIDEO"),
                            "filename_prefix": "video/compat",
                        },
                        slot_variants={
                            "format": "mp4",
                            "format.codec": "h264",
                            "format.codec.encoding": "auto",
                        },
                    ),
                }
            )
            result = await engine.run(
                graph, ["components", "create", "save"], media_sources=(authority,)
            )
            assert result.executed == ("load", "slice", "components", "create", "save")
            images = cast(np.ndarray, result.outputs["components"]["images"].resolve())
            audio = cast(dict[str, object], result.outputs["components"]["audio"].resolve())
            assert images.shape == (3, 64, 64, 3)
            assert cast(np.ndarray, audio["waveform"]).shape[2] > 20_000
            assert audio["sample_rate"] == 48_000
            assert result.outputs["components"]["fps"].resolve() == pytest.approx(4.0)
            assert result.outputs["components"]["bit_depth"].resolve() == "8"
            assert result.outputs["create"]["VIDEO"].meta.entries["container"] is None
            created_probe = cast(
                dict[str, object], result.outputs["create"]["VIDEO"].meta.entries["probe"]
            )
            assert created_probe["bit_depth"] == 10
            assert len(result.artifacts) == 1
            mp4_artifact = result.artifacts[0]
            assert mp4_artifact.node_id == "save"
            assert mp4_artifact.media_type == "video/mp4"
            assert mp4_artifact.virtual_path.startswith("mounts/comfy-output/video/")
            mp4_path = output_root / mp4_artifact.virtual_path.removeprefix("mounts/comfy-output/")
            with av.open(mp4_path) as container:
                assert [frame.format.name for frame in container.decode(video=0)] == [
                    "yuv420p10le"
                ] * 3

            rgba = np.zeros((3, 64, 64, 4), dtype=np.float32)
            rgba[..., 0] = 0.2
            rgba[..., 1] = 0.7
            rgba[..., 2] = 0.4
            rgba[0, ..., 3] = 0.1
            rgba[1, ..., 3] = 0.5
            rgba[2, ..., 3] = 0.9
            webm_result = await engine.run(
                Graph(
                    nodes={
                        "webm": GraphNode(
                            "comfy.SaveWEBM",
                            {
                                "images": rgba.tolist(),
                                "filename_prefix": "video/alpha",
                                "codec": "vp9",
                                "fps": 3.0,
                                "crf": 20.0,
                            },
                        )
                    }
                ),
                ["webm"],
            )
            returned = cast(np.ndarray, webm_result.outputs["webm"]["images"].resolve())
            assert np.array_equal(returned, rgba)
            assert len(webm_result.artifacts) == 1
            webm_artifact = webm_result.artifacts[0]
            assert webm_artifact.node_id == "webm"
            assert webm_artifact.media_type == "video/webm"
            webm_ref = _artifact_ref(webm_artifact, snapshot)
            webm_path = output_root / webm_artifact.virtual_path.removeprefix(
                "mounts/comfy-output/"
            )
            with av.open(webm_path) as container:
                assert container.streams.video[0].metadata["alpha_mode"] == "1"
            loaded = LoadVideo.execute(video=webm_ref)
            loaded_images = cast(np.ndarray, loaded["images"])
            assert loaded_images.shape == rgba.shape
            assert loaded_images[..., 3].min() == pytest.approx(0.1, abs=2 / 255)
            assert loaded_images[..., 3].max() == pytest.approx(0.9, abs=2 / 255)
        finally:
            await worker.close()

    asyncio.run(scenario())
