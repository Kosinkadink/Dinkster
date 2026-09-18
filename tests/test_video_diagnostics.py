"""Native format substitutions use persistent diagnostics with actual output facts."""

import asyncio
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import av
import numpy as np
import pytest
from dinkster_assets import AssetRef
from dinkster_caches import MemoryLRUCache
from dinkster_engine import Engine, EngineEvent
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_media_io import register_media_types
from dinkster_nodes_media_io.video import SaveVideo, SaveVideoFrames, SaveVideoValue
from dinkster_schema import (
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    build_node_types,
    build_schemas,
)
from dinkster_server.events import engine_event_to_wire
from dinkster_values import TypeRegistry, register_core_types
from dinkster_video import assemble_video
from dinkster_workers import InProcessWorker, IsolatedWorker
from PIL import Image

from tests.test_formats_audio_components import _stereo_samples, _wav_audio
from tests.test_video_io import _mount


class FormatSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="test.format-source",
            inputs=(
                InputSpec("alpha", TypeExpr.concrete("core.boolean")),
                InputSpec("audio", TypeExpr.concrete("core.boolean"), default=False),
            ),
            outputs=(OutputSpec("video", TypeExpr.concrete("comfy.VIDEO")),),
        )

    @classmethod
    def execute(cls, alpha: bool, audio: bool = False) -> Mapping[str, object]:
        pixels = np.full((2, 64, 64, 4 if alpha else 3), 0.4, np.float32)
        soundtrack = _wav_audio(80) if audio else None
        return cls.outputs(video=assemble_video(pixels, fps=10, audio=soundtrack))


DIAGNOSTIC_NODES = (FormatSource, SaveVideo, SaveVideoFrames, SaveVideoValue)


@pytest.fixture(params=[False, True], ids=["in-process", "isolated"])
def worker_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    _, snapshot = _mount(tmp_path, monkeypatch)
    registry = TypeRegistry()
    register_core_types(registry)
    register_media_types(registry)
    manifest = tmp_path / "dinkster-pack.toml"
    manifest.write_text(
        '[pack]\nname="video-diagnostics"\n[pack.entry]\n'
        'nodes="tests.test_video_diagnostics:DIAGNOSTIC_NODES"\n'
        'types="dinkster_nodes_media_io:register_media_types"\n'
    )
    return registry, manifest, snapshot, request.param


@pytest.mark.parametrize(
    "kind", ["webm_h264", "mp4_rgba", "gif_rgba", "value_rgba", "audio_endpoint"]
)
def test_native_saver_emits_actual_format_facts_on_its_output_port(worker_setup, kind: str) -> None:
    async def scenario() -> None:
        registry, manifest, snapshot, isolated = worker_setup
        boundary = (
            IsolatedWorker(
                manifest,
                registry,
                extra_env={
                    "PYTHONPATH": os.pathsep.join(
                        (str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", ""))
                    ),
                    "DINKSTER_MOUNTS_SNAPSHOT": str(snapshot),
                },
            )
            if isolated
            else None
        )
        if boundary:
            await boundary.start()
        try:
            events: list[EngineEvent] = []
            engine = Engine(
                schemas=boundary.schemas if boundary else build_schemas(DIAGNOSTIC_NODES),
                registry=registry,
                worker=boundary or InProcessWorker(build_node_types(DIAGNOSTIC_NODES), registry),
                cache=MemoryLRUCache(),
                on_event=events.append,
            )
            animated, value_only = kind == "gif_rgba", kind == "value_rgba"
            audio_endpoint = kind == "audio_endpoint"
            settings = (
                {"format": "gif_pillow"}
                if animated
                else {}
                if value_only
                else {"container": "webm" if kind == "webm_h264" else "mp4", "codec": "h264"}
            )
            if audio_endpoint:
                settings = {"container": "mkv", "codec": "ffv1", "trim_to_audio": True}
            node_type = (
                "dinkster.save_video_frames"
                if animated
                else "dinkster.save_video_value"
                if value_only
                else "dinkster.save_video"
            )
            result = await engine.run(
                Graph(
                    nodes={
                        "source": GraphNode(
                            "test.format-source",
                            {
                                "alpha": kind not in ("webm_h264", "audio_endpoint"),
                                "audio": audio_endpoint,
                            },
                        ),
                        "save": GraphNode(
                            node_type, {"video": Link("source", "video"), **settings}
                        ),
                    }
                ),
                ["save"],
            )
            assert "save" in result.outputs
            (event,) = [event for event in events if event.kind == "value_diagnostics"]
            wire = engine_event_to_wire(event, client_id="client", job_id="job")
            assert wire["type"] == "value_diagnostics" and wire["nodeId"] == "save"
            (record,) = cast(dict[str, Any], wire["detail"])["diagnostics"]
            output_id = "video" if value_only else "asset"
            assert record["code"] == "media_format_fallback"
            assert record["outputId"] == output_id and record["nodeId"] == "save"
            assert record["reason"] == (
                "component_audio_endpoint_unproven" if audio_endpoint else "preserving_cpu_default"
            )
            assert record["requested"] == {
                "container": "gif" if animated else None if value_only else settings["container"],
                "codec": "gif" if animated else None if value_only else settings["codec"],
                "pixelFormat": None,
                "channelLayout": None,
            }
            assert record["effective"] == {
                "container": "webp" if animated else "mkv",
                "codec": "webp" if animated else "h264" if kind == "webm_h264" else "ffv1",
                "pixelFormat": "rgba"
                if animated
                else "yuv420p"
                if kind == "webm_h264"
                else "bgr0"
                if audio_endpoint
                else "bgra",
                "channelLayout": "stereo" if audio_endpoint else None,
            }
            if audio_endpoint:
                assert record["substitutions"][0]["requested"] == {"trim_to_audio": True}
                assert record["substitutions"][0]["effective"] == {"trim_to_audio": False}
            output = result.outputs["save"][output_id]
            assert output.meta.get("valueDiagnostics") == [
                {key: value for key, value in record.items() if key != "nodeId"}
            ]
            asset = cast(AssetRef, output.resolve())
            assert asset.media_type == ("image/webp" if animated else "video/x-matroska")
            with asset.open() as handle:
                if animated:
                    with Image.open(handle) as image:
                        assert image.mode == "RGBA"
                        np.testing.assert_array_equal(np.asarray(image)[..., 3], 102)
                else:
                    with av.open(handle, "r") as opened:
                        stream = opened.streams.video[0]
                        assert stream.codec_context.name == record["effective"]["codec"]
                        assert stream.codec_context.format is not None
                        assert (
                            stream.codec_context.format.name == record["effective"]["pixelFormat"]
                        )
                        alpha = next(opened.decode(video=0)).to_ndarray(format="rgba")[..., 3]
                        np.testing.assert_array_equal(
                            alpha, 255 if kind in ("webm_h264", "audio_endpoint") else 102
                        )
                        if audio_endpoint:
                            assert len(list(opened.decode(video=0))) == 1
            if audio_endpoint:
                with asset.open() as handle, av.open(handle, "r") as opened:
                    samples = _stereo_samples(opened)
                    assert samples.shape == (2, 1600)
                    np.testing.assert_array_equal(samples[:, :80], 0.25)
                    np.testing.assert_array_equal(samples[:, 80:], 0)
            assert not any(event.kind == "node_failed" for event in events)
        finally:
            if boundary:
                await boundary.close()

    asyncio.run(scenario())
