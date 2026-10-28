"""CPU rendering and encoded publication through the public SAM3 preview node."""

from __future__ import annotations

import asyncio
import importlib
import io
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    IMAGE_TYPE,
    SAM3_TRACK_DATA_TYPE,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    initialize_inference_process,
    register_inference_types,
)
from dinkster_assets import AssetVault
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.sam3_preview import SAM3_TrackPreview
from dinkster_schema.reporting import capture_preview_artifacts

from dinkster.compose import PackSpec, ServingComposer


@pytest.fixture(autouse=True)
def cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DINKSTER_ACCELERATOR", "cpu")
    initialize_inference_process()
    management = importlib.import_module("dinkster_inference.model_management")
    monkeypatch.setattr(management, "get_torch_device", lambda: torch.device("cpu"))


def _portable(type_id: str, value: Any) -> Any:
    registry = TypeRegistry()
    register_inference_types(registry)
    spec = registry.spec(type_id)
    return spec.decode(spec.encode(value))


@pytest.fixture
def rendered(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    frames: list[np.ndarray] = []

    class Frame:
        @staticmethod
        def from_ndarray(array: np.ndarray, *, format: str) -> Any:
            assert format == "rgb24"
            frames.append(array.copy())
            return Frame()

        def reformat(self, *, format: str) -> Frame:
            assert format == "yuv420p"
            return self

    class Encoder:
        def __init__(self, buffer: io.BytesIO) -> None:
            self.buffer = buffer

        def __enter__(self) -> Encoder:
            return self

        def __exit__(self, *args: Any) -> None:
            self.buffer.write(b"controlled mp4")

        def add_stream(self, codec: str, *, rate: Fraction) -> Encoder:
            assert codec == "h264" and rate == Fraction(23976, 1000)
            return self

        def encode(self, frame: Any) -> tuple[()]:
            return ()

        def mux(self, packets: Any) -> None:
            pass

    def open_video(buffer: io.BytesIO, *, mode: str, format: str) -> Encoder:
        assert mode == "w" and format == "mp4"
        return Encoder(buffer)

    monkeypatch.setattr(av, "open", open_video)
    monkeypatch.setattr(av, "VideoFrame", Frame)
    return frames


@pytest.mark.parametrize("portable", (False, True))
@pytest.mark.parametrize("packed", (None, "empty"))
def test_empty_tracks_preserve_images_and_use_black_after_short_batch(
    rendered: list[np.ndarray], portable: bool, packed: str | None
) -> None:
    images = torch.tensor([-0.5, 0.25, 1.5]).reshape(1, 1, 1, 3).expand(1, 8, 16, 3).clone()
    original = images.clone()
    tracks = {
        "packed_masks": None if packed is None else torch.empty(3, 0, 2, 1, dtype=torch.uint8),
        "orig_size": (2, 8),
        "n_frames": 3,
    }
    with capture_preview_artifacts() as artifacts:
        result = SAM3_TrackPreview.execute(
            track_data=_portable(SAM3_TRACK_DATA_TYPE, tracks) if portable else tracks,
            images=_portable(IMAGE_TYPE, images) if portable else images,
            fps=23.976,
        )
    assert result == {} and len(artifacts) == 1
    assert artifacts[0].media_type == "video/mp4" and artifacts[0].data == b"controlled mp4"
    assert artifacts[0].name.startswith("sam3_track_preview_")
    assert artifacts[0].name.endswith(".mp4")
    assert len(rendered) == 3
    assert rendered[0].shape == (8, 16, 3)
    np.testing.assert_array_equal(rendered[0], np.tile([0, 63, 255], (8, 16, 1)))
    assert not rendered[1].any() and not rendered[2].any()
    torch.testing.assert_close(images, original, rtol=0, atol=0)


@pytest.mark.parametrize("portable", (False, True))
def test_packed_overlaps_nearest_resize_and_labels_are_nonmutating(
    rendered: list[np.ndarray], portable: bool
) -> None:
    packed = torch.tensor([[[[15]], [[60]]]], dtype=torch.uint8)
    tracks = {"packed_masks": packed, "orig_size": (32, 64), "n_frames": 1, "scores": [0.85, 1.0]}
    original = packed.clone()
    images = torch.full((1, 64, 128, 3), 0.25)
    before = images.clone()
    with capture_preview_artifacts():
        SAM3_TrackPreview.execute(
            track_data=_portable(SAM3_TRACK_DATA_TYPE, tracks) if portable else tracks,
            images=_portable(IMAGE_TYPE, images) if portable else images,
            opacity=0.75,
            fps=23.976,
        )
    frame = rendered[0]
    assert frame.shape == (64, 128, 3)
    # The first packed byte covers x=[0, 64), the second x=[32, 96).
    first = np.array([int((0.25 * 0.25 + value * 0.75) * 255) for value in (0.12, 0.47, 0.71)])
    second = np.array([int((0.25 * 0.25 + value * 0.75) * 255) for value in (1.0, 0.5, 0.05)])
    np.testing.assert_array_equal(frame[2, 8], first)
    np.testing.assert_array_equal(frame[2, 48], first)
    np.testing.assert_array_equal(frame[2, 80], second)
    np.testing.assert_array_equal(frame[2, 112], [63, 63, 63])
    # A 4096-pixel mask caps at four; the resolution font size caps it at three.
    np.testing.assert_array_equal(frame[24, 27], [30, 119, 181])
    np.testing.assert_array_equal(frame[23, 27], [0, 0, 0])
    # Confidence 85 is drawn below object zero, not below the score-one object.
    np.testing.assert_array_equal(frame[44, 24], [30, 119, 181])
    torch.testing.assert_close(packed, original, rtol=0, atol=0)
    torch.testing.assert_close(images, before, rtol=0, atol=0)


def test_absent_images_black_background_and_clipped_number(rendered: list[np.ndarray]) -> None:
    tracks = {
        "packed_masks": torch.tensor([[[[255]]]], dtype=torch.uint8),
        "orig_size": (4, 8),
        "n_frames": 1,
    }
    with capture_preview_artifacts():
        SAM3_TrackPreview.execute(track_data=tracks, opacity=1.0, fps=23.976)
    assert rendered[0].shape == (4, 8, 3)
    np.testing.assert_array_equal(rendered[0][0, 0], [30, 119, 181])


def test_real_mp4_has_all_frames_dimensions_and_fractional_rate() -> None:
    images = torch.stack((torch.zeros(16, 24, 3), torch.ones(16, 24, 3)))
    tracks = {"packed_masks": None, "orig_size": (4, 8), "n_frames": 3}
    with capture_preview_artifacts() as artifacts:
        SAM3_TrackPreview.execute(track_data=tracks, images=images, fps=23.976)
    with av.open(io.BytesIO(artifacts[0].data)) as container:
        stream = container.streams.video[0]
        assert stream.width == 24 and stream.height == 16
        assert stream.average_rate == Fraction(2997, 125)
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    assert len(frames) == 3
    assert not frames[0].any() and not frames[2].any()
    np.testing.assert_array_equal(frames[1], np.full((16, 24, 3), 255, dtype=np.uint8))


class PreviewSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "sam3-preview-fixture.source",
            outputs=(
                OutputSpec("tracks", TypeExpr.concrete(SAM3_TRACK_DATA_TYPE)),
                OutputSpec("images", TypeExpr.concrete(IMAGE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            tracks={
                "packed_masks": torch.zeros(3, 1, 2, 1, dtype=torch.uint8),
                "orig_size": (4, 8),
                "n_frames": 3,
            },
            images=torch.stack((torch.zeros(16, 24, 3), torch.ones(16, 24, 3))).numpy(),
        )


@pytest.mark.parametrize("in_process", (True, False))
def test_native_preview_publishes_through_ordinary_hosts_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, in_process: bool
) -> None:
    root = Path(__file__).resolve().parents[1]
    checkout = root.parents[1]
    native = root / "dinkster-pack.toml"
    module = f"sam3_preview_fixture_{in_process}"
    (tmp_path / f"{module}.py").write_text(
        "from test_sam3_preview_torch import PreviewSource\n"
        "from dinkster_nodes_comfy import register_inference_types\n"
        "NODES = (PreviewSource,)\n",
        encoding="ascii",
    )
    fixture = tmp_path / "fixture.toml"
    fixture.write_text(
        '[pack]\nname="sam3-preview-fixture"\npublisher="dinkster"\n'
        'namespaces=["sam3-preview-fixture"]\n[pack.entry]\n'
        f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
        encoding="ascii",
    )
    vault = AssetVault(tmp_path / "vault")
    monkeypatch.setenv("DINKSTER_ASSET_VAULT", str(vault.root))
    env = {
        "DINKSTER_ACCELERATOR": "cpu",
        "PYTHONPATH": os.pathsep.join(
            (
                str(tmp_path),
                str(Path(__file__).parent),
                str(checkout / "src"),
                *(str(path) for path in sorted((checkout / "packages").glob("*/src"))),
                *sys.path,
            )
        ),
    }

    async def scenario() -> None:
        composer = ServingComposer(worker_env=env)
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest, in_process=True, env=env)
                    if in_process
                    else PackSpec(
                        manifest,
                        worker_group="sam3-preview-fixture",
                        group_manifests=(native, fixture),
                        env=env,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("sam3-preview-fixture.source"),
                    "preview": GraphNode(
                        "SAM3_TrackPreview",
                        {
                            "track_data": Link("source", "tracks"),
                            "images": Link("source", "images"),
                            "fps": 23.976,
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["preview"])
                assert len(result.artifacts) == 1
                artifact = result.artifacts[0]
                assert artifact.node_id == "preview" and artifact.media_type == "video/mp4"
                assert artifact.name.startswith("sam3_track_preview_")
                path = vault.resolve(artifact.digest)
                assert path is not None and path.stat().st_size == artifact.size
                with av.open(path) as container:
                    stream = container.streams.video[0]
                    assert (stream.width, stream.height) == (24, 16)
                    assert stream.average_rate == Fraction(2997, 125)
                    frames = [
                        frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)
                    ]
                assert len(frames) == 3
                assert not frames[0].any() and not frames[2].any()
                np.testing.assert_array_equal(frames[1], np.full((16, 24, 3), 255, dtype=np.uint8))
        finally:
            await composer.close()

    asyncio.run(scenario())
