"""CPU contracts for native LTXV image preprocessing."""

from __future__ import annotations

import asyncio
import os
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_image,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.ltx_video_preprocess import LTXVPreprocess

from dinkster.compose import PackSpec, ServingComposer

_IMAGE = TypeExpr.concrete(IMAGE_TYPE)


def _frames(dtype: torch.dtype) -> torch.Tensor:
    frame0 = torch.arange(5 * 7 * 3, dtype=dtype).reshape(5, 7, 3) / 104
    frame1 = torch.flip(frame0, (0, 1))
    return torch.stack((frame0, frame1))


def _expected_h264(image: torch.Tensor, compression: int) -> torch.Tensor:
    import av

    output_frames = []
    for frame in image:
        pixels = (
            (frame[: (frame.shape[0] // 2) * 2, : (frame.shape[1] // 2) * 2] * 255.0)
            .byte()
            .cpu()
            .numpy()
        )
        with BytesIO() as encoded:
            container = av.open(encoded, "w", format="mp4")
            stream = container.add_stream(
                "libx264", rate=1, options={"crf": str(compression), "preset": "veryfast"}
            )
            stream.height, stream.width = pixels.shape[:2]
            av_frame = av.VideoFrame.from_ndarray(pixels, format="rgb24").reformat(format="yuv420p")
            container.mux(stream.encode(av_frame))
            container.mux(stream.encode())
            container.close()
            with BytesIO(encoded.getvalue()) as decoded:
                container = av.open(decoded)
                stream = next(stream for stream in container.streams if stream.type == "video")
                output_frames.append(next(container.decode(stream)).to_ndarray(format="rgb24"))
                container.close()
    return (
        torch.from_numpy(np.stack(output_frames)).to(dtype=image.dtype, device=image.device) / 255.0
    )


def test_compression_zero_preserves_all_frames_identity_and_values() -> None:
    image = _frames(torch.float64)
    output = LTXVPreprocess.execute(image=image, img_compression=0)["IMAGE"]
    assert output.dtype is torch.float64
    assert output.device == image.device
    assert output.shape == image.shape
    assert torch.equal(output, image)


def test_h264_preprocess_trims_odd_dimensions_preserves_dtype_and_processes_each_frame() -> None:
    image = _frames(torch.float32)
    output = LTXVPreprocess.execute(image=image, img_compression=35)["IMAGE"]
    assert output.shape == (2, 4, 6, 3)
    assert output.dtype is torch.float32
    assert output.device == image.device
    assert torch.all((output >= 0) & (output <= 1))
    assert not torch.equal(output[0], output[1])
    torch.testing.assert_close(output, _expected_h264(image, 35), rtol=0, atol=0)
    assert torch.equal(image, _frames(torch.float32))


def test_h264_preprocess_preserves_float64_dtype() -> None:
    output = LTXVPreprocess.execute(image=_frames(torch.float64), img_compression=23)["IMAGE"]
    assert output.dtype is torch.float64
    assert output.shape == (2, 4, 6, 3)


class LTXPreprocessSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ltx-preprocess-fixture.source",
            outputs=(OutputSpec("image", _IMAGE),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(image=_frames(torch.float32))


class LTXPreprocessConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="ltx-preprocess-fixture.consumer",
            inputs=(InputSpec("image", _IMAGE),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, image: object) -> dict[str, object]:
        image = materialize_image(image)
        expected = _expected_h264(_frames(torch.float32), 35)
        assert image.shape == (2, 4, 6, 3)
        assert image.dtype is torch.float32
        assert image.device.type == "cpu"
        torch.testing.assert_close(image, expected, rtol=0, atol=0)
        return cls.outputs(summary="ltx-preprocess")


@pytest.mark.parametrize("in_process", (True,))
def test_ltx_video_preprocess_executes_twice_through_both_production_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(source_root))
                ),
            }
        )
        try:
            native_manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            (tmp_path / "ltx_preprocess_fixture.py").write_text(
                "from test_ltx_video_preprocess_torch import (\n"
                "    LTXPreprocessConsumer, LTXPreprocessSource,\n"
                ")\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (LTXPreprocessSource, LTXPreprocessConsumer)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                "[pack]\n"
                'name = "ltx-preprocess-fixture"\n'
                'publisher = "dinkster"\n'
                'namespaces = ["ltx-preprocess-fixture"]\n'
                "\n[pack.entry]\n"
                'nodes = "ltx_preprocess_fixture:NODES"\n'
                'types = "ltx_preprocess_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            if in_process:
                await composer.add_pack(PackSpec(manifest=native_manifest, in_process=True))
                await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=True))
            else:
                group_manifests = (native_manifest, fixture_manifest)
                await composer.add_pack(
                    PackSpec(
                        manifest=native_manifest,
                        worker_group="ltx-preprocess-fixture",
                        group_manifests=group_manifests,
                    )
                )
                await composer.add_pack(
                    PackSpec(
                        manifest=fixture_manifest,
                        worker_group="ltx-preprocess-fixture",
                        group_manifests=group_manifests,
                    )
                )
            engine = composer.composition.make_engine(lambda _event: None)
            graph = Graph(
                nodes={
                    "source": GraphNode("ltx-preprocess-fixture.source"),
                    "preprocess": GraphNode(
                        "LTXVPreprocess",
                        {"image": Link("source", "image"), "img_compression": 35},
                    ),
                    "consumer": GraphNode(
                        "ltx-preprocess-fixture.consumer", {"image": Link("preprocess", "IMAGE")}
                    ),
                }
            )
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "ltx-preprocess"
        finally:
            await composer.close()

    asyncio.run(scenario())
