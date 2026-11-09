"""SeedVR2 padding values, boundaries, and ordinary pack execution."""

import asyncio
from pathlib import Path

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
from dinkster_nodes_comfy.seedvr_preprocess import SeedVR2Preprocess

from dinkster.compose import PackSpec, ServingComposer


def frames(count: int, batch: bool = False) -> torch.Tensor:
    shape = (2, count, 3, 17, 4) if batch else (count, 3, 17, 4)
    size = 2 * count * 3 * 17 * 4 if batch else count * 3 * 17 * 4
    return torch.linspace(-0.3, 1.4, size, dtype=torch.float64).reshape(shape)


def expected(image: torch.Tensor) -> torch.Tensor:
    source = image.unsqueeze(0) if image.ndim == 4 else image
    count = source.shape[1]
    length = count if count == 1 else 1 + ((count + 2) // 4) * 4
    output = torch.zeros((source.shape[0], length, 16, 32, 3), dtype=image.dtype)
    for batch in range(source.shape[0]):
        for time in range(length):
            output[batch, time, :3, :17] = source[batch, min(time, count - 1), ..., :3].clamp(0, 1)
    return output


@pytest.mark.parametrize("count", (1, 2, 4, 5, 6, 9))
@pytest.mark.parametrize("batch", (False, True))
def test_preprocess_padding_and_source_preservation(count: int, batch: bool) -> None:
    image = frames(count, batch)
    before = image.clone()
    output = SeedVR2Preprocess.execute(resized_images=image)["images"]
    assert output.dtype == image.dtype and output.device == image.device
    assert output.is_contiguous()
    assert torch.equal(output, expected(image))
    assert torch.equal(image, before)


@pytest.mark.parametrize(
    "shape, message",
    (((2, 3), "4-D or 5-D"), ((1, 1, 3, 3), "shorter edge"), ((0, 3, 17, 4), "at least one frame")),
)
def test_preprocess_invalid_shape(shape: tuple[int, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SeedVR2Preprocess.execute(resized_images=torch.zeros(shape))


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="seedvr-fixture.source",
            outputs=(OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(image=frames(6, True).float())


class Consumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="seedvr-fixture.consumer",
            inputs=(InputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, image: object) -> dict[str, object]:
        output = materialize_image(image)
        assert torch.equal(output, expected(frames(6, True).float()))
        assert output.dtype == torch.float32 and output.device.type == "cpu"
        return cls.outputs(summary="seedvr-padding")


@pytest.mark.parametrize("in_process", (True,))
def test_preprocess_through_unchanged_native_pack_twice(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": str(Path(__file__).parent),
            }
        )
        try:
            (tmp_path / "seedvr_fixture.py").write_text(
                "from test_seedvr_preprocess_torch import Source, Consumer\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (Source, Consumer)\n",
                encoding="ascii",
            )
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                '[pack]\nname="seedvr-fixture"\npublisher="dinkster"\n'
                'namespaces=["seedvr-fixture"]\n[pack.entry]\n'
                'nodes="seedvr_fixture:NODES"\n'
                'types="seedvr_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            native = Path(__file__).parents[1] / "dinkster-pack.toml"
            for path in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=path,
                        in_process=in_process,
                        worker_group=None if in_process else "seedvr-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("seedvr-fixture.source"),
                    "pad": GraphNode(
                        "SeedVR2Preprocess", {"resized_images": Link("source", "image")}
                    ),
                    "consumer": GraphNode(
                        "seedvr-fixture.consumer", {"image": Link("pad", "images")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "seedvr-padding"
        finally:
            await composer.close()

    asyncio.run(scenario())
