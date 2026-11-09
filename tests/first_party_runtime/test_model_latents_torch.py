"""Empty model latents retain shape and metadata through ordinary pack hosts."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_3d_models.latent_model_specific import EmptyTrellis2LatentStructure
from dinkster_api.v1 import (
    CORE_STRING,
    LATENT_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_flux.latent_model_specific import EmptyFlux2LatentImage
from dinkster_graph import Graph, GraphNode, Link
from dinkster_hidream.latent_model_specific import EmptyHiDreamO1LatentImage
from dinkster_hunyuan.latent_model_specific import EmptyLatentHunyuan3Dv2
from dinkster_misc_models.latent_model_specific import EmptyChromaRadianceLatentImage
from dinkster_qwen.latent_model_specific import EmptyQwenImageLayeredLatentImage
from dinkster_sd.latent_model_specific import EmptySD3LatentImage

from dinkster.compose import PackSpec, ServingComposer, _installed_pack_spec

CASES = (
    (
        EmptySD3LatentImage,
        {"width": 48, "height": 80, "batch_size": 2},
        (2, 16, 10, 6),
        {"downscale_ratio_spacial": 8},
    ),
    (EmptyFlux2LatentImage, {"width": 48, "height": 80, "batch_size": 2}, (2, 128, 5, 3), {}),
    (
        EmptyChromaRadianceLatentImage,
        {"width": 48, "height": 80, "batch_size": 2},
        (2, 3, 80, 48),
        {},
    ),
    (EmptyHiDreamO1LatentImage, {"width": 64, "height": 96, "batch_size": 2}, (2, 3, 96, 64), {}),
    (
        EmptyQwenImageLayeredLatentImage,
        {"width": 48, "height": 80, "layers": 2, "batch_size": 2},
        (2, 16, 3, 10, 6),
        {},
    ),
    (
        EmptyQwenImageLayeredLatentImage,
        {"width": 16, "height": 32, "layers": 0, "batch_size": 1},
        (1, 16, 1, 4, 2),
        {},
    ),
    (
        EmptyLatentHunyuan3Dv2,
        {"resolution": 17, "batch_size": 2},
        (2, 64, 17),
        {"type": "hunyuan3dv2"},
    ),
    (EmptyTrellis2LatentStructure, {"batch_size": 2}, (2, 32, 16, 16, 16), {"type": "trellis2"}),
)


@pytest.mark.parametrize(("node", "inputs", "shape", "metadata"), CASES)
def test_model_latent_layout(
    node: type[Node], inputs: dict[str, int], shape: tuple[int, ...], metadata: dict[str, Any]
) -> None:
    result = node.execute(**inputs)["LATENT"]
    samples = result["samples"]
    assert tuple(samples.shape) == shape
    assert samples.device.type == "cpu"
    assert samples.dtype == torch.float32
    assert torch.count_nonzero(samples).item() == 0
    assert {key: value for key, value in result.items() if key != "samples"} == metadata
    with torch.inference_mode():
        samples.fill_(3)
    assert torch.count_nonzero(node.execute(**inputs)["LATENT"]["samples"]).item() == 0


class InspectLatent(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="latent-fixture.inspect",
            inputs=(InputSpec("latent", TypeExpr.concrete(LATENT_TYPE)),),
            outputs=(OutputSpec("text", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, latent: dict[str, Any]) -> dict[str, object]:
        samples = materialize_latent(latent)["samples"]
        return cls.outputs(
            text=json.dumps(
                {
                    "shape": list(samples.shape),
                    "zeros": torch.count_nonzero(samples).item() == 0,
                    "metadata": {k: v for k, v in latent.items() if k != "samples"},
                },
                sort_keys=True,
            )
        )


FIXTURES = (InspectLatent,)


@pytest.mark.parametrize("in_process", (True,))
def test_model_latents_ordinary_manifest(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", ""))
                )
            }
        )
        try:
            (tmp_path / "latent_fixture.py").write_text(
                "from test_model_latents_torch import FIXTURES\n"
                "from dinkster_api.v1 import register_inference_types\n"
                "NODES = FIXTURES\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                '[pack]\nname = "latent-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["latent-fixture"]\n[pack.entry]\n'
                'nodes = "latent_fixture:NODES"\n'
                'types = "latent_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            for identity, module in (
                ("dinkster-latent", "dinkster_latent"),
                ("dinkster-conditioning", "dinkster_conditioning"),
                ("dinkster-sampling", "dinkster_sampling"),
                ("dinkster-video", "dinkster_video"),
                ("dinkster-3d-models", "dinkster_3d_models"),
                ("dinkster-flux", "dinkster_flux"),
                ("dinkster-hidream", "dinkster_hidream"),
                ("dinkster-hunyuan", "dinkster_hunyuan"),
                ("dinkster-qwen", "dinkster_qwen"),
                ("dinkster-misc-models", "dinkster_misc_models"),
                ("dinkster-sd", "dinkster_sd"),
            ):
                await composer.add_pack(
                    _installed_pack_spec(identity, module, in_process=in_process)
                )
            await composer.add_pack(PackSpec(manifest=fixture_manifest, in_process=in_process))
            nodes = {}
            for index, (node, inputs, _, _) in enumerate(CASES):
                nodes[f"source-{index}"] = GraphNode(node.schema().node_type, inputs)
                nodes[f"inspect-{index}"] = GraphNode(
                    "latent-fixture.inspect", {"latent": Link(f"source-{index}", "LATENT")}
                )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(
                    Graph(nodes=nodes), [f"inspect-{index}" for index in range(len(CASES))]
                )
                for index, (_, _, shape, metadata) in enumerate(CASES):
                    assert json.loads(result.outputs[f"inspect-{index}"]["text"].resolve()) == {
                        "shape": list(shape),
                        "zeros": True,
                        "metadata": metadata,
                    }
        finally:
            await composer.close()

    asyncio.run(scenario())
