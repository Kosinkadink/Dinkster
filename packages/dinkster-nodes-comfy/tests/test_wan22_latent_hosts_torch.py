"""Wan 2.2 media values through unchanged native pack hosts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    IMAGE_TYPE,
    LATENT_TYPE,
    VAE_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_api.v1 import (
    register_inference_types as register_inference_types,
)
from dinkster_graph import Graph, GraphNode, Link

from dinkster.compose import PackSpec, ServingComposer


class VideoEncoder:
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        assert frames.shape == (5, 32, 48, 4)
        return torch.full((1, 48, 2, 2, 3), 7.0)


class Source(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wan22-latent-fixture.source",
            outputs=(
                OutputSpec("vae", TypeExpr.concrete(VAE_TYPE)),
                OutputSpec("image", TypeExpr.concrete(IMAGE_TYPE)),
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(vae=VideoEncoder(), image=torch.zeros((5, 32, 48, 4)))


class Consumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="wan22-latent-fixture.consumer",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(LATENT_TYPE)) for name in ("empty", "guided")
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, empty: Any, guided: Any) -> dict[str, object]:
        empty, guided = materialize_latent(empty), materialize_latent(guided)
        assert set(empty) == {"samples"}
        assert empty["samples"].shape == (1, 48, 4, 2, 3)
        assert not empty["samples"].count_nonzero()
        samples, mask = guided["samples"], guided["noise_mask"]
        assert samples.shape == (3, 48, 4, 2, 3)
        assert samples.device.type == "cpu" and samples.dtype == torch.float32
        torch.testing.assert_close(samples[:, :, :2], torch.full((3, 48, 2, 2, 3), 7.0))
        assert mask.shape == (3, 1, 4, 2, 3)
        assert not mask[:, :, :2].count_nonzero()
        assert torch.all(mask[:, :, 2:] == 1)
        # Channel-specific unfilled values distinguish Wan22 normalization from raw zeros.
        assert samples[0, 0, 2, 0, 0].item() == torch.tensor(-0.2289).item()
        assert torch.equal(samples[0], samples[2])
        return cls.outputs(summary="empty-one-guided-three")


HOST_NODES = (Source, Consumer)


@pytest.mark.parametrize("in_process", (True, False))
def test_wan22_runs_twice_in_ordinary_hosts(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        (tmp_path / "wan22_latent_host_entry.py").write_text(
            "from test_wan22_latent_hosts_torch import HOST_NODES, register_inference_types\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="wan22-latent-fixture"\npublisher="dinkster"\n'
            'namespaces=["wan22-latent-fixture"]\n[pack.entry]\n'
            'nodes="wan22_latent_host_entry:HOST_NODES"\n'
            'types="wan22_latent_host_entry:register_inference_types"\n',
            encoding="ascii",
        )
        native = package / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(package / "src"))
                ),
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "wan22-latent-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            inputs = {
                "vae": Link("source", "vae"),
                "width": 48,
                "height": 32,
                "length": 13,
                "batch_size": 3,
            }
            graph = Graph(
                nodes={
                    "source": GraphNode("wan22-latent-fixture.source"),
                    "empty": GraphNode("Wan22ImageToVideoLatent", inputs),
                    "guided": GraphNode(
                        "Wan22ImageToVideoLatent",
                        {**inputs, "start_image": Link("source", "image")},
                    ),
                    "consumer": GraphNode(
                        "wan22-latent-fixture.consumer",
                        {
                            "empty": Link("empty", "LATENT"),
                            "guided": Link("guided", "LATENT"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "empty-one-guided-three"
        finally:
            await composer.close()

    asyncio.run(scenario())
