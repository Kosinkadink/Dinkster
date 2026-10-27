"""CPU null conditioning and ordinary-host transport for Lotus."""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_STRING,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.lotus import LotusConditioning

from dinkster.compose import PackSpec, ServingComposer

_EMBEDDING_HASH = "a7ba15b03c89931a7764a7d4b398201b322e00524d03efaac172b48dc637397d"


def _check(conditioning: Any) -> str:
    assert len(conditioning) == 1 and conditioning[0][1] == {}
    embedding = materialize_latent(conditioning[0][0])
    assert embedding.dtype == torch.float32 and embedding.shape == (1, 2, 1024)
    assert embedding.device.type == "cpu"
    digest = hashlib.sha256(embedding.numpy().tobytes()).hexdigest()
    assert digest == _EMBEDDING_HASH
    return digest


def test_lotus_returns_exact_frozen_embedding_with_independent_storage() -> None:
    first = LotusConditioning.execute()["conditioning"]
    assert _check(first) == _EMBEDDING_HASH
    first[0][0].fill_(12)
    first[0][1]["changed"] = True
    second = LotusConditioning.execute()["conditioning"]
    assert _check(second) == _EMBEDDING_HASH
    assert first[0][0].data_ptr() != second[0][0].data_ptr()
    assert first[0][1] is not second[0][1]


class LotusConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            "lotus-fixture.consume",
            inputs=(InputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),),
            outputs=(OutputSpec("digest", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, conditioning: Any) -> dict[str, object]:
        return cls.outputs(digest=_check(conditioning))


@pytest.mark.parametrize("in_process", (True, False))
def test_lotus_native_manifest_and_public_fixture_in_both_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(Path(__file__).parent), os.environ.get("PYTHONPATH", ""))
                ),
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
            }
        )
        try:
            module = f"lotus_fixture_{in_process}"
            (tmp_path / f"{module}.py").write_text(
                "from test_lotus_torch import LotusConsumer\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (LotusConsumer,)\n",
                encoding="ascii",
            )
            native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            fixture = tmp_path / "dinkster-pack.toml"
            fixture.write_text(
                '[pack]\nname = "lotus-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["lotus-fixture"]\n[pack.entry]\n'
                f'nodes = "{module}:NODES"\ntypes = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(manifest=manifest, in_process=True)
                    if in_process
                    else PackSpec(
                        manifest=manifest,
                        worker_group="lotus-fixture",
                        group_manifests=(native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "lotus": GraphNode("LotusConditioning"),
                    "consumer": GraphNode(
                        "lotus-fixture.consume", {"conditioning": Link("lotus", "conditioning")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["digest"].resolve() == _EMBEDDING_HASH
        finally:
            await composer.close()

    asyncio.run(scenario())
