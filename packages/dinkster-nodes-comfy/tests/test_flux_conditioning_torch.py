"""Flux conditioning preserves tensors and metadata through ordinary pack hosts."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CONDITIONING_TYPE,
    CORE_BOOLEAN,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link

from dinkster.compose import PackSpec, ServingComposer


class ConditioningSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="flux-fixture.source",
            outputs=(OutputSpec("conditioning", TypeExpr.concrete(CONDITIONING_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(
            conditioning=[
                [torch.arange(6).reshape(1, 2, 3).float(), {"guidance": 8.5, "start_percent": 0.2}],
                [torch.ones(1, 2, 3), {"guidance": 4.5, "end_percent": 0.8}],
            ]
        )


class InspectConditioning(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="flux-fixture.inspect",
            inputs=tuple(
                InputSpec(name, TypeExpr.concrete(CONDITIONING_TYPE))
                for name in ("original", "guided", "disabled", "reference")
            ),
            outputs=(OutputSpec("valid", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(
        cls, *, original: Any, guided: Any, disabled: Any, reference: Any
    ) -> dict[str, object]:
        for value, guidance in ((original, (8.5, 4.5)), (guided, (0.0, 0.0))):
            assert value[0][1] == {"guidance": guidance[0], "start_percent": 0.2}
            assert value[1][1] == {"guidance": guidance[1], "end_percent": 0.8}
        assert disabled[0][1] == {"guidance": None, "start_percent": 0.2}
        assert disabled[1][1] == {"guidance": None, "end_percent": 0.8}
        for index, extra in ((0, {"start_percent": 0.2}), (1, {"end_percent": 0.8})):
            assert reference[index][1] == {
                "guidance": None,
                "reference_latents_method": "uxo",
                **extra,
            }
        for value in (original, guided, disabled, reference):
            assert torch.equal(torch.as_tensor(value[0][0]), torch.arange(6).reshape(1, 2, 3))
            assert torch.equal(torch.as_tensor(value[1][0]), torch.ones(1, 2, 3))
        return cls.outputs(valid=True)


@pytest.mark.parametrize("in_process", (True, False))
def test_flux_ordinary_manifest(tmp_path: Path, in_process: bool) -> None:
    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": os.pathsep.join(
                    (str(root / "tests"), str(root / "src"), os.environ.get("PYTHONPATH", ""))
                )
            }
        )
        try:
            (tmp_path / "flux_fixture.py").write_text(
                "from test_flux_conditioning_torch import ConditioningSource, InspectConditioning\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "NODES = (*COMFY_NODES, ConditioningSource, InspectConditioning)\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            manifest.write_text(
                (root / "dinkster-pack.toml")
                .read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["flux-fixture",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", "flux_fixture:NODES")
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    "flux_fixture:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            graph = Graph(
                nodes={
                    "source": GraphNode("flux-fixture.source"),
                    "guided": GraphNode(
                        "FluxGuidance",
                        {"conditioning": Link("source", "conditioning"), "guidance": 0.0},
                    ),
                    "disabled": GraphNode(
                        "FluxDisableGuidance", {"conditioning": Link("guided", "CONDITIONING")}
                    ),
                    "reference": GraphNode(
                        "FluxKontextMultiReferenceLatentMethod",
                        {
                            "conditioning": Link("disabled", "CONDITIONING"),
                            "reference_latents_method": "uxo/uno",
                        },
                    ),
                    "inspect": GraphNode(
                        "flux-fixture.inspect",
                        {
                            "original": Link("source", "conditioning"),
                            "guided": Link("guided", "CONDITIONING"),
                            "disabled": Link("disabled", "CONDITIONING"),
                            "reference": Link("reference", "CONDITIONING"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["valid"].resolve() is True
        finally:
            await composer.close()

    asyncio.run(scenario())
