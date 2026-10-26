"""CPU Chroma Radiance option patch and ordinary-host coverage."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.chroma_options import ChromaRadianceOptions

from dinkster.compose import PackSpec, ServingComposer


class Patcher:
    def __init__(self, wrapper: Any = None) -> None:
        self.model_options: dict[str, Any] = {}
        if wrapper is not None:
            self.model_options["model_function_wrapper"] = wrapper
        self.wrapper: Any = None

    def clone(self) -> Patcher:
        result = Patcher()
        result.model_options = self.model_options.copy()
        return result

    def set_model_unet_function_wrapper(self, wrapper: Any) -> None:
        self.wrapper = wrapper


def _apply_model(input_value: Any, timestep: Any, **c: Any) -> dict[str, Any]:
    return {"input": input_value, "timestep": timestep, "c": c}


def _args(timestep: list[float], c: dict[str, Any]) -> dict[str, Any]:
    return {"input": "input", "timestep": torch.tensor(timestep), "c": c}


def test_no_chroma_options_returns_the_original_model() -> None:
    source = Patcher()
    result = ChromaRadianceOptions.execute(
        model=source,
        preserve_wrapper=True,
        start_sigma=1.0,
        end_sigma=0.0,
        nerf_tile_size=-1,
        force_sequential_txt_ids=False,
    )["MODEL"]
    assert result is source


@pytest.mark.parametrize(
    ("timestep", "active"),
    (
        ([0.1], False),
        ([0.25], True),
        ([0.1, 0.5], True),
        ([0.75], True),
        ([0.9], False),
    ),
)
def test_chroma_options_apply_at_inclusive_max_batch_sigma_boundaries(
    timestep: list[float], active: bool
) -> None:
    source = Patcher()
    patched = ChromaRadianceOptions.execute(
        model=source,
        preserve_wrapper=False,
        start_sigma=0.75,
        end_sigma=0.25,
        nerf_tile_size=64,
        force_sequential_txt_ids=True,
    )["MODEL"]
    original_options = {"existing": "keep"}
    original_c = {"other": "value", "transformer_options": original_options}
    result = patched.wrapper(_apply_model, _args(timestep, original_c))
    assert source.wrapper is None
    assert original_c == {"other": "value", "transformer_options": {"existing": "keep"}}
    assert result["c"]["other"] == "value"
    if active:
        assert result["c"]["transformer_options"] == {
            "existing": "keep",
            "chroma_radiance_options": {"nerf_tile_size": 64, "use_sequential_txt_ids": True},
        }
    else:
        assert result["c"] == original_c


@pytest.mark.parametrize("preserve_wrapper", (True, False))
def test_chroma_options_preserves_or_bypasses_an_existing_wrapper(preserve_wrapper: bool) -> None:
    calls: list[dict[str, Any]] = []

    def old_wrapper(apply_model: Any, args: dict[str, Any]) -> Any:
        calls.append(args)
        return {"old": True, "c": args["c"]}

    patched = ChromaRadianceOptions.execute(
        model=Patcher(old_wrapper),
        preserve_wrapper=preserve_wrapper,
        start_sigma=0.8,
        end_sigma=0.2,
        nerf_tile_size=0,
        force_sequential_txt_ids=False,
    )["MODEL"]
    result = patched.wrapper(_apply_model, _args([0.1], {"other": "value"}))
    if preserve_wrapper:
        assert result == {"old": True, "c": {"other": "value"}}
        assert len(calls) == 1
    else:
        assert result == {
            "input": "input",
            "timestep": torch.tensor([0.1]),
            "c": {"other": "value"},
        }
        assert calls == []


class ModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="chroma-options-fixture.source",
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(MODEL=Patcher())


class Inspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="chroma-options-fixture.inspect",
            inputs=(InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),),
            outputs=(OutputSpec("result", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, model: Patcher) -> Mapping[str, object]:
        result = model.wrapper(_apply_model, _args([0.5], {"other": "value"}))
        return cls.outputs(
            result=str(result["c"]["transformer_options"]["chroma_radiance_options"])
        )


@pytest.mark.parametrize("in_process", (True, False))
def test_chroma_options_execute_twice_in_both_ordinary_manifest_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        native_manifest = package / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (
                        str(tmp_path),
                        str(Path(__file__).parent),
                        str(package / "src"),
                        os.environ.get("PYTHONPATH", ""),
                    )
                ),
            }
        )
        try:
            (tmp_path / "chroma_options_fixture.py").write_text(
                "from test_chroma_options_torch import Inspect, ModelSource\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = (ModelSource, Inspect)\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "dinkster-pack.toml"
            fixture_manifest.write_text(
                '[pack]\nname = "chroma-options-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["chroma-options-fixture"]\n\n[pack.entry]\n'
                'nodes = "chroma_options_fixture:NODES"\n'
                'types = "chroma_options_fixture:register_inference_types"\n',
                encoding="ascii",
            )
            for manifest in (native_manifest, fixture_manifest):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "chroma-options-fixture",
                        group_manifests=() if in_process else (native_manifest, fixture_manifest),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("chroma-options-fixture.source"),
                    "options": GraphNode(
                        "ChromaRadianceOptions",
                        {
                            "model": Link("source", "MODEL"),
                            "preserve_wrapper": False,
                            "start_sigma": 0.8,
                            "end_sigma": 0.2,
                            "nerf_tile_size": 32,
                            "force_sequential_txt_ids": True,
                        },
                    ),
                    "inspect": GraphNode(
                        "chroma-options-fixture.inspect", {"model": Link("options", "MODEL")}
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["result"].resolve() == (
                    "{'nerf_tile_size': 32, 'use_sequential_txt_ids': True}"
                )
        finally:
            await composer.close()

    asyncio.run(scenario())
