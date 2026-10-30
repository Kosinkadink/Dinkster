"""CPU behavior for native context-window nodes."""

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
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.context_windows import (
    ContextWindowsManual,
    LTXVContextWindows,
    WanContextWindowsManual,
)


def _model_patcher() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    return ModelPatcher(torch.nn.Linear(2, 2), torch.device("cpu"), torch.device("cpu"))


@pytest.mark.parametrize(
    ("node", "inputs", "expected"),
    (
        (
            ContextWindowsManual,
            {
                "context_length": 17,
                "context_overlap": 5,
                "context_schedule": "batched",
                "context_stride": 3,
                "closed_loop": True,
                "fuse_method": "overlap-linear",
                "dim": 1,
                "freenoise": False,
                "cond_retain_index_list": "0, 7",
                "split_conds_to_windows": True,
                "latent_retain_index_list": "2, 5",
                "causal_window_fix": False,
            },
            (17, 5, 1, [0, 7], [2, 5], False),
        ),
        (
            WanContextWindowsManual,
            {
                "context_length": 81,
                "context_overlap": 30,
                "context_schedule": "standard_uniform",
                "context_stride": 2,
                "closed_loop": False,
                "fuse_method": "relative",
                "freenoise": True,
                "retain_first_frame": True,
                "split_conds_to_windows": True,
            },
            (21, 7, 2, [0], [], True),
        ),
        (
            LTXVContextWindows,
            {
                "context_length": 145,
                "context_overlap": 40,
                "context_schedule": "looped_uniform",
                "context_stride": 2,
                "closed_loop": True,
                "fuse_method": "flat",
                "freenoise": True,
                "retain_first_frame": True,
                "split_conds_to_windows": True,
            },
            (19, 5, 2, [0], [0], True),
        ),
    ),
)
def test_context_window_nodes_clone_and_configure_model_patcher(
    node: type[Any], inputs: dict[str, object], expected: tuple[object, ...]
) -> None:
    source = _model_patcher()
    source.model_options["transformer_options"]["preserved"] = {"value": 17}
    result = node.execute(model=source, **inputs)["MODEL"]
    handler = result.model_options["context_handler"]
    assert result is not source
    assert "context_handler" not in source.model_options
    assert result.model_options["transformer_options"] == {"preserved": {"value": 17}}
    assert (
        handler.context_length,
        handler.context_overlap,
        handler.dim,
        handler.cond_retain_index_list,
        handler.latent_retain_index_list,
        handler.freenoise,
    ) == expected
    assert handler.split_conds_to_windows is inputs["split_conds_to_windows"]
    assert handler.causal_window_fix is (inputs.get("causal_window_fix") is not False)


class ContextWindowSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="context-window-fixture.source",
            outputs=(OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(model=_model_patcher())


class ContextWindowConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="context-window-fixture.consumer",
            inputs=(
                InputSpec("manual", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("wan", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("ltxv", TypeExpr.concrete(MODEL_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, manual: Any, wan: Any, ltxv: Any) -> Mapping[str, object]:
        expected = (
            (manual, 17, 5, 1, [0, 7], [2, 5], False),
            (wan, 21, 7, 2, [0], [], True),
            (ltxv, 19, 5, 2, [0], [0], True),
        )
        for model, length, overlap, dim, cond_retain, latent_retain, freenoise in expected:
            handler = model.model_options["context_handler"]
            assert (handler.context_length, handler.context_overlap, handler.dim) == (
                length,
                overlap,
                dim,
            )
            assert handler.cond_retain_index_list == cond_retain
            assert handler.latent_retain_index_list == latent_retain
            assert handler.freenoise is freenoise
            assert "ContextWindows_prepare_sampling" in model.wrappers["prepare_sampling"]
            assert (
                "ContextWindows_sampler_sample" in model.wrappers.get("sampler_sample", {})
            ) is freenoise
        return cls.outputs(summary="context-windows")


CONTEXT_WINDOW_FIXTURE_NODES = (ContextWindowSource, ContextWindowConsumer)


@pytest.mark.parametrize("in_process", (True,))
def test_context_windows_execute_twice_with_separate_fixture_pack_in_both_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    from dinkster.compose import PackSpec, ServingComposer

    async def scenario() -> None:
        source_root = Path(__file__).resolve().parents[1] / "src"
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(Path(__file__).parent), str(source_root))),
            }
        )
        try:
            module = f"context_windows_fixture_{int(in_process)}"
            (tmp_path / f"{module}.py").write_text(
                "from test_context_windows_torch import CONTEXT_WINDOW_FIXTURE_NODES\n"
                "from dinkster_nodes_comfy import register_inference_types\n"
                "NODES = CONTEXT_WINDOW_FIXTURE_NODES\n",
                encoding="ascii",
            )
            fixture_manifest = tmp_path / "fixture-dinkster-pack.toml"
            fixture_manifest.write_text(
                '[pack]\nname = "context-window-fixture"\npublisher = "dinkster"\n'
                'namespaces = ["context-window-fixture"]\n'
                f'[pack.entry]\nnodes = "{module}:NODES"\n'
                f'types = "{module}:register_inference_types"\n',
                encoding="ascii",
            )
            native_manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            specs = (
                (
                    PackSpec(manifest=native_manifest, in_process=True),
                    PackSpec(manifest=fixture_manifest, in_process=True),
                )
                if in_process
                else (
                    PackSpec(
                        manifest=native_manifest,
                        worker_group="context-window-fixture",
                        group_manifests=(native_manifest, fixture_manifest),
                    ),
                    PackSpec(
                        manifest=fixture_manifest,
                        worker_group="context-window-fixture",
                        group_manifests=(native_manifest, fixture_manifest),
                    ),
                )
            )
            for spec in specs:
                await composer.add_pack(spec)
            graph = Graph(
                nodes={
                    "source": GraphNode("context-window-fixture.source"),
                    "manual": GraphNode(
                        "ContextWindowsManual",
                        {
                            "model": Link("source", "model"),
                            "context_length": 17,
                            "context_overlap": 5,
                            "context_schedule": "batched",
                            "context_stride": 3,
                            "closed_loop": True,
                            "fuse_method": "overlap-linear",
                            "dim": 1,
                            "freenoise": False,
                            "cond_retain_index_list": "0, 7",
                            "split_conds_to_windows": True,
                            "latent_retain_index_list": "2, 5",
                            "causal_window_fix": False,
                        },
                    ),
                    "wan": GraphNode(
                        "WanContextWindowsManual",
                        {
                            "model": Link("source", "model"),
                            "context_length": 81,
                            "context_overlap": 30,
                            "context_schedule": "standard_uniform",
                            "context_stride": 2,
                            "closed_loop": False,
                            "fuse_method": "relative",
                            "freenoise": True,
                            "retain_first_frame": True,
                            "split_conds_to_windows": True,
                        },
                    ),
                    "ltxv": GraphNode(
                        "LTXVContextWindows",
                        {
                            "model": Link("source", "model"),
                            "context_length": 145,
                            "context_overlap": 40,
                            "context_schedule": "looped_uniform",
                            "context_stride": 2,
                            "closed_loop": True,
                            "fuse_method": "flat",
                            "freenoise": True,
                            "retain_first_frame": True,
                            "split_conds_to_windows": True,
                        },
                    ),
                    "consumer": GraphNode(
                        "context-window-fixture.consumer",
                        {
                            "manual": Link("manual", "MODEL"),
                            "wan": Link("wan", "MODEL"),
                            "ltxv": Link("ltxv", "MODEL"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "context-windows"
        finally:
            await composer.close()

    asyncio.run(scenario())
