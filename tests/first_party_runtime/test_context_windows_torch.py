"""CPU behavior for native context-window nodes."""

from __future__ import annotations

from collections.abc import Mapping
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
from dinkster_ltx.context_windows import LTXVContextWindows
from dinkster_models.context_windows import ContextWindowsManual
from dinkster_wan.context_windows import WanContextWindowsManual


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
