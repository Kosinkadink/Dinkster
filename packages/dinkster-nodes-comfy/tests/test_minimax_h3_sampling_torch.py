"""CPU MiniMax H3 shifts through actual model patchers and ordinary hosts."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
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
from dinkster_nodes_comfy.minimax_h3_sampling import MiniMaxH3SigmaShift

from dinkster.compose import PackSpec, ServingComposer


def _model() -> Any:
    initialize_inference_process()
    from dinkster_inference import model_sampling
    from dinkster_inference.model_patcher import ModelPatcher

    class OriginalSampling(model_sampling.ModelSamplingAV, model_sampling.CONST):
        pass

    model = torch.nn.Linear(2, 3)
    model.model_config = SimpleNamespace(sampling_settings={})
    model.model_sampling = OriginalSampling(model.model_config)
    model.model_sampling.set_parameters(shift=2.0, audio_shift=4.0)
    model.model_sampling.set_noise_scale(0.37)
    patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
    patcher.model_options["transformer_options"]["preserved"] = "source"
    return patcher


def _assert_shifted(model: Any) -> None:
    sampling = model.get_model_object("model_sampling")
    assert sampling.shift == 23.5
    assert sampling.audio_shift == 1.75
    assert sampling.audio_scale == 23.5 / 1.75
    assert sampling.noise_scale == 0.37
    assert model.model_options["transformer_options"] == {
        "preserved": "source",
        "minimax_h3_sigma_shift_video": 23.5,
        "minimax_h3_sigma_shift_audio": 1.75,
    }
    time = torch.tensor([0.125, 0.75], dtype=torch.float64)
    expected = 23.5 * time / (1 + (23.5 - 1) * time)
    torch.testing.assert_close(sampling.sigma(time * 1000), expected, rtol=0, atol=0)


def test_minimax_h3_shifts_clone_and_preserve_noise_and_source() -> None:
    source = _model()
    original = source.get_model_object("model_sampling")
    result = MiniMaxH3SigmaShift.execute(model=source, shift_video=23.5, shift_audio=1.75)["MODEL"]
    assert result is not source
    assert result.model is source.model
    _assert_shifted(result)
    assert source.get_model_object("model_sampling") is original
    assert original.shift == 2.0 and original.audio_shift == 4.0
    assert source.model_options["transformer_options"] == {"preserved": "source"}


class ShiftSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="minimax-shift-fixture.source",
            outputs=(OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(model=_model())


class ShiftConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="minimax-shift-fixture.consumer",
            inputs=(
                InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),
                InputSpec("original", TypeExpr.concrete(MODEL_TYPE)),
            ),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, *, model: Any, original: Any) -> dict[str, object]:
        _assert_shifted(model)
        assert model is not original
        assert original.get_model_object("model_sampling").shift == 2.0
        assert original.model_options["transformer_options"] == {"preserved": "source"}
        return cls.outputs(summary="23.5/1.75/0.37")


@pytest.mark.parametrize("in_process", (True, False))
def test_minimax_h3_shifts_twice_with_native_manifest_in_both_hosts(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        module = f"minimax_shift_fixture_{int(in_process)}"
        (tmp_path / f"{module}.py").write_text(
            "from test_minimax_h3_sampling_torch import ShiftSource, ShiftConsumer\n"
            "from dinkster_nodes_comfy import register_inference_types\n"
            "NODES = (ShiftSource, ShiftConsumer)\n",
            encoding="ascii",
        )
        fixture = tmp_path / "fixture-dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="minimax-shift-fixture"\npublisher="dinkster"\n'
            'namespaces=["minimax-shift-fixture"]\n[pack.entry]\n'
            f'nodes="{module}:NODES"\ntypes="{module}:register_inference_types"\n',
            encoding="ascii",
        )
        native = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
        composer = ServingComposer(
            worker_env={
                "PYTHONPATH": str(Path(__file__).parent),
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
            }
        )
        try:
            for manifest in (native, fixture):
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "minimax-shift-fixture",
                        group_manifests=() if in_process else (native, fixture),
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("minimax-shift-fixture.source"),
                    "shift": GraphNode(
                        "MiniMaxH3SigmaShift",
                        {
                            "model": Link("source", "model"),
                            "shift_video": 23.5,
                            "shift_audio": 1.75,
                        },
                    ),
                    "consumer": GraphNode(
                        "minimax-shift-fixture.consumer",
                        {
                            "model": Link("shift", "MODEL"),
                            "original": Link("source", "model"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for _ in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["summary"].resolve() == "23.5/1.75/0.37"
        finally:
            await composer.close()

    asyncio.run(scenario())
