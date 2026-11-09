"""CPU behavior for native ComfyUI sigma schedule operations."""

from __future__ import annotations

import pytest
import torch
from dinkster_api.v1 import (
    CORE_FLOAT,
    MODEL_TYPE,
    SIGMAS_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    materialize_latent,
)
from dinkster_sampling.sigma_operations import (
    ExtendIntermediateSigmas,
    FlipSigmas,
    ManualSigmas,
    SamplingPercentToSigma,
    SetFirstSigma,
)


def test_flip_empty_and_set_first_clone_behavior() -> None:
    empty = torch.tensor([])
    assert FlipSigmas.execute(sigmas=empty)["sigmas"] is empty
    original = torch.tensor([8.0, 2.0, 0.0])
    flipped = FlipSigmas.execute(sigmas=original)["sigmas"]
    assert torch.equal(flipped, torch.tensor([0.0001, 2.0, 8.0]))
    assert torch.equal(original, torch.tensor([8.0, 2.0, 0.0]))
    replaced = SetFirstSigma.execute(sigmas=original, sigma=11.5)["sigmas"]
    assert torch.equal(replaced, torch.tensor([11.5, 2.0, 0.0]))
    assert replaced.data_ptr() != original.data_ptr()
    assert torch.equal(original, torch.tensor([8.0, 2.0, 0.0]))


@pytest.mark.parametrize(
    ("spacing", "expected"),
    (
        ("linear", (20.0, 10.0, 8.0, 6.0, 4.0, 2.6666667, 1.3333333, 0.0)),
        (
            "cosine",
            (20.0, 10.0, 7.0, 4.8038478, 4.0, 2.0, 0.5358984, 0.0),
        ),
        ("sine", (20.0, 10.0, 9.1961524, 7.0, 4.0, 3.4641016, 2.0, 0.0)),
    ),
)
def test_extend_intermediate_sigmas_uses_asymmetric_float32_interpolation(
    spacing: str, expected: tuple[float, ...]
) -> None:
    result = ExtendIntermediateSigmas.execute(
        sigmas=torch.tensor([20.0, 10.0, 4.0, 0.0], dtype=torch.float64),
        steps=3,
        start_at_sigma=15.0,
        end_at_sigma=3.0,
        spacing=spacing,
    )["sigmas"]
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert torch.allclose(result, torch.tensor(expected, dtype=torch.float32), atol=1e-6)


def test_extend_intermediate_sigmas_keeps_empty_and_unmatched_schedules() -> None:
    assert torch.equal(
        ExtendIntermediateSigmas.execute(
            sigmas=torch.tensor([]),
            steps=2,
            start_at_sigma=-1.0,
            end_at_sigma=12.0,
            spacing="linear",
        )["sigmas"],
        torch.tensor([], dtype=torch.float32),
    )
    source = torch.tensor([20.0, 10.0, 0.0])
    result = ExtendIntermediateSigmas.execute(
        sigmas=source, steps=4, start_at_sigma=9.0, end_at_sigma=1.0, spacing="linear"
    )["sigmas"]
    assert torch.equal(result, source)
    assert result.dtype == torch.float32
    assert torch.equal(source, torch.tensor([20.0, 10.0, 0.0]))


def test_manual_sigmas_matches_reference_number_extraction_and_float32_output() -> None:
    result = ManualSigmas.execute(sigmas="sigma=-1.5, +.25; 2.0 and .75")["sigmas"]
    assert result.dtype == torch.float32
    assert torch.equal(result, torch.tensor([-1.5, 0.25, 2.0, 0.75], dtype=torch.float32))


class _Sampling:
    sigma_min = torch.tensor(0.125)
    sigma_max = torch.tensor(14.0)

    @staticmethod
    def percent_to_sigma(percent: float) -> float:
        return 10.0 - percent


class _Model:
    @staticmethod
    def get_model_object(name: str) -> _Sampling:
        assert name == "model_sampling"
        return _Sampling()


def test_sampling_percent_to_sigma_keeps_interval_and_actual_endpoint_values() -> None:
    model = _Model()
    assert (
        SamplingPercentToSigma.execute(
            model=model, sampling_percent=0.0, return_actual_sigma=False
        )["sigma_value"]
        == 10.0
    )
    assert (
        SamplingPercentToSigma.execute(model=model, sampling_percent=0.0, return_actual_sigma=True)[
            "sigma_value"
        ]
        == 14.0
    )
    assert (
        SamplingPercentToSigma.execute(model=model, sampling_percent=1.0, return_actual_sigma=True)[
            "sigma_value"
        ]
        == 0.125
    )


class SigmaInspect(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sigma-fixture.inspect",
            inputs=(
                InputSpec("source", TypeExpr.concrete(SIGMAS_TYPE)),
                InputSpec("set_first", TypeExpr.concrete(SIGMAS_TYPE)),
                InputSpec("extended", TypeExpr.concrete(SIGMAS_TYPE)),
                InputSpec("flipped", TypeExpr.concrete(SIGMAS_TYPE)),
                InputSpec("high", TypeExpr.concrete(SIGMAS_TYPE)),
                InputSpec("low", TypeExpr.concrete(SIGMAS_TYPE)),
                InputSpec("denoise_high", TypeExpr.concrete(SIGMAS_TYPE)),
                InputSpec("denoise_low", TypeExpr.concrete(SIGMAS_TYPE)),
                InputSpec("endpoint", TypeExpr.concrete(CORE_FLOAT)),
            ),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(
        cls,
        *,
        source: object,
        set_first: object,
        extended: object,
        flipped: object,
        high: object,
        low: object,
        denoise_high: object,
        denoise_low: object,
        endpoint: float,
    ) -> dict[str, object]:
        values = {
            name: materialize_latent(value)
            for name, value in {
                "source": source,
                "set_first": set_first,
                "extended": extended,
                "flipped": flipped,
                "high": high,
                "low": low,
                "denoise_high": denoise_high,
                "denoise_low": denoise_low,
            }.items()
        }
        expected = {
            "source": [7.0, 4.0, 1.0, 0.0],
            "set_first": [9.0, 4.0, 1.0, 0.0],
            "extended": [9.0, 6.5, 4.0, 2.5, 1.0, 0.5, 0.0],
            "flipped": [0.0001, 0.5, 1.0, 2.5, 4.0, 6.5, 9.0],
            "high": [0.0001, 0.5, 1.0, 2.5],
            "low": [2.5, 4.0, 6.5, 9.0],
            "denoise_high": [7.0, 4.0],
            "denoise_low": [4.0, 1.0, 0.0],
        }
        for name, tensor in values.items():
            assert tensor.dtype == torch.float32
            assert torch.allclose(tensor, torch.tensor(expected[name], dtype=torch.float32))
        assert endpoint == 14.0
        return cls.outputs(value=float(values["high"][0] + values["low"][0]))


class SigmaModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sigma-fixture.model",
            outputs=(OutputSpec("model", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(model=_Model())
