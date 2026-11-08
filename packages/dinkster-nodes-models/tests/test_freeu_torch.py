"""CPU execution and manifest transport coverage for FreeU nodes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from dinkster_api.v1 import (
    CORE_FLOAT,
    MODEL_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
)
from dinkster_models.freeu import FreeU, FreeU_V2


class _ModelConfig:
    unet_config = {"model_channels": 2}


class _InnerModel:
    model_config = _ModelConfig()


class Patcher:
    def __init__(self, patch: Any = None) -> None:
        self.model = _InnerModel()
        self.patch = patch

    def clone(self) -> Patcher:
        return Patcher(self.patch)

    def set_model_output_block_patch(self, patch: Any) -> None:
        self.patch = patch


def test_freeu_patches_matching_channel_bands_and_leaves_other_bands_unchanged() -> None:
    source = Patcher()
    patched = FreeU.execute(model=source, b1=3.0, b2=5.0, s1=0.0, s2=0.0)["MODEL"]
    assert isinstance(patched, Patcher)
    assert patched is not source
    assert source.patch is None
    hidden = torch.ones((1, 8, 2, 2))
    skip = torch.ones((1, 8, 2, 2))
    result_hidden, result_skip = patched.patch(hidden, skip, {})
    assert torch.equal(result_hidden[:, :4], torch.full((1, 4, 2, 2), 3.0))
    assert torch.equal(result_hidden[:, 4:], torch.ones((1, 4, 2, 2)))
    assert torch.equal(result_skip, torch.zeros((1, 8, 2, 2)))
    unmatched = torch.ones((1, 7, 2, 2))
    unchanged_hidden, unchanged_skip = patched.patch(unmatched, skip, {})
    assert unchanged_hidden is unmatched
    assert unchanged_skip is skip


def test_freeu_v2_normalizes_each_batch_before_scaling() -> None:
    patched = FreeU_V2.execute(model=Patcher(), b1=3.0, b2=5.0, s1=1.0, s2=1.0)["MODEL"]
    hidden = torch.tensor(
        [
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[0.0, 1.0], [2.0, 3.0]],
                [[9.0, 9.0], [9.0, 9.0]],
                [[9.0, 9.0], [9.0, 9.0]],
                [[7.0, 7.0], [7.0, 7.0]],
                [[7.0, 7.0], [7.0, 7.0]],
                [[7.0, 7.0], [7.0, 7.0]],
                [[7.0, 7.0], [7.0, 7.0]],
            ]
        ]
    )
    result, _ = patched.patch(hidden, torch.ones((1, 8, 2, 2)), {})
    assert torch.equal(result[:, :4, 0, 0], torch.tensor([[0.0, 0.0, 9.0, 9.0]]))
    assert torch.equal(result[:, :4, 1, 1], torch.tensor([[9.0, 9.0, 27.0, 27.0]]))
    assert torch.equal(result[:, 4:], torch.full((1, 4, 2, 2), 7.0))


def test_freeu_uses_cpu_fourier_fallback_for_repeated_unsupported_skips() -> None:
    class Skip:
        device = torch.device("cpu")

        def __init__(self, cpu_value: torch.Tensor) -> None:
            self.cpu_value = cpu_value

        def float(self) -> torch.Tensor:
            raise RuntimeError("fft unavailable")

        def cpu(self) -> torch.Tensor:
            return self.cpu_value

    def expected_filter(value: torch.Tensor, scale: float) -> torch.Tensor:
        frequency = torch.fft.fftshift(torch.fft.fftn(value.float(), dim=(-2, -1)), dim=(-2, -1))
        mask = torch.ones_like(frequency)
        height, width = frequency.shape[-2:]
        row, column = height // 2, width // 2
        mask[..., row - 1 : row + 1, column - 1 : column + 1] = scale
        return torch.fft.ifftn(
            torch.fft.ifftshift(frequency * mask, dim=(-2, -1)), dim=(-2, -1)
        ).real

    source = Patcher()
    patched = FreeU.execute(model=source, b1=2.0, b2=3.0, s1=0.25, s2=0.5)["MODEL"]
    cpu_skip = torch.tensor(
        [[[[0.0, 1.0, 2.0], [3.0, 5.0, 8.0], [13.0, 21.0, 34.0]]]], dtype=torch.float32
    )
    expected_skip = expected_filter(cpu_skip, 0.25)
    expected_hidden = torch.cat((torch.full((1, 4, 3, 3), 2.0), torch.ones((1, 4, 3, 3))), dim=1)
    for _ in range(2):
        hidden, result_skip = patched.patch(torch.ones((1, 8, 3, 3)), Skip(cpu_skip), {})
        assert torch.equal(hidden, expected_hidden)
        assert torch.allclose(result_skip, expected_skip)
    assert source.patch is None


class ModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="freeu-fixture.model",
            inputs=(),
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> Mapping[str, object]:
        return cls.outputs(MODEL=Patcher())


class PatchedModelResult(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="freeu-fixture.result",
            inputs=(InputSpec("model", TypeExpr.concrete(MODEL_TYPE)),),
            outputs=(OutputSpec("value", TypeExpr.concrete(CORE_FLOAT)),),
        )

    @classmethod
    def execute(cls, *, model: Patcher) -> Mapping[str, object]:
        hidden, _ = model.patch(torch.ones((1, 8, 2, 2)), torch.ones((1, 8, 2, 2)), {})
        return cls.outputs(value=float(hidden[0, 0, 0, 0]))
