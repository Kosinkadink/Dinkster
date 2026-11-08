from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    initialize_inference_process,
)
from dinkster_flux.image_control_patches import USOStyleReference


class _Sampling:
    def percent_to_sigma(self, value: float) -> float:
        return 10.0 - value


class _Diffusion(torch.nn.Module):
    def __init__(self, blocks: int = 4) -> None:
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList(torch.nn.Identity() for _ in range(blocks))


class _Host(torch.nn.Module):
    def __init__(self, blocks: int = 4) -> None:
        super().__init__()
        self.model_sampling = _Sampling()
        self.diffusion_model = _Diffusion(blocks)


def _patcher(model: torch.nn.Module) -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    return ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))


class _Projector(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features.sum(dim=0)


def _installed(patcher: Any, name: str) -> Any:
    return patcher.model_options["transformer_options"]["patches"][name][0]


def test_uso_reference_initializes_runtime_and_preserves_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dinkster_flux.image_control_patches as image_controls

    source = _patcher(_Host())
    projector = _patcher(_Projector())
    calls: list[None] = []
    monkeypatch.setattr(image_controls, "initialize_inference_process", lambda: calls.append(None))
    vision = SimpleNamespace(
        all_hidden_states=torch.arange(1 * 21 * 2 * 3, dtype=torch.float32).reshape(1, 21, 2, 3),
        penultimate_hidden_states=torch.full((1, 2, 3), 99.0),
    )
    patched = USOStyleReference.execute(
        model=source, model_patch=projector, clip_vision_output=vision
    )["MODEL"]
    patch = _installed(patched, "post_input")
    text, ids = torch.tensor([[[7.0, 8.0, 9.0]]]), torch.ones(1, 1, 3)
    output = patch({"txt": text, "txt_ids": ids})
    expected = (
        vision.all_hidden_states[:, -20]
        + vision.all_hidden_states[:, -11]
        + vision.penultimate_hidden_states
    )
    assert source.model_options == {"transformer_options": {}}
    assert output["txt"].shape == (1, 3, 3)
    assert torch.equal(output["txt"][:, :2], expected)
    assert torch.equal(output["txt"][:, 2:], text)
    assert torch.equal(output["txt_ids"][:, :2], torch.zeros(1, 2, 3))
    assert torch.equal(output["txt_ids"][:, 2:], ids)
    assert patch.models() == [projector]
    assert patch.to(torch.device("cpu")) is patch
    assert calls
