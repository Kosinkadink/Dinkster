"""CPU behavior coverage for ModelPatchLoader checkpoint dispatch."""

from __future__ import annotations

import pytest
import torch
from dinkster_models.model_patch_loader import (
    ModelPatchLoader,
    _z_image_convert,
)


def test_z_image_conversion_preserves_qkv_order_for_asymmetric_weights() -> None:
    state = {
        "x.attention.to_k.weight": torch.tensor([[10.0], [20.0]]),
        "x.attention.to_q.weight": torch.tensor([[1.0], [2.0]]),
        "x.attention.to_v.weight": torch.tensor([[100.0], [200.0]]),
        "x.attention.to_out.0.weight": torch.tensor([[7.0]]),
    }
    converted = _z_image_convert(state, torch)
    assert torch.equal(
        converted["x.attention.qkv.weight"],
        torch.tensor([[1.0], [2.0], [10.0], [20.0], [100.0], [200.0]]),
    )
    assert torch.equal(converted["x.attention.out.weight"], torch.tensor([[7.0]]))


def test_loader_rejects_unbound_asset() -> None:
    with pytest.raises(TypeError, match="AssetRef or LocalAssetRef"):
        ModelPatchLoader.execute(name=object())
