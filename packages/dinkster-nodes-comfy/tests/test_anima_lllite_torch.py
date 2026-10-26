"""CPU patch execution with actual inference patchers and small conditioning seams."""

import importlib
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_nodes_comfy.anima_lllite import AnimaLLLiteApply


class _Base(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.model_sampling = SimpleNamespace(percent_to_sigma=lambda percent: 10 * (1 - percent))


class _Conditioning(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.cond_in_channels = channels
        self.inpaint_masked_input = False

    def encode_conditioning(self, image: torch.Tensor) -> torch.Tensor:
        return image

    def apply(
        self,
        value: torch.Tensor,
        conditioning: torch.Tensor,
        block: int,
        target: str,
        strength: float,
    ) -> torch.Tensor:
        scale = {
            "self_attn_q_proj": 2,
            "self_attn_k_proj": 3,
            "self_attn_v_proj": 5,
            "cross_attn_q_proj": 7,
            "mlp_layer1": 11,
        }[target]
        return value + conditioning[0, 0, 0, 0] * strength * scale


def _patcher(module: torch.nn.Module) -> Any:
    initialize_inference_process()
    patcher = importlib.import_module("dinkster_inference.model_patcher").ModelPatcher
    return patcher(module, torch.device("cpu"), torch.device("cpu"))


@pytest.mark.parametrize("channels,masked", ((3, False), (3, True), (4, False), (4, True)))
def test_anima_patch_executes_image_mask_attention_and_mlp_without_changing_source(
    channels: int, masked: bool
) -> None:
    source = _patcher(_Base())
    weights = _patcher(_Conditioning(channels))
    image = torch.tensor([0.125, 0.5, 1.0, 0.875]).expand(1, 8, 16, 4).clone()
    mask = torch.ones((1, 8, 16)) if masked else None
    original = image.clone()
    result = AnimaLLLiteApply.execute(
        model=source,
        model_patch=weights,
        image=image,
        mask=mask,
        strength=-2.0,
        start_percent=0.2,
        end_percent=0.7,
    )["MODEL"]
    assert result is not source
    assert source.model_options.get("transformer_options", {}).get("patches", {}) == {}
    patches = result.model_options["transformer_options"]["patches"]
    post = patches["post_input"][0]
    assert post.models() == [weights]
    q, k, v = (torch.tensor([[[value]]]) for value in (1.0, 10.0, 100.0))
    opts = {"sigmas": torch.tensor([8.0]), "model_patch_data": {}, "block_index": 4}
    post({"x": torch.zeros((1, 4, 1, 1, 2)), "transformer_options": opts})
    encoded = opts["model_patch_data"][post]
    expected = torch.tensor(
        [-0.75, 0.0, 1.0] + ([1.0 if masked else -1.0] if channels == 4 else [])
    )
    assert torch.equal(encoded, expected.view(1, channels, 1, 1).expand(1, channels, 8, 16))
    attn = patches["attn1_patch"][0](q, k, v, extra_options=opts)
    assert [attn[key].item() for key in ("q", "k", "v")] == [4.0, 14.5, 107.5]
    cross = patches["attn2_patch"][0](q, k, v, extra_options=opts)
    assert [cross[key].item() for key in ("q", "k", "v")] == [11.5, 10.0, 100.0]
    mlp = patches["mlp_patch"][0]({"x": q, "transformer_options": opts})
    assert mlp["x"].item() == 17.5
    assert torch.equal(q, torch.tensor([[[1.0]]]))
    assert torch.equal(image, original)
    if mask is not None:
        assert torch.equal(mask, torch.ones((1, 8, 16)))
    for sigma in (8.01, 2.99):
        inactive = {"sigmas": torch.tensor([sigma]), "model_patch_data": {}}
        post({"x": torch.zeros((1, 4, 1, 1, 2)), "transformer_options": inactive})
        assert inactive["model_patch_data"] == {}
