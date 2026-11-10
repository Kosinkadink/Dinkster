from __future__ import annotations

from typing import Any

import torch
from dinkster_api.v1 import (
    initialize_inference_process,
)
from dinkster_sd.gligen_conditioning import GLIGENTextBoxApply


class _Tokenizer:
    def tokenize_with_weights(self, text: str, return_word_ids: bool, **kwargs: Any) -> str:
        assert return_word_ids is False
        return text


class _TextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.options: dict[str, Any] = {}

    def reset_clip_options(self) -> None:
        self.options = {}

    def set_clip_options(self, options: dict[str, Any]) -> None:
        self.options.update(options)

    def encode_token_weights(self, tokens: str) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.options["projected_pooled"] is False
        assert self.options["execution_device"] == torch.device("cpu")
        assert tokens == "first textbox\nsecond line"
        return torch.tensor([[999.0]]), torch.tensor([[3.0, -11.0, 0.5]])


def _clip() -> Any:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher
    from dinkster_inference.sd import CLIP

    clip = CLIP(no_init=True)
    clip.cond_stage_model = _TextEncoder()
    clip.patcher = ModelPatcher(
        clip.cond_stage_model, load_device=torch.device("cpu"), offload_device=torch.device("cpu")
    )
    clip.tokenizer = _Tokenizer()
    clip.tokenizer_options = {}
    clip.layer_idx = None
    clip.load_model = lambda tokens: clip.patcher
    return clip


def _gligen() -> Any:
    initialize_inference_process()
    from dinkster_inference.gligen import Gligen
    from dinkster_inference.model_patcher import CoreModelPatcher

    return CoreModelPatcher(
        Gligen([], torch.nn.Identity(), key_dim=3), torch.device("cpu"), torch.device("cpu")
    )


def _conditioning() -> list[list[Any]]:
    return [
        [
            torch.tensor([[-5.0, 13.0]]),
            {
                "source": "prior",
                "strength": 0.3,
                "gligen": ("position", _gligen(), [(torch.tensor([[17.0]]), 5, 9, 1, 2)]),
            },
        ],
        [torch.tensor([[7.0, -2.0]]), {"source": "fresh", "strength": 0.8}],
    ]


def _apply(conditioning: Any, clip: Any, model: Any, **sizes: int) -> Any:
    return GLIGENTextBoxApply.execute(
        conditioning_to=conditioning,
        clip=clip,
        gligen_textbox_model=model,
        text="first textbox\nsecond line",
        **sizes,
    )["CONDITIONING"]


def test_gligen_textbox_uses_unprojected_pool_and_copies_each_metadata_record() -> None:
    source, clip, model = _conditioning(), _clip(), _gligen()
    result = _apply(source, clip, model, width=81, height=63, x=17, y=39)
    assert result is not source and len(result) == 2
    assert len(source[0][1]["gligen"][2]) == 1 and "gligen" not in source[1][1]
    assert source[0][1]["gligen"][1] is not model
    for index, (embedding, values) in enumerate(result):
        assert embedding is source[index][0]
        assert values is not source[index][1]
        assert values["source"] == source[index][1]["source"]
        assert values["strength"] == source[index][1]["strength"]
        kind, bound_model, positions = values["gligen"]
        assert kind == "position" and bound_model is model
        pooled, *box = positions[-1]
        assert box == [7, 10, 4, 2]
        assert torch.equal(pooled, torch.tensor([[3.0, -11.0, 0.5]]))
        assert len(positions) == (2 if index == 0 else 1)
    assert result[0][1]["gligen"][2][0] is source[0][1]["gligen"][2][0]
    assert _apply([], clip, model, width=8, height=8, x=0, y=0) == []
