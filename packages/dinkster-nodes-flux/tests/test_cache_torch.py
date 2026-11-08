from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from dinkster_flux.cache import FluxKVCache


class FluxModel:
    def __init__(self, params: bool = True) -> None:
        self.model = SimpleNamespace(diffusion_model=SimpleNamespace())
        if params:
            self.model.diffusion_model.params = SimpleNamespace()
        self.attention: Any = None
        self.post_input: Any = None
        self.objects: dict[str, object] = {}

    def clone(self) -> FluxModel:
        return FluxModel(hasattr(self.model.diffusion_model, "params"))

    def set_model_attn1_patch(self, callback: Any) -> None:
        self.attention = callback

    def set_model_post_input_patch(self, callback: Any) -> None:
        self.post_input = callback

    def add_object_patch(self, path: str, value: object) -> None:
        self.objects[path] = value


def test_flux_reference_cache_keeps_blocks_independent_and_repeats_changed_batch() -> None:
    q = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)
    k, v = q + 10, q + 30
    image = q.reshape(1, 4, 2)
    for params in (True, False):
        source = FluxModel(params)
        patched = FluxKVCache.execute(model=source)["MODEL"]
        path = "diffusion_model." + ("params." if params else "") + "default_ref_method"
        assert patched.objects == {path: "index_timestep_zero"}
        options = {"reference_image_num_tokens": [1, 1], "block_type": "double", "block_index": 3}
        assert patched.attention(q, k, v, {}) == {}
        assert patched.post_input({"img": image, "transformer_options": options})["img"] is image
        first = patched.attention(q, k, v, options)
        assert torch.equal(first["k"], k) and torch.equal(first["v"], v)
        assert torch.equal(
            patched.post_input({"img": image, "transformer_options": options})["img"],
            image[:, :-2],
        )
        next_q = q[:, :, :1].repeat(3, 1, 1, 1)
        next_k, next_v = next_q + 100, next_q + 200
        second = patched.attention(next_q, next_k, next_v, options)
        assert torch.equal(second["k"], torch.cat((next_k, k[:, :, -2:].repeat(3, 1, 1, 1)), 2))
        assert torch.equal(second["v"], torch.cat((next_v, v[:, :, -2:].repeat(3, 1, 1, 1)), 2))
        other = patched.attention(next_q, next_k, next_v, {**options, "block_index": 4})
        assert torch.equal(other["k"], next_k)
        patched.attention.cleanup()
        reset = patched.attention(next_q, next_k, next_v, options)
        assert torch.equal(reset["k"], next_k)
        assert source.attention is None and source.objects == {}
