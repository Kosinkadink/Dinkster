"""CPU patch execution with actual inference patchers and small conditioning seams."""

import asyncio
import importlib
import os
from pathlib import Path
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


def test_anima_zero_strength_and_inpaint_mask_controls() -> None:
    source = _patcher(_Base())
    weights = _patcher(_Conditioning(4))
    weights.model.inpaint_masked_input = True
    image = torch.tensor([0.125, 0.5, 1.0]).expand(1, 8, 16, 3).clone()
    mask = torch.ones((1, 8, 16))
    for strength in (0.0, -2.0):
        patched = AnimaLLLiteApply.execute(
            model=source,
            model_patch=weights,
            image=image,
            mask=mask,
            strength=strength,
            start_percent=0.2,
            end_percent=0.7,
        )["MODEL"]
        patch = patched.model_options["transformer_options"]["patches"]["post_input"][0]
        options = {"sigmas": torch.tensor([5.0]), "model_patch_data": {}}
        patch({"x": torch.zeros((1, 4, 1, 1, 2)), "transformer_options": options})
        if strength == 0:
            assert options["model_patch_data"] == {}
        else:
            expected = torch.tensor([0.0, 0.0, 0.0, 1.0]).view(1, 4, 1, 1)
            assert torch.equal(options["model_patch_data"][patch], expected.expand(1, 4, 8, 16))
        assert torch.equal(image, torch.tensor([0.125, 0.5, 1.0]).expand(1, 8, 16, 3))
        assert torch.equal(mask, torch.ones((1, 8, 16)))


@pytest.mark.parametrize("in_process", (True,))
def test_anima_executes_production_pack_on_both_ordinary_hosts(in_process: bool) -> None:
    from dinkster_graph import Graph, GraphNode, Link

    from dinkster.compose import PackSpec, ServingComposer

    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        manifests = (
            root / "dinkster-pack.toml",
            root / "tests/fixtures/anima_lllite/dinkster-pack.toml",
        )
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join((str(root / "tests"), str(root / "src"))),
            }
        )
        try:
            for manifest in manifests:
                pack = PackSpec(manifest=manifest, in_process=True)
                if not in_process:
                    pack = PackSpec(
                        manifest=manifest,
                        worker_group="anima-lllite-fixture",
                        group_manifests=manifests,
                    )
                await composer.add_pack(pack)
            graph = Graph(
                nodes={
                    "source": GraphNode("anima-lllite-fixture.source", {}),
                    "patch": GraphNode(
                        "AnimaLLLiteApply",
                        {
                            "model": Link("source", "model"),
                            "model_patch": Link("source", "patch"),
                            "image": Link("source", "image"),
                            "mask": Link("source", "mask"),
                            "strength": -2.0,
                            "start_percent": 0.2,
                            "end_percent": 0.7,
                        },
                    ),
                    "inspect": GraphNode(
                        "anima-lllite-fixture.inspect",
                        {
                            "source": Link("source", "model"),
                            "patched": Link("patch", "MODEL"),
                            "weights": Link("source", "patch"),
                            "image": Link("source", "image"),
                            "mask": Link("source", "mask"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve()
                    == "anima patch values and sources preserved"
                )
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
