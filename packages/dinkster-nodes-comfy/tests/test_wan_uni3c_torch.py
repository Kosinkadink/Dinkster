"""Uni3C patch execution with actual patchers and small control/VAE seams."""

import asyncio
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_nodes_comfy.wan_uni3c import WanUni3CControlnetApply

initialize_inference_process()
WanUni3CControlnet = importlib.import_module("dinkster_inference.ldm.wan.uni3c").WanUni3CControlnet
ModelPatcher = importlib.import_module("dinkster_inference.model_patcher").ModelPatcher


class Base(torch.nn.Module):
    def __init__(self, dim: int | None = 4) -> None:
        super().__init__()
        self.diffusion_model = SimpleNamespace(dim=dim)
        self.model_sampling = SimpleNamespace(percent_to_sigma=lambda percent: 10 * (1 - percent))
        self.latent_format = SimpleNamespace(process_in=lambda latent: latent * 2 + 1)


class Control(WanUni3CControlnet):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.num_layers = 2
        block = torch.nn.Module()
        block.norm1 = torch.nn.Module()
        block.norm1.linear = torch.nn.Linear(4, 1)
        self.controlnet_blocks = torch.nn.ModuleList([block])
        self.inputs: list[torch.Tensor] = []

    def process_input(self, control: torch.Tensor) -> tuple[torch.Tensor, None]:
        self.inputs.append(control.clone())
        return control[:, :1].flatten(2).transpose(1, 2)[:, :2], None

    def forward_block(
        self, block: int, hidden: torch.Tensor, temb: torch.Tensor, freqs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden + 2, hidden + temb[:, :1, None] + block


class Vae:
    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def temporal_compression_decode(self) -> int:
        return 2

    def spacial_compression_encode(self) -> int:
        return 2

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        self.inputs.append(frames.clone())
        return frames[::2, ::2, ::2].mean(dim=-1).unsqueeze(0).unsqueeze(0)


def patcher(module: torch.nn.Module) -> Any:
    return ModelPatcher(module, torch.device("cpu"), torch.device("cpu"))


def video(frames: int = 2) -> torch.Tensor:
    values = torch.arange(frames, dtype=torch.float32) / 2 + 0.125
    rgb = values[:, None, None, None].expand(frames, 2, 4, 3)
    return torch.cat((rgb, torch.full((frames, 2, 4, 1), 99.0)), dim=-1)


def apply(source: Any, control: Any, vae: Any, frames: Any) -> Any:
    return WanUni3CControlnetApply.execute(
        model=source,
        model_patch=control,
        vae=vae,
        render_video=frames,
        strength=-0.5,
        start_percent=0.2,
        end_percent=0.7,
    )["MODEL"]


def inspect(source: Any, patched: Any, control: Any, vae: Any, frames: Any) -> None:
    assert source is not patched
    assert source.model_options == {"transformer_options": {}}
    patch = patched.model_options["transformer_options"]["patches"]["double_block"][0]
    assert patch.models() == [control]
    x = torch.arange(32, dtype=torch.float32).reshape(4, 2, 2, 1, 2)
    original_x = x.clone()
    vec = torch.tensor([1.0, 10.0, 100.0, 1000.0]).reshape(4, 1, 1)
    for sigma in (8.0, 10 * (1 - 0.7)):
        image = torch.full((4, 4, 1), 20.0)
        common = {
            "img": image,
            "x": x,
            "vec": vec,
            "img_offset": 1,
            "transformer_options": {
                "cond_or_uncond": [0, 1],
                "sigmas": torch.tensor([sigma], dtype=torch.float64),
            },
        }
        patch({**common, "block_index": 0})
        # CFG chunks reuse the first two samples, not a separate control pass.
        expected = torch.tensor([1.0, 2.0, 18.0, 19.0]).reshape(2, 2, 1).repeat(2, 1, 1)
        torch.testing.assert_close(image[:, 1:3], 20 - expected * 0.5)
        patch({**common, "block_index": 1})
        torch.testing.assert_close(image[:, 1:3], 20 - expected - 1.5)
        assert torch.equal(image[:, (0, 3)], torch.full((4, 2, 1), 20.0))
        assert patch.temp_data is None
    encoded_input = control.model.inputs[0]
    torch.testing.assert_close(encoded_input[:, :2], x[:2])
    assert torch.count_nonzero(encoded_input[:, 2:20]) == 0
    expected_render = torch.tensor([1.25, 2.25]).reshape(1, 1, 2, 1, 1).expand(2, 1, 2, 1, 2)
    torch.testing.assert_close(encoded_input[:, 20:], expected_render)
    assert len(vae.inputs) == 1
    assert torch.equal(vae.inputs[0][2], frames[1, ..., :3])
    for sigma in (8.01, 2.99):
        value = torch.full((4, 4, 1), 20.0)
        patch(
            {
                "img": value,
                "block_index": 0,
                "transformer_options": {"sigmas": torch.tensor([sigma])},
            }
        )
        assert torch.equal(value, torch.full_like(value, 20))
    assert torch.equal(x, original_x)
    assert torch.equal(frames, video())
    patch.to(torch.device("cpu"))
    assert patch.temp_data is None


def test_uni3c_residuals_cfg_padding_endpoints_and_source_preservation() -> None:
    source, control, vae, frames = patcher(Base()), patcher(Control()), Vae(), video()
    inspect(source, apply(source, control, vae, frames), control, vae, frames)


def test_uni3c_truncates_render_and_reencodes_on_spatial_change() -> None:
    vae = Vae()
    result = apply(patcher(Base()), patcher(Control()), vae, video(5))
    patch = result.model_options["transformer_options"]["patches"]["double_block"][0]
    first = patch.build_controlnet_input(torch.zeros(1, 25, 2, 1, 2), torch.float32, 1)
    assert first.shape == (1, 21, 2, 1, 2)
    assert vae.inputs[0].shape == (3, 2, 4, 3)
    patch.build_controlnet_input(torch.zeros(1, 2, 2, 2, 4), torch.float32, 1)
    assert vae.inputs[1].shape == (3, 4, 8, 3)


@pytest.mark.parametrize("dim", (None, 3))
def test_uni3c_rejects_incompatible_model_width(dim: int | None) -> None:
    with pytest.raises(ValueError, match="Wan models|expects a Wan model with dim"):
        apply(patcher(Base(dim)), patcher(Control()), Vae(), video())


@pytest.mark.parametrize("in_process", (True, False))
def test_uni3c_runs_on_both_ordinary_hosts(in_process: bool) -> None:
    from dinkster_graph import Graph, GraphNode, Link

    from dinkster.compose import PackSpec, ServingComposer

    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        manifests = (
            root / "dinkster-pack.toml",
            root / "tests/fixtures/wan_uni3c/dinkster-pack.toml",
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
                        worker_group="wan-uni3c-fixture",
                        group_manifests=manifests,
                    )
                await composer.add_pack(pack)
            graph = Graph(
                nodes={
                    "source": GraphNode("wan-uni3c-fixture.source", {}),
                    "patch": GraphNode(
                        "WanUni3CControlnetApply",
                        {
                            "model": Link("source", "model"),
                            "model_patch": Link("source", "weights"),
                            "vae": Link("source", "vae"),
                            "render_video": Link("source", "video"),
                            "strength": -0.5,
                            "start_percent": 0.2,
                            "end_percent": 0.7,
                        },
                    ),
                    "inspect": GraphNode(
                        "wan-uni3c-fixture.inspect",
                        {
                            "model": Link("source", "model"),
                            "weights": Link("source", "weights"),
                            "vae": Link("source", "vae"),
                            "video": Link("source", "video"),
                            "patched": Link("patch", "MODEL"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve()
                    == "Uni3C residuals and source video preserved"
                )
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
