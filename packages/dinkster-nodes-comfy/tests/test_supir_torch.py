"""SUPIR patch math with real CPU patchers and small VAE/control seams."""

import asyncio
import importlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_nodes_comfy.supir import SUPIRApply


class Base(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.latent_format = SimpleNamespace(
            process_in=lambda value: value * 2 + 1, process_out=lambda value: (value - 1) / 2
        )


def patcher(module: torch.nn.Module) -> Any:
    initialize_inference_process()
    return importlib.import_module("dinkster_inference.model_patcher").ModelPatcher(
        module, torch.device("cpu"), torch.device("cpu")
    )


class Adapter(torch.nn.Module):
    def forward(
        self, control: torch.Tensor, value: torch.Tensor, *rest: Any, control_scale: float
    ) -> torch.Tensor:
        return value + control * control_scale


class Control(torch.nn.Module):
    def __init__(self, denoise: bool) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.project_modules = torch.nn.ModuleList([Adapter(), Adapter()])
        self.denoise_encoder_sd = {"weight": torch.tensor(5.0)} if denoise else None

    def control_model(self, hint: Any, timestep: Any, x: Any, context: Any, y: Any) -> list[Any]:
        return [hint + 2, hint - 3]


class Vae:
    def __init__(self) -> None:
        model = torch.nn.Module()
        encoder = torch.nn.Module()
        encoder.register_parameter("weight", torch.nn.Parameter(torch.ones(())))
        model.add_module("encoder", encoder)
        self.patcher = patcher(model)
        self.fail_encode = False

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        if self.fail_encode:
            raise ValueError("controlled encode failure")
        encoded = image.sum(dim=-1, keepdim=True).movedim(-1, 1) * 2
        if "encoder.weight" in self.patcher.patches:
            encoded = encoded + 5
        return encoded

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return latent.movedim(1, -1).expand(-1, -1, -1, 3)


@pytest.mark.parametrize("denoise", (False, True))
def test_supir_patch_guidance_restore_threshold_and_source_preservation(denoise: bool) -> None:
    source, weights, vae = patcher(Base()), patcher(Control(denoise)), Vae()
    original_patcher = vae.patcher
    image = torch.tensor([0.25, 0.5, 1.0, 0.9]).expand(1, 2, 3, 4).clone()
    original_image = image.clone()
    result = SUPIRApply.execute(
        model=source,
        model_patch=weights,
        vae=vae,
        image=image,
        strength_start=2.0,
        strength_end=0.5,
        restore_cfg=1.0,
        restore_cfg_s_tmin=0.25,
    )["MODEL"]
    assert result is not source
    assert not source.model_options.get("transformer_options", {}).get("patches", {})
    assert "sampler_post_cfg_function" not in source.model_options
    assert vae.patcher is original_patcher and not original_patcher.patches
    assert torch.equal(image, original_image)
    middle = result.model_options["transformer_options"]["patches"]["middle_block_after_patch"][0]
    for sigma, scale in ((0.0, 0.5), (14.6146, 2.0)):
        value = torch.full((1, 1, 2, 3), 4.0)
        actual = middle(
            {
                "h": value,
                "x": value,
                "timesteps": torch.tensor([0.0]),
                "context": None,
                "y": None,
                "transformer_options": {"sigmas": torch.tensor([sigma], dtype=torch.float64)},
            }
        )["h"]
        expected_hint = 18.0 if denoise else 8.0
        assert torch.equal(actual, torch.full_like(value, 4 + (expected_hint - 3) * scale))
    restore = result.model_options["sampler_post_cfg_function"][0]
    denoised = torch.tensor([10.0, 20.0, 30.0]).view(3, 1, 1, 1).expand(3, 1, 2, 3).clone()
    for sigma in (torch.tensor(0.25), torch.tensor([0.25])):
        assert restore({"denoised": denoised, "sigma": sigma}) is denoised
    actual = restore({"denoised": denoised, "sigma": torch.tensor([7.3073])})
    center = 103.0 if denoise else 43.0
    assert torch.equal(actual, (denoised + center) / 2)
    assert torch.equal(denoised[:, 0, 0, 0], torch.tensor([10.0, 20.0, 30.0]))


def test_supir_disabled_restore_and_encode_failure_restore_vae_patcher() -> None:
    source, weights, vae = patcher(Base()), patcher(Control(True)), Vae()
    original_patcher = vae.patcher
    image = torch.zeros((1, 2, 3, 3))
    result = SUPIRApply.execute(
        model=source,
        model_patch=weights,
        vae=vae,
        image=image,
        strength_start=0.0,
        strength_end=0.0,
        restore_cfg=0.0,
        restore_cfg_s_tmin=0.25,
    )["MODEL"]
    assert "sampler_post_cfg_function" not in result.model_options
    vae.fail_encode = True
    with pytest.raises(ValueError, match="controlled encode failure"):
        SUPIRApply.execute(
            model=source,
            model_patch=weights,
            vae=vae,
            image=image,
            strength_start=1.0,
            strength_end=1.0,
            restore_cfg=1.0,
            restore_cfg_s_tmin=0.25,
        )
    assert vae.patcher is original_patcher and not original_patcher.patches


@pytest.mark.parametrize("in_process", (True, False))
def test_supir_executes_on_both_ordinary_hosts(in_process: bool) -> None:
    from dinkster_graph import Graph, GraphNode, Link

    from dinkster.compose import PackSpec, ServingComposer

    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        manifests = (
            root / "dinkster-pack.toml",
            root / "tests/fixtures/supir/dinkster-pack.toml",
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
                        worker_group="supir-fixture",
                        group_manifests=manifests,
                    )
                await composer.add_pack(pack)
            graph = Graph(
                nodes={
                    "source": GraphNode("supir-fixture.source", {}),
                    "patch": GraphNode(
                        "SUPIRApply",
                        {
                            "model": Link("source", "model"),
                            "model_patch": Link("source", "weights"),
                            "vae": Link("source", "vae"),
                            "image": Link("source", "image"),
                            "strength_start": 2.0,
                            "strength_end": 0.5,
                            "restore_cfg": 1.0,
                            "restore_cfg_s_tmin": 0.25,
                        },
                    ),
                    "inspect": GraphNode(
                        "supir-fixture.inspect",
                        {
                            "source": Link("source", "model"),
                            "patched": Link("patch", "MODEL"),
                            "weights": Link("source", "weights"),
                            "vae": Link("source", "vae"),
                            "image": Link("source", "image"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve()
                    == "SUPIR control and restoration tensors preserved"
                )
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
