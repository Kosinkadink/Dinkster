"""CPU behavior for native standard sampler nodes."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import initialize_inference_process
from dinkster_nodes_comfy.standard_sampling import KSampler, KSamplerAdvanced


class _LatentFormat:
    latent_channels = 1
    spacial_downscale_ratio = 8
    temporal_downscale_ratio = 1
    latent_dimensions = 2

    def fix_empty_latent(self, samples: torch.Tensor) -> torch.Tensor:
        return samples


class _SamplingSpace:
    sigma_max = 1.0
    sigma_min = 0.0

    def timestep(self, sigma: float) -> float:
        return sigma

    def sigma(self, timestep: float) -> float:
        return timestep

    def noise_scaling(
        self, sigma: torch.Tensor, noise: torch.Tensor, latent: torch.Tensor, _maximum: bool
    ) -> torch.Tensor:
        return latent + sigma.reshape((-1, 1, 1, 1)) * noise

    def inverse_noise_scaling(self, _sigma: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        return samples


class _CurrentPatcher:
    def prepare_state(self, *_args: object) -> None:
        pass

    def prepare_hook_patches_current_keyframe(self, *_args: object) -> None:
        pass

    def get_free_memory(self, *_args: object) -> float:
        return 1e30

    def apply_hooks(self, *_args: object, **_kwargs: object) -> dict[str, object]:
        return {}


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.current_patcher = _CurrentPatcher()
        self.latent_format = _LatentFormat()
        self.model_sampling = _SamplingSpace()
        self.inputs: list[torch.Tensor] = []

    def get_dtype(self) -> torch.dtype:
        return torch.float32

    def memory_required(self, *_args: object, **_kwargs: object) -> int:
        return 0

    def process_latent_in(self, samples: torch.Tensor) -> torch.Tensor:
        return samples

    def process_latent_out(self, samples: torch.Tensor) -> torch.Tensor:
        return samples

    def extra_conds(self, **_kwargs: object) -> dict[str, object]:
        return {}

    def extra_conds_shapes(self, **_kwargs: object) -> dict[str, tuple[int, ...]]:
        return {}

    def scale_latent_inpaint(
        self, *, latent_image: torch.Tensor, **_kwargs: object
    ) -> torch.Tensor:
        return latent_image

    def apply_model(
        self, samples: torch.Tensor, _sigma: torch.Tensor, **_kwargs: object
    ) -> torch.Tensor:
        self.inputs.append(samples.clone())
        return samples * 0.25


def _model() -> tuple[Any, _TinyModel]:
    initialize_inference_process()
    from dinkster_inference.model_patcher import ModelPatcher

    raw = _TinyModel()
    return ModelPatcher(raw, torch.device("cpu"), torch.device("cpu")), raw


def _conditioning() -> list[list[object]]:
    return [[None, {}]]


def _latent() -> dict[str, object]:
    return {
        "samples": torch.tensor([[[[0.25, -0.5, 1.0], [2.0, -3.0, 0.75]]]], dtype=torch.float32),
        "batch_index": (2, 0),
        "noise_mask": torch.tensor([[[[1.0, 0.0, 0.5], [0.25, 0.75, 1.0]]]]),
        "downscale_ratio_spacial": 8,
        "downscale_ratio_temporal": 1,
        "preserved": "metadata",
    }


def _batch_noise(samples: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = [torch.randn((1, *samples.shape[1:]), generator=generator) for _ in range(3)]
    return torch.cat((draws[2], draws[0]))


@pytest.mark.parametrize("denoise", (1.0, 0.5, 0.0))
def test_ksampler_runs_pinned_cpu_engine_with_seed_batch_mask_and_source_preservation(
    denoise: float,
) -> None:
    model, raw = _model()
    source = _latent()
    original = source["samples"].clone()
    mask = source["noise_mask"]

    result = KSampler.execute(
        model=model,
        seed=739,
        steps=1,
        cfg=1.0,
        sampler_name="euler",
        scheduler="normal",
        positive=_conditioning(),
        negative=_conditioning(),
        latent_image=source,
        denoise=denoise,
    )["LATENT"]

    if denoise == 0:
        assert raw.inputs == []
        torch.testing.assert_close(result["samples"], original)
        return
    expected_source = original.repeat(2, 1, 1, 1)
    noise = _batch_noise(expected_source, 739)
    expected_input = (expected_source + noise * denoise) * mask + expected_source * (1.0 - mask)
    expected = expected_input * 0.25 * mask + expected_source * (1.0 - mask)
    assert len(raw.inputs) == 1
    torch.testing.assert_close(raw.inputs[0], expected_input)
    torch.testing.assert_close(result["samples"], expected)
    assert torch.equal(source["samples"], original)


@pytest.mark.parametrize("leftover,scale", (("enable", 0.375), ("disable", 0.75)))
def test_advanced_partial_interval_distinguishes_leftover_from_full_denoise(
    leftover: str, scale: float
) -> None:
    model, raw = _model()
    source = _latent()
    original = source["samples"].clone()
    result = KSamplerAdvanced.execute(
        model=model,
        add_noise="disable",
        noise_seed=918,
        steps=2,
        cfg=1.0,
        sampler_name="euler",
        scheduler="normal",
        positive=_conditioning(),
        negative=_conditioning(),
        latent_image=source,
        start_at_step=0,
        end_at_step=1,
        return_with_leftover_noise=leftover,
    )["LATENT"]
    assert len(raw.inputs) == 1
    torch.testing.assert_close(result["samples"], original * (1 - scale * source["noise_mask"]))
    assert torch.equal(source["samples"], original)
    assert result["preserved"] == "metadata"
    assert "downscale_ratio_spacial" not in result
    assert "downscale_ratio_temporal" not in result


def test_ksampler_advanced_uses_noise_and_step_boundaries() -> None:
    model, raw = _model()
    source = _latent()
    original = source["samples"].clone()

    result = KSamplerAdvanced.execute(
        model=model,
        add_noise="disable",
        noise_seed=918,
        steps=2,
        cfg=1.0,
        sampler_name="euler",
        scheduler="normal",
        positive=_conditioning(),
        negative=_conditioning(),
        latent_image=source,
        start_at_step=1,
        end_at_step=2,
        return_with_leftover_noise="enable",
    )["LATENT"]

    assert len(raw.inputs) == 1
    torch.testing.assert_close(raw.inputs[0], original)
    torch.testing.assert_close(
        result["samples"],
        original * 0.25 * source["noise_mask"] + original * (1.0 - source["noise_mask"]),
    )
    assert torch.equal(source["samples"], original)


@pytest.mark.parametrize("in_process", (True, False))
def test_standard_samplers_execute_on_both_ordinary_hosts(in_process: bool) -> None:
    from dinkster_graph import Graph, GraphNode, Link

    from dinkster.compose import PackSpec, ServingComposer

    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        manifests = (
            root / "dinkster-pack.toml",
            root / "tests/fixtures/standard_sampling/dinkster-pack.toml",
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
                        worker_group="standard-sampling-fixture",
                        group_manifests=manifests,
                    )
                await composer.add_pack(pack)
            inputs = {
                "model": Link("source", "model"),
                "steps": 1,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": Link("source", "conditioning"),
                "negative": Link("source", "conditioning"),
                "latent_image": Link("source", "latent"),
            }
            graph = Graph(
                nodes={
                    "source": GraphNode("standard-sampling-fixture.source", {}),
                    "standard": GraphNode("KSampler", {**inputs, "seed": 739, "denoise": 0.5}),
                    "advanced": GraphNode(
                        "KSamplerAdvanced",
                        {
                            **inputs,
                            "steps": 2,
                            "add_noise": "disable",
                            "noise_seed": 918,
                            "start_at_step": 0,
                            "end_at_step": 1,
                            "return_with_leftover_noise": "enable",
                        },
                    ),
                    "inspect": GraphNode(
                        "standard-sampling-fixture.inspect",
                        {
                            "source": Link("source", "latent"),
                            "standard": Link("standard", "LATENT"),
                            "advanced": Link("advanced", "LATENT"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert (
                    result.outputs["inspect"]["summary"].resolve()
                    == "sampling tensors and source metadata preserved"
                )
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
