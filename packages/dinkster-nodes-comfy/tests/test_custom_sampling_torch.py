"""CPU behavior for native custom sampler nodes."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import MultiStreamLatent
from dinkster_nodes_comfy import custom_sampling
from dinkster_nodes_comfy.noise import DisableNoise, RandomNoise


class _Sample:
    def __init__(self) -> None:
        self.noise_args: tuple[object, ...] | None = None

    def fix_empty_latent_channels(
        self, _model: object, samples: torch.Tensor, *_ratios: object
    ) -> torch.Tensor:
        return samples + 1

    def prepare_noise(self, samples: torch.Tensor, seed: int, batch_index: object) -> torch.Tensor:
        self.noise_args = (seed, batch_index)
        return samples * 0 + torch.tensor([[[[2.0, -3.0]]], [[[5.0, 7.0]]]])

    def prepare_empty_noise(self, samples: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(samples)

    def sample_custom(
        self,
        model: object,
        noise: torch.Tensor,
        cfg: float,
        sampler: object,
        sigmas: torch.Tensor,
        positive: object,
        negative: object,
        samples: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        del model, sampler, sigmas, positive, negative
        callback = kwargs["callback"]
        assert callable(callback)
        callback(0, samples - noise, samples, 2)
        assert cfg == 2.5 and kwargs["noise_mask"] is not None and kwargs["seed"] == 19
        return samples + noise


class _Model:
    def __init__(self) -> None:
        self.model = SimpleNamespace(process_latent_out=lambda tensor: tensor * 10)


class _LatentFormat:
    latent_channels = 1
    latent_dimensions = 2
    spacial_downscale_ratio = 8
    temporal_downscale_ratio = 1

    def fix_empty_latent(self, samples: torch.Tensor) -> torch.Tensor:
        return samples


class _ModelSampling:
    sigma_max = 1.0

    def noise_scaling(
        self, sigma: torch.Tensor, noise: torch.Tensor, latent: torch.Tensor, _maximum: bool
    ) -> torch.Tensor:
        return latent + noise * sigma

    def inverse_noise_scaling(self, _sigma: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        return samples


class _RealModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.latent_format = _LatentFormat()
        self.model_sampling = _ModelSampling()

    def get_dtype(self) -> torch.dtype:
        return torch.float32

    def get_model_object(self, name: str) -> object:
        return getattr(self, name)

    def memory_required(self, *_args: object, **_kwargs: object) -> int:
        return 0

    def extra_conds(self, **_kwargs: object) -> dict[str, object]:
        return {}

    def scale_latent_inpaint(self, **values: torch.Tensor) -> torch.Tensor:
        return values["latent_image"]

    def process_latent_in(self, samples: torch.Tensor) -> torch.Tensor:
        return samples

    def process_latent_out(self, samples: torch.Tensor) -> torch.Tensor:
        return samples * 3

    def apply_model(
        self, samples: torch.Tensor, _sigma: torch.Tensor, **_kwargs: object
    ) -> torch.Tensor:
        return torch.tensor([[[[3.0, -7.0]]], [[[11.0, 5.0]]]], dtype=samples.dtype)


def test_sampler_custom_uses_actual_cpu_cfg_guider_and_sampler() -> None:
    from dinkster_api.v1 import initialize_inference_process

    initialize_inference_process()
    from dinkster_inference import samplers
    from dinkster_inference.model_patcher import ModelPatcher

    patcher = ModelPatcher(_RealModel(), torch.device("cpu"), torch.device("cpu"))
    patcher.model.current_patcher = patcher
    output = custom_sampling.SamplerCustom.execute(
        model=patcher,
        add_noise=False,
        noise_seed=99,
        cfg=1.0,
        positive=[],
        negative=[],
        sampler=samplers.ksampler("euler"),
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image={
            "samples": torch.tensor([[[[2.0, -4.0]]], [[[8.0, 1.0]]]]),
            "batch_index": [6, 2],
            "noise_mask": torch.tensor([[[[1.0, 0.0]]]]),
            "origin": "real-cpu",
        },
    )
    assert output["output"]["batch_index"] == [6, 2]
    assert output["output"]["origin"] == "real-cpu"
    expected_masked = torch.tensor([[[[0.0, -12.0]]], [[[0.0, 3.0]]]])
    torch.testing.assert_close(output["output"]["samples"], expected_masked)
    torch.testing.assert_close(output["denoised_output"]["samples"], expected_masked)


def test_sampler_custom_preserves_source_metadata_and_returns_callback_denoised_tensor(
    monkeypatch: Any,
) -> None:
    sample = _Sample()
    monkeypatch.setattr(custom_sampling, "_runtime", lambda: sample)
    latent = {
        "samples": torch.tensor([[[[1.0, -4.0]]], [[[3.0, 8.0]]]]),
        "batch_index": [9, 2],
        "noise_mask": torch.tensor([[[[0.0, 1.0]]]]),
        "downscale_ratio_spacial": 8,
        "downscale_ratio_temporal": 2,
        "source": "kept",
    }
    output = custom_sampling.SamplerCustom.execute(
        model=_Model(),
        add_noise=True,
        noise_seed=19,
        cfg=2.5,
        positive=object(),
        negative=object(),
        sampler=object(),
        sigmas=torch.tensor([3.0, 1.0, 0.0]),
        latent_image=latent,
    )
    assert sample.noise_args == (19, [9, 2])
    assert output["output"]["source"] == "kept"
    assert output["output"]["batch_index"] == [9, 2]
    assert "downscale_ratio_spacial" not in output["output"]
    torch.testing.assert_close(
        output["output"]["samples"], torch.tensor([[[[4.0, -6.0]]], [[[9.0, 16.0]]]])
    )
    torch.testing.assert_close(
        output["denoised_output"]["samples"], torch.tensor([[[[0.0, 0.0]]], [[[-10.0, 20.0]]]])
    )


def test_advanced_sampler_uses_the_guider_and_preserves_batch_metadata(monkeypatch: Any) -> None:
    sample = _Sample()
    monkeypatch.setattr(custom_sampling, "_runtime", lambda: sample)

    class Noise:
        seed = 41

        def generate_noise(self, latent: dict[str, object]) -> torch.Tensor:
            assert latent["batch_index"] == [4, 1]
            return torch.tensor([[[[4.0, -2.0]]], [[[3.0, 9.0]]]])

    class Guider:
        model_patcher = _Model()

        def sample(
            self,
            noise: torch.Tensor,
            samples: torch.Tensor,
            sampler: object,
            sigmas: torch.Tensor,
            **kwargs: object,
        ) -> torch.Tensor:
            del sampler, sigmas
            assert kwargs["denoise_mask"] is not None and kwargs["seed"] == 41
            kwargs["callback"](0, samples - noise, samples, 1)
            return samples + noise * 2

    output = custom_sampling.SamplerCustomAdvanced.execute(
        noise=Noise(),
        guider=Guider(),
        sampler=object(),
        sigmas=torch.tensor([2.0, 0.0]),
        latent_image={
            "samples": torch.tensor([[[[2.0, 5.0]]], [[[7.0, -1.0]]]]),
            "batch_index": [4, 1],
            "noise_mask": torch.ones((1, 1, 1, 2)),
            "label": "source",
        },
    )
    assert output["output"]["batch_index"] == [4, 1]
    assert output["output"]["label"] == "source"
    torch.testing.assert_close(
        output["output"]["samples"], torch.tensor([[[[11.0, 2.0]]], [[[14.0, 18.0]]]])
    )
    torch.testing.assert_close(
        output["denoised_output"]["samples"], torch.tensor([[[[-10.0, 80.0]]], [[[50.0, -90.0]]]])
    )


@pytest.mark.parametrize("noise_node", (RandomNoise, DisableNoise))
def test_advanced_sampler_restores_asymmetric_av_streams_after_pinned_sampling(
    monkeypatch: Any, noise_node: type[Any]
) -> None:
    class Sample:
        def fix_empty_latent_channels(
            self, _model: object, samples: object, *_ratios: object
        ) -> object:
            assert getattr(samples, "is_nested", False)
            return samples

    monkeypatch.setattr(custom_sampling, "_runtime", lambda: Sample())
    video = torch.arange(12, dtype=torch.float64).reshape(1, 3, 2, 2, 1)
    audio = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
    samples = MultiStreamLatent.from_pairs((("video", video), ("audio", audio)))
    masks = MultiStreamLatent.from_pairs(
        (("video", torch.ones_like(video)), ("audio", torch.zeros_like(audio)))
    )

    class Guider:
        model_patcher = _Model()

        def sample(self, noise: object, latent: object, *_args: object, **kwargs: object) -> object:
            assert getattr(noise, "is_nested", False)
            assert getattr(latent, "is_nested", False)
            assert getattr(kwargs["denoise_mask"], "is_nested", False)
            kwargs["callback"](0, latent, latent, 1)
            return latent

    noise = (
        noise_node.execute(noise_seed=19)["NOISE"]
        if noise_node is RandomNoise
        else noise_node.execute()["NOISE"]
    )
    output = custom_sampling.SamplerCustomAdvanced.execute(
        noise=noise,
        guider=Guider(),
        sampler=object(),
        sigmas=torch.tensor([1.0, 0.0]),
        latent_image={"samples": samples, "noise_mask": masks, "batch_index": [3]},
    )
    for name in ("output", "denoised_output"):
        result = output[name]["samples"]
        assert type(result) is MultiStreamLatent
        assert result.roles == ("video", "audio")
        assert result.by_role("video").shape == video.shape
        assert result.by_role("video").dtype == video.dtype
        assert result.by_role("audio").shape == audio.shape
        assert result.by_role("audio").dtype == audio.dtype


@pytest.mark.parametrize("in_process", (True, False))
@pytest.mark.parametrize("mode", ("noise", "empty", "advanced"))
def test_custom_sampling_on_ordinary_hosts(mode: str, in_process: bool) -> None:
    from dinkster_graph import Graph, GraphNode, Link

    from dinkster.compose import PackSpec, ServingComposer

    async def scenario() -> None:
        root = Path(__file__).resolve().parents[1]
        manifests = (
            root / "dinkster-pack.toml",
            root / "tests/fixtures/custom_sampling/dinkster-pack.toml",
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
                        worker_group="custom-sampling-fixture",
                        group_manifests=manifests,
                    )
                await composer.add_pack(pack)
            inputs: dict[str, Any] = {
                "sampler": Link("source", "sampler"),
                "sigmas": Link("source", "sigmas"),
                "latent_image": Link("source", "latent"),
            }
            if mode == "advanced":
                inputs.update(noise=Link("source", "noise"), guider=Link("source", "guider"))
            else:
                inputs.update(
                    model=Link("source", "model"),
                    positive=Link("source", "positive"),
                    negative=Link("source", "negative"),
                    cfg=2.0,
                    noise_seed=41,
                    add_noise=mode == "noise",
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("custom-sampling-fixture.source", {}),
                    "sample": GraphNode(
                        "SamplerCustomAdvanced" if mode == "advanced" else "SamplerCustom",
                        inputs,
                    ),
                    "inspect": GraphNode(
                        "custom-sampling-fixture.consumer",
                        {
                            "output": Link("sample", "output"),
                            "denoised_output": Link("sample", "denoised_output"),
                            "source": Link("source", "latent"),
                            "add_noise": mode != "empty",
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _: None)
            for attempt in range(2):
                result = await engine.run(graph, ["inspect"])
                assert result.outputs["inspect"]["sampled"].resolve() is True
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
