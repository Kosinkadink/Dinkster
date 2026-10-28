"""CPU behavior for the native VOID DDIM sampler."""

from __future__ import annotations

import asyncio
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_BOOLEAN,
    GUIDER_TYPE,
    LATENT_TYPE,
    NOISE_TYPE,
    SAMPLER_TYPE,
    SIGMAS_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
    materialize_latent,
)
from dinkster_api.v1 import register_inference_types as register_inference_types
from dinkster_nodes_comfy.void_sampler import VOIDSampler


def test_void_sampler_schema_is_torch_free() -> None:
    schema = VOIDSampler.define_schema()

    assert schema.node_type == "VOIDSampler"
    assert schema.category == "model/sampling/samplers"
    assert schema.inputs == ()
    assert [(output.id, output.type) for output in schema.outputs] == [
        ("SAMPLER", TypeExpr.concrete(SAMPLER_TYPE))
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dinkster_nodes_comfy.void_sampler import VOIDSampler; "
            "VOIDSampler.schema(); assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("schedule_dtype", "positive_result"),
    (("float32", 0.29626840353012085), ("float64", 0.2962684631347656)),
)
def test_void_sampler_runs_alpha_space_ddim_through_public_sampler_boundary(
    schedule_dtype: str, positive_result: float
) -> None:
    import torch

    sampler = VOIDSampler.execute()["SAMPLER"]
    from dinkster_inference.samplers import Sampler

    assert isinstance(sampler, Sampler)

    noise = torch.tensor([[[[2.0, -3.0]]]], dtype=torch.float64)
    original_noise = noise.clone()
    sigmas = torch.tensor((2.0, 1.0, 0.25, 0.0), dtype=getattr(torch, schedule_dtype))
    model_options = {"marker": "preserved"}
    calls: list[tuple[torch.Tensor, torch.Tensor, object, object]] = []
    callbacks: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []

    def model(
        x: torch.Tensor, sigma: torch.Tensor, *, model_options: object, seed: object
    ) -> torch.Tensor:
        calls.append((x.clone(), sigma.clone(), model_options, seed))
        return x * 0.25 + sigma.reshape((-1, 1, 1, 1)) * 0.125

    def callback(index: int, denoised: torch.Tensor, x: torch.Tensor, total: int) -> None:
        callbacks.append((index, denoised.clone(), x.clone(), total))

    result = sampler.sample(
        model,
        sigmas,
        {"model_options": model_options, "seed": 947},
        callback,
        noise,
        latent_image=torch.full_like(noise, 17.0),
        denoise_mask=torch.ones_like(noise),
        disable_pbar=True,
    )

    # Hand-derived from F bc5f880dbdd43e426077f42ccec53bb577ffee9f,
    # nodes_void.py:411-445. Eliminating epsilon gives an affine update:
    # x1 = ((sqrt(5)/2 + 1/8)*x0 + 1/8)/sqrt(2),
    # x2 = ((sqrt(2) + 3/4)*x1 + 3/8)/sqrt(17), x3 = x2/4 + 1/32.
    # Independently round scalar operations via struct.pack, using alpha=1/5,1/2,16/17.
    # Float32 torch.pow roots are 0.44721361994743347,0.7071067690849304,0.9701425433158875;
    # float64 scalars use double roots before conversion. No second sampler is the oracle.
    expected = torch.tensor([positive_result, -0.2804364860057831]).reshape_as(noise)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.dtype is torch.float32
    assert torch.equal(noise, original_noise)
    assert [(index, total) for index, _, _, total in callbacks] == [(0, 3), (1, 3), (2, 3)]
    for index, denoised, x, _total in callbacks:
        torch.testing.assert_close(x, calls[index][0], rtol=0, atol=0)
        torch.testing.assert_close(
            denoised,
            calls[index][0] * 0.25 + calls[index][1].reshape((-1, 1, 1, 1)) * 0.125,
            rtol=0,
            atol=0,
        )
    assert [tuple(sigma.tolist()) for _, sigma, _, _ in calls] == [(2.0,), (1.0,), (0.25,)]
    assert [(options, seed) for _, _, options, seed in calls] == [(model_options, 947)] * 3

    ignored_result = sampler.sample(
        model,
        sigmas,
        {"model_options": model_options, "seed": 947},
        None,
        original_noise,
        latent_image=torch.full_like(noise, -91.0),
        denoise_mask=torch.zeros_like(noise),
        disable_pbar=True,
    )
    torch.testing.assert_close(ignored_result, result, rtol=0, atol=0)


def test_void_sampler_returns_float32_noise_without_steps() -> None:
    import torch

    sampler = VOIDSampler.execute()["SAMPLER"]
    noise = torch.tensor([[[[1.25, -4.5]]]], dtype=torch.float64)
    original_noise = noise.clone()
    calls: list[Any] = []

    def model(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append((args, kwargs))
        raise AssertionError("the no-step sampler must not call the model")

    result = sampler.sample(
        model,
        torch.tensor((0.0,), dtype=torch.float32),
        {},
        lambda *args: calls.append(args),
        noise,
        latent_image=torch.ones_like(noise),
        denoise_mask=torch.ones_like(noise),
        disable_pbar=True,
    )

    torch.testing.assert_close(result, noise.to(torch.float32), rtol=0, atol=0)
    assert result.dtype is torch.float32
    assert torch.equal(noise, original_noise)
    assert calls == []


class _VoidModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from test_custom_sampling_torch import _LatentFormat, _ModelSampling

        self.weight = torch.nn.Parameter(torch.ones(()))
        self.latent_format = _LatentFormat()
        self.model_sampling = _ModelSampling()

    def get_dtype(self) -> torch.dtype:
        return torch.float32

    def memory_required(self, *_args: object, **_kwargs: object) -> int:
        return 0

    def extra_conds(self, **kwargs: object) -> dict[str, object]:
        conds = importlib.import_module("dinkster_inference.conds")
        return {"c_crossattn": conds.CONDCrossAttn(kwargs["cross_attn"])}

    def extra_conds_shapes(self, **_kwargs: object) -> dict[str, tuple[int, ...]]:
        return {}

    def scale_latent_inpaint(self, **values: torch.Tensor) -> torch.Tensor:
        return values["latent_image"]

    def process_latent_in(self, samples: torch.Tensor) -> torch.Tensor:
        return samples

    def process_latent_out(self, samples: torch.Tensor) -> torch.Tensor:
        return samples

    def apply_model(
        self, samples: torch.Tensor, _sigma: torch.Tensor, **_kwargs: object
    ) -> torch.Tensor:
        return _kwargs["c_crossattn"].mean(dim=(1, 2)).reshape(-1, 1, 1, 1).expand_as(samples)


class _VoidNoise:
    seed = 41

    def generate_noise(self, latent: dict[str, object]) -> torch.Tensor:
        return materialize_latent(latent)["samples"].new_tensor([[[[2.0, -4.0]]]])


class VoidSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="void-fixture.source",
            idempotent=True,
            outputs=tuple(
                OutputSpec(name, TypeExpr.concrete(kind))
                for name, kind in (
                    ("guider", GUIDER_TYPE),
                    ("noise", NOISE_TYPE),
                    ("sigmas", SIGMAS_TYPE),
                    ("latent", LATENT_TYPE),
                )
            ),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        initialize_inference_process()
        from dinkster_inference.model_patcher import ModelPatcher
        from dinkster_inference.samplers import CFGGuider

        patcher = ModelPatcher(_VoidModel(), torch.device("cpu"), torch.device("cpu"))
        patcher.model.current_patcher = patcher
        guider = CFGGuider(patcher)
        guider.set_conds([[torch.full((1, 1, 1), 2.0), {}]], [[torch.full((1, 1, 1), -1.0), {}]])
        guider.set_cfg(2.0)
        return cls.outputs(
            guider=guider,
            noise=_VoidNoise(),
            sigmas=torch.tensor([1.0, 0.0]),
            latent={
                "samples": torch.tensor([[[[5.0, -7.0]]]]),
                "noise_mask": torch.ones((1, 1, 1, 2)),
                "origin": "void-source",
            },
        )


class VoidConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="void-fixture.consumer",
            idempotent=True,
            inputs=(
                InputSpec("output", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec("denoised", TypeExpr.concrete(LATENT_TYPE)),
                InputSpec("source", TypeExpr.concrete(LATENT_TYPE)),
            ),
            outputs=(OutputSpec("complete", TypeExpr.concrete(CORE_BOOLEAN)),),
        )

    @classmethod
    def execute(cls, *, output: object, denoised: object, source: object) -> dict[str, object]:
        output, denoised, source = map(materialize_latent, (output, denoised, source))
        expected = torch.full((1, 1, 1, 2), 5.0)
        assert torch.equal(output["samples"], expected)
        assert torch.equal(denoised["samples"], expected)
        assert output["origin"] == denoised["origin"] == source["origin"] == "void-source"
        assert torch.equal(source["samples"], torch.tensor([[[[5.0, -7.0]]]]))
        return cls.outputs(complete=True)


HOST_NODES = (VoidSource, VoidConsumer)


@pytest.mark.parametrize("in_process", [True, False])
def test_void_sampler_graph_runs_through_native_advanced_sampler_and_cache(
    in_process: bool, tmp_path: Path
) -> None:
    async def scenario() -> None:
        package = Path(__file__).resolve().parents[1]
        (tmp_path / "void_entry.py").write_text(
            "from test_void_sampler_torch import HOST_NODES, register_inference_types\n",
            encoding="ascii",
        )
        fixture = tmp_path / "dinkster-pack.toml"
        fixture.write_text(
            '[pack]\nname="void-fixture"\npublisher="dinkster"\nnamespaces=["void-fixture"]\n[pack.entry]\nnodes="void_entry:HOST_NODES"\ntypes="void_entry:register_inference_types"\n',
            encoding="ascii",
        )
        native = package / "dinkster-pack.toml"
        from dinkster_graph import Graph, GraphNode, Link

        from dinkster.compose import PackSpec, ServingComposer

        manifests = (native, fixture)
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": os.pathsep.join(
                    (str(tmp_path), str(Path(__file__).parent), str(package / "src"))
                ),
            }
        )
        try:
            for manifest in manifests:
                await composer.add_pack(
                    PackSpec(
                        manifest=manifest,
                        in_process=in_process,
                        worker_group=None if in_process else "void-fixture",
                        group_manifests=() if in_process else manifests,
                    )
                )
            graph = Graph(
                nodes={
                    "source": GraphNode("void-fixture.source"),
                    "sampler": GraphNode("VOIDSampler"),
                    "sample": GraphNode(
                        "SamplerCustomAdvanced",
                        {
                            "noise": Link("source", "noise"),
                            "guider": Link("source", "guider"),
                            "sampler": Link("sampler", "SAMPLER"),
                            "sigmas": Link("source", "sigmas"),
                            "latent_image": Link("source", "latent"),
                        },
                    ),
                    "consumer": GraphNode(
                        "void-fixture.consumer",
                        {
                            "output": Link("sample", "output"),
                            "denoised": Link("sample", "denoised_output"),
                            "source": Link("source", "latent"),
                        },
                    ),
                }
            )
            engine = composer.composition.make_engine(lambda _: None)
            for attempt in range(2):
                result = await engine.run(graph, ["consumer"])
                assert result.outputs["consumer"]["complete"].resolve() is True
                assert set(result.cached) == (set(graph.nodes) if attempt else set())
        finally:
            await composer.close()

    asyncio.run(scenario())
