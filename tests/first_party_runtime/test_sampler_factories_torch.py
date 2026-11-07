"""CPU behavior for native sampler factory nodes."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    MODEL_TYPE,
    SAMPLER_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    initialize_inference_process,
)
from dinkster_graph import Graph, GraphNode, Link
from dinkster_nodes_comfy.sampler_factories import (
    SAMPLER_FACTORY_NODES,
    SAMPLER_NAMES,
    KSamplerSelect,
    SamplerDPMAdaptative,
    SamplerDPMPP_2M_SDE,
    SamplerDPMPP_2S_Ancestral,
    SamplerDPMPP_3M_SDE,
    SamplerDPMPP_SDE,
    SamplerER_SDE,
    SamplerEulerAncestral,
    SamplerEulerAncestralCFGPP,
    SamplerLMS,
    SamplerSASolver,
    SamplerSEEDS2,
)

from dinkster.compose import PackSpec, ServingComposer


class ModelSampling:
    def percent_to_sigma(self, percent: float) -> float:
        return 3.0 - percent * 2.0


class Model:
    def get_model_object(self, name: str) -> ModelSampling:
        assert name == "model_sampling"
        return ModelSampling()


def _sampler(node: type[Any], **inputs: Any) -> Any:
    return node.execute(**inputs)["SAMPLER"]


def test_root_sampler_factory_schemas_do_not_import_torch_or_inference() -> None:
    assert "torch" in sys.modules
    assert len(SAMPLER_FACTORY_NODES) == 12
    assert tuple(node.define_schema().node_type for node in SAMPLER_FACTORY_NODES) == (
        "KSamplerSelect",
        "SamplerEulerAncestral",
        "SamplerEulerAncestralCFGPP",
        "SamplerLMS",
        "SamplerDPMPP_3M_SDE",
        "SamplerDPMPP_2M_SDE",
        "SamplerDPMPP_SDE",
        "SamplerDPMPP_2S_Ancestral",
        "SamplerDPMAdaptative",
        "SamplerER_SDE",
        "SamplerSASolver",
        "SamplerSEEDS2",
    )


def test_sampler_factories_construct_pinned_runtime_samplers_with_exact_options() -> None:
    initialize_inference_process()
    from dinkster_inference import samplers

    assert tuple(samplers.SAMPLER_NAMES) == SAMPLER_NAMES
    cases = (
        (KSamplerSelect, {"sampler_name": "euler"}, "sample_euler", {}),
        (
            SamplerEulerAncestral,
            {"eta": 0.375, "s_noise": 1.25},
            "sample_euler_ancestral",
            {"eta": 0.375, "s_noise": 1.25},
        ),
        (
            SamplerEulerAncestralCFGPP,
            {"eta": 0.625, "s_noise": 2.5},
            "sample_euler_ancestral_cfg_pp",
            {"eta": 0.625, "s_noise": 2.5},
        ),
        (SamplerLMS, {"order": 7}, "sample_lms", {"order": 7}),
        (
            SamplerDPMPP_3M_SDE,
            {"eta": 0.4, "s_noise": 1.3, "noise_device": "cpu"},
            "sample_dpmpp_3m_sde",
            {"eta": 0.4, "s_noise": 1.3},
        ),
        (
            SamplerDPMPP_2M_SDE,
            {"solver_type": "heun", "eta": 0.4, "s_noise": 1.3, "noise_device": "cpu"},
            "sample_dpmpp_2m_sde",
            {"eta": 0.4, "s_noise": 1.3, "solver_type": "heun"},
        ),
        (
            SamplerDPMPP_SDE,
            {"eta": 0.4, "s_noise": 1.3, "r": 0.27, "noise_device": "cpu"},
            "sample_dpmpp_sde",
            {"eta": 0.4, "s_noise": 1.3, "r": 0.27},
        ),
        (
            SamplerDPMPP_2S_Ancestral,
            {"eta": 0.4, "s_noise": 1.3},
            "sample_dpmpp_2s_ancestral",
            {"eta": 0.4, "s_noise": 1.3},
        ),
        (
            SamplerDPMAdaptative,
            {
                "order": 2,
                "rtol": 0.04,
                "atol": 0.005,
                "h_init": 0.07,
                "pcoeff": 0.2,
                "icoeff": 0.8,
                "dcoeff": 0.1,
                "accept_safety": 0.75,
                "eta": 0.3,
                "s_noise": 1.1,
            },
            "dpm_adaptive_function",
            {
                "order": 2,
                "rtol": 0.04,
                "atol": 0.005,
                "h_init": 0.07,
                "pcoeff": 0.2,
                "icoeff": 0.8,
                "dcoeff": 0.1,
                "accept_safety": 0.75,
                "eta": 0.3,
                "s_noise": 1.1,
            },
        ),
        (
            SamplerER_SDE,
            {"solver_type": "ER-SDE", "max_stage": 2, "eta": 0.5, "s_noise": 1.3},
            "sample_er_sde",
            {"s_noise": 1.3, "max_stage": 2},
        ),
        (
            SamplerSASolver,
            {
                "model": Model(),
                "eta": 0.75,
                "sde_start_percent": 0.25,
                "sde_end_percent": 0.75,
                "s_noise": 1.4,
                "predictor_order": 2,
                "corrector_order": 3,
                "use_pece": True,
                "simple_order_2": True,
            },
            "sample_sa_solver",
            {
                "s_noise": 1.4,
                "predictor_order": 2,
                "corrector_order": 3,
                "use_pece": True,
                "simple_order_2": True,
            },
        ),
        (
            SamplerSEEDS2,
            {"solver_type": "phi_2", "eta": 0.4, "s_noise": 1.3, "r": 0.27},
            "sample_seeds_2",
            {"eta": 0.4, "s_noise": 1.3, "r": 0.27, "solver_type": "phi_2"},
        ),
    )
    for node, inputs, function_name, expected_options in cases:
        sampler = _sampler(node, **inputs)
        assert callable(sampler.sampler_function)
        assert sampler.sampler_function.__name__ == function_name
        for name, value in expected_options.items():
            assert sampler.extra_options[name] == value


def test_er_sde_and_sa_solver_closures_preserve_solver_boundaries() -> None:
    er_sde = _sampler(SamplerER_SDE, solver_type="ER-SDE", max_stage=2, eta=0.5, s_noise=1.3)
    reverse = _sampler(
        SamplerER_SDE, solver_type="Reverse-time SDE", max_stage=2, eta=0.5, s_noise=1.3
    )
    ode = _sampler(SamplerER_SDE, solver_type="ER-SDE", max_stage=2, eta=0.0, s_noise=1.3)
    values = torch.tensor((0.25, 2.0), dtype=torch.float32)
    torch.testing.assert_close(
        er_sde.extra_options["noise_scaler"](values),
        values * ((values**0.3).exp() + 10.0) ** 0.5,
    )
    torch.testing.assert_close(reverse.extra_options["noise_scaler"](values), values**1.5)
    assert ode.extra_options["s_noise"] == 0.0
    assert ode.extra_options["noise_scaler"](values) is values

    solver = _sampler(
        SamplerSASolver,
        model=Model(),
        eta=0.75,
        sde_start_percent=0.25,
        sde_end_percent=0.75,
        s_noise=1.4,
        predictor_order=2,
        corrector_order=3,
        use_pece=True,
        simple_order_2=True,
    )
    tau = solver.extra_options["tau_func"]
    assert tau(2.5) == 0.75
    assert tau(1.5) == 0.75
    assert tau(0.5) == 0.0

    explicit_ode = _sampler(SamplerER_SDE, solver_type="ODE", max_stage=1, eta=0.9, s_noise=2.0)
    assert explicit_ode.extra_options["s_noise"] == 0.0
    assert explicit_ode.extra_options["noise_scaler"](values) is values


def test_sde_noise_device_gpu_names_are_selected_without_gpu_execution() -> None:
    assert (
        _sampler(
            SamplerDPMPP_3M_SDE, eta=0.4, s_noise=1.3, noise_device="gpu"
        ).sampler_function.__name__
        == "sample_dpmpp_3m_sde_gpu"
    )
    assert (
        _sampler(
            SamplerDPMPP_2M_SDE,
            solver_type="heun",
            eta=0.4,
            s_noise=1.3,
            noise_device="gpu",
        ).sampler_function.__name__
        == "sample_dpmpp_2m_sde_gpu"
    )
    assert (
        _sampler(
            SamplerDPMPP_SDE, eta=0.4, s_noise=1.3, r=0.27, noise_device="gpu"
        ).sampler_function.__name__
        == "sample_dpmpp_sde_gpu"
    )


def test_euler_ancestral_factory_callable_uses_asymmetric_inputs() -> None:
    sampler = _sampler(SamplerEulerAncestral, eta=0.375, s_noise=1.25)
    noise = torch.tensor([[[[0.2, -0.4], [1.1, -0.8]]]], dtype=torch.float32)

    class SamplingModel:
        inner_model = SimpleNamespace(inner_model=SimpleNamespace(model_sampling=object()))

        def __call__(
            self, value: torch.Tensor, _sigma: torch.Tensor, **_kwargs: Any
        ) -> torch.Tensor:
            return torch.full_like(value, 0.25)

    result = sampler.sampler_function(
        SamplingModel(),
        noise,
        torch.tensor((1.5, 0.75)),
        extra_args={"seed": 739},
        disable=True,
        **sampler.extra_options,
    )
    sigma_up = 0.375 * (0.75**2 * (1.5**2 - 0.75**2) / 1.5**2) ** 0.5
    sigma_down = (0.75**2 - sigma_up**2) ** 0.5
    ancestral_noise = torch.randn_like(noise, generator=torch.Generator().manual_seed(740))
    expected = noise * (sigma_down / 1.5) + 0.25 * (1 - sigma_down / 1.5)
    torch.testing.assert_close(result, expected + ancestral_noise * 1.25 * sigma_up)


class ModelSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sampler-factory.model",
            outputs=(OutputSpec("MODEL", TypeExpr.concrete(MODEL_TYPE)),),
        )

    @classmethod
    def execute(cls) -> dict[str, object]:
        return cls.outputs(MODEL=Model())


class SamplerConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        sampler = TypeExpr.concrete(SAMPLER_TYPE)
        return NodeSchema(
            node_type="sampler-factory.consumer",
            inputs=tuple(InputSpec(name, sampler) for name in _factory_graph_inputs()),
            outputs=(OutputSpec("summary", TypeExpr.concrete(CORE_STRING)),),
        )

    @classmethod
    def execute(cls, **inputs: Any) -> dict[str, object]:
        assert set(inputs) == set(_factory_graph_inputs())
        _assert_hosted_sampler_contracts(inputs)
        return cls.outputs(
            summary=",".join(
                sorted(sampler.sampler_function.__name__ for sampler in inputs.values())
            )
        )


class _RuntimeModel:
    inner_model = SimpleNamespace(inner_model=SimpleNamespace(model_sampling=object()))

    def __call__(self, value: torch.Tensor, _sigma: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        return torch.full_like(value, 0.25)


def _assert_hosted_sampler_contracts(inputs: dict[str, Any]) -> None:
    expected = {
        "select": ("sample_euler", {}),
        "euler_ancestral": ("sample_euler_ancestral", {"eta": 0.375, "s_noise": 1.25}),
        "euler_ancestral_cfgpp": (
            "sample_euler_ancestral_cfg_pp",
            {"eta": 0.625, "s_noise": 2.5},
        ),
        "lms": ("sample_lms", {"order": 7}),
        "dpmpp_3m_sde": ("sample_dpmpp_3m_sde", {"eta": 0.4, "s_noise": 1.3}),
        "dpmpp_2m_sde": (
            "sample_dpmpp_2m_sde",
            {"eta": 0.4, "s_noise": 1.3, "solver_type": "heun"},
        ),
        "dpmpp_sde": ("sample_dpmpp_sde", {"eta": 0.4, "s_noise": 1.3, "r": 0.27}),
        "dpmpp_2s_ancestral": (
            "sample_dpmpp_2s_ancestral",
            {"eta": 0.4, "s_noise": 1.3},
        ),
        "dpm_adaptative": (
            "dpm_adaptive_function",
            {
                "order": 2,
                "rtol": 0.04,
                "atol": 0.005,
                "h_init": 0.07,
                "pcoeff": 0.2,
                "icoeff": 0.8,
                "dcoeff": 0.1,
                "accept_safety": 0.75,
                "eta": 0.3,
                "s_noise": 1.1,
            },
        ),
        "er_sde": ("sample_er_sde", {"s_noise": 1.3, "max_stage": 2}),
        "sa_solver": (
            "sample_sa_solver",
            {
                "s_noise": 1.4,
                "predictor_order": 2,
                "corrector_order": 3,
                "use_pece": True,
                "simple_order_2": True,
            },
        ),
        "seeds2": (
            "sample_seeds_2",
            {"eta": 0.4, "s_noise": 1.3, "r": 0.27, "solver_type": "phi_2"},
        ),
    }
    for name, (function_name, options) in expected.items():
        sampler = inputs[name]
        assert callable(sampler.sample)
        assert sampler.sampler_function.__name__ == function_name
        if name == "er_sde":
            assert sampler.extra_options.keys() == {"s_noise", "noise_scaler", "max_stage"}
            assert sampler.extra_options["s_noise"] == options["s_noise"]
            assert sampler.extra_options["max_stage"] == options["max_stage"]
        elif name == "sa_solver":
            assert sampler.extra_options.keys() == {"tau_func", *options}
            for option_name, option_value in options.items():
                assert sampler.extra_options[option_name] == option_value
        else:
            assert sampler.extra_options.keys() == options.keys()
            assert sampler.extra_options == options

    values = torch.tensor((0.125, 2.75), dtype=torch.float32)
    er_scaler = inputs["er_sde"].extra_options["noise_scaler"]
    torch.testing.assert_close(er_scaler(values), values * ((values**0.3).exp() + 10.0) ** 0.5)

    tau = inputs["sa_solver"].extra_options["tau_func"]
    assert tau(2.5) == 0.75
    assert tau(1.5) == 0.75
    assert tau(2.5001) == 0.0
    assert tau(1.4999) == 0.0

    sampler = inputs["euler_ancestral"]
    noise = torch.tensor([[[[0.2, -0.4], [1.1, -0.8]]]], dtype=torch.float32)
    result = sampler.sampler_function(
        _RuntimeModel(),
        noise,
        torch.tensor((1.5, 0.75)),
        extra_args={"seed": 739},
        disable=True,
        **sampler.extra_options,
    )
    sigma_up = 0.375 * (0.75**2 * (1.5**2 - 0.75**2) / 1.5**2) ** 0.5
    sigma_down = (0.75**2 - sigma_up**2) ** 0.5
    random = torch.randn(4, generator=torch.Generator().manual_seed(740)).reshape_as(noise)
    expected_euler = noise * (sigma_down / 1.5) + 0.25 * (1 - sigma_down / 1.5)
    torch.testing.assert_close(result, expected_euler + random * 1.25 * sigma_up)


def _factory_graph_inputs() -> tuple[str, ...]:
    return (
        "select",
        "euler_ancestral",
        "euler_ancestral_cfgpp",
        "lms",
        "dpmpp_3m_sde",
        "dpmpp_2m_sde",
        "dpmpp_sde",
        "dpmpp_2s_ancestral",
        "dpm_adaptative",
        "er_sde",
        "sa_solver",
        "seeds2",
    )


@pytest.mark.parametrize("in_process", (True, False))
def test_all_sampler_factories_cross_ordinary_manifest_hosts_twice(
    tmp_path: Path, in_process: bool
) -> None:
    async def scenario() -> None:
        composer = ServingComposer(
            worker_env={
                "CUDA_VISIBLE_DEVICES": "",
                "DINKSTER_ACCELERATOR": "cpu",
                "PYTHONPATH": str(Path(__file__).parent),
            }
        )
        try:
            module = f"sampler_factories_fixture_{int(in_process)}"
            (tmp_path / f"{module}.py").write_text(
                "from test_sampler_factories_torch import ModelSource, SamplerConsumer\n"
                "from dinkster_nodes_comfy import COMFY_NODES, register_inference_types\n"
                "from dinkster_nodes_comfy.device_routing import device_choices\n"
                "NODES = (*COMFY_NODES, ModelSource, SamplerConsumer)\n",
                encoding="ascii",
            )
            manifest = tmp_path / "dinkster-pack.toml"
            native_manifest = Path(__file__).resolve().parents[1] / "dinkster-pack.toml"
            manifest.write_text(
                native_manifest.read_text(encoding="ascii")
                .replace("namespaces = [", 'namespaces = ["sampler-factory",')
                .replace("dinkster_nodes_comfy:COMFY_NODES", f"{module}:NODES")
                .replace(
                    "dinkster_nodes_comfy.device_routing:device_choices",
                    f"{module}:device_choices",
                )
                .replace(
                    "dinkster_nodes_comfy:register_inference_types",
                    f"{module}:register_inference_types",
                ),
                encoding="ascii",
            )
            await composer.add_pack(PackSpec(manifest=manifest, in_process=in_process))
            nodes = {
                "model": GraphNode("sampler-factory.model"),
                "select": GraphNode("KSamplerSelect", {"sampler_name": "euler"}),
                "euler_ancestral": GraphNode(
                    "SamplerEulerAncestral", {"eta": 0.375, "s_noise": 1.25}
                ),
                "euler_ancestral_cfgpp": GraphNode(
                    "SamplerEulerAncestralCFGPP", {"eta": 0.625, "s_noise": 2.5}
                ),
                "lms": GraphNode("SamplerLMS", {"order": 7}),
                "dpmpp_3m_sde": GraphNode(
                    "SamplerDPMPP_3M_SDE", {"eta": 0.4, "s_noise": 1.3, "noise_device": "cpu"}
                ),
                "dpmpp_2m_sde": GraphNode(
                    "SamplerDPMPP_2M_SDE",
                    {"solver_type": "heun", "eta": 0.4, "s_noise": 1.3, "noise_device": "cpu"},
                ),
                "dpmpp_sde": GraphNode(
                    "SamplerDPMPP_SDE",
                    {"eta": 0.4, "s_noise": 1.3, "r": 0.27, "noise_device": "cpu"},
                ),
                "dpmpp_2s_ancestral": GraphNode(
                    "SamplerDPMPP_2S_Ancestral", {"eta": 0.4, "s_noise": 1.3}
                ),
                "dpm_adaptative": GraphNode(
                    "SamplerDPMAdaptative",
                    {
                        "order": 2,
                        "rtol": 0.04,
                        "atol": 0.005,
                        "h_init": 0.07,
                        "pcoeff": 0.2,
                        "icoeff": 0.8,
                        "dcoeff": 0.1,
                        "accept_safety": 0.75,
                        "eta": 0.3,
                        "s_noise": 1.1,
                    },
                ),
                "er_sde": GraphNode(
                    "SamplerER_SDE",
                    {"solver_type": "ER-SDE", "max_stage": 2, "eta": 0.5, "s_noise": 1.3},
                ),
                "sa_solver": GraphNode(
                    "SamplerSASolver",
                    {
                        "model": Link("model", "MODEL"),
                        "eta": 0.75,
                        "sde_start_percent": 0.25,
                        "sde_end_percent": 0.75,
                        "s_noise": 1.4,
                        "predictor_order": 2,
                        "corrector_order": 3,
                        "use_pece": True,
                        "simple_order_2": True,
                    },
                ),
                "seeds2": GraphNode(
                    "SamplerSEEDS2", {"solver_type": "phi_2", "eta": 0.4, "s_noise": 1.3, "r": 0.27}
                ),
            }
            nodes["consumer"] = GraphNode(
                "sampler-factory.consumer",
                {name: Link(name, "SAMPLER") for name in _factory_graph_inputs()},
            )
            engine = composer.composition.make_engine(lambda _event: None)
            for attempt in range(2):
                result = await engine.run(Graph(nodes=nodes), ["consumer"])
                assert "sample_euler_ancestral" in result.outputs["consumer"]["summary"].resolve()
                if attempt:
                    assert "consumer" in result.cached
        finally:
            await composer.close()

    asyncio.run(scenario())
