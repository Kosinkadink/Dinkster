"""Exact pinned-source trajectory proofs for torch-native UniPC."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference import Parameterization, SamplerInfo, SolverStateEvent, StepEvent
from dinkster_inference.sampling import run_step_begin_solver
from dinkster_inference_torch import solvers as torch_solvers  # noqa: I001
from golden_files import assert_reference_tensor, load_platform_golden

GOLDEN = load_platform_golden(Path(__file__).parent / "goldens/unipc_trajectory_goldens.json")


class _Model:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
        self.calls.append(x.detach().clone())
        return x * (1.0 / (1.0 + sigma)) + (x * x) * (0.05 / (1.0 + sigma))


def _tensor(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).reshape(1, 1, 2, 2)


@pytest.mark.parametrize("name", ["uni_pc", "uni_pc_bh2"])
def test_torch_unipc_matches_every_executed_reference_seam(name: str) -> None:
    case: dict[str, Any] = GOLDEN["cases"][name]
    model = _Model()
    steps: list[torch.Tensor] = []
    events: list[StepEvent] = []
    variant = "bh1" if name == "uni_pc" else "bh2"
    solver = torch_solvers._uni_pc(  # pyright: ignore[reportPrivateUsage]
        variant, trace=lambda _, value: steps.append(value.detach().clone())
    )
    result = solver(
        model,
        _tensor(GOLDEN["initial_noise"]),
        tuple(GOLDEN["sigmas"]),
        SamplerInfo(Parameterization.EPS),
        on_step=events.append,
    )

    assert_reference_tensor(model.calls[0], _tensor(case["first_call"]))
    assert len(model.calls) == len(case["model_calls"]) == 20
    for actual, expected in zip(model.calls, case["model_calls"], strict=True):
        assert_reference_tensor(actual, _tensor(expected))
    assert len(steps) == len(case["steps"]) == 20
    for actual, expected in zip(steps, case["steps"], strict=True):
        assert_reference_tensor(actual, _tensor(expected))
    assert_reference_tensor(result, _tensor(case["final"]))
    assert [(event.step, event.total, event.sigma) for event in events] == [
        (index, 20, GOLDEN["sigmas"][index]) for index in range(20)
    ]


@pytest.mark.parametrize("name", ["uni_pc", "uni_pc_bh2"])
def test_torch_registry_routes_unipc_through_reference_kernels(name: str) -> None:
    descriptor = torch_solvers.torch_sampler_registry().get(f"dinkster.{name}")
    assert descriptor is not None
    model = _Model()
    result = descriptor.build()(
        model,
        _tensor(GOLDEN["initial_noise"]),
        tuple(GOLDEN["sigmas"]),
        SamplerInfo(Parameterization.EPS),
    )
    case: dict[str, Any] = GOLDEN["cases"][name]
    assert_reference_tensor(model.calls[0], _tensor(case["first_call"]))
    assert_reference_tensor(result, _tensor(case["final"]))


@pytest.mark.parametrize("name", ["uni_pc", "uni_pc_bh2"])
def test_torch_unipc_preserves_flow_terminal_schedule_mutation(name: str) -> None:
    descriptor = torch_solvers.torch_sampler_registry().get(f"dinkster.{name}")
    assert descriptor is not None
    solver = descriptor.build()
    sigmas = tuple(GOLDEN["sigmas"])
    initial = _tensor(GOLDEN["initial_noise"])

    eps = solver(_Model(), initial, sigmas, SamplerInfo(Parameterization.EPS))
    flow = solver(_Model(), initial, sigmas, SamplerInfo(Parameterization.FLOW))
    image_flow = solver(
        _Model(),
        initial,
        sigmas,
        SamplerInfo(Parameterization.IMAGE_TO_IMAGE_FLOW),
    )
    terminal = torch.tensor(0.001, dtype=torch.float32)

    assert torch.equal(flow, eps / (1.0 - terminal))
    assert torch.equal(image_flow, flow)


def test_torch_registry_routes_ddim_through_exact_default_euler() -> None:
    registry = torch_solvers.torch_sampler_registry()
    ddim_descriptor = registry.get("ddim")
    euler_descriptor = registry.get("euler")
    assert ddim_descriptor is not None and euler_descriptor is not None
    assert ddim_descriptor.options == ()
    assert ddim_descriptor.random_inpaint_noise
    sigmas = (1.0, 0.6, 0.2, 0.0)
    initial = _tensor(GOLDEN["initial_noise"])
    info = SamplerInfo(Parameterization.EPS)
    assert torch.equal(
        ddim_descriptor.build()(_Model(), initial, sigmas, info),
        euler_descriptor.build()(_Model(), initial, sigmas, info),
    )


def test_torch_euler_materializes_the_device_schedule_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = torch_solvers.torch_sampler_registry().get("euler")
    assert descriptor is not None
    sigmas = (1.0, 0.6, 0.2, 0.0)
    initial = _tensor(GOLDEN["initial_noise"])
    calls: list[object] = []
    original = torch.tensor

    def tracked(data: object, *args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(data)
        return original(data, *args, **kwargs)

    monkeypatch.setattr(torch, "tensor", tracked)
    descriptor.build()(_Model(), initial, sigmas, SamplerInfo(Parameterization.EPS))

    assert calls == [sigmas]


@pytest.mark.parametrize("name", ["euler", "ddim", "uni_pc", "uni_pc_bh2"])
def test_torch_solver_state_matches_the_reported_current_value(name: str) -> None:
    descriptor = torch_solvers.torch_sampler_registry().get(f"dinkster.{name}")
    assert descriptor is not None
    states: list[SolverStateEvent[object]] = []
    model = _Model()
    descriptor.build()(
        model,
        _tensor(GOLDEN["initial_noise"]),
        (1.0, 0.6, 0.2, 0.0),
        SamplerInfo(Parameterization.EPS, on_state=states.append),
    )
    assert len(states) == 3
    for index, state in enumerate(states):
        assert state.step == index
        assert state.total == 3
        assert state.phase == "pre_update"
        assert type(state.current) is torch.Tensor
        assert type(state.denoised) is torch.Tensor
        expected = _Model()(state.current, state.sigma)
        assert torch.equal(state.denoised, expected)


@pytest.mark.parametrize("name", ["euler", "ddim", "uni_pc", "uni_pc_bh2"])
def test_torch_solver_state_callback_exceptions_stop_sampling(name: str) -> None:
    descriptor = torch_solvers.torch_sampler_registry().get(f"dinkster.{name}")
    assert descriptor is not None

    def stop(_event: SolverStateEvent[object]) -> None:
        raise RuntimeError("stop callback")

    with pytest.raises(RuntimeError, match="stop callback"):
        descriptor.build()(
            _Model(),
            _tensor(GOLDEN["initial_noise"]),
            (1.0, 0.6, 0.2, 0.0),
            SamplerInfo(Parameterization.EPS, on_state=stop),
        )


@pytest.mark.parametrize(
    "name",
    ["euler", "dpmpp_2m", "ddim", "uni_pc", "uni_pc_bh2"],
)
def test_outer_step_callback_precedes_and_covers_every_model_evaluation(name: str) -> None:
    descriptor = torch_solvers.torch_sampler_registry().get(f"dinkster.{name}")
    assert descriptor is not None
    active_step: int | None = None
    events: list[tuple[str, int]] = []

    class Model:
        def __call__(self, x: torch.Tensor, sigma: float) -> torch.Tensor:
            del sigma
            assert active_step is not None
            events.append(("model", active_step))
            return x * 0.75

    def begin(index: int) -> None:
        nonlocal active_step
        active_step = index
        events.append(("begin", index))

    run_step_begin_solver(
        descriptor.build(),
        Model(),
        _tensor(GOLDEN["initial_noise"]),
        (1.0, 0.8, 0.6, 0.4, 0.0),
        SamplerInfo(Parameterization.EPS, on_state=lambda _event: None),
        noise=None,
        on_step=None,
        on_step_begin=begin,
    )
    assert [value for kind, value in events if kind == "begin"] == [0, 1, 2, 3]
    latest = -1
    model_steps: list[int] = []
    for kind, value in events:
        if kind == "begin":
            latest = value
        else:
            assert value == latest
            model_steps.append(value)
    assert model_steps
    if name.startswith("uni_pc"):
        assert any(left == right for left, right in zip(model_steps, model_steps[1:], strict=False))


@pytest.mark.parametrize("name", ["uni_pc", "uni_pc_bh2"])
@pytest.mark.parametrize(
    ("suffix", "sigmas"),
    [("terminal_zero", (1.0, 0.0)), ("terminal_nonzero", (1.0, 0.5))],
)
def test_torch_unipc_short_schedule_matches_executed_reference(
    name: str, suffix: str, sigmas: tuple[float, float]
) -> None:
    case: dict[str, Any] = GOLDEN["edge_cases"][f"{name}_{suffix}"]
    model = _Model()
    steps: list[torch.Tensor] = []
    variant = "bh1" if name == "uni_pc" else "bh2"
    result = torch_solvers._uni_pc(  # pyright: ignore[reportPrivateUsage]
        variant, trace=lambda _, value: steps.append(value.detach().clone())
    )(
        model,
        _tensor(GOLDEN["initial_noise"]),
        sigmas,
        SamplerInfo(Parameterization.EPS),
    )
    assert len(model.calls) == 1
    assert torch.equal(model.calls[0], _tensor(case["first_call"]))
    assert len(steps) == 1
    assert torch.equal(steps[0], _tensor(case["steps"][0]))
    assert torch.equal(result, _tensor(case["final"]))


@pytest.mark.parametrize("name", ["uni_pc", "uni_pc_bh2"])
def test_torch_unipc_empty_and_single_schedules_are_identity(name: str) -> None:
    descriptor = torch_solvers.torch_sampler_registry().get(f"dinkster.{name}")
    assert descriptor is not None
    solver = descriptor.build()
    source = _tensor(GOLDEN["initial_noise"])
    for sigmas in ((), (1.0,)):
        events: list[StepEvent] = []
        assert (
            solver(
                _Model(),
                source,
                sigmas,
                SamplerInfo(Parameterization.EPS),
                on_step=events.append,
            )
            is source
        )
        assert events == []


@pytest.mark.parametrize("name", ["uni_pc", "uni_pc_bh2"])
@pytest.mark.parametrize(
    "sigmas",
    [
        (1.0, math.inf),
        (1.0, math.nan),
        (1.0, -0.5),
        (1.0, 1.0, 0.0),
        (1.00000001, 1.0, 0.0),
        (1e39, 1.0, 0.0),
    ],
)
def test_torch_unipc_preserves_degenerate_schedule_refusals(
    name: str, sigmas: tuple[float, ...]
) -> None:
    descriptor = torch_solvers.torch_sampler_registry().get(f"dinkster.{name}")
    assert descriptor is not None
    with pytest.raises(ValueError, match=rf"^{name} degenerate schedule"):
        descriptor.build()(
            _Model(),
            _tensor(GOLDEN["initial_noise"]),
            sigmas,
            SamplerInfo(Parameterization.EPS),
        )
