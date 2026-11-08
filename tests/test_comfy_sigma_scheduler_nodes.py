"""Public schemas for native ComfyUI sigma scheduler nodes."""

from dinkster_sampling import NODES
from dinkster_sampling.sigma_schedulers import (
    BasicScheduler,
    BetaSamplingScheduler,
    ExponentialScheduler,
    KarrasScheduler,
    LaplaceScheduler,
    PolyexponentialScheduler,
    SDTurboScheduler,
    VPScheduler,
)
from dinkster_schema import NumberWidget

SIGMA_SCHEDULER_NODES = (
    BasicScheduler,
    KarrasScheduler,
    ExponentialScheduler,
    PolyexponentialScheduler,
    LaplaceScheduler,
    VPScheduler,
    BetaSamplingScheduler,
    SDTurboScheduler,
)


def test_native_sigma_scheduler_nodes_are_registered_with_comfy_schemas() -> None:
    nodes = {node.schema().node_type: node.schema() for node in SIGMA_SCHEDULER_NODES}
    expected_inputs = {
        "dinkster.sampling.basic_scheduler": ("model", "scheduler", "steps", "denoise"),
        "dinkster.sampling.karras_scheduler": ("steps", "sigma_max", "sigma_min", "rho"),
        "dinkster.sampling.exponential_scheduler": ("steps", "sigma_max", "sigma_min"),
        "dinkster.sampling.polyexponential_scheduler": ("steps", "sigma_max", "sigma_min", "rho"),
        "dinkster.sampling.laplace_scheduler": ("steps", "sigma_max", "sigma_min", "mu", "beta"),
        "dinkster.sampling.vp_scheduler": ("steps", "beta_d", "beta_min", "eps_s"),
        "dinkster.sampling.beta_sampling_scheduler": ("model", "steps", "alpha", "beta"),
        "dinkster.sampling.sd_turbo_scheduler": ("model", "steps", "denoise"),
    }
    assert {
        name: tuple(port.id for port in nodes[name].inputs) for name in expected_inputs
    } == expected_inputs
    assert all(nodes[name].outputs[0].id == "SIGMAS" for name in expected_inputs)
    assert set(SIGMA_SCHEDULER_NODES).issubset(NODES)
    for schema in nodes.values():
        for port in schema.inputs:
            if (
                port.type.types == ("FLOAT",)
                and port.id != "denoise"
                and isinstance(port.widget, NumberWidget)
            ):
                assert port.widget.round is False
