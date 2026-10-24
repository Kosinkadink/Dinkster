"""Public schemas for native ComfyUI sigma scheduler nodes."""

from dinkster_nodes_comfy import COMFY_NODES


def test_native_sigma_scheduler_nodes_are_registered_with_comfy_schemas() -> None:
    nodes = {node.schema().node_type: node.schema() for node in COMFY_NODES}
    expected_inputs = {
        "BasicScheduler": ("model", "scheduler", "steps", "denoise"),
        "KarrasScheduler": ("steps", "sigma_max", "sigma_min", "rho"),
        "ExponentialScheduler": ("steps", "sigma_max", "sigma_min"),
        "PolyexponentialScheduler": ("steps", "sigma_max", "sigma_min", "rho"),
        "LaplaceScheduler": ("steps", "sigma_max", "sigma_min", "mu", "beta"),
        "VPScheduler": ("steps", "beta_d", "beta_min", "eps_s"),
        "BetaSamplingScheduler": ("model", "steps", "alpha", "beta"),
        "SDTurboScheduler": ("model", "steps", "denoise"),
    }
    assert {
        name: tuple(port.id for port in nodes[name].inputs) for name in expected_inputs
    } == expected_inputs
    assert all(nodes[name].outputs[0].id == "SIGMAS" for name in expected_inputs)
