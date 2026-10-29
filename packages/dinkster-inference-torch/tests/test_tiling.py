"""Stage 5 slice 1: tiled_apply vs the executed reference.

Every golden in goldens/tiling_goldens.json was produced by RUNNING
comfy/utils.py tiled_scale_multidim @ 947c2749
(tools/gen_tiling_goldens.py) on recorded inputs with named
deterministic codec stand-ins. The tests replay the same inputs
through dinkster_inference.tiling.plan_tiles +
dinkster_inference_torch.tiling.tiled_apply and require exact agreement -
positions, edge clamping, feather math, and accumulation all fold
into the output tensor, so output equality pins the whole pipeline.

Deliberate loud deviations (documented in both tiling.py halves) are
tested directly: no baked-in inference_mode (gradients flow), and
uncovered output raises instead of dividing 0/0 into NaNs.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference.tiling import (
    CausalScale,
    LinearScale,
    Scale,
    plan_tiles,
)
from dinkster_inference_torch import TileApplyError, tiled_apply
from golden_files import load_platform_golden

GOLDENS = load_platform_golden(Path(__file__).parent / "goldens" / "tiling_goldens.json")


def dec(spec: dict[str, Any]) -> torch.Tensor:
    data = torch.tensor(spec["data"], dtype=torch.float32)
    return data.reshape(spec["shape"]).to(getattr(torch, spec["dtype"]))


# Named pinned codec stand-ins mirroring tools/gen_tiling_goldens.py
# FUNCTIONS - golden cases reference them by name.


def up2_mix3(x: torch.Tensor) -> torch.Tensor:
    up = x.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    m = up.mean(1, keepdim=True)
    return torch.cat([m * 0.5, m, m * 1.5], dim=1) + 0.25


def down8_mix4(x: torch.Tensor) -> torch.Tensor:
    h = round(x.shape[-2] / 8)
    w = round(x.shape[-1] / 8)
    pooled = torch.nn.functional.adaptive_avg_pool2d(x, (h, w))
    m = pooled.mean(1, keepdim=True)
    return torch.cat([m, m * -1.0, m * 2.0, m * 0.5], dim=1)


def up4_1d(x: torch.Tensor) -> torch.Tensor:
    return x.repeat_interleave(4, dim=-1) * 0.5 + 0.1


def causal_up_video(x: torch.Tensor) -> torch.Tensor:
    frames = x.repeat_interleave(4, dim=2)[:, :, 3:]
    up = frames.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    m = up.mean(1, keepdim=True)
    return torch.cat([m, m * 0.5, m * -0.25], dim=1)


def causal_down_video(x: torch.Tensor) -> torch.Tensor:
    sub = x[:, :, ::4]
    b, c, tt, h, w = sub.shape
    hh, ww = round(h / 8), round(w / 8)
    pooled = torch.nn.functional.adaptive_avg_pool2d(sub.reshape(b, c * tt, h, w), (hh, ww))
    pooled = pooled.reshape(b, c, tt, hh, ww)
    m = pooled.mean(1, keepdim=True)
    return torch.cat([m, m * 2.0, m * -1.0, m + 0.5], dim=1)


def up2_tilenorm(x: torch.Tensor) -> torch.Tensor:
    """2D decode stand-in whose output depends on the WHOLE tile
    (subtracts the tile-global mean): the three aspect-sweep passes
    produce genuinely different values, so the sweep goldens pin pass
    order and averaging, not just single-pass accumulation."""
    up = x.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    m = up.mean(1, keepdim=True) - up.mean()
    return torch.cat([m * 0.5, m, m * 1.5], dim=1) + 0.25


def down4_tilenorm(x: torch.Tensor) -> torch.Tensor:
    """2D encode stand-in, tile-globally sensitive like
    up2_tilenorm."""
    h = round(x.shape[-2] / 4)
    w = round(x.shape[-1] / 4)
    pooled = torch.nn.functional.adaptive_avg_pool2d(x, (h, w)) - x.mean()
    m = pooled.mean(1, keepdim=True)
    return torch.cat([m, m * -1.0, m * 2.0, m * 0.5], dim=1)


FUNCTIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "up2_mix3": up2_mix3,
    "down8_mix4": down8_mix4,
    "up4_1d": up4_1d,
    "causal_up_video": causal_up_video,
    "causal_down_video": causal_down_video,
    "up2_tilenorm": up2_tilenorm,
    "down4_tilenorm": down4_tilenorm,
}


def build_scales(specs: list[dict[str, Any]]) -> tuple[Scale, ...]:
    out: list[Scale] = []
    for spec in specs:
        if spec["kind"] == "linear":
            out.append(LinearScale(spec["factor"]))
        else:
            out.append(CausalScale(spec["factor"]))
    return tuple(out)


def replay(case: dict[str, Any]) -> torch.Tensor:
    samples = dec(case["samples"])
    plan = plan_tiles(
        tuple(samples.shape[2:]),
        tuple(case["tile"]),
        overlap=tuple(case["overlap"]),
        scale=build_scales(case["scales"]),
        downscale=case["downscale"],
    )
    return tiled_apply(
        samples,
        FUNCTIONS[case["function"]],
        plan,
        out_channels=case["out_channels"],
    )


# ------------------------------------------------------ golden replay


@pytest.mark.parametrize("case", GOLDENS["cases"], ids=lambda c: c["name"])
def test_tiled_apply_golden(case: dict[str, Any]) -> None:
    expected = dec(case["expected"])
    actual = replay(case)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_golden_meta_pins_reference() -> None:
    assert GOLDENS["_meta"]["reference_commit"].startswith("b78cec87")


# ------------------------------------------------- direct expectations


def _identity_plan(shape: tuple[int, ...], tile: tuple[int, ...]):
    return plan_tiles(
        shape,
        tile,
        overlap=2,
        scale=LinearScale(1),
    )


def test_identity_function_reproduces_input() -> None:
    # feather-weighted accumulation of overlapping identical tiles
    # must normalize back to the input exactly at every element
    x = torch.randn(1, 3, 20, 20)
    plan = _identity_plan((20, 20), (8, 8))
    assert not plan.single_tile
    out = tiled_apply(x, lambda a: a, plan, out_channels=3)
    torch.testing.assert_close(out, x, rtol=0, atol=1e-6)


def test_single_tile_calls_function_once_per_batch_item() -> None:
    x = torch.randn(3, 2, 4, 4)
    plan = plan_tiles((4, 4), (8, 8), overlap=2, scale=LinearScale(1))
    assert plan.single_tile
    calls = 0

    def counted() -> None:
        nonlocal calls
        calls += 1

    out = tiled_apply(x, lambda a: a * 2, plan, out_channels=2, on_tile=counted)
    torch.testing.assert_close(out, x * 2)
    assert calls == 3


def test_on_tile_fires_per_tile_per_batch_item() -> None:
    x = torch.randn(2, 1, 20)
    plan = plan_tiles((20,), (8,), overlap=2, scale=LinearScale(1))
    calls = 0

    def counted() -> None:
        nonlocal calls
        calls += 1

    tiled_apply(x, lambda a: a, plan, out_channels=1, on_tile=counted)
    assert calls == 2 * len(plan.tiles)


def test_output_dtype_and_device_arguments() -> None:
    x = torch.randn(1, 1, 20)
    plan = plan_tiles((20,), (8,), overlap=2, scale=LinearScale(1))
    out = tiled_apply(
        x,
        lambda a: a,
        plan,
        out_channels=1,
        dtype=torch.float64,
    )
    assert out.dtype == torch.float64
    assert out.device.type == "cpu"


def test_shape_mismatch_raises() -> None:
    x = torch.randn(1, 1, 16)
    plan = plan_tiles((20,), (8,), overlap=2, scale=LinearScale(1))
    with pytest.raises(TileApplyError, match="does not match"):
        tiled_apply(x, lambda a: a, plan, out_channels=1)


def test_uncovered_output_raises_instead_of_nan() -> None:
    # a function that returns less content than the scale rules
    # predict leaves accumulated weight at zero; the reference
    # divides 0/0 into NaNs, Dinkster refuses
    x = torch.randn(1, 1, 20)
    plan = plan_tiles((20,), (8,), overlap=2, scale=LinearScale(4))

    def too_small(a: torch.Tensor) -> torch.Tensor:
        return a.repeat_interleave(2, dim=-1)  # x2, plan expects x4

    with pytest.raises(TileApplyError, match="uncovered"):
        tiled_apply(x, too_small, plan, out_channels=1)


def test_gradients_flow_through_tiled_execution() -> None:
    # deliberate deviation from the reference's @torch.inference_mode:
    # the substrate must stay training-compatible
    x = torch.randn(1, 2, 12, 12, requires_grad=True)
    weight = torch.randn(2, 2, 1, 1, requires_grad=True)
    plan = plan_tiles((12, 12), (8, 8), overlap=4, scale=LinearScale(1))

    def conv(a: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(a, weight)

    out = tiled_apply(x, conv, plan, out_channels=2)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


def test_worker_thread_execution() -> None:
    # Dinkster's engine is async + threaded; the tiler must not depend
    # on main-thread state
    import threading

    case = GOLDENS["cases"][0]
    expected = dec(case["expected"])
    results: list[torch.Tensor] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(replay(case))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=60)
    assert not thread.is_alive()
    assert not errors
    torch.testing.assert_close(results[0], expected, rtol=0, atol=0)
