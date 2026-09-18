"""Generate tiled-scale goldens from the ComfyUI reference.

Runs the REFERENCE comfy/utils.py tiled_scale_multidim @ the audited
baseline and writes packages/dinkster-inference-torch/tests/goldens/
tiling_goldens.json. dinkster_inference.tiling.plan_tiles +
dinkster_inference_torch.tiling.tiled_apply are pinned against these
outputs - the oracle is the reference code itself, never a
re-derivation. Positions, edge clamping, feather math, and
accumulation all fold into the output tensor, so output equality pins
the whole pipeline.

Codec stand-ins are named deterministic functions (FUNCTIONS below,
duplicated by name in test_tiling.py - the same registry-by-name
convention as gen_patch_goldens.py): pure indexing/pooling/affine
torch ops, no RNG at apply time, so replays are torch-build-stable.
Input tensors are stored IN the golden file.

Scale rules are recorded as data ({"kind": "linear"|"causal",
"factor": f}) and expanded here into the reference's argument shapes:
linear -> plain numbers, causal -> the video lambda pair
(max(0, a*f-(f-1)) decode / max(0, floor((a+f-1)/f)) encode) with a
numeric index formula, exactly the forms comfy/sd.py passes
@ 947c2749.

Usage (needs a torch interpreter that imports the pinned checkout;
the workspace root venv is deliberately torch-free):

    PYTHONPATH=../ComfyUI /path/to/torch-venv/bin/python tools/gen_tiling_goldens.py

Run from the Dinkster repo root with ../ComfyUI at the recorded baseline
commit (recorded in the output for provenance).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import types
from pathlib import Path

import comfy_aimdo
import torch

# Same venv-skew stubs as gen_patch_goldens.py: modules imported at
# scope by comfy.memory_management but never called here.
for _sub in ("host_buffer", "vram_buffer"):
    _name = f"comfy_aimdo.{_sub}"
    try:
        __import__(_name)
    except ModuleNotFoundError:
        _stub = types.ModuleType(_name)
        setattr(comfy_aimdo, _sub, _stub)
        sys.modules[_name] = _stub

import comfy.utils  # noqa: E402
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = platform_golden_path(
    REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "tiling_goldens.json",
    torch.__version__,
)

_gen = torch.Generator().manual_seed(0x71E5)


def t(*shape: int) -> torch.Tensor:
    return torch.randn(shape, generator=_gen, dtype=torch.float32)


def enc(x: torch.Tensor) -> dict:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).removeprefix("torch."),
        "data": x.to(torch.float32).flatten().tolist(),
    }


# Named pinned codec stand-ins shared with the Dinkster replay side
# (test_tiling.py FUNCTIONS). Deterministic, shape-driven, defined for
# any tile geometry the plans produce.


def up2_mix3(x: torch.Tensor) -> torch.Tensor:
    """2D decode stand-in: spatial x2, channels -> 3."""
    up = x.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    m = up.mean(1, keepdim=True)
    return torch.cat([m * 0.5, m, m * 1.5], dim=1) + 0.25


def down8_mix4(x: torch.Tensor) -> torch.Tensor:
    """2D encode stand-in: spatial -> round(size/8), channels -> 4."""
    h = round(x.shape[-2] / 8)
    w = round(x.shape[-1] / 8)
    pooled = torch.nn.functional.adaptive_avg_pool2d(x, (h, w))
    m = pooled.mean(1, keepdim=True)
    return torch.cat([m, m * -1.0, m * 2.0, m * 0.5], dim=1)


def up4_1d(x: torch.Tensor) -> torch.Tensor:
    """1D (audio-style) decode stand-in: length x4, channels kept."""
    return x.repeat_interleave(4, dim=-1) * 0.5 + 0.1


def causal_up_video(x: torch.Tensor) -> torch.Tensor:
    """3D causal decode stand-in: t -> 4t-3, spatial x2, channels -> 3."""
    frames = x.repeat_interleave(4, dim=2)[:, :, 3:]
    up = frames.repeat_interleave(2, dim=-2).repeat_interleave(2, dim=-1)
    m = up.mean(1, keepdim=True)
    return torch.cat([m, m * 0.5, m * -0.25], dim=1)


def causal_down_video(x: torch.Tensor) -> torch.Tensor:
    """3D causal encode stand-in: t -> ceil(t/4), spatial ->
    round(size/8), channels -> 4."""
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


FUNCTIONS = {
    "up2_mix3": up2_mix3,
    "down8_mix4": down8_mix4,
    "up4_1d": up4_1d,
    "causal_up_video": causal_up_video,
    "causal_down_video": causal_down_video,
    "up2_tilenorm": up2_tilenorm,
    "down4_tilenorm": down4_tilenorm,
}


def reference_scale_args(scales: list[dict], downscale: bool):
    """Expand recorded scale data into the reference's
    upscale_amount / index_formulas entries (comfy/sd.py forms)."""
    amounts = []
    indexes = []
    for spec in scales:
        f = spec["factor"]
        if spec["kind"] == "linear":
            amounts.append(f)
        elif not downscale:
            amounts.append(lambda a, f=f: max(0, a * f - (f - 1)))
        else:
            amounts.append(lambda a, f=f: max(0, math.floor((a + f - 1) / f)))
        indexes.append(f)
    return amounts, indexes


# name, input shape, tile, overlap, scales, downscale, out_channels,
# function name
CASES = [
    # non-divisible 2D upscale: edge clamping in both dims
    (
        "up2_2d",
        (1, 2, 21, 17),
        (8, 8),
        (4, 4),
        [{"kind": "linear", "factor": 2}] * 2,
        False,
        3,
        "up2_mix3",
    ),
    # per-batch loop
    (
        "up2_2d_batch2",
        (2, 2, 13, 11),
        (8, 8),
        (4, 4),
        [{"kind": "linear", "factor": 2}] * 2,
        False,
        3,
        "up2_mix3",
    ),
    # whole input fits one tile: the fast path
    (
        "single_tile",
        (1, 2, 6, 6),
        (8, 8),
        (4, 4),
        [{"kind": "linear", "factor": 2}] * 2,
        False,
        3,
        "up2_mix3",
    ),
    # one dim fits its tile ([0] positions branch), the other tiles
    (
        "one_dim_tiled",
        (1, 2, 6, 20),
        (8, 8),
        (4, 4),
        [{"kind": "linear", "factor": 2}] * 2,
        False,
        3,
        "up2_mix3",
    ),
    # 1D content (audio-shaped)
    ("up4_1d", (1, 2, 300), (128,), (32,), [{"kind": "linear", "factor": 4}], False, 2, "up4_1d"),
    # encode direction: numeric scales divide (downscale=True)
    (
        "down8_2d",
        (1, 2, 56, 40),
        (32, 32),
        (8, 8),
        [{"kind": "linear", "factor": 8}] * 2,
        True,
        4,
        "down8_mix4",
    ),
    # 3D causal decode: lambda size rule + numeric index formula
    (
        "causal_up_video",
        (1, 2, 4, 6, 6),
        (3, 4, 4),
        (1, 2, 2),
        [
            {"kind": "causal", "factor": 4},
            {"kind": "linear", "factor": 2},
            {"kind": "linear", "factor": 2},
        ],
        False,
        3,
        "causal_up_video",
    ),
    # 3D causal encode; time size 12 makes the last time tile 4
    # frames -> 1 output frame == feather, hitting the reference's
    # feather >= mask dim skip branch
    (
        "causal_down_video",
        (1, 1, 12, 24, 24),
        (5, 16, 16),
        (1, 8, 8),
        [
            {"kind": "causal", "factor": 4},
            {"kind": "linear", "factor": 8},
            {"kind": "linear", "factor": 8},
        ],
        True,
        4,
        "causal_down_video",
    ),
]


# The reference's 2D three-aspect seam sweep (comfy/sd.py
# decode_tiled_/encode_tiled_ @ 947c2749), executed via
# comfy.utils.tiled_scale exactly as sd.py composes it.
# tiled_scale(samples, fn, tile_x, tile_y) maps to
# tile=(tile_y, tile_x): with (t0, t1) = (tile_y, tile_x), decode
# accumulates (2t0, t1/2) + (t0/2, 2t1) + (t0, t1), encode
# accumulates (t0, t1) += (t0/2, 2t1) += (2t0, t1/2), both / 3.
# name, input shape, (tile_x, tile_y), overlap, factor, decode?,
# out_channels, function name
SWEEP_CASES = [
    ("sweep_decode_2d", (1, 2, 13, 11), (6, 8), 2, 2, True, 3, "up2_tilenorm"),
    # every variant's positions/lengths stay multiples of the factor
    # so no pass hits the reference's rounding-gap NaN case
    ("sweep_encode_2d", (1, 3, 48, 40), (16, 16), 4, 4, False, 4, "down4_tilenorm"),
]


def reference_sweep(
    samples: torch.Tensor,
    fn,
    tile_x: int,
    tile_y: int,
    overlap: int,
    factor: int,
    out_c: int,
    *,
    decode: bool,
) -> torch.Tensor:
    ts = comfy.utils.tiled_scale
    if decode:
        return (
            ts(
                samples,
                fn,
                tile_x // 2,
                tile_y * 2,
                overlap,
                upscale_amount=factor,
                out_channels=out_c,
            )
            + ts(
                samples,
                fn,
                tile_x * 2,
                tile_y // 2,
                overlap,
                upscale_amount=factor,
                out_channels=out_c,
            )
            + ts(samples, fn, tile_x, tile_y, overlap, upscale_amount=factor, out_channels=out_c)
        ) / 3.0
    out = ts(samples, fn, tile_x, tile_y, overlap, upscale_amount=(1 / factor), out_channels=out_c)
    out += ts(
        samples,
        fn,
        tile_x * 2,
        tile_y // 2,
        overlap,
        upscale_amount=(1 / factor),
        out_channels=out_c,
    )
    out += ts(
        samples,
        fn,
        tile_x // 2,
        tile_y * 2,
        overlap,
        upscale_amount=(1 / factor),
        out_channels=out_c,
    )
    out /= 3.0
    return out


def main() -> None:
    cases = []
    for name, in_shape, tile, overlap, scales, downscale, out_c, fn_name in CASES:
        samples = t(*in_shape)
        amounts, indexes = reference_scale_args(scales, downscale)
        expected = comfy.utils.tiled_scale_multidim(
            samples,
            FUNCTIONS[fn_name],
            tile=tile,
            overlap=list(overlap),
            upscale_amount=amounts,
            out_channels=out_c,
            output_device="cpu",
            downscale=downscale,
            index_formulas=indexes,
        )
        if not bool(torch.isfinite(expected).all()):
            raise RuntimeError(f"case {name}: reference output not finite")
        cases.append(
            {
                "name": name,
                "samples": enc(samples),
                "tile": list(tile),
                "overlap": list(overlap),
                "scales": scales,
                "downscale": downscale,
                "out_channels": out_c,
                "function": fn_name,
                "expected": enc(expected),
            }
        )

    sweep_cases = []
    for name, in_shape, xy, overlap, factor, decode, out_c, fn_name in SWEEP_CASES:
        samples = t(*in_shape)
        tile_x, tile_y = xy
        # the reference's += composition runs under ComfyUI's global
        # inference-mode execution context (tiled_scale_multidim is
        # @torch.inference_mode(), so its output only accepts in-place
        # updates inside one)
        with torch.inference_mode():
            expected = reference_sweep(
                samples,
                FUNCTIONS[fn_name],
                tile_x,
                tile_y,
                overlap,
                factor,
                out_c,
                decode=decode,
            )
        if not bool(torch.isfinite(expected).all()):
            raise RuntimeError(f"case {name}: reference output not finite")
        sweep_cases.append(
            {
                "name": name,
                "samples": enc(samples),
                # plugin geometry: tile dims are (dim0, dim1) =
                # (tile_y, tile_x), overlap per-dim
                "tile": [tile_y, tile_x],
                "overlap": [overlap, overlap],
                "factor": factor,
                "decode": decode,
                "out_channels": out_c,
                "function": fn_name,
                "expected": enc(expected),
            }
        )

    commit = subprocess.run(
        ["git", "-C", str(REPO.parent / "ComfyUI"), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    OUT.write_bytes(
        (
            json.dumps(
                {
                    "_meta": {
                        "reference_commit": commit,
                        "torch": torch.__version__,
                        "generator": "tools/gen_tiling_goldens.py",
                        **tuple_provenance(torch.__version__),
                    },
                    "cases": cases,
                    "sweep_cases": sweep_cases,
                },
                indent=1,
            )
            + "\n"
        ).encode()
    )
    print(f"wrote {len(cases)} cases + {len(sweep_cases)} sweep cases to {OUT}")


if __name__ == "__main__":
    main()
