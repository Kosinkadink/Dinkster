"""Generate conditioning goldens by executing pinned ComfyUI (b78cec87).

Runs CPU-only helpers from node_helpers.py, nodes.py, comfy/hooks.py, and
comfy/samplers.py, plus the conditioning value nodes (combine, average,
concat, multiply, zero-out) on small deterministic tensors whose raw bytes
are recorded for bit-exact replay. The source citations governing each
behavior are recorded in the output. Run with a torch interpreter inside a
clean checkout of the pinned ComfyUI commit:

    COMFYUI_ROOT=/path/to/ComfyUI \
      /path/to/ComfyUI/venv/bin/python tools/gen_conditioning_goldens.py

Tensor bytes are recorded little-endian; the generator refuses big-endian
hosts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

PINNED_COMFYUI = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "conditioning_goldens.json"
COMFYUI = Path(os.environ.get("COMFYUI_ROOT", "/home/kosin/ComfyUI")).resolve()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(COMFYUI), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_checkout() -> None:
    head = _git("rev-parse", "HEAD")
    if head != PINNED_COMFYUI:
        raise RuntimeError(f"ComfyUI HEAD {head} != pinned {PINNED_COMFYUI}")
    if _git("status", "--porcelain"):
        raise RuntimeError("ComfyUI reference checkout must be clean")


_validate_checkout()
sys.path.insert(0, str(COMFYUI))

from comfy.cli_args import args as _comfy_args  # noqa: E402

# Golden math executes on CPU tensors; forcing CPU keeps the import working
# on hosts whose torch build has no CUDA support.
_comfy_args.cpu = True

import comfy.hooks  # noqa: E402
import comfy.model_sampling  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy_extras.nodes_video_model  # noqa: E402
import node_helpers  # noqa: E402
import nodes  # noqa: E402
import torch  # noqa: E402


def _validate_import_roots() -> None:
    for module in (
        comfy.hooks,
        comfy.model_sampling,
        comfy.samplers,
        comfy_extras.nodes_video_model,
        node_helpers,
        nodes,
    ):
        module_path = Path(module.__file__).resolve()
        if not module_path.is_relative_to(COMFYUI):
            raise RuntimeError(f"imported {module.__name__} from outside {COMFYUI}")


_validate_import_roots()

if sys.byteorder != "little":
    raise RuntimeError("golden tensor bytes are recorded little-endian")


def _metadata(value: list[object]) -> dict[str, object]:
    return dict(value[1])


def _active(metadata: dict[str, object], sigma: float) -> bool:
    prepared = {**metadata, "model_conds": {}, "uuid": "synthetic"}
    result = comfy.samplers.get_area_and_mult(
        prepared,
        torch.zeros((1, 1, 1, 1)),
        torch.tensor([sigma]),
    )
    return result is not None


def _tensor_record(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "data_hex": tensor.contiguous().numpy().tobytes().hex(),
    }


def _ramp(shape: tuple[int, ...], dtype: torch.dtype, start: float, step: float) -> torch.Tensor:
    count = 1
    for dim in shape:
        count *= dim
    values = torch.arange(count, dtype=torch.float64) * step + start
    return values.to(dtype).reshape(shape)


def _payload_math_cases() -> dict[str, object]:
    average = nodes.ConditioningAverage().addWeighted
    concat = nodes.ConditioningConcat().concat
    multiply = nodes.ConditioningMultiply().multiply
    zero_out = nodes.ConditioningZeroOut().zero_out

    def average_case(
        dtype: torch.dtype,
        to_tokens: int,
        from_tokens: int,
        strength: float,
        *,
        with_to_pooled: bool,
    ) -> dict[str, object]:
        features = 4
        to_text = _ramp((1, to_tokens, features), dtype, -1.5, 0.37)
        from_text = _ramp((1, from_tokens, features), dtype, 2.0, -0.61)
        from_pooled = _ramp((1, features), dtype, 0.25, 1.13)
        to_metadata: dict[str, object] = {}
        inputs: dict[str, object] = {
            "to_text": _tensor_record(to_text),
            "from_text": _tensor_record(from_text),
            "from_pooled": _tensor_record(from_pooled),
        }
        if with_to_pooled:
            to_pooled = _ramp((1, features), dtype, -0.8, 0.29)
            to_metadata["pooled_output"] = to_pooled
            inputs["to_pooled"] = _tensor_record(to_pooled)
        out = average(
            [[to_text, to_metadata]],
            [[from_text, {"pooled_output": from_pooled}]],
            strength,
        )[0]
        return {
            "strength": strength,
            "inputs": inputs,
            "outputs": {
                "text": _tensor_record(out[0][0]),
                "pooled": _tensor_record(out[0][1]["pooled_output"]),
            },
        }

    def concat_case(dtype: torch.dtype) -> dict[str, object]:
        to_text = _ramp((1, 2, 3), dtype, 0.1, 0.77)
        from_text = _ramp((1, 4, 3), dtype, -2.3, 0.41)
        out = concat([[to_text, {}]], [[from_text, {}]])[0]
        return {
            "inputs": {
                "to_text": _tensor_record(to_text),
                "from_text": _tensor_record(from_text),
            },
            "outputs": {"text": _tensor_record(out[0][0])},
        }

    def multiply_case(dtype: torch.dtype, multiplier: float) -> dict[str, object]:
        text = _ramp((1, 2, 3), dtype, -1.1, 0.53)
        pooled = _ramp((1, 3), dtype, 0.7, -0.89)
        out = multiply([[text, {"pooled_output": pooled}]], multiplier)[0]
        return {
            "multiplier": multiplier,
            "inputs": {"text": _tensor_record(text), "pooled": _tensor_record(pooled)},
            "outputs": {
                "text": _tensor_record(out[0][0]),
                "pooled": _tensor_record(out[0][1]["pooled_output"]),
            },
        }

    def zero_out_case() -> dict[str, object]:
        text = _ramp((1, 2, 3), torch.float32, 0.9, 1.31)
        pooled = _ramp((1, 3), torch.float32, -0.4, 0.57)
        scale = _ramp((2,), torch.float32, 0.5, 0.25)
        out = zero_out([[text, {"pooled_output": pooled, "conditioning_scale": scale}]])[0]
        return {
            "inputs": {
                "text": _tensor_record(text),
                "pooled": _tensor_record(pooled),
                "conditioning_scale": _tensor_record(scale),
            },
            "outputs": {
                "text": _tensor_record(out[0][0]),
                "pooled": _tensor_record(out[0][1]["pooled_output"]),
                "conditioning_scale": _tensor_record(out[0][1]["conditioning_scale"]),
            },
        }

    return {
        "average_truncate_f32": average_case(torch.float32, 3, 5, 0.37, with_to_pooled=True),
        "average_pad_pooled_inherit_f32": average_case(
            torch.float32, 5, 3, 0.62, with_to_pooled=False
        ),
        "average_truncate_f16": average_case(torch.float16, 3, 5, 0.37, with_to_pooled=True),
        "average_pad_pooled_inherit_f16": average_case(
            torch.float16, 4, 2, 0.37, with_to_pooled=False
        ),
        "concat_f32": concat_case(torch.float32),
        "concat_f16": concat_case(torch.float16),
        "multiply_f32": multiply_case(torch.float32, -1.7),
        "multiply_f16": multiply_case(torch.float16, 0.37),
        "zero_out_f32": zero_out_case(),
    }


def main() -> None:
    payload_a = object()
    payload_b = object()
    nested = ["kept-by-reference"]
    base = [[payload_a, {"nested": nested, "old": 1}]]
    cloned = node_helpers.conditioning_set_values(base, {"old": 2})

    combined = nodes.ConditioningCombine().combine(
        [[payload_a, {"id": "left"}]], [[payload_b, {"id": "right"}]]
    )[0]

    area_cells = nodes.ConditioningSetArea().append(
        [[payload_a, {}]], width=80, height=64, x=24, y=16, strength=1.25
    )[0]
    area_percent = nodes.ConditioningSetAreaPercentage().append(
        [[payload_a, {}]],
        width=0.5,
        height=0.5,
        x=0.5,
        y=0.5,
        strength=0.75,
    )[0]
    area_percent_video = comfy_extras.nodes_video_model.ConditioningSetAreaPercentageVideo().append(
        [[payload_a, {}]],
        width=0.5,
        height=0.25,
        temporal=0.75,
        x=0.125,
        y=0.0,
        z=0.25,
        strength=0.6,
    )[0]
    area_percent_video_tiny = (
        comfy_extras.nodes_video_model.ConditioningSetAreaPercentageVideo().append(
            [[payload_a, {}]],
            width=0.01,
            height=0.01,
            temporal=0.01,
            x=0.125,
            y=0.0,
            z=0.25,
            strength=0.6,
        )[0]
    )
    resolved_percent = [_metadata(area_percent[0])]
    comfy.samplers.resolve_areas_and_cond_masks_multidim(
        resolved_percent, [7, 9], torch.device("cpu")
    )
    resolved_percent_video = [_metadata(area_percent_video[0])]
    comfy.samplers.resolve_areas_and_cond_masks_multidim(
        resolved_percent_video, [8, 12, 16], torch.device("cpu")
    )
    resolved_percent_video_tiny = [_metadata(area_percent_video_tiny[0])]
    comfy.samplers.resolve_areas_and_cond_masks_multidim(
        resolved_percent_video_tiny, [8, 12, 16], torch.device("cpu")
    )
    video_area_and_mult = comfy.samplers.get_area_and_mult(
        {**resolved_percent_video[0], "model_conds": {}, "uuid": "video-area-golden"},
        torch.zeros((1, 1, 8, 12, 16)),
        torch.tensor([0.5]),
    )
    assert video_area_and_mult is not None

    mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    mask_default = nodes.ConditioningSetMask().append([[payload_a, {}]], mask, "default", 0.6)[0]
    mask_bounds = nodes.ConditioningSetMask().append([[payload_a, {}]], mask, "mask bounds", 0.6)[0]

    range_node = nodes.ConditioningSetTimestepRange().set_range([[payload_a, {}]], 0.5, 0.5)[0]
    range_hook = comfy.hooks.set_timesteps_for_conditioning([[payload_a, {}]], (0.5, 0.5))
    flow = comfy.model_sampling.ModelSamplingDiscreteFlow()
    model = SimpleNamespace(model_sampling=flow)
    range_cases: dict[str, object] = {}
    for name, value in (("node", range_node), ("hook", range_hook)):
        prepared = [_metadata(value[0])]
        comfy.samplers.calculate_start_end_timesteps(model, prepared)
        sigma = float(flow.percent_to_sigma(0.5))
        range_cases[name] = {
            "metadata": prepared[0],
            "probes": [
                {"sigma": sigma + 1e-6, "active": _active(prepared[0], sigma + 1e-6)},
                {"sigma": sigma, "active": _active(prepared[0], sigma)},
                {"sigma": sigma - 1e-6, "active": _active(prepared[0], sigma - 1e-6)},
            ],
        }

    spaces = {
        "discrete": comfy.model_sampling.ModelSamplingDiscrete(),
        "flow": flow,
        "flux": comfy.model_sampling.ModelSamplingFlux(),
        "continuous_edm": comfy.model_sampling.ModelSamplingContinuousEDM(),
    }
    percents = (0.0, 0.25, 0.5, 0.75, 1.0)
    sigma_goldens = {
        name: [float(space.percent_to_sigma(percent)) for percent in percents]
        for name, space in spaces.items()
    }

    document = {
        "comfyui_commit": PINNED_COMFYUI,
        "citations": {
            "clone": "node_helpers.py:8-22",
            "combine": "nodes.py:82-93",
            "area": (
                "nodes.py:163-205; comfy_extras/nodes_video_model.py:126-153; "
                "comfy/samplers.py:33-116,760-786"
            ),
            "mask": "nodes.py:223-248",
            "range": "nodes.py:275-290; comfy/hooks.py:713-717; comfy/samplers.py:37-44,816-839",
            "sigma": "comfy/model_sampling.py:176-182,225-233,279-284,372-377",
            "payload_math": "nodes.py:80-184,272-299",
        },
        "clone": {
            "source_unchanged": base[0][1]["old"] == 1,
            "metadata_dict_copied": cloned[0][1] is not base[0][1],
            "payload_shared": cloned[0][0] is payload_a,
            "nested_metadata_shared": cloned[0][1]["nested"] is nested,
        },
        "combine_order": [entry[1]["id"] for entry in combined],
        "area": {
            "latent_cells": list(_metadata(area_cells[0])["area"]),
            "percent_stored": list(_metadata(area_percent[0])["area"]),
            "percent_video_stored": list(_metadata(area_percent_video[0])["area"]),
            "percent_resolved_7x9": list(resolved_percent[0]["area"]),
            "percent_video_resolved_8x12x16": list(resolved_percent_video[0]["area"]),
            "percent_video_tiny_resolved_8x12x16": list(resolved_percent_video_tiny[0]["area"]),
            "percent_video_get_area_and_mult": {
                "area": list(video_area_and_mult.area),
                "input_shape": list(video_area_and_mult.input_x.shape),
                "multiplier": _tensor_record(video_area_and_mult.mult),
            },
        },
        "mask": {
            "default": {
                "shape": list(_metadata(mask_default[0])["mask"].shape),
                "strength": _metadata(mask_default[0])["mask_strength"],
                "set_area_to_bounds": _metadata(mask_default[0])["set_area_to_bounds"],
            },
            "bounds": {
                "shape": list(_metadata(mask_bounds[0])["mask"].shape),
                "strength": _metadata(mask_bounds[0])["mask_strength"],
                "set_area_to_bounds": _metadata(mask_bounds[0])["set_area_to_bounds"],
            },
        },
        "zero_width_range": range_cases,
        "percent_probes": list(percents),
        "sigma_goldens": sigma_goldens,
        "payload_math": _payload_math_cases(),
    }
    OUT.write_text(
        json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
