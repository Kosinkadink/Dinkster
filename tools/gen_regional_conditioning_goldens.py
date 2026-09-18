"""Generate S4-B2.1 regional goldens from pinned ComfyUI source.

Usage (the interpreter must provide torch):

    .venv-gpu/bin/python tools/gen_regional_conditioning_goldens.py \
        --comfy-git ../ComfyUI

The shared ComfyUI checkout need not move to the audited commit. The generator
reads ``comfy/samplers.py`` and ``comfy/conds.py`` with ``git show``, compiles the exact pinned
``get_mask_aabb``, ``resolve_areas_and_cond_masks_multidim``,
``add_area_dims``, ``get_area_and_mult``, ``cond_equal_size``, and
``can_concat_cond`` definitions, and executes them on CPU. The final one-list
group/pop loop pins the reference sufficient-memory/max-concat path: one fixed
output per region, every peer compatible with the first pending item selected,
and reverse order within that maximal group. The B2.3a extension records the
physical compatible order, cross-attention LCM factors, pinned reciprocal-floor result,
and accelerated-attention memory formula without importing live ComfyUI.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import logging
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from golden_platform import platform_golden_path, platform_provenance

REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent
BASE_OUT = REPO / "packages/dinkster-inference-torch/tests/goldens/regional_goldens.json"
OUT = platform_golden_path(
    BASE_OUT,
    torch.__version__,
)
GROUPED_OUT = REPO / "packages/dinkster-inference-torch/tests/goldens/grouped_regional_goldens.json"
FUNCTIONS = {
    "_calc_cond_batch",
    "add_area_dims",
    "can_concat_cond",
    "cond_cat",
    "cond_equal_size",
    "get_area_and_mult",
    "get_mask_aabb",
    "resolve_areas_and_cond_masks_multidim",
}
CLASSES = {"CONDRegular", "CONDCrossAttn"}


def _source(comfy_git: Path, path: str) -> str:
    resolved = subprocess.run(
        ["git", "rev-parse", REFERENCE_COMMIT],
        cwd=comfy_git,
        encoding="utf-8",
        capture_output=True,
        check=True,
    ).stdout.strip()
    if resolved != REFERENCE_COMMIT:
        raise SystemExit(f"reference resolved to {resolved}, expected {REFERENCE_COMMIT}")
    return subprocess.run(
        ["git", "show", f"{REFERENCE_COMMIT}:{path}"],
        cwd=comfy_git,
        encoding="utf-8",
        capture_output=True,
        check=True,
    ).stdout


def _common_upscale(
    samples: torch.Tensor, width: int, height: int, method: str, crop: str
) -> torch.Tensor:
    assert method == "bilinear" and crop == "none"
    return torch.nn.functional.interpolate(samples, size=(height, width), mode=method)


def _reference_functions(sampler_source: str, cond_source: str) -> dict[str, Any]:
    body: list[ast.stmt] = [
        node
        for source in (sampler_source, cond_source)
        for node in ast.parse(source).body
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in FUNCTIONS)
        or isinstance(node, ast.ClassDef)
        and node.name in CLASSES
    ]
    function_names = {
        node.name for node in body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    class_names = {node.name for node in body if isinstance(node, ast.ClassDef)}
    if function_names != FUNCTIONS or class_names != CLASSES:
        raise SystemExit("pinned source no longer contains the expected functions and classes")
    module = ast.Module(body=body, type_ignores=[])
    namespace: dict[str, Any] = {
        "BaseModel": object,
        "collections": collections,
        "logging": logging,
        "math": math,
        "torch": torch,
        "comfy": SimpleNamespace(
            utils=SimpleNamespace(
                common_upscale=_common_upscale,
                repeat_to_batch_size=lambda value, _batch: value,
            )
        ),
    }
    exec(compile(module, f"{REFERENCE_COMMIT}:regional-reference", "exec"), namespace)
    return namespace


def _reference_memory_function(model_source: str) -> Any:
    parsed = ast.parse(model_source)
    base = next(
        node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "BaseModel"
    )
    method = next(
        node
        for node in base.body
        if isinstance(node, ast.FunctionDef) and node.name == "memory_required"
    )
    pinned = ast.ClassDef(
        name="PinnedMemory",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    namespace: dict[str, Any] = {
        "comfy": SimpleNamespace(
            model_management=SimpleNamespace(
                xformers_enabled=lambda: True,
                pytorch_attention_flash_attention=lambda: False,
                dtype_size=lambda dtype: torch.empty((), dtype=dtype).element_size(),
            )
        ),
        "math": math,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[pinned], type_ignores=[])),
            f"{REFERENCE_COMMIT}:memory-reference",
            "exec",
        ),
        namespace,
    )
    return namespace["PinnedMemory"]


def _grouped_reference_facts(ref: dict[str, Any]) -> dict[str, object]:
    class Patcher:
        def __init__(self, free: float) -> None:
            self.free = free
            self.applied_hooks: list[int | None] = []

        def prepare_state(self, _timestep: torch.Tensor, _options: dict[str, object]) -> None:
            return None

        def prepare_hook_patches_current_keyframe(
            self, _timestep: torch.Tensor, _hooks: object, _options: dict[str, object]
        ) -> None:
            return None

        def get_free_memory(self, _device: torch.device) -> float:
            return self.free

        def apply_hooks(self, *, hooks: object | None) -> dict[str, object]:
            self.applied_hooks.append(None if hooks is None else id(hooks))
            return {}

    class Model:
        def __init__(self, free: float) -> None:
            self.current_patcher = Patcher(free)
            self.memory_attempts: list[int] = []
            self.forward_orders: list[list[str]] = []
            self.context_lengths: list[int] = []

        def memory_required(
            self, input_shape: list[int], *, cond_shapes: dict[str, list[torch.Size]]
        ) -> float:
            del cond_shapes
            self.memory_attempts.append(input_shape[0])
            return float(input_shape[0] * 100)

        def apply_model(
            self, input_x: torch.Tensor, _timestep: torch.Tensor, **conditioning: object
        ) -> torch.Tensor:
            options = conditioning["transformer_options"]
            assert isinstance(options, dict)
            uuids = options["uuids"]
            assert isinstance(uuids, list)
            order = [str(value) for value in uuids]
            self.forward_orders.append(order)
            context = conditioning["c_crossattn"]
            assert isinstance(context, torch.Tensor)
            self.context_lengths.append(context.shape[1])
            return torch.cat(
                [torch.full_like(input_x[:1], float(int(value) + 1)) for value in order]
            )

    def condition(uuid: str, tokens: int, hooks: object | None = None) -> dict[str, object]:
        value: dict[str, object] = {
            "uuid": uuid,
            "model_conds": {"c_crossattn": ref["CONDCrossAttn"](torch.zeros((1, tokens, 2)))},
        }
        if hooks is not None:
            value["hooks"] = hooks
        return value

    x = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    timestep = torch.ones(1, dtype=torch.float32)
    grouped = Model(10_000.0)
    grouped_outputs = ref["_calc_cond_batch"](
        grouped,
        [
            [condition("0", 3), condition("1", 6)],
            [condition("2", 3)],
        ],
        x,
        timestep,
        {},
    )
    boundary = Model(300.0)
    ref["_calc_cond_batch"](
        boundary,
        [[condition(str(index), 3) for index in range(4)]],
        x,
        timestep,
        {},
    )
    reciprocal_boundary = Model(400.0)
    ref["_calc_cond_batch"](
        reciprocal_boundary,
        [[condition(str(index), 3) for index in range(6)]],
        x,
        timestep,
        {},
    )
    hook_a = object()
    hook_b = object()
    serial = Model(10_000.0)
    ref["_calc_cond_batch"](
        serial,
        [[condition("0", 3, hook_a), condition("1", 3, hook_b)]],
        x,
        timestep,
        {},
    )
    return {
        "compatible_physical_order": grouped.forward_orders[0],
        "mixed_role_outputs": [float(value[0, 0, 0, 0]) for value in grouped_outputs],
        "forward_count": len(grouped.forward_orders),
        "lcm": grouped.context_lengths[0],
        "repeat_factors": [grouped.context_lengths[0] // value for value in (3, 6, 3)],
        "incompatible_repeat_factor": math.lcm(3, 15) // 3,
        "reciprocal_floor_counts_n4": boundary.memory_attempts[:3],
        "reciprocal_floor_boundary": reciprocal_boundary.memory_attempts[:3],
        "reciprocal_floor_first_order": reciprocal_boundary.forward_orders[0],
        "serial_hook_forward_orders": serial.forward_orders,
    }


def _tensor(value: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "data": value.reshape(-1).tolist(),
    }


def _condition(
    uuid: str,
    *,
    area: tuple[int, int, int, int] | None = None,
    mask: torch.Tensor | None = None,
    strength: float = 1.0,
    mask_strength: float = 1.0,
    timestep_start: float | None = None,
    tokens: int = 3,
    features: int = 4,
    pooled_features: int | None = None,
) -> dict[str, object]:
    value: dict[str, object] = {
        "uuid": uuid,
        "model_conds": {},
        "strength": strength,
        "reference_tokens": tokens,
        "reference_features": features,
        "reference_pooled_features": pooled_features,
    }
    if area is not None:
        value["area"] = area
    if mask is not None:
        value["mask"] = mask
        value["mask_strength"] = mask_strength
    if timestep_start is not None:
        value["timestep_start"] = timestep_start
    return value


def _projected_order(
    ref: dict[str, Any],
    conditions: tuple[dict[str, object], ...],
    x: torch.Tensor,
    timestep: torch.Tensor,
) -> list[Any]:
    pending = []
    for condition in conditions:
        projected = ref["get_area_and_mult"](condition, x, timestep)
        assert projected is not None
        batch = projected.input_x.shape[0]
        model_conds: dict[str, object] = {
            "c_crossattn": ref["CONDCrossAttn"](
                torch.zeros(
                    (
                        batch,
                        int(condition["reference_tokens"]),
                        int(condition["reference_features"]),
                    )
                )
            )
        }
        pooled = condition["reference_pooled_features"]
        if pooled is not None:
            model_conds["y"] = ref["CONDRegular"](torch.zeros((batch, int(pooled))))
        pending.append(projected._replace(conditioning=model_conds))

    ordered = []
    while pending:
        first = pending[0]
        compatible = [
            index for index, item in enumerate(pending) if ref["can_concat_cond"](item, first)
        ]
        compatible.reverse()
        for index in compatible:
            ordered.append(pending.pop(index))
    return ordered


def _accumulate(ordered: list[Any], x: torch.Tensor, values: dict[str, float]) -> torch.Tensor:
    output = torch.zeros_like(x)
    counts = torch.ones_like(x) * 1e-37
    for projected in ordered:
        value = torch.full_like(projected.input_x, values[projected.uuid])
        area = projected.area
        out_view = output
        count_view = counts
        if area is not None:
            out_view = out_view.narrow(2, area[2], area[0]).narrow(3, area[3], area[1])
            count_view = count_view.narrow(2, area[2], area[0]).narrow(3, area[3], area[1])
        out_view += value * projected.mult
        count_view += projected.mult
    output /= counts
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-git", type=Path, required=True)
    args = parser.parse_args()
    sampler_source = _source(args.comfy_git.resolve(), "comfy/samplers.py")
    cond_source = _source(args.comfy_git.resolve(), "comfy/conds.py")
    model_source = _source(args.comfy_git.resolve(), "comfy/model_base.py")
    if "area * comfy.model_management.dtype_size(dtype) * 0.01" not in model_source:
        raise SystemExit("pinned accelerated-attention memory formula is missing")
    ref = _reference_functions(sampler_source, cond_source)
    grouped_facts = _grouped_reference_facts(ref)
    memory_class = _reference_memory_function(model_source)
    memory_model = memory_class()
    memory_model.memory_usage_factor = 1.0
    memory_model.memory_usage_factor_conds = ()
    memory_model.memory_usage_shape_process = {}
    memory_model.get_dtype_inference = lambda: torch.float32
    grouped_facts["sd15_working_memory_bytes"] = memory_model.memory_required([3, 1, 12, 12])

    percentage = [{"area": ("percentage", 0.31, 0.49, 0.26, 0.51)}]
    ref["resolve_areas_and_cond_masks_multidim"](percentage, [7, 9], torch.device("cpu"))

    resize_mask = torch.tensor([[[0.0, 1.0], [0.5, 0.25]]], dtype=torch.float32)
    resize_case = [{"mask": resize_mask}]
    ref["resolve_areas_and_cond_masks_multidim"](resize_case, [3, 5], torch.device("cpu"))

    aabb_mask = torch.zeros((2, 12, 13), dtype=torch.float32)
    aabb_mask[0, 4:6, 7:10] = -1.0
    aabb_case = [{"mask": aabb_mask, "set_area_to_bounds": True}]
    ref["resolve_areas_and_cond_masks_multidim"](aabb_case, [12, 13], torch.device("cpu"))
    zero_case = [{"mask": torch.zeros((1, 3, 4), dtype=torch.float32), "set_area_to_bounds": True}]
    ref["resolve_areas_and_cond_masks_multidim"](zero_case, [3, 4], torch.device("cpu"))

    x = torch.zeros((1, 1, 12, 12), dtype=torch.float32)
    timestep = torch.tensor([0.5], dtype=torch.float32)
    no_mask = ref["get_area_and_mult"](
        _condition("area", area=(8, 8, 2, 2), strength=0.75), x, timestep
    )
    small_flush = ref["get_area_and_mult"](
        _condition("small-flush", area=(3, 4, 0, 8), strength=0.5), x, timestep
    )
    mask = torch.linspace(0.0, 1.0, 144, dtype=torch.float32).reshape(1, 12, 12)
    with_mask = ref["get_area_and_mult"](
        _condition(
            "mask",
            area=(6, 7, 3, 2),
            mask=mask,
            strength=0.8,
            mask_strength=0.5,
        ),
        x,
        timestep,
    )

    overlap_conditions = (
        _condition("first", area=(8, 8, 0, 0)),
        _condition("second", area=(8, 8, 4, 4)),
    )
    overlap_order = _projected_order(ref, overlap_conditions, x, timestep)
    output = _accumulate(overlap_order, x, {"first": 1.0, "second": 3.0})

    ones = torch.ones((1, 12, 12), dtype=torch.float32)
    heterogeneous_conditions = (
        _condition("heterogeneous-0", area=(10, 10, 0, 0), mask=ones),
        _condition("heterogeneous-1", area=(8, 8, 1, 1), mask=ones),
        _condition("heterogeneous-2", area=(6, 6, 2, 2), mask=ones),
    )
    heterogeneous_order = _projected_order(ref, heterogeneous_conditions, x, timestep)
    heterogeneous_output = _accumulate(
        heterogeneous_order,
        x,
        {"heterogeneous-0": 1.0, "heterogeneous-1": 2.0, "heterogeneous-2": 4.0},
    )
    mixed_conditions = (
        _condition("A0", mask=ones, tokens=3),
        _condition("B", mask=ones, tokens=15),
        _condition("A1", mask=ones, tokens=6),
    )
    mixed_order = _projected_order(ref, mixed_conditions, x, timestep)
    mixed_output = _accumulate(mixed_order, x, {"A0": -1e20, "B": 3.0, "A1": 1e20})
    inactive = ref["get_area_and_mult"](_condition("inactive", timestep_start=0.25), x, timestep)
    payload = {
        "reference": {
            "repo": "ComfyUI",
            "commit": REFERENCE_COMMIT,
            "paths": ["comfy/samplers.py", "comfy/conds.py"],
            "source_sha256": {
                "comfy/samplers.py": hashlib.sha256(sampler_source.encode()).hexdigest(),
                "comfy/conds.py": hashlib.sha256(cond_source.encode()).hexdigest(),
            },
            "device": "cpu",
            **platform_provenance(torch.__version__),
        },
        "percent_area": list(percentage[0]["area"]),
        "mask_resize": _tensor(resize_case[0]["mask"]),
        "mask_aabb_area": list(aabb_case[0]["area"]),
        "mask_aabb_zero_area": list(zero_case[0]["area"]),
        "no_mask_multiplier": _tensor(no_mask.mult),
        "small_flush_multiplier": _tensor(small_flush.mult),
        "mask_multiplier": _tensor(with_mask.mult),
        "overlap": {"order": [item.uuid for item in overlap_order], "output": _tensor(output)},
        "heterogeneous_groups": {
            "order": [item.uuid for item in heterogeneous_order],
            "output": _tensor(heterogeneous_output),
        },
        "mixed_groups": {
            "order": [item.uuid for item in mixed_order],
            "output": _tensor(mixed_output),
        },
        "inactive_is_none": inactive is None,
        "uncovered_is_zero": bool(torch.all(output[:, :, :4, 8:] == 0)),
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", newline="\n")
    grouped_payload = {
        "reference": {
            "repo": "ComfyUI",
            "commit": REFERENCE_COMMIT,
            "paths": ["comfy/samplers.py", "comfy/conds.py", "comfy/model_base.py"],
            "source_sha256": {
                "comfy/samplers.py": hashlib.sha256(sampler_source.encode()).hexdigest(),
                "comfy/conds.py": hashlib.sha256(cond_source.encode()).hexdigest(),
                "comfy/model_base.py": hashlib.sha256(model_source.encode()).hexdigest(),
            },
            "device": "cpu",
            **platform_provenance(torch.__version__),
        },
        "facts": grouped_facts,
    }
    if OUT == BASE_OUT:
        GROUPED_OUT.write_text(
            json.dumps(grouped_payload, indent=2, sort_keys=True) + "\n", newline="\n"
        )
    print(OUT)
    if OUT == BASE_OUT:
        print(GROUPED_OUT)


if __name__ == "__main__":
    main()
