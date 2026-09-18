"""Eager scaled-patch execution proofs."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import gc
import json
import struct
import sys
import threading
import weakref
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from dinkster_inference import (
    FLOAT32,
    AdapterPatch,
    ComponentPlan,
    DiffPatch,
    LinearToConv2D,
    ModelAsLoraPatch,
    NestedPatch,
    PatchEntry,
    PatchOffset,
    PatchSet,
    SetPatch,
)
from dinkster_inference_torch import (
    CastOperations,
    LoHaAdapter,
    LoRAAdapter,
    ScaledPatchError,
    enroll_component,
    scaled_patch_context,
)
from dinkster_inference_torch import scaled_patches as scaled_module
from dinkster_inference_torch.apply import apply_patches, patch_stored_weight
from golden_files import load_platform_golden

GOLDENS = load_platform_golden(Path(__file__).parent / "goldens/scaled_patch_goldens.json")


def _tensor(record: dict[str, object], *, device: torch.device | None = None) -> torch.Tensor:
    dtype = getattr(torch, cast(str, record["dtype"]))
    return torch.tensor(record["data"], dtype=dtype, device=device).reshape(
        cast("list[int]", record["shape"])
    )


def _linear(
    *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
) -> torch.nn.Linear:
    layer = torch.nn.Linear(3, 2, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [[0.25, -0.5, 0.75], [-1.0, 0.5, 0.125]],
                device=device,
                dtype=dtype,
            )
        )
    return layer


def _conv(
    *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32
) -> torch.nn.Conv2d:
    layer = torch.nn.Conv2d(
        2,
        3,
        3,
        stride=2,
        padding=2,
        dilation=2,
        bias=False,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        layer.weight.copy_(
            torch.arange(layer.weight.numel(), device=device, dtype=dtype).reshape(
                layer.weight.shape
            )
            / 37
        )
    return layer


def _diff_patch(
    diff: torch.Tensor | None = None, *, strength: float = 1.0
) -> PatchSet[torch.Tensor]:
    if diff is None:
        diff = torch.tensor([[-0.125, 0.375, 0.5], [0.75, -0.25, 0.625]], dtype=torch.float32)
    return PatchSet({"weight": (PatchEntry(DiffPatch(diff), strength=strength),)})


def _scaled_entries(patch: PatchSet[torch.Tensor], scale: float, key: str = "weight"):
    return tuple(replace(entry, strength=entry.strength * scale) for entry in patch.entries(key))


def _ordinary_linear(
    layer: torch.nn.Linear,
    value: torch.Tensor,
    patch: PatchSet[torch.Tensor],
    scales: torch.Tensor,
    key: str = "weight",
) -> torch.Tensor:
    values = scales.tolist()
    if len(values) == 1:
        weight = patch_stored_weight(layer.weight, _scaled_entries(patch, values[0], key), key=key)
        return F.linear(value, weight.to(value.dtype), layer.bias)
    return torch.cat(
        [
            F.linear(
                value[index : index + 1],
                patch_stored_weight(layer.weight, _scaled_entries(patch, scale, key), key=key).to(
                    value.dtype
                ),
                layer.bias,
            )
            for index, scale in enumerate(values)
        ]
    )


def _ordinary_conv(
    layer: torch.nn.Conv2d,
    value: torch.Tensor,
    patch: PatchSet[torch.Tensor],
    scales: torch.Tensor,
) -> torch.Tensor:
    values = scales.tolist()
    return torch.cat(
        [
            F.conv2d(
                value if len(values) == 1 else value[index : index + 1],
                patch_stored_weight(layer.weight, _scaled_entries(patch, scale), key="weight").to(
                    value.dtype
                ),
                None if layer.bias is None else layer.bias.to(value.dtype),
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
            )
            for index, scale in enumerate(values)
        ]
    )


def _bytes(value: torch.Tensor) -> list[int]:
    return value.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist()


def _write_float32_safetensors(path: Path, key: str, value: torch.Tensor) -> None:
    value = value.to(torch.float32).contiguous()
    payload = bytes(value.untyped_storage())[: value.numel() * value.element_size()]
    header = json.dumps(
        {
            key: {
                "dtype": "F32",
                "shape": list(value.shape),
                "data_offsets": [0, len(payload)],
            }
        }
    ).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def _assert_code(code: str, call: Any) -> None:
    with pytest.raises(ScaledPatchError) as raised:
        call()
    assert raised.value.code == code
    assert str(raised.value) == f"scaled-patch:{code}"


@pytest.mark.parametrize(
    ("shape", "scales"),
    [
        ((3, 3), [0.0, 1.0, -0.5]),
        ((3, 4, 3), [0.0, 1.0, -0.5]),
        ((3, 2, 2, 3), [0.0, 1.0, -0.5]),
        ((3, 2, 3), [0.25]),
    ],
)
def test_scaled_diff_zero_unit_heterogeneous_and_trailing_broadcast(
    shape: tuple[int, ...], scales: list[float]
) -> None:
    layer = _linear()
    value = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape) / 7
    diff = torch.tensor([[-0.25, 0.5, 0.75], [1.0, -0.125, 0.25]])
    scale = torch.tensor(scales)
    baseline_weight = _bytes(layer.weight)
    baseline = layer(value)

    patch = _diff_patch(diff, strength=0.6)
    expected = _ordinary_linear(layer, value, patch, scale)
    with scaled_patch_context(layer, patch, scale, lambda: False):
        actual = layer(value)
        assert torch.equal(actual, expected)

    assert _bytes(layer.weight) == baseline_weight
    assert torch.equal(layer(value), baseline)
    if scales[0] == 0.0:
        # The zero-scale lane composes the unpatched weight bit-exactly, but
        # its output comes from a single-lane GEMM while the baseline slices
        # a full-batch GEMM; on macOS Accelerate those 2D dispatches disagree
        # by up to one float32 ULP (observed: one element of the 2D case; the
        # batched 3D/4D dispatches match bit-exactly).
        if sys.platform == "darwin" and len(shape) == 2:
            different = actual[0] != baseline[0]
            assert torch.count_nonzero(different).item() <= 1
            if torch.any(different):
                assert torch.equal(
                    torch.nextafter(baseline[0][different], actual[0][different]),
                    actual[0][different],
                )
        else:
            assert torch.equal(actual[0], baseline[0])


def test_pinned_lora_and_full_diff_goldens_alpha_rank_and_strength() -> None:
    inputs = GOLDENS["inputs"]
    layer = _linear()
    value = _tensor(inputs["input"])
    up = _tensor(inputs["up"])
    down = _tensor(inputs["down"])
    alpha = cast(float, inputs["alpha"])
    multiplier = cast(float, inputs["multiplier"])
    lora = PatchSet(
        {
            "weight": (
                PatchEntry(
                    AdapterPatch(LoRAAdapter(up, down, alpha=alpha)),
                    strength=multiplier,
                ),
            )
        }
    )
    expected = _ordinary_linear(layer, value, lora, torch.ones(1))
    with scaled_patch_context(layer, lora, torch.ones(1), lambda: False):
        assert torch.equal(layer(value), expected)

    diff = _tensor(inputs["diff"])
    scales = _tensor(inputs["scales"])
    strength = cast(float, inputs["diff_strength"])
    patch = _diff_patch(diff, strength=strength)
    expected = _ordinary_linear(layer, value, patch, scales)
    with scaled_patch_context(layer, patch, scales, lambda: False):
        assert torch.equal(layer(value), expected)


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_constant_linear_lora_matches_ordinary_materialization(strength: float) -> None:
    layer = _linear(dtype=torch.float16)
    value = torch.tensor([[1.0, -2.0, 0.5], [-0.25, 0.75, 2.0]], dtype=torch.float16)
    up = torch.tensor([[0.75], [-0.125]], dtype=torch.float32)
    down = torch.tensor([[0.5, -0.25, 0.375]], dtype=torch.float32)
    patch = PatchSet({"weight": (PatchEntry(AdapterPatch(LoRAAdapter(up, down)), strength=1.0),)})
    scale = torch.tensor([strength])
    expected = _ordinary_linear(layer, value, patch, scale)

    with scaled_patch_context(layer, patch, scale, lambda: False):
        assert torch.equal(layer(value), expected)


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_constant_convolution_lora_matches_ordinary_materialization(strength: float) -> None:
    layer = _conv(dtype=torch.float16)
    value = torch.arange(2 * 2 * 9 * 8, dtype=torch.float16).reshape(2, 2, 9, 8) / 29
    up = torch.arange(6, dtype=torch.float32).reshape(3, 2, 1, 1) / 17
    down = torch.arange(36, dtype=torch.float32).reshape(2, 2, 3, 3) / 23
    patch = PatchSet({"weight": (PatchEntry(AdapterPatch(LoRAAdapter(up, down)), strength=1.0),)})
    scale = torch.tensor([strength])
    expected = _ordinary_conv(layer, value, patch, scale)

    with scaled_patch_context(layer, patch, scale, lambda: False):
        assert torch.equal(layer(value), expected)


def _assert_lora_cross_dtype_parity(
    device: torch.device,
    kind: str,
    model_dtype: torch.dtype,
    source_dtype: torch.dtype,
) -> None:
    if kind == "linear":
        layer = _linear(device=device, dtype=model_dtype)
        value = torch.tensor(
            [[1.0, -2.0, 0.5], [-0.25, 0.75, 2.0]],
            device=device,
            dtype=model_dtype,
        )
        up = (torch.arange(4, dtype=torch.float32).reshape(2, 2) / 17).to(source_dtype)
        down = (torch.arange(6, dtype=torch.float32).reshape(2, 3) / 23).to(source_dtype)
    else:
        layer = _conv(device=device, dtype=model_dtype)
        value = (
            torch.arange(2 * 2 * 9 * 8, device=device, dtype=model_dtype).reshape(2, 2, 9, 8) / 29
        )
        up = (torch.arange(6, dtype=torch.float32).reshape(3, 2, 1, 1) / 17).to(source_dtype)
        down = (torch.arange(36, dtype=torch.float32).reshape(2, 2, 3, 3) / 23).to(source_dtype)
    patch = PatchSet({"weight": (PatchEntry(AdapterPatch(LoRAAdapter(up, down)), strength=1.0),)})
    weight_before = _bytes(layer.weight)
    up_float = up.to(device=device, dtype=torch.float32)
    down_float = down.to(device=device, dtype=torch.float32)
    lora_diff = torch.mm(up_float.flatten(start_dim=1), down_float.flatten(start_dim=1)).reshape(
        layer.weight.shape
    )
    if source_dtype is not torch.float32:
        low_precision_diff = torch.mm(
            up.to(device=device).flatten(start_dim=1),
            down.to(device=device).flatten(start_dim=1),
        ).reshape(layer.weight.shape)
        assert not torch.equal(low_precision_diff.to(torch.float32), lora_diff)

    for strength in (0.0, 0.5, 1.0):
        scale = torch.tensor([strength])
        ordinary = (
            _ordinary_linear(layer, value, patch, scale)
            if isinstance(layer, torch.nn.Linear)
            else _ordinary_conv(layer, value, patch, scale)
        )
        reference_weight = (layer.weight.detach().to(torch.float32) + strength * lora_diff).to(
            model_dtype
        )
        reference = (
            F.linear(value, reference_weight)
            if isinstance(layer, torch.nn.Linear)
            else F.conv2d(
                value,
                reference_weight,
                None,
                layer.stride,
                layer.padding,
                layer.dilation,
                layer.groups,
            )
        )
        assert torch.equal(ordinary, reference)
        with scaled_patch_context(layer, patch, scale, lambda: False):
            assert torch.equal(layer(value), reference)
        assert _bytes(layer.weight) == weight_before
    assert up.dtype is source_dtype and down.dtype is source_dtype


@pytest.mark.parametrize("kind", ("linear", "conv"))
@pytest.mark.parametrize("model_dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("source_dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_ordinary_and_scheduled_lora_cross_dtype_matrix_on_cpu(
    kind: str,
    model_dtype: torch.dtype,
    source_dtype: torch.dtype,
) -> None:
    _assert_lora_cross_dtype_parity(torch.device("cpu"), kind, model_dtype, source_dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA capability required")
@pytest.mark.parametrize("kind", ("linear", "conv"))
@pytest.mark.parametrize("model_dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("source_dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_ordinary_and_scheduled_lora_cross_dtype_matrix_on_cuda(
    kind: str,
    model_dtype: torch.dtype,
    source_dtype: torch.dtype,
) -> None:
    _assert_lora_cross_dtype_parity(torch.device("cuda:0"), kind, model_dtype, source_dtype)
    torch.cuda.synchronize()


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_converted_linear_uses_checkpoint_source_authority(tmp_path: Path, strength: float) -> None:
    source = torch.tensor([[0.1234567, -0.3333333, 0.8765432], [0.6543211, -0.1111111, 0.2222222]])
    path = tmp_path / "linear.safetensors"
    _write_float32_safetensors(path, "source.weight", source)
    layer = torch.nn.Linear(3, 2, bias=False, dtype=torch.float16)
    layer.weight = torch.nn.Parameter(source.to(torch.float16))
    layer.__dict__["_dinkster_component_plan"] = ComponentPlan(
        component="diffusion",
        path=path,
        config=None,
        keys={"weight": "source.weight"},
        dtypes={"weight": FLOAT32},
        quant={},
    )
    layer.__dict__["_dinkster_storage_converted"] = True
    value = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float16)
    up = torch.tensor([[0.75], [-0.125]])
    down = torch.tensor([[0.5, -0.25, 0.375]])
    patch = PatchSet({"weight": (PatchEntry(AdapterPatch(LoRAAdapter(up, down)), strength=1.0),)})
    entries = _scaled_entries(patch, strength)
    expected_weight = apply_patches(
        source.to(torch.float32, copy=True),
        entries,
        key="weight",
        intermediate_dtype=torch.float32,
        original_weight=source,
    ).to(torch.float16)

    with scaled_patch_context(layer, patch, torch.tensor([strength]), lambda: False):
        assert torch.equal(layer(value), F.linear(value, expected_weight))


@pytest.mark.parametrize("scheduled_scale", [0.0, 0.5, 1.0])
def test_converted_linear_composes_base_and_scheduled_loras(
    tmp_path: Path, scheduled_scale: float
) -> None:
    source = torch.tensor([[0.1234567, -0.3333333, 0.8765432], [0.6543211, -0.1111111, 0.2222222]])
    path = tmp_path / "stacked-linear.safetensors"
    _write_float32_safetensors(path, "source.weight", source)
    base = PatchSet(
        {
            "weight": (
                PatchEntry(
                    AdapterPatch(
                        LoRAAdapter(
                            torch.tensor([[0.75], [-0.125]]),
                            torch.tensor([[0.5, -0.25, 0.375]]),
                        )
                    ),
                    strength=0.4,
                ),
            )
        }
    )
    scheduled = PatchSet(
        {
            "weight": (
                PatchEntry(
                    AdapterPatch(
                        LoRAAdapter(
                            torch.tensor([[-0.25], [0.625]]),
                            torch.tensor([[0.125, 0.5, -0.75]]),
                        )
                    ),
                    strength=0.7,
                ),
            )
        }
    )
    base_weight = apply_patches(
        source.to(torch.float32, copy=True),
        base.entries("weight"),
        key="weight",
        intermediate_dtype=torch.float32,
        original_weight=source,
    ).to(torch.float16)
    layer = torch.nn.Linear(3, 2, bias=False, dtype=torch.float16)
    layer.weight = torch.nn.Parameter(base_weight)
    layer.__dict__["_dinkster_component_plan"] = ComponentPlan(
        component="diffusion",
        path=path,
        config=None,
        keys={"weight": "source.weight"},
        dtypes={"weight": FLOAT32},
        quant={},
    )
    layer.__dict__["_dinkster_storage_converted"] = True
    layer.__dict__["_dinkster_base_patch_set"] = base
    expected_entries = (
        *base.entries("weight"),
        *_scaled_entries(scheduled, scheduled_scale),
    )
    expected_weight = apply_patches(
        source.to(torch.float32, copy=True),
        expected_entries,
        key="weight",
        intermediate_dtype=torch.float32,
        original_weight=source,
    ).to(torch.float16)
    value = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float16)

    with scaled_patch_context(layer, scheduled, torch.tensor([scheduled_scale]), lambda: False):
        assert torch.equal(layer(value), F.linear(value, expected_weight))


@pytest.mark.parametrize("scheduled_scale", [0.0, 0.5, 1.0])
def test_unconverted_linear_does_not_reapply_base_lora(scheduled_scale: float) -> None:
    source = torch.tensor([[0.1234567, -0.3333333, 0.8765432], [0.6543211, -0.1111111, 0.2222222]])
    base = PatchSet(
        {
            "weight": (
                PatchEntry(
                    AdapterPatch(
                        LoRAAdapter(
                            torch.tensor([[0.75], [-0.125]]),
                            torch.tensor([[0.5, -0.25, 0.375]]),
                        )
                    ),
                    strength=0.4,
                ),
            )
        }
    )
    scheduled = PatchSet(
        {
            "weight": (
                PatchEntry(
                    AdapterPatch(
                        LoRAAdapter(
                            torch.tensor([[-0.25], [0.625]]),
                            torch.tensor([[0.125, 0.5, -0.75]]),
                        )
                    ),
                    strength=0.7,
                ),
            )
        }
    )
    base_weight = apply_patches(
        source.to(torch.float32, copy=True),
        base.entries("weight"),
        key="weight",
        intermediate_dtype=torch.float32,
    )
    layer = torch.nn.Linear(3, 2, bias=False)
    layer.weight = torch.nn.Parameter(base_weight)
    layer.__dict__["_dinkster_storage_converted"] = False
    layer.__dict__["_dinkster_base_patch_set"] = None
    expected_weight = apply_patches(
        base_weight.to(torch.float32, copy=True),
        _scaled_entries(scheduled, scheduled_scale),
        key="weight",
        intermediate_dtype=torch.float32,
    )
    value = torch.tensor([[1.0, -2.0, 0.5]])

    with scaled_patch_context(layer, scheduled, torch.tensor([scheduled_scale]), lambda: False):
        assert torch.equal(layer(value), F.linear(value, expected_weight))


def test_converted_convolution_applies_planned_source_transform(tmp_path: Path) -> None:
    source = torch.tensor(
        [[0.1234567, -0.3333333], [0.8765432, 0.6543211], [-0.1111111, 0.2222222]]
    )
    path = tmp_path / "conv.safetensors"
    _write_float32_safetensors(path, "source.weight", source)
    layer = torch.nn.Conv2d(2, 3, 1, bias=False, dtype=torch.float16)
    layer.weight = torch.nn.Parameter(source.reshape(3, 2, 1, 1).to(torch.float16))
    layer.__dict__["_dinkster_component_plan"] = ComponentPlan(
        component="diffusion",
        path=path,
        config=None,
        keys={"weight": "source.weight"},
        dtypes={"weight": FLOAT32},
        quant={},
        transforms={"weight": LinearToConv2D()},
    )
    layer.__dict__["_dinkster_storage_converted"] = True
    value = torch.arange(2 * 2 * 3 * 4, dtype=torch.float16).reshape(2, 2, 3, 4) / 13
    up = torch.tensor([[[[0.75]]], [[[-0.125]]], [[[0.5]]]])
    down = torch.tensor([[[[0.5]], [[-0.25]]]])
    patch = PatchSet({"weight": (PatchEntry(AdapterPatch(LoRAAdapter(up, down)), strength=0.5),)})
    original = source.reshape(3, 2, 1, 1)
    expected_weight = apply_patches(
        original.to(torch.float32, copy=True),
        patch.entries("weight"),
        key="weight",
        intermediate_dtype=torch.float32,
        original_weight=original,
    ).to(torch.float16)

    with scaled_patch_context(layer, patch, torch.ones(1), lambda: False):
        assert torch.equal(layer(value), F.conv2d(value, expected_weight))


def test_convolution_lora_strength_scales_each_batch_row() -> None:
    layer = _conv()
    value = torch.arange(2 * 2 * 9 * 8, dtype=torch.float32).reshape(2, 2, 9, 8) / 29
    up = torch.tensor(
        [
            [[[0.25]], [[-0.5]]],
            [[[0.75]], [[0.125]]],
            [[[-0.25]], [[1.0]]],
        ]
    )
    down = torch.arange(2 * 2 * 3 * 3, dtype=torch.float32).reshape(2, 2, 3, 3) / 41
    patch = PatchSet(
        {
            "weight": (
                PatchEntry(
                    AdapterPatch(LoRAAdapter(up, down, alpha=1.0)),
                    strength=0.6,
                ),
            )
        }
    )
    scale = torch.tensor([0.0, 1.25])
    expected = _ordinary_conv(layer, value, patch, scale)
    baseline_weight = _bytes(layer.weight)

    with scaled_patch_context(layer, patch, scale, lambda: False):
        assert torch.equal(layer(value), expected)

    assert _bytes(layer.weight) == baseline_weight
    assert not layer._forward_hooks


def test_cast_convolution_lora_uses_compute_dtype_through_offloaded_residency() -> None:
    model = CastOperations(torch.float32).conv2d(2, 3, 3, padding=1)
    model.weight = torch.nn.Parameter(
        torch.arange(model.weight.numel(), dtype=torch.float16).reshape(model.weight.shape) / 31
    )
    model.bias = None
    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    mechanism.partially_unload(mechanism.loaded_bytes())
    up = torch.arange(6, dtype=torch.float16).reshape(3, 2, 1, 1) / 17
    down = torch.arange(36, dtype=torch.float16).reshape(2, 2, 3, 3) / 23
    patch = PatchSet({"weight": (PatchEntry(AdapterPatch(LoRAAdapter(up, down)), strength=0.5),)})
    value = torch.arange(2 * 2 * 5 * 4, dtype=torch.float32).reshape(2, 2, 5, 4) / 19
    ordinary = CastOperations(torch.float32).conv2d(2, 3, 3, padding=1)
    ordinary.weight = torch.nn.Parameter(model.weight.detach().clone())
    ordinary.bias = None
    ordinary_mechanism = enroll_component(
        ordinary,
        load_device="cpu",
        offload_device="cpu",
        patch_set=patch,
    )
    ordinary_mechanism.partially_load(None)
    ordinary_mechanism.partially_unload(ordinary_mechanism.loaded_bytes())
    expected = ordinary(value)

    with scaled_patch_context(model, patch, torch.ones(1), lambda: False):
        assert torch.equal(model(value), expected)

    assert mechanism.loaded_bytes() == 0
    assert model.weight.dtype is torch.float16
    assert not model._forward_hooks


def test_multiple_entries_and_targets_preserve_declared_addition_order() -> None:
    class Pair(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = _linear()
            self.second = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.second.weight.copy_(torch.tensor([[0.25, -0.5], [-1.0, 0.125]]))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.second(self.first(value))

    model = Pair()
    value = torch.tensor([[1e10, 1.0, -1e10], [-2.0, 3.0, 4.0]])
    first_a = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    first_b = torch.tensor([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    second = torch.tensor([[0.25, -0.5], [0.75, 0.125]])
    patches = PatchSet(
        {
            "first.weight": (
                PatchEntry(DiffPatch(first_a)),
                PatchEntry(DiffPatch(first_b)),
            ),
            "second.weight": (PatchEntry(DiffPatch(second)),),
        }
    )
    scale = torch.tensor([1.0, -0.5])
    first_expected = _ordinary_linear(model.first, value, patches, scale, "first.weight")
    expected = _ordinary_linear(model.second, first_expected, patches, scale, "second.weight")
    with scaled_patch_context(model, patches, scale, lambda: False):
        assert torch.equal(model(value), expected)


def test_one_shot_wrapper_preserves_per_target_compute_placement() -> None:
    class MixedDtype(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.single = torch.nn.Linear(1, 1, bias=False, dtype=torch.float32)
            self.double_layer = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
            with torch.no_grad():
                self.single.weight.fill_(2.0)
                self.double_layer.weight.fill_(3.0)

        def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.single(value), self.double_layer(value.to(torch.float64))

    model = MixedDtype()
    patches = PatchSet(
        {
            "single.weight": (PatchEntry(DiffPatch(torch.ones((1, 1)))),),
            "double_layer.weight": (PatchEntry(DiffPatch(torch.ones((1, 1)))),),
        }
    )
    value = torch.ones((1, 1))
    with scaled_patch_context(model, patches, torch.tensor([2.0]), lambda: False):
        single, double = model(value)
        assert torch.equal(single, torch.tensor([[4.0]]))
        assert torch.equal(double, torch.tensor([[5.0]], dtype=torch.float64))
    assert not model.single._forward_hooks
    assert not model.double_layer._forward_hooks


def test_one_shot_castlinear_uses_compute_dtype_through_offloaded_residency() -> None:
    model = CastOperations(torch.float32).linear(1, 1, bias=False)
    model.weight = torch.nn.Parameter(torch.tensor([[2.0]], dtype=torch.float16))
    mechanism = enroll_component(model, load_device="cpu", offload_device="cpu")
    mechanism.partially_load(None)
    mechanism.partially_unload(mechanism.loaded_bytes())
    assert mechanism.loaded_bytes() == 0
    patch = PatchSet({"weight": (PatchEntry(DiffPatch(torch.ones((1, 1), dtype=torch.float16))),)})
    value = torch.ones((1, 1), dtype=torch.float32)
    with scaled_patch_context(model, patch, torch.tensor([2.0]), lambda: False):
        output = model(value)
        assert output.dtype is torch.float32
        assert torch.equal(output, torch.tensor([[4.0]]))
    assert mechanism.loaded_bytes() == 0
    assert model.weight.dtype is torch.float16
    assert not model._forward_hooks


@pytest.mark.parametrize("kind", ("diff", "lora-up", "lora-down"))
@pytest.mark.parametrize(
    ("execution_dtype", "source_dtype", "overflow_value"),
    (
        (torch.float16, torch.float32, 1e10),
        (torch.bfloat16, torch.float64, 1e300),
    ),
)
def test_finite_payload_overflow_refuses_after_staging_without_hooks(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    execution_dtype: torch.dtype,
    source_dtype: torch.dtype,
    overflow_value: float,
) -> None:
    layer = _linear(dtype=execution_dtype)
    overflow = torch.tensor(overflow_value, dtype=source_dtype)
    assert torch.isfinite(overflow)
    if kind == "diff":
        patch = _diff_patch(overflow.expand(2, 3).clone())
    else:
        up = torch.ones((2, 1), dtype=source_dtype)
        down = torch.ones((1, 3), dtype=source_dtype)
        if kind == "lora-up":
            up = overflow.expand(2, 1).clone()
        else:
            down = overflow.expand(1, 3).clone()
        patch = PatchSet({"weight": (PatchEntry(AdapterPatch(LoRAAdapter(up, down))),)})
    staged_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    original_finite = scaled_module._require_finite

    def observe(value: torch.Tensor, code: str) -> None:
        if code == "staged-payload-finite":
            staged_refs.append(weakref.ref(value))
        original_finite(value, code)

    monkeypatch.setattr(scaled_module, "_require_finite", observe)
    with pytest.raises(ScaledPatchError, match="staged-payload-finite"):
        scaled_patch_context(layer, patch, torch.ones(1), lambda: False).__enter__()
    assert not layer._forward_hooks and layer not in scaled_module._ACTIVE_MODELS
    with pytest.raises(ScaledPatchError, match="staged-payload-finite"):
        scaled_module.prepare_scaled_patches(layer, patch, "cpu", execution_dtype, lambda: False)
    assert not layer._forward_hooks and layer not in scaled_module._ACTIVE_MODELS
    gc.collect()
    assert staged_refs and all(reference() is None for reference in staged_refs)


@pytest.mark.parametrize(
    ("execution_dtype", "source_dtype", "overflow_value"),
    (
        (torch.float16, torch.float32, 1e10),
        (torch.bfloat16, torch.float64, 1e300),
    ),
)
def test_finite_scale_overflow_refuses_after_staging_without_hooks(
    monkeypatch: pytest.MonkeyPatch,
    execution_dtype: torch.dtype,
    source_dtype: torch.dtype,
    overflow_value: float,
) -> None:
    layer = _linear(dtype=execution_dtype)
    patch = _diff_patch(torch.ones((2, 3)))
    scale = torch.tensor([overflow_value], dtype=source_dtype)
    assert torch.isfinite(scale).all()
    staged_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    original_finite = scaled_module._require_finite

    def observe(value: torch.Tensor, code: str) -> None:
        if code == "staged-scale-finite":
            staged_refs.append(weakref.ref(value))
        original_finite(value, code)

    monkeypatch.setattr(scaled_module, "_require_finite", observe)
    with pytest.raises(ScaledPatchError, match="staged-scale-finite"):
        scaled_patch_context(layer, patch, scale, lambda: False).__enter__()
    assert not layer._forward_hooks and layer not in scaled_module._ACTIVE_MODELS
    prepared = scaled_module.prepare_scaled_patches(
        layer, patch, "cpu", execution_dtype, lambda: False
    )
    with pytest.raises(ScaledPatchError, match="staged-scale-finite"):
        prepared.activate(scale).__enter__()
    assert not prepared._active
    assert not layer._forward_hooks and layer not in scaled_module._ACTIVE_MODELS
    prepared.close()
    gc.collect()
    assert staged_refs and all(reference() is None for reference in staged_refs)


def test_mixed_target_scale_overflow_refuses_before_guard_or_hook_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MixedDtype(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = _linear(dtype=torch.float64)
            self.second = _linear(dtype=torch.float16)

    model = MixedDtype()
    patch = PatchSet(
        {
            "first.weight": (PatchEntry(DiffPatch(torch.ones((2, 3)))),),
            "second.weight": (PatchEntry(DiffPatch(torch.ones((2, 3)))),),
        }
    )
    scale = torch.tensor([1e300], dtype=torch.float64)
    assert torch.isfinite(scale).all()
    registrations = 0
    original_register = torch.nn.Linear.register_forward_hook

    def observe_register(layer: torch.nn.Linear, *args: Any, **kwargs: Any):
        nonlocal registrations
        registrations += 1
        return original_register(layer, *args, **kwargs)

    monkeypatch.setattr(torch.nn.Linear, "register_forward_hook", observe_register)
    with pytest.raises(ScaledPatchError, match="staged-scale-finite"):
        scaled_patch_context(model, patch, scale, lambda: False).__enter__()
    assert registrations == 0
    assert not model.first._forward_hooks and not model.second._forward_hooks
    assert model not in scaled_module._ACTIVE_MODELS


@pytest.mark.parametrize("kind", ("diff", "lora-up", "lora-down"))
def test_prepared_payload_version_drift_refuses_before_activation(kind: str) -> None:
    layer = _linear()
    if kind == "diff":
        payload = torch.ones((2, 3))
        patch = _diff_patch(payload)
    else:
        up = torch.ones((2, 1))
        down = torch.ones((1, 3))
        payload = up if kind == "lora-up" else down
        patch = PatchSet({"weight": (PatchEntry(AdapterPatch(LoRAAdapter(up, down))),)})
    prepared = scaled_module.prepare_scaled_patches(
        layer, patch, "cpu", torch.float32, lambda: False
    )
    payload.add_(0.25)
    with pytest.raises(ScaledPatchError, match="payload-drift"):
        prepared.activate(torch.ones(1)).__enter__()
    assert not layer._forward_hooks and layer not in scaled_module._ACTIVE_MODELS
    prepared.close()


def test_prepared_convolution_geometry_drift_refuses_before_activation() -> None:
    layer = _conv()
    up = torch.ones((3, 1, 1, 1))
    down = torch.ones((1, 2, 3, 3))
    patch = PatchSet({"weight": (PatchEntry(AdapterPatch(LoRAAdapter(up, down))),)})
    prepared = scaled_module.prepare_scaled_patches(
        layer, patch, "cpu", torch.float32, lambda: False
    )
    layer.stride = (1, 1)

    with pytest.raises(ScaledPatchError, match="conv-geometry-drift"):
        prepared.activate(torch.ones(1)).__enter__()

    assert not layer._forward_hooks and layer not in scaled_module._ACTIVE_MODELS
    prepared.close()


def test_inference_payload_without_version_counter_refuses_deterministically() -> None:
    layer = _linear()
    with torch.inference_mode():
        payload = torch.ones((2, 3))
    with pytest.raises(ScaledPatchError, match="tensor-version"):
        scaled_module.prepare_scaled_patches(
            layer, _diff_patch(payload), "cpu", torch.float32, lambda: False
        )
    assert not layer._forward_hooks and layer not in scaled_module._ACTIVE_MODELS


def test_scale_zero_and_all_exit_paths_restore_byte_identical_baseline() -> None:
    layer = _linear()
    value = torch.randn(2, 3)
    state_before = {key: _bytes(tensor) for key, tensor in layer.state_dict().items()}
    output_before = layer(value).clone()
    patch = _diff_patch()
    revision = patch.revision
    payload = cast(DiffPatch[torch.Tensor], patch.entries("weight")[0].value).value
    payload_before = _bytes(payload)
    with scaled_patch_context(layer, patch, torch.zeros(1), lambda: False):
        assert torch.equal(layer(value), output_before)
    with pytest.raises(RuntimeError, match="body"):
        with scaled_patch_context(layer, patch, torch.ones(1), lambda: False):
            raise RuntimeError("body")
    assert {key: _bytes(tensor) for key, tensor in layer.state_dict().items()} == state_before
    assert torch.equal(layer(value), output_before)
    assert patch.revision == revision
    assert _bytes(payload) == payload_before
    assert not layer._forward_hooks


@pytest.mark.parametrize(
    ("patch", "code"),
    [
        (PatchSet({}), "empty-patch-set"),
        (PatchSet({"weight": ()}), "empty-target"),
        (PatchSet({"bias": (PatchEntry(DiffPatch(torch.ones(2, 3))),)}), "target-name"),
        (
            PatchSet({"missing.weight": (PatchEntry(DiffPatch(torch.ones(2, 3))),)}),
            "target-resolution",
        ),
        (PatchSet({"weight": (PatchEntry(SetPatch(torch.ones(2, 3))),)}), "patch-kind"),
        (
            PatchSet({"weight": (PatchEntry(ModelAsLoraPatch(torch.ones(2, 3))),)}),
            "patch-kind",
        ),
        (
            PatchSet({"weight": (PatchEntry(NestedPatch(torch.ones(2, 3), entries=())),)}),
            "patch-kind",
        ),
        (
            PatchSet({"weight": (PatchEntry(DiffPatch(torch.ones(2, 3), pad_weight=True)),)}),
            "diff-pad",
        ),
        (
            PatchSet(
                {"weight": (PatchEntry(DiffPatch(torch.ones(2, 3), pad_weight=cast(bool, 0))),)}
            ),
            "diff-pad",
        ),
        (PatchSet({"weight": (PatchEntry(DiffPatch(torch.ones(2, 2))),)}), "diff-shape"),
        (
            PatchSet({"weight": (PatchEntry(DiffPatch(torch.ones(2, 3)), strength=float("nan")),)}),
            "strength",
        ),
        (
            PatchSet({"weight": (PatchEntry(DiffPatch(torch.full((2, 3), float("inf")))),)}),
            "diff-finite",
        ),
        (
            PatchSet({"weight": (PatchEntry(DiffPatch(torch.ones(2, 3)), strength_model=0.5),)}),
            "strength-model",
        ),
        (
            PatchSet(
                {"weight": (PatchEntry(DiffPatch(torch.ones(2, 3)), offset=PatchOffset(0, 0, 1)),)}
            ),
            "offset",
        ),
        (
            PatchSet(
                {"weight": (PatchEntry(DiffPatch(torch.ones(2, 3)), function=lambda value: value),)}
            ),
            "function",
        ),
    ],
)
def test_strict_patch_refusal_matrix_precedes_hook_side_effects(
    patch: PatchSet[torch.Tensor], code: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    layer = _linear()
    called = False
    original = layer.register_forward_hook

    def register(*args: Any, **kwargs: Any):
        nonlocal called
        called = True
        return original(*args, **kwargs)

    monkeypatch.setattr(layer, "register_forward_hook", register)
    _assert_code(
        code,
        lambda: scaled_patch_context(layer, patch, torch.ones(1), lambda: False).__enter__(),
    )
    assert not called
    assert not layer._forward_hooks


@pytest.mark.parametrize(
    ("adapter", "code"),
    [
        (LoRAAdapter(torch.ones(2, 1), torch.ones(1, 3), mid=torch.ones(1, 1, 1, 1)), "lora-mid"),
        (LoRAAdapter(torch.ones(2, 1), torch.ones(1, 3), dora_scale=torch.ones(2)), "lora-dora"),
        (LoRAAdapter(torch.ones(2, 1), torch.ones(1, 3), reshape=(2, 3)), "lora-reshape"),
        (LoRAAdapter(torch.ones(2, 0), torch.ones(0, 3)), "lora-rank"),
        (LoRAAdapter(torch.ones(2, 2), torch.ones(1, 3)), "lora-shape"),
        (LoRAAdapter(torch.ones(2, 1), torch.ones(1, 3), alpha=float("inf")), "lora-alpha"),
        (
            LoRAAdapter(torch.full((2, 1), float("nan")), torch.ones(1, 3)),
            "lora-finite",
        ),
        (LoRAAdapter(torch.ones(2, 1, dtype=torch.int64), torch.ones(1, 3)), "lora-tensor"),
    ],
)
def test_lora_refusal_matrix(adapter: LoRAAdapter, code: str) -> None:
    patch = PatchSet({"weight": (PatchEntry(AdapterPatch(adapter)),)})
    _assert_code(
        code,
        lambda: scaled_patch_context(_linear(), patch, torch.ones(1), lambda: False).__enter__(),
    )


def test_convolution_lora_refuses_unsupported_geometry_before_hooks() -> None:
    grouped = torch.nn.Conv2d(2, 2, 1, groups=2, bias=False)
    grouped_patch = PatchSet(
        {
            "weight": (
                PatchEntry(
                    AdapterPatch(LoRAAdapter(torch.ones(2, 1, 1, 1), torch.ones(1, 1, 1, 1)))
                ),
            )
        }
    )
    _assert_code(
        "conv-groups",
        lambda: scaled_patch_context(
            grouped, grouped_patch, torch.ones(1), lambda: False
        ).__enter__(),
    )

    reflected = torch.nn.Conv2d(2, 3, 3, padding=1, padding_mode="reflect", bias=False)
    reflected_patch = PatchSet(
        {
            "weight": (
                PatchEntry(
                    AdapterPatch(LoRAAdapter(torch.ones(3, 1, 1, 1), torch.ones(1, 2, 3, 3)))
                ),
            )
        }
    )
    _assert_code(
        "conv-padding-mode",
        lambda: scaled_patch_context(
            reflected, reflected_patch, torch.ones(1), lambda: False
        ).__enter__(),
    )
    assert not grouped._forward_hooks and not reflected._forward_hooks


@pytest.mark.parametrize(
    "adapter",
    (
        LoRAAdapter(torch.ones(3, 2, 2, 1), torch.ones(2, 2, 3, 3)),
        LoRAAdapter(torch.ones(3, 2, 1, 1), torch.ones(2, 2, 1, 1)),
    ),
)
def test_convolution_lora_refuses_non_1x1_up_or_wrong_down_kernel(
    adapter: LoRAAdapter,
) -> None:
    conv = torch.nn.Conv2d(2, 3, 3, padding=1, bias=False)
    patch = PatchSet({"weight": (PatchEntry(AdapterPatch(adapter)),)})
    _assert_code(
        "lora-shape",
        lambda: scaled_patch_context(conv, patch, torch.ones(1), lambda: False).__enter__(),
    )
    assert not conv._forward_hooks


def test_other_adapter_custom_module_convolution_diff_alias_and_existing_hooks_refuse() -> None:
    loha = LoHaAdapter(
        torch.ones(2, 1),
        torch.ones(1, 3),
        torch.ones(2, 1),
        torch.ones(1, 3),
    )
    other = PatchSet({"weight": (PatchEntry(AdapterPatch(loha)),)})
    _assert_code(
        "patch-kind",
        lambda: scaled_patch_context(_linear(), other, torch.ones(1), lambda: False).__enter__(),
    )

    class Custom(torch.nn.Linear):
        pass

    custom = Custom(3, 2, bias=False)
    _assert_code(
        "target-type",
        lambda: scaled_patch_context(
            custom, _diff_patch(), torch.ones(1), lambda: False
        ).__enter__(),
    )
    conv = torch.nn.Conv2d(3, 2, 1, bias=False)
    conv_patch = PatchSet({"weight": (PatchEntry(DiffPatch(torch.ones_like(conv.weight))),)})
    _assert_code(
        "diff-target",
        lambda: scaled_patch_context(conv, conv_patch, torch.ones(1), lambda: False).__enter__(),
    )

    shared = _linear()
    aliased = torch.nn.Module()
    aliased.a = shared
    aliased.b = shared
    alias_patch = PatchSet({"a.weight": (PatchEntry(DiffPatch(torch.ones(2, 3))),)})
    _assert_code(
        "shared-module",
        lambda: scaled_patch_context(
            aliased, alias_patch, torch.ones(1), lambda: False
        ).__enter__(),
    )
    hooked = _linear()
    handle = hooked.register_forward_hook(lambda _m, _a, output: output)
    try:
        _assert_code(
            "target-hooked",
            lambda: scaled_patch_context(
                hooked, _diff_patch(), torch.ones(1), lambda: False
            ).__enter__(),
        )
    finally:
        handle.remove()
    pre_hooked = _linear()
    handle = pre_hooked.register_forward_pre_hook(lambda _module, inputs: inputs)
    try:
        _assert_code(
            "target-hooked",
            lambda: scaled_patch_context(
                pre_hooked, _diff_patch(), torch.ones(1), lambda: False
            ).__enter__(),
        )
    finally:
        handle.remove()


def test_fp8_and_shared_storage_targets_refuse() -> None:
    fp8 = _linear()
    fp8.weight = torch.nn.Parameter(fp8.weight.detach().to(torch.float8_e4m3fn))
    _assert_code(
        "target-weight",
        lambda: scaled_patch_context(fp8, _diff_patch(), torch.ones(1), lambda: False).__enter__(),
    )

    model = torch.nn.Module()
    model.a = _linear()
    model.b = _linear()
    model.b.weight = torch.nn.Parameter(model.a.weight.detach())
    patch = PatchSet({"a.weight": (PatchEntry(DiffPatch(torch.ones(2, 3))),)})
    _assert_code(
        "shared-weight",
        lambda: scaled_patch_context(model, patch, torch.ones(1), lambda: False).__enter__(),
    )


@pytest.mark.parametrize(
    ("scale", "code"),
    [
        (torch.tensor(1.0), "scale-vector"),
        (torch.empty(0), "scale-vector"),
        (torch.ones(2, 1), "scale-vector"),
        (torch.ones(1, dtype=torch.int64), "scale-vector"),
        (torch.tensor([float("nan")]), "scale-finite"),
    ],
)
def test_scale_refusal_matrix(scale: torch.Tensor, code: str) -> None:
    _assert_code(
        code,
        lambda: scaled_patch_context(_linear(), _diff_patch(), scale, lambda: False).__enter__(),
    )


def test_input_output_batch_dtype_and_shape_contract_refusals() -> None:
    layer = _linear()
    with scaled_patch_context(layer, _diff_patch(), torch.ones(3), lambda: False):
        _assert_code("scale-batch", lambda: layer(torch.ones(2, 3)))
    assert not layer._forward_hooks

    layer = _linear()
    layer.forward = (
        lambda value: torch.zeros(  # type: ignore[method-assign]
            (*value.shape[:-1], 3), dtype=value.dtype, device=value.device
        )
    )
    _assert_code(
        "target-forward",
        lambda: scaled_patch_context(
            layer, _diff_patch(), torch.ones(1), lambda: False
        ).__enter__(),
    )

    layer = _linear()
    layer.forward = (
        lambda value: torch.zeros(  # type: ignore[method-assign]
            (*value.shape[:-1], 2), dtype=torch.float64, device=value.device
        )
    )
    _assert_code(
        "target-forward",
        lambda: scaled_patch_context(
            layer, _diff_patch(), torch.ones(1), lambda: False
        ).__enter__(),
    )

    layer = _linear()
    with scaled_patch_context(layer, _diff_patch(), torch.ones(1), lambda: False):
        _assert_code("input-contract", lambda: layer(input=torch.ones(2, 3)))


def test_cancellation_before_install_in_hook_between_entries_and_cleanup() -> None:
    layer = _linear()
    _assert_code(
        "cancelled",
        lambda: scaled_patch_context(layer, _diff_patch(), torch.ones(1), lambda: True).__enter__(),
    )
    assert not layer._forward_hooks

    entries = (
        PatchEntry(DiffPatch(torch.ones(2, 3))),
        PatchEntry(DiffPatch(torch.full((2, 3), 2.0))),
    )
    for stop_at in (3, 5):
        calls = 0

        def cancel(stop: int = stop_at) -> bool:
            nonlocal calls
            calls += 1
            return calls == stop

        with scaled_patch_context(layer, PatchSet({"weight": entries}), torch.ones(1), cancel):
            _assert_code("cancelled", lambda: layer(torch.ones(1, 3)))
        assert not layer._forward_hooks
        with scaled_patch_context(layer, _diff_patch(), torch.ones(1), lambda: False):
            layer(torch.ones(1, 3))


def test_prepared_activation_cancellation_precedes_hook_publication() -> None:
    cancelled = False
    layer = _linear()
    prepared = scaled_module.prepare_scaled_patches(
        layer, _diff_patch(), "cpu", torch.float32, lambda: cancelled
    )
    cancelled = True
    with pytest.raises(ScaledPatchError, match="cancelled"):
        prepared.activate(torch.ones(1)).__enter__()
    assert not layer._forward_hooks
    assert not prepared._active
    prepared.close()


def test_registration_failure_is_transactional_and_guard_is_reusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Module()
    model.a = _linear()
    model.b = _linear()
    patch = PatchSet(
        {
            "a.weight": (PatchEntry(DiffPatch(torch.ones(2, 3))),),
            "b.weight": (PatchEntry(DiffPatch(torch.ones(2, 3))),),
        }
    )

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("register")

    staged_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    original_stage = scaled_module._stage_targets

    def observe(*args: Any, **kwargs: Any):
        staged = original_stage(*args, **kwargs)
        for target in staged:
            staged_refs.append(weakref.ref(target.scale))
            for entry in target.entries:
                staged_refs.append(weakref.ref(entry.first))
                if entry.second is not None:
                    staged_refs.append(weakref.ref(entry.second))
        return staged

    monkeypatch.setattr(scaled_module, "_stage_targets", observe)
    monkeypatch.setattr(model.b, "register_forward_hook", fail)
    with pytest.raises(RuntimeError, match="register"):
        with scaled_patch_context(model, patch, torch.ones(1), lambda: False):
            pass
    assert not model.a._forward_hooks
    assert not model.b._forward_hooks
    gc.collect()
    assert staged_refs and all(reference() is None for reference in staged_refs)
    with scaled_patch_context(model.a, _diff_patch(), torch.ones(1), lambda: False):
        pass


def test_nested_concurrent_same_model_refusal_and_reuse() -> None:
    layer = _linear()
    patch = _diff_patch()
    with scaled_patch_context(layer, patch, torch.ones(1), lambda: False):
        _assert_code(
            "active-model",
            lambda: scaled_patch_context(layer, patch, torch.ones(1), lambda: False).__enter__(),
        )

    context = scaled_patch_context(layer, patch, torch.ones(1), lambda: False)
    with context:
        _assert_code("context-reentry", context.__enter__)
    _assert_code("context-reentry", context.__enter__)

    entered = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with scaled_patch_context(layer, patch, torch.ones(1), lambda: False):
            entered.set()
            assert release.wait(5)

    thread = threading.Thread(target=hold)
    thread.start()
    assert entered.wait(5)
    _assert_code(
        "active-model",
        lambda: scaled_patch_context(layer, patch, torch.ones(1), lambda: False).__enter__(),
    )
    release.set()
    thread.join(5)
    assert not thread.is_alive()
    with scaled_patch_context(layer, patch, torch.ones(1), lambda: False):
        pass


def test_parent_reservation_blocks_child_before_first_hook_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Parent(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.child = _linear()

    parent = Parent()
    parent_patch = PatchSet({"child.weight": (PatchEntry(DiffPatch(torch.ones((2, 3)))),)})
    child_patch = _diff_patch()
    reserved = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []
    original_register = parent.child.register_forward_hook

    def pause_registration(hook: Any, **kwargs: Any):
        reserved.set()
        assert release.wait(5)
        return original_register(hook, **kwargs)

    monkeypatch.setattr(parent.child, "register_forward_hook", pause_registration)

    def hold_parent() -> None:
        try:
            with scaled_patch_context(parent, parent_patch, torch.ones(1), lambda: False):
                pass
        except BaseException as error:
            failures.append(error)

    thread = threading.Thread(target=hold_parent)
    thread.start()
    assert reserved.wait(5)
    _assert_code(
        "active-model",
        lambda: scaled_patch_context(
            parent.child, child_patch, torch.ones(1), lambda: False
        ).__enter__(),
    )
    release.set()
    thread.join(5)
    assert not thread.is_alive() and failures == []
    assert parent not in scaled_module._ACTIVE_MODELS
    assert parent.child not in scaled_module._ACTIVE_MODELS
    assert not parent.child._forward_hooks


def test_structural_payload_scale_and_target_identity_drift_refuse() -> None:
    layer = _linear()
    adapter = LoRAAdapter(torch.ones(2, 1), torch.ones(1, 3), alpha=1.0)
    patch = PatchSet({"weight": (PatchEntry(AdapterPatch(adapter)),)})
    with scaled_patch_context(layer, patch, torch.ones(1), lambda: False):
        adapter.alpha = 2.0
        _assert_code("patch-set-drift", lambda: layer(torch.ones(1, 3)))
    adapter.alpha = 1.0

    scale = torch.ones(1)
    with scaled_patch_context(layer, patch, scale, lambda: False):
        scale.resize_(2)
        _assert_code("scale-drift", lambda: layer(torch.ones(1, 3)))

    layer = _linear()
    with scaled_patch_context(layer, _diff_patch(), torch.ones(1), lambda: False):
        layer.weight = torch.nn.Parameter(layer.weight.detach().clone())
        _assert_code("target-drift", lambda: layer(torch.ones(1, 3)))


def test_hook_error_compile_refusal_and_staged_tensor_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _linear()
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    with pytest.raises(ScaledPatchError, match="scaled-patch:compiled"):
        with scaled_patch_context(layer, _diff_patch(), torch.ones(1), lambda: False):
            layer(torch.ones(1, 3))
    assert not layer._forward_hooks
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)

    staged_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    original = scaled_module._stage_targets

    def observe(*args: Any, **kwargs: Any):
        staged = original(*args, **kwargs)
        for target in staged:
            staged_refs.append(weakref.ref(target.scale))
            for entry in target.entries:
                staged_refs.append(weakref.ref(entry.first))
                if entry.second is not None:
                    staged_refs.append(weakref.ref(entry.second))
        return staged

    monkeypatch.setattr(scaled_module, "_stage_targets", observe)
    with scaled_patch_context(layer, _diff_patch(), torch.ones(1), lambda: False):
        layer(torch.ones(1, 3))
    gc.collect()
    assert staged_refs
    assert all(reference() is None for reference in staged_refs)
    assert not layer._forward_hooks


def test_convolution_scheduling_exports_only_public_api() -> None:
    assert scaled_module.__all__ == [
        "PreparedScaledPatches",
        "ScaledPatchError",
        "prepare_scaled_patches",
        "scaled_patch_context",
    ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA capability required")
def test_scaled_patch_cuda_checkout_source() -> None:
    repo = Path(__file__).resolve().parents[3]
    assert Path(scaled_module.__file__).resolve().is_relative_to(repo)
    device = torch.device("cuda:0")
    layer = _linear(device=device, dtype=torch.float32)
    value = torch.tensor([[1.0, -2.0, 0.5], [-0.25, 0.75, 2.0]], device=device)
    diff = torch.tensor([[-0.125, 0.375, 0.5], [0.75, -0.25, 0.625]], device="cpu")
    scale = torch.tensor([0.0, 1.25], device="cpu")
    baseline = layer(value)
    weight_before = _bytes(layer.weight)
    with scaled_patch_context(layer, _diff_patch(diff, strength=0.6), scale, lambda: False):
        expected = baseline + scale.to(device)[:, None] * F.linear(value, diff.to(device)) * 0.6
        assert torch.equal(layer(value), expected)
    torch.cuda.synchronize()
    assert _bytes(layer.weight) == weight_before
    assert torch.equal(layer(value), baseline)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA capability required")
def test_convolution_lora_cuda_strength_and_cleanup() -> None:
    device = torch.device("cuda:0")
    layer = _conv(device=device)
    value = torch.arange(2 * 2 * 9 * 8, device=device, dtype=torch.float32).reshape(2, 2, 9, 8) / 29
    up = torch.arange(6, dtype=torch.float32).reshape(3, 2, 1, 1) / 17
    down = torch.arange(36, dtype=torch.float32).reshape(2, 2, 3, 3) / 23
    patch = PatchSet({"weight": (PatchEntry(AdapterPatch(LoRAAdapter(up, down)), strength=0.5),)})
    scale = torch.tensor([0.0, 1.25])
    baseline = layer(value)
    expected = _ordinary_conv(layer, value, patch, scale)
    weight_before = _bytes(layer.weight)

    with scaled_patch_context(layer, patch, scale, lambda: False):
        assert torch.equal(layer(value), expected)

    torch.cuda.synchronize()
    assert _bytes(layer.weight) == weight_before
    assert torch.equal(layer(value), baseline)
    assert not layer._forward_hooks
