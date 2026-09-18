"""Native latent operators against goldens from pinned ComfyUI e20d433a.

The goldens in goldens/latent_ops_e20d433a.json are produced end to end by
the pinned upstream nodes (tools/gen_latent_ops_goldens.py). The native
executors use the production tensor resize helpers without a ComfyUI shim.

The exact float values are platform-dependent (torch.linspace and the
interpolation kernels differ by ULPs across torch builds), so the fixture
goes through the platform-tuple golden loader: the canonical file is
Linux-minted and non-Linux hosts mint a suffixed fixture with the
generator.
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    CancellationToken,
    Conditioning,
    GuidanceCondition,
    GuidanceEvaluationRequest,
    GuidancePlanContext,
    GuidancePrediction,
    GuidancePredictions,
    GuidancePredictionSource,
    GuidanceRole,
    ProgressScope,
    SamplingExecutionContext,
)
from dinkster_inference_torch.guidance import GuidanceExecutor, GuidanceRegistry
from golden_files import load_platform_golden

ROOT = Path(__file__).resolve().parents[3]
for source in sorted((ROOT / "packages").glob("*/src")):
    sys.path.insert(0, str(source))

native_arm = importlib.import_module("dinkster_compat_comfy.native_arm")

GOLDEN_PATH = Path(__file__).parent / "goldens" / "latent_ops_e20d433a.json"
GOLDEN = load_platform_golden(GOLDEN_PATH)


def _tensor(record: dict[str, Any]) -> torch.Tensor:
    values = torch.tensor([float(value) for value in record["values"]], dtype=torch.float32)
    return values.reshape(record["shape"])


def _source(name: str) -> torch.Tensor:
    return _tensor(GOLDEN["sources"][name])


def _latent(name: str) -> dict[str, object]:
    return {"samples": _source(name)}


def _mask(name: str) -> torch.Tensor:
    return _source(name)


def _case_runs() -> dict[str, tuple[Any, dict[str, object]]]:
    runs: dict[str, tuple[Any, dict[str, object]]] = {
        "add": (
            native_arm.GenerationLatentCombine,
            {"samples1": _latent("latent_a"), "samples2": _latent("latent_b"), "operation": "add"},
        ),
        "add:reshape-repeat": (
            native_arm.GenerationLatentCombine,
            {
                "samples1": _latent("latent_a"),
                "samples2": _latent("latent_small"),
                "operation": "add",
            },
        ),
        "subtract": (
            native_arm.GenerationLatentCombine,
            {
                "samples1": _latent("latent_a"),
                "samples2": _latent("latent_b"),
                "operation": "subtract",
            },
        ),
        "interpolate": (
            native_arm.GenerationLatentMix,
            {
                "samples1": _latent("latent_wide"),
                "samples2": _latent("latent_wide_b"),
                "operation": "interpolate",
                "factor": 0.35,
            },
        ),
        "interpolate:reshape": (
            native_arm.GenerationLatentMix,
            {
                "samples1": _latent("latent_wide"),
                "samples2": _latent("latent_small"),
                "operation": "interpolate",
                "factor": 0.7,
            },
        ),
        "blend": (
            native_arm.GenerationLatentMix,
            {
                "samples1": _latent("latent_a"),
                "samples2": _latent("latent_b"),
                "operation": "blend",
                "factor": 0.65,
            },
        ),
        "blend:mismatched": (
            native_arm.GenerationLatentMix,
            {
                "samples1": _latent("latent_a"),
                "samples2": _latent("latent_small"),
                "operation": "blend",
                "factor": 0.4,
            },
        ),
        "multiply": (
            native_arm.GenerationLatentMultiply,
            {"samples": _latent("latent_a"), "multiplier": -1.75},
        ),
        "crop": (
            native_arm.GenerationLatentCrop,
            {"samples": _latent("latent_wide"), "width": 64, "height": 64, "x": 32, "y": 8},
        ),
        "crop:clamped": (
            native_arm.GenerationLatentCrop,
            {"samples": _latent("latent_wide"), "width": 64, "height": 64, "x": 120, "y": 112},
        ),
        "resize:center": (
            native_arm.GenerationLatentResize,
            {
                "samples": _latent("latent_wide"),
                "method": "bilinear",
                "width": 96,
                "height": 64,
                "crop": "center",
            },
        ),
        "resize:width-zero": (
            native_arm.GenerationLatentResize,
            {
                "samples": _latent("latent_wide"),
                "method": "bilinear",
                "width": 0,
                "height": 96,
                "crop": "disabled",
            },
        ),
        "resize:height-zero": (
            native_arm.GenerationLatentResize,
            {
                "samples": _latent("latent_wide"),
                "method": "bilinear",
                "width": 96,
                "height": 0,
                "crop": "disabled",
            },
        ),
        "resize_by:bislerp-1.5x": (
            native_arm.GenerationLatentResizeBy,
            {"samples": _latent("latent_wide"), "method": "bislerp", "scale_by": 1.5},
        ),
        "resize_by:nearest-half": (
            native_arm.GenerationLatentResizeBy,
            {"samples": _latent("latent_wide"), "method": "nearest-exact", "scale_by": 0.5},
        ),
        "composite": (
            native_arm.GenerationLatentComposite,
            {
                "destination": _latent("latent_wide"),
                "source": _latent("latent_patch"),
                "x": 32,
                "y": 16,
                "feather": 0,
            },
        ),
        "composite:feather": (
            native_arm.GenerationLatentComposite,
            {
                "destination": _latent("latent_wide"),
                "source": _latent("latent_patch"),
                "x": 32,
                "y": 16,
                "feather": 16,
            },
        ),
        "composite_masked": (
            native_arm.GenerationLatentCompositeMasked,
            {
                "destination": _latent("latent_wide"),
                "source": _latent("latent_patch"),
                "x": 8,
                "y": 16,
                "resize_source": False,
            },
        ),
        "composite_masked:mask": (
            native_arm.GenerationLatentCompositeMasked,
            {
                "destination": _latent("latent_wide"),
                "source": _latent("latent_patch"),
                "x": 8,
                "y": 16,
                "resize_source": False,
                "mask": _mask("mask_small"),
            },
        ),
        "composite_masked:resize": (
            native_arm.GenerationLatentCompositeMasked,
            {
                "destination": _latent("latent_wide"),
                "source": _latent("latent_patch"),
                "x": 0,
                "y": 0,
                "resize_source": True,
                "mask": _mask("mask_small"),
            },
        ),
        "concat:x": (
            native_arm.GenerationLatentConcat,
            {"samples1": _latent("latent_a"), "samples2": _latent("latent_b"), "dim": "x"},
        ),
        "concat:-x": (
            native_arm.GenerationLatentConcat,
            {"samples1": _latent("latent_a"), "samples2": _latent("latent_b"), "dim": "-x"},
        ),
        "concat:t": (
            native_arm.GenerationLatentConcat,
            {
                "samples1": _latent("latent_video"),
                "samples2": _latent("latent_video_b"),
                "dim": "t",
            },
        ),
        "concat:batch-repeat": (
            native_arm.GenerationLatentConcat,
            {"samples1": _latent("latent_a"), "samples2": _latent("latent_patch"), "dim": "x"},
        ),
        "cut:x": (
            native_arm.GenerationLatentCut,
            {"samples": _latent("latent_wide"), "dim": "x", "index": 3, "amount": 4},
        ),
        "cut:negative": (
            native_arm.GenerationLatentCut,
            {"samples": _latent("latent_wide"), "dim": "y", "index": -5, "amount": 9},
        ),
        "cut:t": (
            native_arm.GenerationLatentCut,
            {"samples": _latent("latent_video"), "dim": "t", "index": 1, "amount": 2},
        ),
        "cut:t-4d": (
            native_arm.GenerationLatentCut,
            {"samples": _latent("latent_a"), "dim": "t", "index": 0, "amount": 1},
        ),
        "cut_to_batch:t": (
            native_arm.GenerationLatentCutToBatch,
            {"samples": _latent("latent_video"), "dim": "t", "slice_size": 2},
        ),
        "cut_to_batch:x": (
            native_arm.GenerationLatentCutToBatch,
            {"samples": _latent("latent_a"), "dim": "x", "slice_size": 3},
        ),
        "cut_to_batch:passthrough": (
            native_arm.GenerationLatentCutToBatch,
            {"samples": _latent("latent_a"), "dim": "t", "slice_size": 2},
        ),
        "cut_to_batch:oversize": (
            native_arm.GenerationLatentCutToBatch,
            {"samples": _latent("latent_a"), "dim": "x", "slice_size": 10},
        ),
    }
    for angle in ("none", "90", "180", "270"):
        runs[f"rotate:{angle}"] = (
            native_arm.GenerationLatentRotate,
            {"samples": _latent("latent_a"), "angle": angle},
        )
    for axis in ("vertical", "horizontal"):
        runs[f"flip:{axis}"] = (
            native_arm.GenerationLatentFlip,
            {"samples": _latent("latent_a"), "axis": axis},
        )
    for method in ("nearest-exact", "bilinear", "area", "bicubic", "bislerp"):
        runs[f"resize:{method}"] = (
            native_arm.GenerationLatentResize,
            {
                "samples": _latent("latent_wide"),
                "method": method,
                "width": 96,
                "height": 64,
                "crop": "disabled",
            },
        )
    return runs


def test_latent_ops_match_comfy_goldens() -> None:
    assert GOLDEN["baseline"] == "e20d433a4966dcc88fa5abbae6ace824cb78b263"
    runs = _case_runs()
    assert set(runs) == set(GOLDEN["cases"])
    for name, (node, kwargs) in runs.items():
        result = node.execute(**kwargs)["latent"]
        actual = cast("torch.Tensor", cast("dict[str, object]", result)["samples"])
        torch.testing.assert_close(actual, _tensor(GOLDEN["cases"][name]), rtol=0, atol=0, msg=name)


def test_latent_ops_copy_latent_dicts_and_preserve_metadata() -> None:
    latent = {"samples": _source("latent_a"), "batch_index": [0, 1], "metadata": "preserved"}
    other = _latent("latent_b")
    result = native_arm.GenerationLatentCombine.execute(
        samples1=latent, samples2=other, operation="add"
    )["latent"]
    assert result is not latent
    assert result["batch_index"] == [0, 1]
    assert result["metadata"] == "preserved"
    assert torch.equal(cast("torch.Tensor", latent["samples"]), _source("latent_a"))


def test_latent_resize_and_cut_to_batch_pass_through_unchanged() -> None:
    latent = _latent("latent_wide")
    resized = native_arm.GenerationLatentResize.execute(
        samples=latent, method="bilinear", width=0, height=0, crop="disabled"
    )["latent"]
    assert resized is latent

    flat = _latent("latent_a")
    passed = native_arm.GenerationLatentCutToBatch.execute(samples=flat, dim="t", slice_size=2)[
        "latent"
    ]
    assert passed is flat


def test_latent_ops_reject_invalid_widgets_and_carriers() -> None:
    latent = _latent("latent_a")
    with pytest.raises(ValueError, match="operation"):
        native_arm.GenerationLatentCombine.execute(
            samples1=latent, samples2=_latent("latent_b"), operation="divide"
        )
    with pytest.raises(ValueError, match="factor"):
        native_arm.GenerationLatentMix.execute(
            samples1=latent,
            samples2=_latent("latent_b"),
            operation="blend",
            factor=1.5,
        )
    with pytest.raises(ValueError, match="scale_by"):
        native_arm.GenerationLatentResizeBy.execute(samples=latent, method="bilinear", scale_by=0.0)
    with pytest.raises(TypeError, match="samples"):
        native_arm.GenerationLatentMultiply.execute(samples={"samples": [1.0, 2.0]}, multiplier=1.0)
    with pytest.raises(TypeError, match="mask"):
        native_arm.GenerationLatentCompositeMasked.execute(
            destination=latent,
            source=_latent("latent_b"),
            x=0,
            y=0,
            resize_source=False,
            mask=[0.5],
        )


def _meta_latent(
    name: str, mask: str | None = None, batch_index: list[int] | None = None
) -> dict[str, object]:
    value: dict[str, object] = {"samples": _source(name)}
    if mask is not None:
        value["noise_mask"] = _source(mask)
    if batch_index is not None:
        value["batch_index"] = list(batch_index)
    return value


def _assert_latent_record(actual: dict[str, object], record: dict[str, Any], name: str) -> None:
    samples = cast("torch.Tensor", actual["samples"])
    torch.testing.assert_close(samples, _tensor(record["samples"]), rtol=0, atol=0, msg=name)
    assert ("noise_mask" in actual) == ("noise_mask" in record), name
    if "noise_mask" in record:
        mask = cast("torch.Tensor", actual["noise_mask"])
        torch.testing.assert_close(mask, _tensor(record["noise_mask"]), rtol=0, atol=0, msg=name)
    assert ("batch_index" in actual) == ("batch_index" in record), name
    if "batch_index" in record:
        assert list(cast("list[int]", actual["batch_index"])) == record["batch_index"], name


def _batch_case_runs() -> dict[str, tuple[Any, dict[str, object]]]:
    return {
        "from_batch": (
            native_arm.GenerationLatentFromBatch,
            {
                "samples": _meta_latent("latent_a", mask="mask_batch2"),
                "batch_index": 1,
                "length": 2,
            },
        ),
        "from_batch:negative": (
            native_arm.GenerationLatentFromBatch,
            {
                "samples": _meta_latent("latent_a", mask="mask_one", batch_index=[5, 9]),
                "batch_index": -1,
                "length": 1,
            },
        ),
        "from_batch:mask-repeat": (
            native_arm.GenerationLatentFromBatch,
            {
                "samples": _meta_latent("latent_batch3", mask="mask_batch2"),
                "batch_index": 1,
                "length": 2,
            },
        ),
        "repeat": (
            native_arm.GenerationLatentRepeat,
            {
                "samples": _meta_latent("latent_a", mask="mask_batch2", batch_index=[3, 7]),
                "amount": 3,
            },
        ),
        "repeat:mask-single": (
            native_arm.GenerationLatentRepeat,
            {"samples": _meta_latent("latent_a", mask="mask_one"), "amount": 2},
        ),
        "seed_behavior:fixed": (
            native_arm.GenerationLatentSeedBehavior,
            {"samples": _meta_latent("latent_a", batch_index=[4, 9]), "behavior": "fixed"},
        ),
        "seed_behavior:fixed-default": (
            native_arm.GenerationLatentSeedBehavior,
            {"samples": _meta_latent("latent_a"), "behavior": "fixed"},
        ),
        "seed_behavior:random": (
            native_arm.GenerationLatentSeedBehavior,
            {"samples": _meta_latent("latent_a", batch_index=[4, 9]), "behavior": "random"},
        ),
        "batch": (
            native_arm.GenerationLatentBatch,
            {
                "latents": {
                    "latent_1": _meta_latent("latent_a", batch_index=[2, 3]),
                    "latent_2": _meta_latent("latent_small"),
                }
            },
        ),
        "batch:multi": (
            native_arm.GenerationLatentBatch,
            {
                "latents": {
                    "latent_1": _meta_latent("latent_a", batch_index=[2, 3]),
                    "latent_2": _meta_latent("latent_small"),
                    "latent_3": _meta_latent("latent_b"),
                }
            },
        ),
        "set_noise_mask": (
            native_arm.GenerationLatentSetNoiseMask,
            {"samples": _meta_latent("latent_a"), "mask": _mask("mask_small")},
        ),
        "replace_frames": (
            native_arm.GenerationLatentReplaceFrames,
            {
                "destination": _meta_latent("latent_video", batch_index=[1]),
                "index": 2,
                "source": _meta_latent("latent_video_short", batch_index=[7]),
            },
        ),
        "replace_frames:negative": (
            native_arm.GenerationLatentReplaceFrames,
            {
                "destination": _meta_latent("latent_video"),
                "index": -2,
                "source": _meta_latent("latent_video_short"),
            },
        ),
        "replace_frames:oob-start": (
            native_arm.GenerationLatentReplaceFrames,
            {
                "destination": _meta_latent("latent_video", batch_index=[1]),
                "index": 6,
                "source": _meta_latent("latent_video_short"),
            },
        ),
        "replace_frames:oob-length": (
            native_arm.GenerationLatentReplaceFrames,
            {
                "destination": _meta_latent("latent_video", batch_index=[1]),
                "index": 5,
                "source": _meta_latent("latent_video_short"),
            },
        ),
        "rebatch:merge": (
            native_arm.GenerationLatentRebatch,
            {"latents": [_meta_latent("latent_a"), _meta_latent("latent_b")], "batch_size": 3},
        ),
        "rebatch:masked": (
            native_arm.GenerationLatentRebatch,
            {
                "latents": [
                    _meta_latent("latent_a", mask="mask_batch2"),
                    _meta_latent("latent_b"),
                ],
                "batch_size": 4,
            },
        ),
        "rebatch:mixed-dims": (
            native_arm.GenerationLatentRebatch,
            {"latents": [_meta_latent("latent_a"), _meta_latent("latent_small")], "batch_size": 4},
        ),
        "rebatch:split": (
            native_arm.GenerationLatentRebatch,
            {"latents": [_meta_latent("latent_a", batch_index=[5, 6])], "batch_size": 1},
        ),
    }


def test_latent_batch_ops_match_comfy_goldens() -> None:
    runs = _batch_case_runs()
    assert set(runs) == set(GOLDEN["metadata_cases"])
    for name, (node, kwargs) in runs.items():
        golden = GOLDEN["metadata_cases"][name]
        if isinstance(golden, list):
            results = cast("list[dict[str, object]]", node.execute(**kwargs)["latents"])
            assert len(results) == len(golden), name
            for position, (actual, record) in enumerate(zip(results, golden, strict=True)):
                _assert_latent_record(actual, record, f"{name}[{position}]")
        else:
            result = cast("dict[str, object]", node.execute(**kwargs)["latent"])
            _assert_latent_record(result, golden, name)


def test_latent_batch_ops_copy_dicts_and_route_metadata() -> None:
    latent = {
        "samples": _source("latent_a"),
        "noise_mask": _source("mask_one"),
        "metadata": "preserved",
    }
    result = cast(
        "dict[str, object]",
        native_arm.GenerationLatentFromBatch.execute(samples=latent, batch_index=0, length=1)[
            "latent"
        ],
    )
    assert result is not latent
    assert result["metadata"] == "preserved"
    assert torch.equal(cast("torch.Tensor", latent["samples"]), _source("latent_a"))

    source = {
        "samples": _source("latent_video_short"),
        "metadata": "from-source",
    }
    replaced = cast(
        "dict[str, object]",
        native_arm.GenerationLatentReplaceFrames.execute(
            destination=_meta_latent("latent_video"), index=0, source=source
        )["latent"],
    )
    assert replaced is not source
    assert replaced["metadata"] == "from-source"

    destination = _meta_latent("latent_video", batch_index=[1])
    passthrough = native_arm.GenerationLatentReplaceFrames.execute(
        destination=destination, index=0, source=None
    )["latent"]
    assert passthrough is destination


def test_latent_batch_ops_reject_invalid_widgets_and_carriers() -> None:
    latent = _meta_latent("latent_a")
    with pytest.raises(ValueError, match="length"):
        native_arm.GenerationLatentFromBatch.execute(samples=latent, batch_index=0, length=0)
    with pytest.raises(ValueError, match="amount"):
        native_arm.GenerationLatentRepeat.execute(samples=latent, amount=65)
    with pytest.raises(ValueError, match="behavior"):
        native_arm.GenerationLatentSeedBehavior.execute(samples=latent, behavior="bogus")
    with pytest.raises(ValueError, match="batch_size"):
        native_arm.GenerationLatentRebatch.execute(latents=[latent], batch_size=0)
    with pytest.raises(ValueError, match="index"):
        native_arm.GenerationLatentReplaceFrames.execute(
            destination=_meta_latent("latent_video"),
            index=20_000,
            source=_meta_latent("latent_video_short"),
        )
    with pytest.raises(ValueError, match="latents"):
        native_arm.GenerationLatentBatch.execute(latents={})
    with pytest.raises(TypeError, match="latents member"):
        native_arm.GenerationLatentBatch.execute(latents={"latent_1": [1.0]})
    with pytest.raises(TypeError, match="latents"):
        native_arm.GenerationLatentRebatch.execute(latents=latent, batch_size=1)
    with pytest.raises(TypeError, match="mask"):
        native_arm.GenerationLatentSetNoiseMask.execute(samples=latent, mask=[0.5])


def _tonemap_operation(multiplier: float) -> object:
    return native_arm.GenerationLatentOperationTonemapReinhard.execute(multiplier=multiplier)[
        "operation"
    ]


def _sharpen_operation(sharpen_radius: int, sigma: float, alpha: float) -> object:
    return native_arm.GenerationLatentOperationSharpen.execute(
        sharpen_radius=sharpen_radius, sigma=sigma, alpha=alpha
    )["operation"]


def _operation_case_runs() -> dict[str, dict[str, object]]:
    return {
        "apply:tonemap": {
            "samples": _latent("latent_a"),
            "operation": _tonemap_operation(1.0),
        },
        "apply:tonemap-low": {
            "samples": _latent("latent_wide"),
            "operation": _tonemap_operation(0.35),
        },
        "apply:tonemap-high": {
            "samples": _latent("latent_batch3"),
            "operation": _tonemap_operation(2.5),
        },
        "apply:tonemap-video": {
            "samples": _latent("latent_video"),
            "operation": _tonemap_operation(1.0),
        },
        "apply:sharpen": {
            "samples": _latent("latent_wide"),
            "operation": _sharpen_operation(9, 1.0, 0.1),
        },
        "apply:sharpen-small": {
            "samples": _latent("latent_a"),
            "operation": _sharpen_operation(3, 0.5, 0.25),
        },
        "apply:sharpen-radius1": {
            "samples": _latent("latent_small"),
            "operation": _sharpen_operation(1, 1.0, 0.5),
        },
        "apply:tonemap-metadata": {
            "samples": _meta_latent("latent_a", mask="mask_batch2", batch_index=[3, 7]),
            "operation": _tonemap_operation(0.8),
        },
    }


def test_latent_operation_ops_match_comfy_goldens() -> None:
    runs = _operation_case_runs()
    assert set(runs) == set(GOLDEN["operation_cases"])
    for name, kwargs in runs.items():
        golden = GOLDEN["operation_cases"][name]
        result = cast(
            "dict[str, object]",
            native_arm.GenerationLatentApplyOperation.execute(**kwargs)["latent"],
        )
        if "values" in golden:
            actual = cast("torch.Tensor", result["samples"])
            torch.testing.assert_close(actual, _tensor(golden), rtol=0, atol=0, msg=name)
        else:
            _assert_latent_record(result, golden, name)


def test_latent_apply_operation_copies_dict_and_preserves_metadata() -> None:
    latent = {"samples": _source("latent_a"), "batch_index": [0, 1], "metadata": "preserved"}
    result = cast(
        "dict[str, object]",
        native_arm.GenerationLatentApplyOperation.execute(
            samples=latent, operation=_tonemap_operation(1.0)
        )["latent"],
    )
    assert result is not latent
    assert result["batch_index"] == [0, 1]
    assert result["metadata"] == "preserved"
    assert torch.equal(cast("torch.Tensor", latent["samples"]), _source("latent_a"))


def test_latent_operation_ops_reject_invalid_widgets_and_carriers() -> None:
    latent = _latent("latent_a")
    with pytest.raises(TypeError, match="operation"):
        native_arm.GenerationLatentApplyOperation.execute(
            samples=latent, operation="tonemap_reinhard"
        )
    with pytest.raises(TypeError, match="samples"):
        native_arm.GenerationLatentApplyOperation.execute(
            samples={"samples": [1.0]}, operation=_tonemap_operation(1.0)
        )
    with pytest.raises(ValueError, match="multiplier"):
        native_arm.GenerationLatentOperationTonemapReinhard.execute(multiplier=101.0)
    with pytest.raises(ValueError, match="sharpen_radius"):
        native_arm.GenerationLatentOperationSharpen.execute(sharpen_radius=0, sigma=1.0, alpha=0.1)
    with pytest.raises(ValueError, match="sigma"):
        native_arm.GenerationLatentOperationSharpen.execute(sharpen_radius=9, sigma=0.0, alpha=0.1)
    with pytest.raises(ValueError, match="alpha"):
        native_arm.GenerationLatentOperationSharpen.execute(sharpen_radius=9, sigma=1.0, alpha=5.5)


def test_latent_operation_descriptors_reject_mutation() -> None:
    tonemap = _tonemap_operation(1.0)
    sharpen = _sharpen_operation(9, 1.0, 0.1)
    for descriptor, field, value in (
        (tonemap, "kind", "sharpen"),
        (tonemap, "params", ()),
        (sharpen, "params", ()),
    ):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(descriptor, field, value)


def _cfg_case_runs() -> dict[str, tuple[tuple[object, ...], float]]:
    return {
        "cfg:tonemap": ((_tonemap_operation(0.8),), 3.5),
        "cfg:tonemap-cfg1": ((_tonemap_operation(0.8),), 1.0),
        "cfg:sharpen": ((_sharpen_operation(5, 0.7, 0.3),), 3.5),
        "cfg:sharpen-cfg1": ((_sharpen_operation(5, 0.7, 0.3),), 1.0),
        "cfg:tonemap-then-sharpen": (
            (_tonemap_operation(1.2), _sharpen_operation(3, 0.5, 0.25)),
            3.5,
        ),
    }


class _StubHandle:
    def require_active(self) -> None:
        return None


def _chained_transform_pairs(operations: tuple[object, ...]) -> tuple[tuple[str, object], ...]:
    model: object = native_arm._NativeModelOverlay(_StubHandle(), (), {}, None, None, ())
    for operation in operations:
        model = native_arm.GenerationLatentApplyOperationCFG.execute(
            model=model, operation=operation
        )["model"]
    return cast("Any", model).guidance_transforms


def _cfg_replay(pairs: tuple[tuple[str, object], ...], cfg: float) -> torch.Tensor:
    x = _source("latent_a")
    values = {"c": _source("latent_b"), "u": _source("latent_c")}
    token = CancellationToken(lambda: False)

    def evaluate(
        request: GuidanceEvaluationRequest[torch.Tensor],
    ) -> GuidancePredictions[torch.Tensor]:
        return GuidancePredictions(
            tuple(
                GuidancePrediction(lane.id, values[lane.id].clone(), GuidancePredictionSource.MODEL)
                for lane in request.plan.lanes
            )
        )

    executor = GuidanceExecutor(GuidanceRegistry(cast("Any", pairs)))
    conditions = (
        GuidanceCondition("c", GuidanceRole.CONDITIONAL, Conditioning(torch.empty(0), None)),
        GuidanceCondition("u", GuidanceRole.UNCONDITIONAL, Conditioning(torch.empty(0), None)),
    )
    execution = SamplingExecutionContext(
        (0.625, 0.0), 0, 0, 0.625, 1, token, ProgressScope(token), {}
    )
    context = GuidancePlanContext(
        x,
        torch.tensor([0.625] * x.shape[0], dtype=torch.float32),
        cfg,
        conditions,
        False,
        execution,
    )
    return executor.execute(context, evaluate).denoised


def test_latent_apply_operation_cfg_matches_comfy_goldens() -> None:
    runs = _cfg_case_runs()
    assert set(runs) == set(GOLDEN["cfg_cases"])
    for name, (operations, cfg) in runs.items():
        denoised = _cfg_replay(_chained_transform_pairs(operations), cfg)
        torch.testing.assert_close(
            denoised, _tensor(GOLDEN["cfg_cases"][name]), rtol=0, atol=0, msg=name
        )


def test_latent_apply_operation_cfg_empty_chain_is_plain_cfg() -> None:
    assert _chained_transform_pairs(()) == ()
    cond, uncond = _source("latent_b"), _source("latent_c")
    expected = uncond + (cond - uncond) * 3.5
    torch.testing.assert_close(_cfg_replay((), 3.5), expected, rtol=0, atol=0)


def test_latent_apply_operation_cfg_appends_operations_in_chain_order() -> None:
    overlay = native_arm._NativeModelOverlay(_StubHandle(), (), {}, None, None, ())
    tonemap = _tonemap_operation(1.0)
    sharpen = _sharpen_operation(3, 0.5, 0.25)
    first = native_arm.GenerationLatentApplyOperationCFG.execute(model=overlay, operation=tonemap)[
        "model"
    ]
    second = native_arm.GenerationLatentApplyOperationCFG.execute(model=first, operation=sharpen)[
        "model"
    ]
    assert overlay.guidance_transforms == ()
    assert [pair[0] for pair in first.guidance_transforms] == [
        "dinkster.latent.apply_operation_cfg:0"
    ]
    assert [pair[0] for pair in second.guidance_transforms] == [
        "dinkster.latent.apply_operation_cfg:0",
        "dinkster.latent.apply_operation_cfg:1",
    ]
    descriptors = [
        descriptor
        for _, contribution in second.guidance_transforms
        for descriptor in contribution.pre_cfg
    ]
    assert [descriptor.id for descriptor in descriptors] == [
        "dinkster.latent-operation.0",
        "dinkster.latent-operation.1",
    ]
    assert [descriptor.order for descriptor in descriptors] == [0, 1]
    assert all(not descriptor.requires_uncond for descriptor in descriptors)


def test_latent_apply_operation_cfg_rejects_invalid_operation() -> None:
    with pytest.raises(TypeError, match="operation"):
        native_arm.GenerationLatentApplyOperationCFG.execute(
            model=object(), operation="tonemap_reinhard"
        )
