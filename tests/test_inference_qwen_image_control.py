"""Torch-free maintained Qwen Image ControlNet header proofs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    QWEN_IMAGE_DIFFSYNTH,
    QWEN_IMAGE_DIFFSYNTH_INPAINT,
    QWEN_IMAGE_FUN_CONTROL,
    QWEN_IMAGE_INSTANTX_CONTROL,
    QWEN_IMAGE_INSTANTX_INPAINT_CONTROL,
    DType,
    QwenImageControlConfig,
    QwenImageDiffSynthConfig,
    TensorGeometry,
    WeightEntry,
    detect_qwen_image_control,
    detect_qwen_image_diffsynth,
    plan_qwen_image_control,
    plan_qwen_image_diffsynth,
    qwen_image_control_layout,
    qwen_image_diffsynth_layout,
    require_qwen_image_control_layout,
)


@dataclass
class HeaderSource:
    geometries: dict[str, TensorGeometry]
    path: Path = Path("control.safetensors")

    def keys(self) -> Sequence[str]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        raise AssertionError("Qwen Image control detection must remain header-only")


class PlanningHeaderSource(HeaderSource):
    def metadata(self) -> Mapping[str, str]:
        return {}


def source_for(config: QwenImageControlConfig, *, dtype: DType = BFLOAT16) -> HeaderSource:
    return HeaderSource(
        {
            key: TensorGeometry(shape, dtype)
            for key, shape in qwen_image_control_layout(config).items()
        }
    )


def diffsynth_source_for(
    config: QwenImageDiffSynthConfig,
    *,
    dtype: DType = BFLOAT16,
) -> HeaderSource:
    return HeaderSource(
        {
            key: TensorGeometry(shape, dtype)
            for key, shape in qwen_image_diffsynth_layout(config).items()
        }
    )


@pytest.mark.parametrize("config", (QWEN_IMAGE_DIFFSYNTH, QWEN_IMAGE_DIFFSYNTH_INPAINT))
def test_exact_diffsynth_layouts_detect_without_payload_reads(
    config: QwenImageDiffSynthConfig,
) -> None:
    source = diffsynth_source_for(config)
    assert detect_qwen_image_diffsynth(source) is config
    layout = qwen_image_diffsynth_layout(config)
    assert len(layout) == 362
    assert layout["img_in.weight"] == (3072, config.input_features)
    assert layout["controlnet_blocks.0.y_rms.weight"] == (3072,)
    assert layout["controlnet_blocks.59.output_proj.weight"] == (3072, 3072)


def test_diffsynth_detection_fails_closed() -> None:
    source = diffsynth_source_for(QWEN_IMAGE_DIFFSYNTH)
    source.geometries["foreign.weight"] = TensorGeometry((1,), BFLOAT16)
    assert detect_qwen_image_diffsynth(source) is None

    source = diffsynth_source_for(QWEN_IMAGE_DIFFSYNTH_INPAINT)
    source.geometries.pop("controlnet_blocks.59.output_proj.bias")
    assert detect_qwen_image_diffsynth(source) is None

    source = diffsynth_source_for(QWEN_IMAGE_DIFFSYNTH)
    source.geometries["img_in.weight"] = TensorGeometry((3072, 65), BFLOAT16)
    assert detect_qwen_image_diffsynth(source) is None

    source = diffsynth_source_for(QWEN_IMAGE_DIFFSYNTH)
    source.geometries["controlnet_blocks.8.input_proj.weight"] = TensorGeometry(
        (3072, 3071), BFLOAT16
    )
    assert detect_qwen_image_diffsynth(source) is None

    source = diffsynth_source_for(QWEN_IMAGE_DIFFSYNTH, dtype=FLOAT16)
    assert detect_qwen_image_diffsynth(source) is QWEN_IMAGE_DIFFSYNTH
    source.geometries["img_in.bias"] = TensorGeometry((3072,), DType("int4", 4, "int"))
    assert detect_qwen_image_diffsynth(source) is None


@pytest.mark.parametrize("config", (QWEN_IMAGE_DIFFSYNTH, QWEN_IMAGE_DIFFSYNTH_INPAINT))
def test_diffsynth_planner_binds_complete_layout_and_asset_identity(
    config: QwenImageDiffSynthConfig,
) -> None:
    source = PlanningHeaderSource(diffsynth_source_for(config).geometries)
    plan = plan_qwen_image_diffsynth(source, asset_digest="blake3:" + "1" * 64)
    assert plan.source_role == "qwen_image_diffsynth"
    assert plan.patch.config is config
    assert tuple(plan.patch.keys) == tuple(qwen_image_diffsynth_layout(config))
    assert plan.identity_components == (plan.patch,)


def test_diffsynth_planner_accepts_float32_storage() -> None:
    source = PlanningHeaderSource(
        diffsynth_source_for(QWEN_IMAGE_DIFFSYNTH, dtype=FLOAT32).geometries
    )
    plan = plan_qwen_image_diffsynth(source, asset_digest="blake3:" + "1" * 64)
    assert set(plan.patch.dtypes.values()) == {FLOAT32}


def test_diffsynth_planner_refuses_noncanonical_identity_and_incomplete_state() -> None:
    source = PlanningHeaderSource(diffsynth_source_for(QWEN_IMAGE_DIFFSYNTH).geometries)
    with pytest.raises(ValueError, match="canonical blake3"):
        plan_qwen_image_diffsynth(source, asset_digest="sha256:" + "0" * 64)
    source.geometries.pop("controlnet_blocks.14.x_rms.weight")
    with pytest.raises(ValueError, match="exact maintained"):
        plan_qwen_image_diffsynth(source, asset_digest="blake3:" + "0" * 64)


@pytest.mark.parametrize(
    "config",
    (
        QWEN_IMAGE_INSTANTX_CONTROL,
        QWEN_IMAGE_INSTANTX_INPAINT_CONTROL,
        QWEN_IMAGE_FUN_CONTROL,
    ),
)
def test_exact_control_layouts_detect_without_payload_reads(
    config: QwenImageControlConfig,
) -> None:
    source = source_for(config)
    assert detect_qwen_image_control(source) is config
    assert require_qwen_image_control_layout(source) == (
        config,
        qwen_image_control_layout(config),
    )


def test_layouts_pin_maintained_signatures_and_complete_key_counts() -> None:
    instant = qwen_image_control_layout(QWEN_IMAGE_INSTANTX_CONTROL)
    inpaint = qwen_image_control_layout(QWEN_IMAGE_INSTANTX_INPAINT_CONTROL)
    fun = qwen_image_control_layout(QWEN_IMAGE_FUN_CONTROL)
    assert len(instant) == len(inpaint) == 2051
    assert instant["controlnet_x_embedder.weight"] == (3072, 64)
    assert inpaint["controlnet_x_embedder.weight"] == (3072, 68)
    assert "proj_out.weight" not in instant
    assert instant["transformer_blocks.59.img_mlp.net.0.proj.weight"] == (12288, 3072)
    assert len(fun) == 174
    assert fun["control_img_in.weight"] == (3072, 132)
    assert fun["control_blocks.0.before_proj.weight"] == (3072, 3072)
    assert "control_blocks.1.before_proj.weight" not in fun
    assert fun["control_blocks.4.after_proj.weight"] == (3072, 3072)


def test_detection_fails_closed_on_foreign_missing_shape_or_dtype_state() -> None:
    source = source_for(QWEN_IMAGE_FUN_CONTROL)
    source.geometries["foreign.weight"] = TensorGeometry((1,), BFLOAT16)
    assert detect_qwen_image_control(source) is None

    source = source_for(QWEN_IMAGE_INSTANTX_CONTROL)
    source.geometries.pop("controlnet_blocks.59.bias")
    assert detect_qwen_image_control(source) is None

    source = source_for(QWEN_IMAGE_INSTANTX_INPAINT_CONTROL)
    source.geometries["controlnet_x_embedder.weight"] = TensorGeometry((3072, 65), BFLOAT16)
    assert detect_qwen_image_control(source) is None

    source = source_for(QWEN_IMAGE_FUN_CONTROL)
    source.geometries["control_img_in.weight"] = TensorGeometry((3072, 132), FLOAT16)
    assert detect_qwen_image_control(source) is QWEN_IMAGE_FUN_CONTROL
    source.geometries["control_img_in.weight"] = TensorGeometry(
        (3072, 132),
        DType("int4", 4, "int"),
    )
    assert detect_qwen_image_control(source) is None


def test_require_layout_names_detection_refusal() -> None:
    with pytest.raises(ValueError, match="exact maintained"):
        require_qwen_image_control_layout(HeaderSource({}))
    forged = QwenImageControlConfig("fun", 132, 5, (0, 12, 24, 36, 48))
    with pytest.raises(ValueError, match="exact Qwen Image control profile"):
        qwen_image_control_layout(forged)


@pytest.mark.parametrize(
    "config",
    (
        QWEN_IMAGE_INSTANTX_CONTROL,
        QWEN_IMAGE_INSTANTX_INPAINT_CONTROL,
        QWEN_IMAGE_FUN_CONTROL,
    ),
)
def test_control_planner_maps_complete_layout_and_binds_asset_identity(
    config: QwenImageControlConfig,
) -> None:
    source = PlanningHeaderSource(source_for(config).geometries)
    plan = plan_qwen_image_control(source, asset_digest="blake3:" + "0" * 64)
    assert plan.source_role == "qwen_image_control"
    assert plan.control.config is config
    assert tuple(plan.control.keys) == tuple(qwen_image_control_layout(config))
    assert plan.identity_components == (plan.control,)


def test_control_planner_accepts_float32_storage() -> None:
    source = PlanningHeaderSource(source_for(QWEN_IMAGE_FUN_CONTROL, dtype=FLOAT32).geometries)
    plan = plan_qwen_image_control(source, asset_digest="blake3:" + "0" * 64)
    assert set(plan.control.dtypes.values()) == {FLOAT32}


def test_control_planner_refuses_noncanonical_identity_and_incomplete_state() -> None:
    source = PlanningHeaderSource(source_for(QWEN_IMAGE_FUN_CONTROL).geometries)
    with pytest.raises(ValueError, match="canonical blake3"):
        plan_qwen_image_control(source, asset_digest="sha256:" + "0" * 64)
    source.geometries.pop("control_blocks.4.after_proj.bias")
    with pytest.raises(ValueError, match="exact maintained"):
        plan_qwen_image_control(source, asset_digest="blake3:" + "0" * 64)
