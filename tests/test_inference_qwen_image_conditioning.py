"""Qwen Image S6A torch-free conditioning contract tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
from dinkster_inference.qwen_image_conditioning import (
    QWEN_IMAGE_WAN21_NORMALIZATION,
    QwenImageAttentionMask,
    QwenImageConditioning,
    QwenImageCrossAttention,
    QwenImageLatentSnapshot,
    QwenImageReferenceMethod,
    plan_qwen_image_references,
)


def latent(
    height: int,
    width: int,
    *,
    batch: int = 1,
    frames: int = 1,
) -> QwenImageLatentSnapshot:
    return QwenImageLatentSnapshot((batch, 16, frames, height, width))


def test_wan21_normalization_facts_are_exact_immutable_and_invertible() -> None:
    facts = QWEN_IMAGE_WAN21_NORMALIZATION
    assert facts.channels == 16
    assert facts.scale_factor == 1.0
    assert facts.mean == (
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921,
    )
    assert facts.std == (
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.9160,
    )
    assert facts.process_in_formula == "(latent - mean) * scale_factor / std"
    assert facts.process_out_formula == "latent * std / scale_factor + mean"
    for channel, value in enumerate((-2.0, 0.0, 3.5, 9.0)):
        normalized = facts.process_in_scalar(channel, value)
        assert facts.process_out_scalar(channel, normalized) == pytest.approx(value)
    with pytest.raises(FrozenInstanceError):
        facts.channels = 1  # type: ignore[misc]


@pytest.mark.parametrize("channel", (-1, 16, True, 1.0))
def test_wan21_normalization_refuses_invalid_channels(channel: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        QWEN_IMAGE_WAN21_NORMALIZATION.process_in_scalar(channel, 0.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", (True, float("nan"), float("inf")))
def test_wan21_normalization_refuses_non_finite_or_bool_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        QWEN_IMAGE_WAN21_NORMALIZATION.process_out_scalar(0, value)  # type: ignore[arg-type]


def test_conditioning_snapshots_order_mask_and_cross_attention_immutably() -> None:
    references = [latent(5, 7), latent(8, 4)]
    conditioning = QwenImageConditioning(
        cross_attention=QwenImageCrossAttention((1, 13, 3584)),
        attention_mask=QwenImageAttentionMask((1, 13), "non_floating"),
        references=references,
        reference_method="index",
    )
    references.reverse()
    assert conditioning.cross_attention.owner == "c_crossattn"
    assert conditioning.attention_mask is not None
    assert conditioning.attention_mask.owner == "attention_mask"
    assert conditioning.references == (latent(5, 7), latent(8, 4))
    assert conditioning.reference_method == "index"
    with pytest.raises(FrozenInstanceError):
        conditioning.reference_method = "negative_index"  # type: ignore[misc]


@pytest.mark.parametrize(
    "shape",
    (
        (1, 15, 1, 8, 8),
        (1, 16, 0, 8, 8),
        (1, 16, 1, -1, 8),
        (1, 16, 1, 8),
        [1, 16, 1, 8, 8],
        (True, 16, 1, 8, 8),
    ),
)
def test_reference_latent_refuses_wrong_rank_channel_or_extent(shape: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        QwenImageLatentSnapshot(shape)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("cross_shape", "mask_shape", "mask_kind"),
    (
        ((1, 4, 3583), None, None),
        ((0, 4, 3584), None, None),
        ((1, 4, 3584), (2, 4), "non_floating"),
        ((1, 4, 3584), (1, 3), "floating"),
        ((1, 4, 3584), (1, 4), "int"),
    ),
)
def test_conditioning_refuses_cross_attention_and_mask_mismatch(
    cross_shape: tuple[int, int, int],
    mask_shape: tuple[int, int] | None,
    mask_kind: str | None,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        cross = QwenImageCrossAttention(cross_shape)
        mask = (
            None if mask_shape is None else QwenImageAttentionMask(mask_shape, mask_kind)  # type: ignore[arg-type]
        )
        QwenImageConditioning(cross, mask, (), "index")


def test_conditioning_refuses_reference_batch_cardinality_and_method() -> None:
    cross = QwenImageCrossAttention((2, 5, 3584))
    with pytest.raises(ValueError, match="batch"):
        QwenImageConditioning(cross, None, (latent(8, 8),), "index")
    with pytest.raises(ValueError, match="method"):
        QwenImageConditioning(
            QwenImageCrossAttention((1, 5, 3584)),
            None,
            (latent(8, 8),),
            "index_timestep_zero",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("method", "indices"),
    (("index", (1, 2, 3)), ("negative_index", (-1, -2, -3))),
)
def test_index_reference_plans_preserve_order_and_exact_token_offsets(
    method: QwenImageReferenceMethod, indices: tuple[int, ...]
) -> None:
    target = latent(7, 9, frames=2)
    references = (latent(5, 7), latent(4, 4, frames=3), latent(2, 9))
    plan = plan_qwen_image_references(target, references, method)
    assert plan.target_token_count == 2 * 4 * 5
    assert plan.reference_token_counts == (3 * 4, 3 * 2 * 2, 1 * 1 * 5)
    assert tuple(item.index for item in plan.references) == indices
    assert tuple(item.temporal_id_range for item in plan.references) == (
        (indices[0], indices[0]),
        (0, 2),
        (indices[2], indices[2]),
    )
    assert tuple(item.token_offset for item in plan.references) == (40, 52, 64)
    assert all((item.height_offset, item.width_offset) == (0, 0) for item in plan.references)
    assert plan.total_image_tokens == 69
    assert (plan.flow_multiplier, plan.flow_shift) == (1.0, 1.15)


def test_spatial_offset_plan_matches_reference_packing_order() -> None:
    references = (latent(8, 4), latent(3, 9), latent(4, 4))
    plan = plan_qwen_image_references(latent(8, 8), references, "spatial_offset")
    assert plan.reference_token_counts == (8, 10, 4)
    assert tuple(
        (item.index, item.height_offset, item.width_offset) for item in plan.references
    ) == ((1, 0, 0), (1, 8, 0), (1, 0, 9))
    assert tuple(
        (item.patch_height_offset, item.patch_width_offset) for item in plan.references
    ) == ((0, 0), (4, 0), (0, 5))


def test_reference_plan_snapshots_inputs_and_refuses_mixed_batches() -> None:
    refs = [latent(4, 4)]
    plan = plan_qwen_image_references(latent(8, 8), refs, "index")
    refs.clear()
    assert len(plan.references) == 1
    with pytest.raises(ValueError, match="batch"):
        plan_qwen_image_references(latent(8, 8), (latent(4, 4, batch=2),), "index")
    with pytest.raises(FrozenInstanceError):
        plan.total_image_tokens = 0  # type: ignore[misc]


def test_fixed_contract_fields_cannot_be_replaced() -> None:
    with pytest.raises(ValueError, match="exact Wan21"):
        replace(QWEN_IMAGE_WAN21_NORMALIZATION, scale_factor=2.0)
    with pytest.raises(ValueError, match="exact Qwen Image flow"):
        replace(
            plan_qwen_image_references(latent(8, 8), (), "index"),
            flow_shift=1.0,
        )
    one_token = plan_qwen_image_references(latent(1, 1), (), "index")
    with pytest.raises(TypeError, match="target token count"):
        replace(one_token, target_token_count=True)
    with pytest.raises(TypeError, match="reference token counts"):
        replace(one_token, reference_token_counts=[])
    with pytest.raises(TypeError, match="total image tokens"):
        replace(one_token, total_image_tokens=True)


@pytest.mark.parametrize(
    "changes",
    (
        {"index": -1},
        {"height_offset": 2, "patch_height_offset": 1},
        {"token_offset": 0},
    ),
)
def test_reference_plan_refuses_replaced_derived_placement_facts(
    changes: dict[str, int],
) -> None:
    plan = plan_qwen_image_references(latent(8, 8), (latent(4, 4),), "index")
    changed = replace(plan.references[0], **changes)
    with pytest.raises(ValueError, match="exact packing plan"):
        replace(plan, references=(changed,))


def test_reference_placement_refuses_mismatched_patch_offset() -> None:
    plan = plan_qwen_image_references(latent(8, 8), (latent(8, 4), latent(3, 9)), "spatial_offset")
    with pytest.raises(ValueError, match="patch height offset"):
        replace(plan.references[1], patch_height_offset=0)


def test_reference_plan_refuses_replaced_method_or_target_batch() -> None:
    plan = plan_qwen_image_references(latent(8, 8), (latent(4, 4),), "index")
    with pytest.raises(ValueError, match="exact packing plan"):
        replace(plan, method="negative_index")
    with pytest.raises(ValueError, match="batch"):
        replace(plan, target=latent(8, 8, batch=2))
