"""Basic conditioning carrier adapter proofs."""

from __future__ import annotations

import pytest
import torch
from dinkster_inference import (
    AreaDescriptor,
    AreaUnits,
    Conditioning,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    ConditionScaleVector,
    MaskDescriptor,
    PayloadBinding,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    make_conditioning_carrier,
)
from dinkster_inference_torch import (
    CONDITIONING_POOLED_SPACE,
    CONDITIONING_TEXT_SPACE,
    basic_conditioning_to_carrier,
    materialize_basic_conditioning,
    tensor_to_payload_binding,
)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_round_trip_without_pooled(dtype: torch.dtype) -> None:
    embeddings = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4).to(dtype)
    carrier = basic_conditioning_to_carrier(Conditioning(embeddings))
    restored = materialize_basic_conditioning(carrier, device="cpu")
    assert restored.pooled is None
    assert restored.embeddings.dtype == dtype
    assert restored.embeddings.device == torch.device("cpu")
    assert torch.equal(restored.embeddings, embeddings)


def test_round_trip_with_pooled() -> None:
    embeddings = torch.randn(1, 77, 768)
    pooled = torch.randn(1, 1280, dtype=torch.float16)
    carrier = basic_conditioning_to_carrier(Conditioning(embeddings, pooled))
    restored = materialize_basic_conditioning(carrier, device=torch.device("cpu"))
    assert torch.equal(restored.embeddings, embeddings)
    assert restored.pooled is not None
    assert restored.pooled.dtype == torch.float16
    assert torch.equal(restored.pooled, pooled)


def test_carrier_shape_is_one_full_schedule_record() -> None:
    carrier = basic_conditioning_to_carrier(Conditioning(torch.zeros(1, 2, 4)))
    (record,) = carrier.conditioning.records
    assert record.schedule == PercentRange(0.0, 1.0)
    assert record.area is None
    assert record.mask is None
    assert record.scale_vector is None
    assert record.extension_metadata == ()
    assert [channel for channel, _ in record.channels] == [ConditioningChannel.TEXT]
    (descriptor,) = [payload for _, payload in record.channels]
    assert descriptor.space == CONDITIONING_TEXT_SPACE


def test_token_layout_passes_through() -> None:
    layout = TokenLayoutDescriptor(
        "dinkster.test",
        1,
        ("text",),
        (TokenSegmentDescriptor("prompt", "text", 0, 2),),
    )
    carrier = basic_conditioning_to_carrier(Conditioning(torch.zeros(1, 2, 4)), token_layout=layout)
    (record,) = carrier.conditioning.records
    assert record.token_layout == layout
    restored = materialize_basic_conditioning(carrier, device="cpu")
    assert restored.embeddings.shape == (1, 2, 4)


def test_to_carrier_rejects_non_conditioning() -> None:
    with pytest.raises(TypeError, match="must be a Conditioning"):
        basic_conditioning_to_carrier(object())  # type: ignore[arg-type]


def test_materialize_rejects_non_carrier() -> None:
    with pytest.raises(TypeError, match="must be a ConditioningCarrier"):
        materialize_basic_conditioning(object(), device="cpu")  # type: ignore[arg-type]


def _text_descriptor_and_binding() -> tuple[PayloadDescriptor, PayloadBinding]:
    binding = tensor_to_payload_binding("text", torch.zeros(1, 2, 4), space=CONDITIONING_TEXT_SPACE)
    descriptor = PayloadDescriptor(
        PayloadReference("text"), binding.shape, binding.dtype, binding.space
    )
    return descriptor, binding


def _carrier_with(record: ConditioningRecord, *extra: PayloadBinding) -> ConditioningCarrier:
    _, text_binding = _text_descriptor_and_binding()
    return make_conditioning_carrier(ConditioningSet((record,)), (text_binding, *extra))


def _basic_record(**overrides: object) -> ConditioningRecord:
    descriptor, _ = _text_descriptor_and_binding()
    fields: dict[str, object] = {"channels": ((ConditioningChannel.TEXT, descriptor),)}
    fields.update(overrides)
    return ConditioningRecord(**fields)  # type: ignore[arg-type]


def test_materialize_refuses_multiple_records() -> None:
    record = _basic_record()
    _, text_binding = _text_descriptor_and_binding()
    carrier = make_conditioning_carrier(ConditioningSet((record, record)), (text_binding,))
    with pytest.raises(ValueError, match="exactly one record"):
        materialize_basic_conditioning(carrier, device="cpu")


def test_materialize_refuses_area() -> None:
    record = _basic_record(area=AreaDescriptor(0.5, 0.5, 0.0, 0.0, AreaUnits.PERCENT))
    with pytest.raises(ValueError, match="cannot carry an area"):
        materialize_basic_conditioning(_carrier_with(record), device="cpu")


def test_materialize_refuses_mask() -> None:
    mask_binding = PayloadBinding("mask", (2, 2), "U8", "mask", b"mask")
    record = _basic_record(mask=MaskDescriptor(PayloadReference("mask")))
    with pytest.raises(ValueError, match="cannot carry a mask"):
        materialize_basic_conditioning(_carrier_with(record, mask_binding), device="cpu")


def test_materialize_refuses_scale_vector() -> None:
    scale_binding = tensor_to_payload_binding("scale", torch.ones(1), space="scale")
    scale_descriptor = PayloadDescriptor(
        PayloadReference("scale"), scale_binding.shape, scale_binding.dtype, scale_binding.space
    )
    record = _basic_record(scale_vector=ConditionScaleVector(scale_descriptor))
    with pytest.raises(ValueError, match="condition scale vector"):
        materialize_basic_conditioning(_carrier_with(record, scale_binding), device="cpu")


def test_materialize_refuses_extension_metadata() -> None:
    record = _basic_record(extension_metadata=(("pack/flag", True),))
    with pytest.raises(ValueError, match="extension metadata"):
        materialize_basic_conditioning(_carrier_with(record), device="cpu")


def test_materialize_refuses_partial_schedule() -> None:
    record = _basic_record(schedule=PercentRange(0.2, 0.8))
    with pytest.raises(ValueError, match="full schedule"):
        materialize_basic_conditioning(_carrier_with(record), device="cpu")


def test_materialize_refuses_missing_text_channel() -> None:
    pooled_binding = tensor_to_payload_binding(
        "pooled", torch.zeros(1, 4), space=CONDITIONING_POOLED_SPACE
    )
    descriptor = PayloadDescriptor(
        PayloadReference("pooled"),
        pooled_binding.shape,
        pooled_binding.dtype,
        pooled_binding.space,
    )
    record = ConditioningRecord(channels=((ConditioningChannel.POOLED, descriptor),))
    carrier = make_conditioning_carrier(ConditioningSet((record,)), (pooled_binding,))
    with pytest.raises(ValueError, match="must carry a TEXT channel"):
        materialize_basic_conditioning(carrier, device="cpu")


def test_materialize_refuses_extra_channel() -> None:
    text_descriptor, text_binding = _text_descriptor_and_binding()
    vision_binding = tensor_to_payload_binding("vision", torch.zeros(1, 4), space="vision")
    vision_descriptor = PayloadDescriptor(
        PayloadReference("vision"),
        vision_binding.shape,
        vision_binding.dtype,
        vision_binding.space,
    )
    record = ConditioningRecord(
        channels=(
            (ConditioningChannel.TEXT, text_descriptor),
            (ConditioningChannel.VISION_EMBEDDING, vision_descriptor),
        )
    )
    carrier = make_conditioning_carrier(ConditioningSet((record,)), (text_binding, vision_binding))
    with pytest.raises(ValueError, match="cannot carry channels: vision_embedding"):
        materialize_basic_conditioning(carrier, device="cpu")


def test_materialize_refuses_wrong_payload_space() -> None:
    binding = tensor_to_payload_binding("text", torch.zeros(1, 2, 4), space="text")
    descriptor = PayloadDescriptor(
        PayloadReference("text"), binding.shape, binding.dtype, binding.space
    )
    record = ConditioningRecord(channels=((ConditioningChannel.TEXT, descriptor),))
    carrier = make_conditioning_carrier(ConditioningSet((record,)), (binding,))
    with pytest.raises(ValueError, match="payload space must be 'conditioning-text'"):
        materialize_basic_conditioning(carrier, device="cpu")


def test_materialize_refuses_unbound_payload() -> None:
    descriptor, _ = _text_descriptor_and_binding()
    record = ConditioningRecord(channels=((ConditioningChannel.TEXT, descriptor),))
    carrier = ConditioningCarrier(ConditioningSet((record,)), ())
    with pytest.raises(ValueError, match="conditioning-wire:unbound-reference"):
        materialize_basic_conditioning(carrier, device="cpu")


def test_materialize_refuses_non_content_addressed_bindings() -> None:
    descriptor, binding = _text_descriptor_and_binding()
    record = ConditioningRecord(channels=((ConditioningChannel.TEXT, descriptor),))
    carrier = ConditioningCarrier(ConditioningSet((record,)), (binding,))
    with pytest.raises(ValueError, match="conditioning-wire:content-id-mismatch"):
        materialize_basic_conditioning(carrier, device="cpu")


def test_materialize_refuses_unknown_extra_binding() -> None:
    canonical = basic_conditioning_to_carrier(Conditioning(torch.zeros(1, 2, 4)))
    extra = tensor_to_payload_binding("extra", torch.ones(1), space="extra")
    tampered = ConditioningCarrier(canonical.conditioning, (*canonical.bindings, extra))
    with pytest.raises(ValueError, match="conditioning-wire:"):
        materialize_basic_conditioning(tampered, device="cpu")


def test_materialize_refuses_duplicate_bindings() -> None:
    canonical = basic_conditioning_to_carrier(Conditioning(torch.zeros(1, 2, 4)))
    binding = canonical.bindings[0]
    tampered = ConditioningCarrier(canonical.conditioning, (binding, binding))
    with pytest.raises(ValueError, match="conditioning-wire:duplicate-content-id"):
        materialize_basic_conditioning(tampered, device="cpu")
