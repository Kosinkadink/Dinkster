"""Adapters between text-encoder results and canonical conditioning carriers.

A basic carrier holds exactly one full-schedule record with a TEXT channel
and an optional POOLED channel - the direct product of one
``FamilyRuntime.encode_text`` call. Materialization refuses every richer
carrier (regional, masked, scaled, scheduled, or extension-bearing) because
collapsing one into a plain ``Conditioning`` would silently discard
semantics; consumers of rich carriers must speak records directly.
"""

from __future__ import annotations

import torch
from dinkster_inference import (
    Conditioning,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningPayloadBinding,
    ConditioningRecord,
    ConditioningSet,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    TokenLayoutDescriptor,
    encode_conditioning_carrier,
    make_conditioning_carrier,
)

from .payloads import payload_binding_to_tensor, tensor_to_payload_binding

CONDITIONING_TEXT_SPACE = "conditioning-text"
CONDITIONING_POOLED_SPACE = "conditioning-pooled"

_CHANNEL_SPACES = {
    ConditioningChannel.TEXT: CONDITIONING_TEXT_SPACE,
    ConditioningChannel.POOLED: CONDITIONING_POOLED_SPACE,
}


def basic_conditioning_to_carrier(
    value: Conditioning[torch.Tensor],
    *,
    token_layout: TokenLayoutDescriptor | None = None,
    reference_prefix: str = "",
) -> ConditioningCarrier:
    """One encode_text result -> a canonical single-record carrier."""

    if not isinstance(value, Conditioning):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"value must be a Conditioning, got {type(value).__name__}")
    if type(reference_prefix) is not str:
        raise TypeError("reference prefix must be a string")
    prefix = f"{reference_prefix}-" if reference_prefix else ""
    text = tensor_to_payload_binding(
        f"{prefix}text", value.embeddings, space=CONDITIONING_TEXT_SPACE
    )
    bindings = [text]
    channels: list[tuple[ConditioningChannel, PayloadDescriptor]] = [
        (
            ConditioningChannel.TEXT,
            PayloadDescriptor(
                PayloadReference(text.reference_id), text.shape, text.dtype, text.space
            ),
        )
    ]
    if value.pooled is not None:
        pooled = tensor_to_payload_binding(
            f"{prefix}pooled", value.pooled, space=CONDITIONING_POOLED_SPACE
        )
        bindings.append(pooled)
        channels.append(
            (
                ConditioningChannel.POOLED,
                PayloadDescriptor(
                    PayloadReference(pooled.reference_id),
                    pooled.shape,
                    pooled.dtype,
                    pooled.space,
                ),
            )
        )
    record = ConditioningRecord(channels=tuple(channels), token_layout=token_layout)
    return make_conditioning_carrier(ConditioningSet((record,)), tuple(bindings))


def materialize_basic_conditioning(
    carrier: ConditioningCarrier,
    *,
    device: torch.device | str,
) -> Conditioning[torch.Tensor]:
    """One basic carrier -> tensors on ``device``; refuses richer carriers."""

    if not isinstance(carrier, ConditioningCarrier):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(f"carrier must be a ConditioningCarrier, got {type(carrier).__name__}")
    encode_conditioning_carrier(carrier)
    records = carrier.conditioning.records
    if len(records) != 1:
        raise ValueError(f"basic conditioning carries exactly one record, got {len(records)}")
    record = records[0]
    if record.area is not None:
        raise ValueError("basic conditioning cannot carry an area")
    if record.mask is not None:
        raise ValueError("basic conditioning cannot carry a mask")
    if record.scale_vector is not None:
        raise ValueError("basic conditioning cannot carry a condition scale vector")
    if record.extension_metadata:
        raise ValueError("basic conditioning cannot carry extension metadata")
    schedule = record.schedule
    if not isinstance(schedule, PercentRange) or (
        schedule.start_percent,
        schedule.end_percent,
    ) != (0.0, 1.0):
        raise ValueError("basic conditioning must cover the full schedule")
    channels = dict(record.channels)
    text_descriptor = channels.pop(ConditioningChannel.TEXT, None)
    if text_descriptor is None:
        raise ValueError("basic conditioning must carry a TEXT channel")
    pooled_descriptor = channels.pop(ConditioningChannel.POOLED, None)
    if channels:
        names = ", ".join(sorted(channel.value for channel in channels))
        raise ValueError(f"basic conditioning cannot carry channels: {names}")
    bindings = {binding.reference_id: binding for binding in carrier.bindings}
    embeddings = _materialize(text_descriptor, bindings, ConditioningChannel.TEXT, device)
    pooled = (
        None
        if pooled_descriptor is None
        else _materialize(pooled_descriptor, bindings, ConditioningChannel.POOLED, device)
    )
    return Conditioning(embeddings, pooled)


def _materialize(
    descriptor: PayloadDescriptor,
    bindings: dict[str, ConditioningPayloadBinding],
    channel: ConditioningChannel,
    device: torch.device | str,
) -> torch.Tensor:
    expected_space = _CHANNEL_SPACES[channel]
    if descriptor.space != expected_space:
        raise ValueError(
            f"{channel.value} channel payload space must be {expected_space!r}, "
            f"got {descriptor.space!r}"
        )
    # Canonical carrier validation already proved every descriptor's
    # reference is bound by a byte-identical binding.
    return payload_binding_to_tensor(bindings[descriptor.reference.id]).to(device)


__all__ = [
    "CONDITIONING_POOLED_SPACE",
    "CONDITIONING_TEXT_SPACE",
    "basic_conditioning_to_carrier",
    "materialize_basic_conditioning",
]
