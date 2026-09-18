"""Torch-free conditioning records and executed ComfyUI goldens."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import MappingProxyType

import pytest
from dinkster_inference import (
    BUILTIN_METADATA_MERGE_TABLE,
    CONDITIONING_RECORD_BOUNDARY,
    EMPTY_RANGE,
    AreaDescriptor,
    AreaUnits,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    ConditionScaleVector,
    ContinuousEDMSigmas,
    DiscreteSigmas,
    FlowSigmas,
    FluxFlowSigmas,
    MaskDescriptor,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    RegionDescriptor,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    canonical_conditioning_set,
)

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "conditioning_goldens.json").read_text())


def payload(name: str, shape: tuple[int, ...] = (1, 77, 768)) -> PayloadDescriptor:
    return PayloadDescriptor(PayloadReference(name), shape, "float32", "worker:test")


def layout() -> TokenLayoutDescriptor:
    return TokenLayoutDescriptor(
        "dinkster.sd15",
        1,
        ("clip_l",),
        (TokenSegmentDescriptor("prompt", "clip_l", 0, 77),),
    )


def full_record(name: str = "text") -> ConditioningRecord:
    text = payload(name)
    scale = payload(f"{name}-scale", (1,))
    mask_ref = PayloadReference(f"{name}-mask")
    return ConditioningRecord(
        channels=((ConditioningChannel.TEXT, text),),
        area=AreaDescriptor(8, 10, 2, 3, AreaUnits.LATENT_CELLS, 1.25),
        mask=MaskDescriptor(mask_ref, 0.6, True),
        schedule=PercentRange(0.2, 0.8),
        scale_vector=ConditionScaleVector(scale),
        token_layout=layout(),
        extension_metadata=(
            ("pack.z/data", {"nested": [1, True, None]}),
            ("pack.a/payload", mask_ref),
        ),
    )


def test_executed_comfyui_composition_and_descriptor_goldens() -> None:
    assert GOLDENS["comfyui_commit"] == "b78cec879b9460d5cb25228a83a942fb78d2cd24"
    assert GOLDENS["clone"] == {
        "metadata_dict_copied": True,
        "nested_metadata_shared": True,
        "payload_shared": True,
        "source_unchanged": True,
    }
    assert GOLDENS["combine_order"] == ["left", "right"]
    area_goldens = GOLDENS["area"]
    assert {
        key: value
        for key, value in area_goldens.items()
        if key != "percent_video_get_area_and_mult"
    } == {
        "latent_cells": [8, 10, 2, 3],
        "percent_stored": ["percentage", 0.5, 0.5, 0.5, 0.5],
        "percent_video_stored": ["percentage", 0.75, 0.25, 0.5, 0.25, 0.0, 0.125],
        "percent_resolved_7x9": [4, 4, 4, 4],
        "percent_video_resolved_8x12x16": [6, 3, 8, 2, 0, 2],
        "percent_video_tiny_resolved_8x12x16": [1, 1, 1, 2, 0, 2],
    }
    applied_video = area_goldens["percent_video_get_area_and_mult"]
    assert applied_video["area"] == [6, 3, 8, 2, 0, 2]
    assert applied_video["input_shape"] == [1, 1, 6, 3, 8]
    multiplier = applied_video["multiplier"]
    assert multiplier["dtype"] == "float32"
    assert multiplier["shape"] == [1, 1, 6, 3, 8]
    assert hashlib.sha256(bytes.fromhex(multiplier["data_hex"])).hexdigest() == (
        "a1b236e58608ead153c54ed4e48e0d7840dabea8140d8283c2d6036cc0e48431"
    )
    assert GOLDENS["mask"]["default"] == {
        "shape": [1, 2, 2],
        "strength": 0.6,
        "set_area_to_bounds": False,
    }
    assert GOLDENS["mask"]["bounds"]["set_area_to_bounds"] is True

    cells = GOLDENS["area"]["latent_cells"]
    latent = AreaDescriptor(cells[0], cells[1], cells[2], cells[3], AreaUnits.LATENT_CELLS, 1.25)
    assert (latent.height, latent.width, latent.y, latent.x) == tuple(
        GOLDENS["area"]["latent_cells"]
    )
    mask_ref = PayloadReference("golden-mask")
    for name in ("default", "bounds"):
        golden = GOLDENS["mask"][name]
        descriptor = MaskDescriptor(
            mask_ref,
            golden["strength"],
            golden["set_area_to_bounds"],
        )
        assert descriptor.strength == golden["strength"]
        assert descriptor.set_area_to_bounds is golden["set_area_to_bounds"]


def test_percent_area_rounding_matches_executed_comfyui_golden() -> None:
    stored = GOLDENS["area"]["percent_stored"]
    assert stored[0] == "percentage"
    area = AreaDescriptor(stored[1], stored[2], stored[3], stored[4], AreaUnits.PERCENT, 0.75)
    assert list(area.materialize_percent(7, 9)) == GOLDENS["area"]["percent_resolved_7x9"]


def test_percent_video_area_preserves_pinned_comfyui_axis_order() -> None:
    stored = GOLDENS["area"]["percent_video_stored"]
    assert stored[0] == "percentage"
    area = AreaDescriptor(
        height=stored[2],
        width=stored[3],
        y=stored[5],
        x=stored[6],
        units=AreaUnits.PERCENT,
        strength=0.6,
        temporal=stored[1],
        z=stored[4],
    )
    assert (area.temporal, area.height, area.width, area.z, area.y, area.x) == tuple(stored[1:])
    assert (
        list(area.materialize_percent_video(8, 12, 16))
        == GOLDENS["area"]["percent_video_resolved_8x12x16"]
    )
    tiny = AreaDescriptor(
        height=0.01,
        width=0.01,
        y=0.0,
        x=0.125,
        units=AreaUnits.PERCENT,
        temporal=0.01,
        z=0.25,
    )
    assert (
        list(tiny.materialize_percent_video(8, 12, 16))
        == GOLDENS["area"]["percent_video_tiny_resolved_8x12x16"]
    )
    with pytest.raises(ValueError, match="video materialization"):
        area.materialize_percent(12, 16)


def test_temporal_area_requires_paired_percent_fields() -> None:
    with pytest.raises(ValueError, match="provided together"):
        AreaDescriptor(0.5, 0.5, 0.0, 0.0, AreaUnits.PERCENT, temporal=0.5)
    with pytest.raises(ValueError, match="percent units"):
        AreaDescriptor(8, 8, 0, 0, AreaUnits.LATENT_CELLS, temporal=0.5, z=0.0)
    with pytest.raises(TypeError, match="must be floats"):
        AreaDescriptor(0.5, 0.5, 0.0, 0.0, AreaUnits.PERCENT, temporal=1, z=0.0)


def test_percent_to_sigma_matches_executed_comfyui_numeric_goldens() -> None:
    spaces = {
        "discrete": DiscreteSigmas.linear_beta(),
        "flow": FlowSigmas(),
        "flux": FluxFlowSigmas(),
        "continuous_edm": ContinuousEDMSigmas(),
    }
    for name, space in spaces.items():
        expected = GOLDENS["sigma_goldens"][name]
        actual = [space.percent_to_sigma(value) for value in GOLDENS["percent_probes"]]
        assert len(actual) == len(expected)
        for ours, reference in zip(actual, expected, strict=True):
            assert math.isclose(ours, reference, rel_tol=3e-6, abs_tol=1e-7)


def test_zero_width_range_is_closed_and_active_for_node_and_hook_goldens() -> None:
    value = PercentRange(0.5, 0.5)
    for surface in ("node", "hook"):
        probes = GOLDENS["zero_width_range"][surface]["probes"]
        assert [value.is_active(case["sigma"], FlowSigmas()) for case in probes] == [
            case["active"] for case in probes
        ]
    assert value.to_percent_pair() == (0.5, 0.5)
    assert value is not EMPTY_RANGE


def test_distinct_empty_range_has_all_five_adjudicated_properties() -> None:
    left = PercentRange(0.0, 0.2)
    right = PercentRange(0.8, 1.0)
    empty = left.intersect(right)
    assert empty is EMPTY_RANGE
    assert all(not empty.is_active(sigma, FlowSigmas()) for sigma in (-1.0, 0.0, 0.5, 1.0, 2.0))
    assert empty.intersect(left) is EMPTY_RANGE
    assert left.intersect(EMPTY_RANGE) is EMPTY_RANGE
    assert PercentRange(0.4, 0.4) is not EMPTY_RANGE
    with pytest.raises(ValueError, match="no percent-pair"):
        empty.to_percent_pair()
    record = ConditioningRecord(((ConditioningChannel.TEXT, payload("x")),), schedule=empty)
    assert canonical_conditioning_set(ConditioningSet((record,))) == (
        '{"format":"dinkster-conditioning-set-v1","records":[{"area":null,'
        '"channels":[{"id":"text","payload":{"dtype":"float32","ref":"x",'
        '"shape":[1,77,768],"space":"worker:test"}}],"extension_metadata":{},'
        '"mask":null,"scale_vector":null,"schedule":{"kind":"empty"},'
        '"token_layout":null}]}'
    )


def test_range_intersection_is_closed_and_deterministic() -> None:
    assert PercentRange(0.1, 0.5).intersect(PercentRange(0.5, 0.9)) == PercentRange(0.5, 0.5)
    assert PercentRange(0.1, 0.8).intersect(PercentRange(0.2, 0.7)) == PercentRange(0.2, 0.7)


@pytest.mark.parametrize(
    "start,end,error",
    [
        (-0.1, 0.5, ValueError),
        (0.5, 1.1, ValueError),
        (0.8, 0.2, ValueError),
        (math.nan, 0.5, ValueError),
        (0.0, math.inf, ValueError),
        (0, 1.0, TypeError),
    ],
)
def test_range_constructor_refuses_invalid_values(
    start: float, end: float, error: type[Exception]
) -> None:
    with pytest.raises(error):
        PercentRange(start, end)


def test_clone_deep_copies_metadata_and_shares_payload_descriptors() -> None:
    source = full_record()
    cloned_set = ConditioningSet((source,)).clone()
    cloned = cloned_set.records[0]
    assert cloned == source
    assert cloned is not source
    assert cloned.channels[0][1] is source.channels[0][1]
    assert cloned.area is not source.area
    assert cloned.mask is not source.mask
    assert cloned.mask is not None and source.mask is not None
    assert cloned.mask.payload is source.mask.payload
    assert cloned.schedule is not source.schedule
    assert cloned.scale_vector is not source.scale_vector
    assert cloned.scale_vector is not None and source.scale_vector is not None
    assert cloned.scale_vector.values is source.scale_vector.values
    assert cloned.token_layout is not source.token_layout
    source_map = source.extension_metadata[0][1]
    cloned_map = cloned.extension_metadata[0][1]
    assert isinstance(source_map, Mapping) and isinstance(cloned_map, Mapping)
    assert cloned_map is not source_map


def test_combine_is_ordered_record_concatenation_without_element_merge() -> None:
    left = full_record("left")
    right = full_record("right")
    combined = ConditioningSet((left,)).combine(ConditioningSet((right,)))
    assert combined.records == (left, right)
    assert combined.records[0] is left
    assert combined.records[1] is right


def test_for_region_replaces_area_and_mask_and_preserves_every_other_kind() -> None:
    source = full_record()
    replacement = RegionDescriptor(
        AreaDescriptor(4, 5, 1, 2, AreaUnits.LATENT_CELLS),
        MaskDescriptor(PayloadReference("replacement-mask"), 0.4, False),
    )
    transformed = ConditioningSet((source,)).for_region(replacement).records[0]
    assert transformed.area is replacement.area
    assert transformed.mask is replacement.mask
    assert transformed.schedule == source.schedule
    assert transformed.scale_vector == source.scale_vector
    assert transformed.token_layout == source.token_layout
    assert transformed.extension_metadata == source.extension_metadata
    assert transformed.channels == source.channels


def test_builtin_metadata_merge_table_is_complete_and_cited() -> None:
    assert set(BUILTIN_METADATA_MERGE_TABLE) == {
        "area",
        "mask",
        "schedule",
        "scale_vector",
        "token_layout",
    }
    for operations in BUILTIN_METADATA_MERGE_TABLE.values():
        assert set(operations) == {"clone", "combine", "for_region"}
        assert all(
            ".py:" in citation or "Dinkster divergence" in citation
            for citation in operations.values()
        )


def test_all_channels_are_declarative_and_deterministically_ordered() -> None:
    channels = tuple((channel, payload(channel.value)) for channel in ConditioningChannel)
    record = ConditioningRecord(channels)
    assert [channel.value for channel, _ in record.channels] == [
        "text",
        "pooled",
        "concat_latent",
        "vision_embedding",
        "reference_vision_embedding",
        "audio_embedding",
        "control_video",
        "reference_motion",
        "pose_text",
        "pose_vision_embedding",
        "pose_latent",
        "face_pixels",
        "control_hint",
        "reference_latent",
        "scail_reference_latent",
        "scail_reference_mask",
        "scail_driving_mask",
        "camera",
    ]
    serialized = canonical_conditioning_set(ConditioningSet((record,)))
    assert [item["id"] for item in json.loads(serialized)["records"][0]["channels"]] == [
        channel.value for channel in ConditioningChannel
    ]


def test_extension_metadata_is_immutable_preserve_only_and_canonical() -> None:
    reference = PayloadReference("meta-payload")
    source = ConditioningRecord(
        ((ConditioningChannel.TEXT, payload("text")),),
        extension_metadata=(
            ("z.pack/value", {"b": [2, 1], "a": reference}),
            ("a.pack/flag", True),
        ),
    )
    frozen = source.extension_metadata[0][1]
    assert isinstance(frozen, MappingProxyType)
    with pytest.raises(TypeError):
        frozen["new"] = 1  # type: ignore[index]
    clone = source.clone()
    region = RegionDescriptor(area=AreaDescriptor(1, 1, 0, 0, AreaUnits.LATENT_CELLS))
    transformed = source.for_region(region)
    assert clone.extension_metadata == source.extension_metadata
    assert transformed.extension_metadata == source.extension_metadata
    canonical = json.loads(canonical_conditioning_set(ConditioningSet((source,))))
    metadata = canonical["records"][0]["extension_metadata"]
    assert metadata["a.pack/flag"] == {"type": "bool", "value": True}
    assert metadata["z.pack/value"]["type"] == "mapping"


def test_extension_canonical_tags_prevent_payload_reference_mapping_collision() -> None:
    reference = ConditioningRecord(
        ((ConditioningChannel.TEXT, payload("text")),),
        extension_metadata=(("pack/value", PayloadReference("x")),),
    )
    mapping = ConditioningRecord(
        ((ConditioningChannel.TEXT, payload("text")),),
        extension_metadata=(("pack/value", {"payload_ref": "x"}),),
    )
    assert canonical_conditioning_set(ConditioningSet((reference,))) != (
        canonical_conditioning_set(ConditioningSet((mapping,)))
    )


def test_canonical_serialization_normalizes_signed_zero() -> None:
    negative = ConditioningRecord(
        ((ConditioningChannel.TEXT, payload("text")),),
        area=AreaDescriptor(-0.0, 0.5, -0.0, 0.5, AreaUnits.PERCENT, -0.0),
        mask=MaskDescriptor(PayloadReference("mask"), -0.0),
        schedule=PercentRange(-0.0, 1.0),
        extension_metadata=(("pack/zero", -0.0),),
    )
    positive = ConditioningRecord(
        ((ConditioningChannel.TEXT, payload("text")),),
        area=AreaDescriptor(0.0, 0.5, 0.0, 0.5, AreaUnits.PERCENT, 0.0),
        mask=MaskDescriptor(PayloadReference("mask"), 0.0),
        schedule=PercentRange(0.0, 1.0),
        extension_metadata=(("pack/zero", 0.0),),
    )
    assert negative == positive
    assert canonical_conditioning_set(ConditioningSet((negative,))) == (
        canonical_conditioning_set(ConditioningSet((positive,)))
    )


def test_every_behavior_field_participates_in_canonical_serialization() -> None:
    base = full_record()
    assert base.area is not None
    assert base.mask is not None
    assert base.token_layout is not None
    base_bytes = canonical_conditioning_set(ConditioningSet((base,)))
    variants = (
        replace(base, channels=((ConditioningChannel.TEXT, payload("other")),)),
        replace(base, area=replace(base.area, strength=2.0)),
        replace(base, mask=replace(base.mask, set_area_to_bounds=False)),
        replace(base, schedule=PercentRange(0.3, 0.8)),
        replace(base, scale_vector=ConditionScaleVector(payload("other-scale", (1,)))),
        replace(base, token_layout=replace(base.token_layout, version=2)),
        replace(base, extension_metadata=(("pack/value", "other"),)),
    )
    assert all(
        canonical_conditioning_set(ConditioningSet((variant,))) != base_bytes
        for variant in variants
    )
    video_area = AreaDescriptor(
        0.25,
        0.5,
        0.0,
        0.125,
        AreaUnits.PERCENT,
        temporal=0.75,
        z=0.25,
    )
    video = replace(base, area=video_area)
    video_bytes = canonical_conditioning_set(ConditioningSet((video,)))
    for area in (replace(video_area, temporal=0.5), replace(video_area, z=0.5)):
        assert (
            canonical_conditioning_set(ConditioningSet((replace(video, area=area),))) != video_bytes
        )


@pytest.mark.parametrize(
    "metadata,error",
    [
        (("missing-slash", 1), ValueError),
        (("/key", 1), ValueError),
        (("pack/", 1), ValueError),
        (("pack/key", object()), TypeError),
        (("pack/key", math.inf), ValueError),
    ],
)
def test_extension_metadata_refuses_invalid_vocabulary(
    metadata: tuple[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        ConditioningRecord(
            ((ConditioningChannel.TEXT, payload("text")),),
            extension_metadata=(metadata,),  # type: ignore[arg-type]
        )


def test_extension_metadata_refuses_collisions() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ConditioningRecord(
            ((ConditioningChannel.TEXT, payload("text")),),
            extension_metadata=(("pack/key", 1), ("pack/key", 2)),
        )


def test_scale_vector_reuses_guidance_type_and_validates_declarative_shape() -> None:
    descriptor = payload("scale", (4,))
    record = ConditioningRecord(
        ((ConditioningChannel.TEXT, payload("text")),),
        scale_vector=ConditionScaleVector(descriptor),
    )
    assert record.scale_vector == ConditionScaleVector(descriptor)
    with pytest.raises(ValueError, match="rank 1"):
        ConditioningRecord(
            ((ConditioningChannel.TEXT, payload("text")),),
            scale_vector=ConditionScaleVector(payload("bad", (1, 1))),
        )
    with pytest.raises(ValueError, match="non-empty rank 1"):
        ConditioningRecord(
            ((ConditioningChannel.TEXT, payload("text")),),
            scale_vector=ConditionScaleVector(payload("empty", (0,))),
        )


def test_token_layout_refuses_unsupported_family_or_version() -> None:
    value = layout()
    value.require_supported("dinkster.sd15", (1,))
    with pytest.raises(ValueError, match="unsupported token layout"):
        value.require_supported("dinkster.sdxl", (1,))
    with pytest.raises(ValueError, match="unsupported token layout"):
        value.require_supported("dinkster.sd15", (2,))


def test_descriptor_and_collection_refusal_paths_and_immutability() -> None:
    with pytest.raises(TypeError, match="string"):
        PayloadReference(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PayloadDescriptor(PayloadReference("x"), (), "float32", "test")
    with pytest.raises(ValueError):
        AreaDescriptor(0, 1, 0, 0, AreaUnits.LATENT_CELLS)
    with pytest.raises(ValueError):
        MaskDescriptor(PayloadReference("mask"), math.nan)
    with pytest.raises(ValueError, match="duplicate conditioning channel"):
        item = (ConditioningChannel.TEXT, payload("text"))
        ConditioningRecord((item, item))
    with pytest.raises(ValueError):
        RegionDescriptor()
    with pytest.raises(ValueError, match="at least one channel"):
        ConditioningRecord(())
    with pytest.raises(TypeError, match="schedule"):
        ConditioningRecord(
            ((ConditioningChannel.TEXT, payload("text")),),
            schedule="not-a-range",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="intersect"):
        PercentRange(0.0, 1.0).intersect("not-a-range")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="region"):
        full_record().for_region("not-a-region")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ConditioningSet([])  # type: ignore[arg-type]
    record = full_record()
    with pytest.raises(FrozenInstanceError):
        record.area = None  # type: ignore[misc]


def test_records_name_the_carrier_boundary_and_module_is_torch_free() -> None:
    import dinkster_inference.conditioning as module

    assert CONDITIONING_RECORD_BOUNDARY == "dinkster-conditioning-carrier-v1"
    assert "import torch" not in inspect.getsource(module)
    assert ConditioningRecord.__module__ == "dinkster_inference.conditioning"
