"""CPU coverage for portable SAM3 track data through ordinary pack hosts."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
import torch
from dinkster_api.v1 import (
    CORE_STRING,
    SAM3_TRACK_DATA_TYPE,
    InputSpec,
    Node,
    NodeSchema,
    OutputSpec,
    TypeExpr,
    TypeRegistry,
    materialize_latent,
)
from dinkster_vision_sam3 import register_types


def register_sam3_types(registry: TypeRegistry) -> None:
    with registry.registering_pack("dinkster-vision-sam3", ("SAM3_TRACK_DATA",), allow_bare=True):
        register_types(registry)


def _track(case: str) -> dict[str, object]:
    if case == "asymmetric-present-scores":
        return {
            "packed_masks": torch.arange(30, dtype=torch.uint8).reshape(2, 3, 5),
            "orig_size": [13, 29],
            "n_frames": 7,
            "scores": [0.25, (0.5, 0.75)],
            "metadata": {"labels": ("left", "right"), "indices": [3, 1]},
        }
    if case == "zero-objects":
        return {
            "packed_masks": torch.empty((0, 0, 5), dtype=torch.uint8),
            "orig_size": (3, 5),
            "n_frames": 0,
            "scores": (),
            "metadata": {"empty": [(), []]},
        }
    if case == "tensor-scores":
        return {
            "packed_masks": torch.tensor([[[7, 3, 11]]], dtype=torch.uint8),
            "orig_size": [5, 3],
            "n_frames": 1,
            "scores": torch.tensor([0.125, 0.875], dtype=torch.float32),
            "metadata": {"score_names": ("first", "second")},
        }
    if case == "none-scores":
        return {
            "packed_masks": None,
            "orig_size": (17, 11),
            "n_frames": 2,
            "scores": None,
            "metadata": {"source": ["prompt", ("point", 4)]},
        }
    if case == "none-absent-scores":
        return {
            "packed_masks": None,
            "orig_size": (17, 11),
            "n_frames": 2,
            "metadata": {"source": ["prompt", ("point", 4)]},
        }
    raise ValueError(f"unknown SAM3 track case: {case}")


def _track_summary(value: dict[str, object]) -> str:
    packed = value["packed_masks"]
    if packed is None:
        packed_summary = "none"
    else:
        tensor = packed
        assert isinstance(tensor, torch.Tensor)
        packed_summary = ";".join(
            (
                str(tensor.dtype),
                str(tuple(tensor.shape)),
                hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest(),
            )
        )
    return repr(
        (
            packed_summary,
            value["orig_size"],
            value["n_frames"],
            value.get("scores", "absent"),
            tuple(sorted(value["metadata"].items())),
        )
    )


def _assert_structure(actual: object, expected: object) -> None:
    assert type(actual) is type(expected)
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert set(actual) == set(expected)
        for key in expected:
            _assert_structure(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple))
        assert len(actual) == len(expected)
        for received, wanted in zip(actual, expected, strict=True):
            _assert_structure(received, wanted)
    else:
        assert actual == expected


def _assert_track(value: dict[str, object], case: str) -> None:
    expected = _track(case)
    assert set(value) == set(expected)
    _assert_structure(value["orig_size"], expected["orig_size"])
    assert value["n_frames"] == expected["n_frames"]
    assert type(value["n_frames"]) is int
    _assert_structure(value.get("scores", "absent"), expected.get("scores", "absent"))
    if "scores" in expected:
        assert type(value["scores"]) is type(expected["scores"])
    _assert_structure(value["metadata"], expected["metadata"])
    packed = value["packed_masks"]
    expected_packed = expected["packed_masks"]
    if expected_packed is None:
        assert packed is None
        return
    assert isinstance(packed, torch.Tensor)
    assert isinstance(expected_packed, torch.Tensor)
    assert packed.dtype == expected_packed.dtype
    assert packed.shape == expected_packed.shape
    assert torch.equal(packed, expected_packed)


@pytest.mark.parametrize(
    "case",
    (
        "asymmetric-present-scores",
        "zero-objects",
        "tensor-scores",
        "none-scores",
        "none-absent-scores",
    ),
)
def test_public_sam3_track_codec_materializes_torch_data_without_mutating_source(case: str) -> None:
    source = _track(case)
    packed = source["packed_masks"]
    before = packed.clone() if isinstance(packed, torch.Tensor) else None
    registry = TypeRegistry()
    register_sam3_types(registry)
    spec = registry.spec(SAM3_TRACK_DATA_TYPE)

    restored = materialize_latent(spec.decode(spec.encode(source)))

    assert isinstance(restored, dict)
    _assert_track(restored, case)
    _assert_track(source, case)
    if before is not None:
        assert isinstance(packed, torch.Tensor)
        assert torch.equal(packed, before)


SAM3_TRACK = TypeExpr.concrete(SAM3_TRACK_DATA_TYPE)
STRING = TypeExpr.concrete(CORE_STRING)


class SAM3TrackSource(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sam3-public-source.source",
            inputs=(InputSpec("case", STRING),),
            outputs=(OutputSpec("track", SAM3_TRACK),),
        )

    @classmethod
    def execute(cls, *, case: str) -> dict[str, object]:
        return cls.outputs(track=_track(case))


class SAM3TrackConsumer(Node):
    @classmethod
    def define_schema(cls) -> NodeSchema:
        return NodeSchema(
            node_type="sam3-public-consumer.consumer",
            inputs=(InputSpec("track", SAM3_TRACK), InputSpec("case", STRING)),
            outputs=(OutputSpec("summary", STRING),),
        )

    @classmethod
    def execute(cls, *, track: Any, case: str) -> dict[str, object]:
        restored = materialize_latent(track)
        assert isinstance(restored, dict)
        _assert_track(restored, case)
        return cls.outputs(summary=_track_summary(restored))
