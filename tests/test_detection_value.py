from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from dinkster_api.v1 import TypeRegistry
from dinkster_nodes_image import DETECTION_TYPE, Detection, Region, register_image_types


def _registry() -> TypeRegistry:
    registry = TypeRegistry()
    register_image_types(registry)
    return registry


def _mask(height: int = 3, width: int = 4) -> np.ndarray:
    values = np.arange(height * width, dtype=np.float32)
    return (values / max(1.0, float(values.size - 1))).reshape(height, width)


def test_detection_validates_fields() -> None:
    region = Region(1, 2, 3, 4)
    with pytest.raises(ValueError, match="non-empty string"):
        Detection(label="", score=0.5, region=region)
    with pytest.raises(ValueError, match="non-empty string"):
        Detection(label=None, score=0.5, region=region)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a number"):
        Detection(label="cat", score="high", region=region)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        Detection(label="cat", score=1.5, region=region)
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        Detection(label="cat", score=float("nan"), region=region)
    with pytest.raises(TypeError, match="must be a Region"):
        Detection(label="cat", score=0.5, region=(1, 2, 3, 4))  # type: ignore[arg-type]


def test_detection_validates_mask() -> None:
    region = Region(0, 0, 4, 3)
    with pytest.raises(ValueError, match="non-empty HW shape"):
        Detection(label="cat", score=0.5, region=region, mask=np.zeros((1, 3, 4), np.float32))
    with pytest.raises(ValueError, match="must be finite"):
        Detection(label="cat", score=0.5, region=region, mask=np.full((3, 4), np.nan, np.float32))
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        Detection(label="cat", score=0.5, region=region, mask=np.full((3, 4), 2.0, np.float32))


def test_detection_is_immutable_and_mask_is_read_only() -> None:
    detection = Detection(label="cat", score=0.5, region=Region(0, 0, 4, 3), mask=_mask())
    with pytest.raises(FrozenInstanceError):
        detection.label = "dog"  # type: ignore[misc]
    assert detection.mask is not None
    assert not detection.mask.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        detection.mask[0, 0] = 1.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        detection.mask.setflags(write=True)


def test_detection_mask_never_aliases_caller_storage() -> None:
    source = _mask()
    detection = Detection(label="cat", score=0.5, region=Region(0, 0, 4, 3), mask=source)
    assert detection.mask is not None
    assert not np.shares_memory(source, detection.mask)
    original = detection.mask.copy()
    source[0, 0] = 1.0
    np.testing.assert_array_equal(detection.mask, original)


def test_detection_label_length_is_bounded_and_round_trips() -> None:
    region = Region(0, 0, 4, 3)
    with pytest.raises(ValueError, match="at most 1024 characters"):
        Detection(label="x" * 1025, score=0.5, region=region)
    spec = _registry().spec(DETECTION_TYPE)
    longest = Detection(label="\u00e9" * 1024, score=0.5, region=region)
    assert spec.decode(spec.encode(longest)) == longest


def test_detection_label_rejects_surrogate_code_points() -> None:
    region = Region(0, 0, 4, 3)
    with pytest.raises(ValueError, match="surrogate code points"):
        Detection(label="\ud800", score=0.5, region=region)
    with pytest.raises(ValueError, match="surrogate code points"):
        Detection(label="cat\udfff", score=0.5, region=region)
    with pytest.raises(ValueError, match="surrogate code points"):
        Detection(label="\ud800\udfff", score=0.5, region=region)
    spec = _registry().spec(DETECTION_TYPE)
    non_bmp = Detection(label="\U000103ff\U0001f600", score=0.5, region=region)
    assert spec.decode(spec.encode(non_bmp)) == non_bmp


def test_detection_normalizes_int_region_coordinates_and_round_trips() -> None:
    spec = _registry().spec(DETECTION_TYPE)
    huge = Detection(label="x", score=1.0, region=Region(10**308, 0, 10**308, 1))
    assert all(
        type(value) is float
        for value in (huge.region.x, huge.region.y, huge.region.width, huge.region.height)
    )
    assert huge == Detection(label="x", score=1.0, region=Region(1e308, 0.0, 1e308, 1.0))
    assert spec.decode(spec.encode(huge)) == huge
    small = Detection(label="x", score=1.0, region=Region(1, 2, 3, 4))
    assert spec.decode(spec.encode(small)) == small


def test_detection_equality_and_area() -> None:
    region = Region(1, 2, 3, 4)
    assert Detection("cat", 0.5, region) == Detection("cat", 0.5, region)
    assert Detection("cat", 0.5, region) != Detection("dog", 0.5, region)
    assert Detection("cat", 0.5, region, _mask()) == Detection("cat", 0.5, region, _mask())
    assert Detection("cat", 0.5, region, _mask()) != Detection("cat", 0.5, region)
    assert Detection("cat", 0.5, region).area == 12.0


def test_detection_codec_round_trip_without_mask() -> None:
    spec = _registry().spec(DETECTION_TYPE)
    detection = Detection(label="cat", score=0.25, region=Region(1.5, 2.5, 3, 4))
    encoded = spec.encode(detection)
    assert encoded == spec.encode(detection)
    decoded = spec.decode(encoded)
    assert decoded == detection


def test_detection_codec_round_trip_with_mask() -> None:
    spec = _registry().spec(DETECTION_TYPE)
    detection = Detection(label="cat", score=1.0, region=Region(0, 0, 4, 3), mask=_mask())
    decoded = spec.decode(spec.encode(detection))
    assert isinstance(decoded, Detection)
    assert decoded == detection
    assert decoded.mask is not None
    assert decoded.mask.dtype == np.float32
    assert decoded.mask.shape == (3, 4)


def test_detection_codec_rejects_invalid_frames() -> None:
    spec = _registry().spec(DETECTION_TYPE)
    detection = Detection(label="cat", score=1.0, region=Region(0, 0, 4, 3), mask=_mask())
    encoded = spec.encode(detection)
    with pytest.raises(ValueError, match="invalid dinkster.detection codec framing"):
        spec.decode(b"not a detection")
    version_offset = encoded.index(b"\x00") + 1
    with pytest.raises(ValueError, match="unsupported dinkster.detection codec version"):
        spec.decode(encoded[:version_offset] + bytes((99,)) + encoded[version_offset + 1 :])
    with pytest.raises(ValueError, match="mask payload is truncated"):
        spec.decode(encoded[:-4])
    with pytest.raises(ValueError, match="unexpected trailing bytes"):
        spec.decode(spec.encode(Detection("cat", 1.0, Region(0, 0, 4, 3))) + b"junk")


def test_detection_coercion_and_meta() -> None:
    registry = _registry()
    spec = registry.spec(DETECTION_TYPE)
    assert spec.coerce is not None
    coerced = spec.coerce(
        {"label": "cat", "score": 0.5, "region": {"x": 1, "y": 2, "width": 3, "height": 4}}
    )
    assert coerced == Detection("cat", 0.5, Region(1, 2, 3, 4))
    with pytest.raises(TypeError, match="expects an object"):
        spec.coerce([])
    with pytest.raises(ValueError, match="requires label, score, region"):
        spec.coerce({"label": "cat"})
    with pytest.raises(ValueError, match="requires label, score, region"):
        spec.coerce({"label": "cat", "score": 0.5, "region": {}, "extra": 1})
    assert spec.meta is not None
    meta = spec.meta(Detection("cat", 0.5, Region(1, 2, 3, 4), _mask()))
    assert meta == {
        "label": "cat",
        "score": 0.5,
        "region": {"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0},
        "mask": [3, 4],
    }
