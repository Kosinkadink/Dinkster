"""Region and detection codecs shared across pack boundaries.

Detections cross interpreter boundaries in both directions: detection
provider packs produce them in isolated workers, and image-pack nodes
consume and transform them in their own workers. Both sides therefore
need the value classes and byte contract from a host requirement rather
than from any single pack, matching the image-array codec's placement.

The detection frame is a magic-and-version prefix, a 4-byte little-endian
header size, an ASCII JSON header (label, score, region, optional mask
shape), then the mask's raw little-endian float32 bytes when present.

numpy is a call-time requirement, not a package dependency: dinkster-values
stays dependency-free, and every interpreter that actually moves
detections already ships it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    "DETECTION_TYPE",
    "REGION_TYPE",
    "Detection",
    "Region",
    "coerce_detection",
    "coerce_region",
    "decode_detection",
    "decode_region",
    "detection_meta",
    "encode_detection",
    "encode_region",
    "region_meta",
]

REGION_TYPE = "dinkster.region"
DETECTION_TYPE = "dinkster.detection"

DETECTION_CODEC_MAGIC = b"DINKSTER-DETECTION\x00"
DETECTION_CODEC_VERSION = 1

# The header carries one label plus fixed fields; anything bigger is garbage.
_DETECTION_HEADER_LIMIT = 64 * 1024

# JSON escaping expands a character to at most 12 header bytes (a non-BMP
# character escapes to a UTF-16 surrogate pair), so this keeps every accepted
# label decodable within the header size bound.
_DETECTION_LABEL_LIMIT = 1024


def _numpy() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - both venvs ship numpy
        raise RuntimeError("the detection codec needs numpy in this interpreter") from exc
    return numpy


@dataclass(frozen=True)
class Region:
    """An axis-aligned pixel rectangle in left/top/width/height form."""

    x: int | float
    y: int | float
    width: int | float
    height: int | float

    def __post_init__(self) -> None:
        values: tuple[object, ...] = (self.x, self.y, self.width, self.height)
        if any(type(value) not in (int, float) for value in values):
            raise TypeError("region coordinates must be numbers")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("region coordinates must be finite")
        if self.width < 0 or self.height < 0:
            raise ValueError("region width and height must be non-negative")

    @property
    def right(self) -> int | float:
        return self.x + self.width

    @property
    def bottom(self) -> int | float:
        return self.y + self.height

    def to_record(self) -> dict[str, int | float]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


def coerce_region(obj: object) -> Region:
    if isinstance(obj, Region):
        return obj
    if not isinstance(obj, Mapping):
        raise TypeError(f"{REGION_TYPE} expects an object, got {type(obj).__name__}")
    record = cast("Mapping[str, object]", obj)
    if set(record) != {"x", "y", "width", "height"}:
        raise ValueError(f"{REGION_TYPE} requires exactly x, y, width, and height")
    return Region(
        x=cast("int | float", record["x"]),
        y=cast("int | float", record["y"]),
        width=cast("int | float", record["width"]),
        height=cast("int | float", record["height"]),
    )


def encode_region(obj: object) -> bytes:
    region = coerce_region(obj)
    return json.dumps(region.to_record(), sort_keys=True, separators=(",", ":")).encode("ascii")


def decode_region(data: bytes) -> object:
    try:
        record = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {REGION_TYPE} payload") from exc
    return coerce_region(record)


def region_meta(obj: object) -> Mapping[str, object]:
    return coerce_region(obj).to_record()


def _normalize_detection_mask(mask: object) -> np.ndarray:
    np = _numpy()
    array = np.asarray(mask, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"detection mask must have non-empty HW shape, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("detection mask values must be finite")
    if float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise ValueError("detection mask values must be within [0, 1]")
    # Copy into an immutable bytes buffer so the value never aliases caller
    # storage and WRITEABLE cannot be re-enabled on it or through a base.
    return np.frombuffer(array.tobytes(order="C"), dtype=np.float32).reshape(array.shape)


@dataclass(frozen=True, eq=False)
class Detection:
    """One labeled, scored region, optionally with a full-frame soft mask."""

    label: str
    score: float
    region: Region
    mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        if type(self.label) is not str or not self.label:
            raise ValueError("detection label must be a non-empty string")
        if len(self.label) > _DETECTION_LABEL_LIMIT:
            raise ValueError(f"detection label must be at most {_DETECTION_LABEL_LIMIT} characters")
        if any("\ud800" <= character <= "\udfff" for character in self.label):
            # Surrogates are not Unicode scalar values, and JSON collapses an
            # adjacent escaped surrogate pair into the equivalent non-BMP
            # character, so such labels cannot round-trip through the codec.
            raise ValueError("detection label must not contain surrogate code points")
        if type(self.score) not in (int, float):
            raise TypeError("detection score must be a number")
        if not math.isfinite(float(self.score)) or not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("detection score must be within [0, 1]")
        if type(self.region) is not Region:
            raise TypeError("detection region must be a Region")
        object.__setattr__(self, "score", float(self.score))
        region = self.region
        coordinates = (region.x, region.y, region.width, region.height)
        if any(type(value) is int for value in coordinates):
            # Normalize int coordinates so the codec's float record decodes
            # equal to the constructed value.
            object.__setattr__(self, "region", Region(*(float(value) for value in coordinates)))
        if self.mask is not None:
            object.__setattr__(self, "mask", _normalize_detection_mask(self.mask))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Detection):
            return NotImplemented
        if (self.label, self.score, self.region) != (other.label, other.score, other.region):
            return False
        if self.mask is None or other.mask is None:
            return self.mask is None and other.mask is None
        return bool(_numpy().array_equal(self.mask, other.mask))

    @property
    def area(self) -> float:
        return float(self.region.width) * float(self.region.height)

    def to_record(self) -> dict[str, object]:
        return {
            "label": self.label,
            "score": float(self.score),
            "region": self.region.to_record(),
            "mask": None if self.mask is None else [int(n) for n in self.mask.shape],
        }


def coerce_detection(obj: object) -> Detection:
    if isinstance(obj, Detection):
        return obj
    if not isinstance(obj, Mapping):
        raise TypeError(f"{DETECTION_TYPE} expects an object, got {type(obj).__name__}")
    record = cast("Mapping[str, object]", obj)
    keys = set(record)
    if not keys.issuperset(("label", "score", "region")) or not keys.issubset(
        ("label", "score", "region", "mask")
    ):
        raise ValueError(f"{DETECTION_TYPE} requires label, score, region, and optionally mask")
    mask = record.get("mask")
    return Detection(
        label=cast("str", record["label"]),
        score=cast("float", record["score"]),
        region=coerce_region(record["region"]),
        mask=None if mask is None else cast("np.ndarray", mask),
    )


def encode_detection(obj: object) -> bytes:
    detection = coerce_detection(obj)
    header = json.dumps(
        detection.to_record(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    frame = [
        DETECTION_CODEC_MAGIC,
        bytes((DETECTION_CODEC_VERSION,)),
        len(header).to_bytes(4, "little"),
        header,
    ]
    if detection.mask is not None:
        frame.append(detection.mask.astype("<f4", copy=False).tobytes(order="C"))
    return b"".join(frame)


def decode_detection(data: bytes) -> object:
    np = _numpy()
    data = bytes(data)
    offset = len(DETECTION_CODEC_MAGIC)
    if not data.startswith(DETECTION_CODEC_MAGIC) or len(data) < offset + 5:
        raise ValueError(f"invalid {DETECTION_TYPE} codec framing")
    if data[offset] != DETECTION_CODEC_VERSION:
        raise ValueError(f"unsupported {DETECTION_TYPE} codec version")
    header_size = int.from_bytes(data[offset + 1 : offset + 5], "little")
    if header_size > _DETECTION_HEADER_LIMIT:
        raise ValueError(f"{DETECTION_TYPE} codec header exceeds its size bound")
    body = offset + 5 + header_size
    if body > len(data):
        raise ValueError(f"{DETECTION_TYPE} codec header is truncated")
    try:
        record: object = json.loads(data[offset + 5 : body])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {DETECTION_TYPE} codec header") from exc
    if not isinstance(record, Mapping):
        raise ValueError(f"invalid {DETECTION_TYPE} codec header")
    header = cast("Mapping[str, object]", record)
    shape = header.get("mask")
    mask: np.ndarray | None = None
    if shape is None:
        if len(data) != body:
            raise ValueError(f"{DETECTION_TYPE} codec frame carries unexpected trailing bytes")
    else:
        if (
            not isinstance(shape, list)
            or len(cast("list[object]", shape)) != 2
            or any(type(n) is not int or n < 1 for n in cast("list[object]", shape))
        ):
            raise ValueError(f"{DETECTION_TYPE} codec mask shape is invalid")
        height, width = cast("list[int]", shape)
        expected = height * width * np.dtype("<f4").itemsize
        if len(data) != body + expected:
            raise ValueError(f"{DETECTION_TYPE} codec mask payload is truncated")
        mask = np.frombuffer(data, dtype="<f4", count=height * width, offset=body).reshape(
            height, width
        )
    fields = {key: value for key, value in header.items() if key != "mask"}
    return coerce_detection({**fields, "mask": mask})


def detection_meta(obj: object) -> Mapping[str, object]:
    return coerce_detection(obj).to_record()
