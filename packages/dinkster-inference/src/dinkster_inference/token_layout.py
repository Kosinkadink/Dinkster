"""Torch-free token-layout vocabulary for the global model sequence.

Semantic facts only: what each global model-token row means. Layouts are
derived from declared packing geometry, never measured from live tensors.
Distinct from ``LatentPackLayout`` (latent packing) and from shard-mechanics
metadata; the global row index is the join key between them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .sequence_partition import translate_segments

__all__ = [
    "ModelTokenLayout",
    "ModelTokenSegment",
    "TokenGridTransform",
    "TokenLayoutError",
    "TokenRowSpan",
    "TokenRowTable",
    "localize_row_spans",
    "map_transforms",
    "validate_row_spans",
]


class TokenLayoutError(ValueError):
    pass


def _require_identifier(name: str, value: str) -> None:
    """Digest lines join identifiers with ``|``, so separator and control
    characters would let distinct fact sets collide on one digest."""
    if type(value) is not str or not value or value != value.strip():
        raise TokenLayoutError(f"{name} must be a non-empty string without surrounding space")
    if "|" in value or any(character.isspace() and character != " " for character in value):
        raise TokenLayoutError(f"{name} must not contain '|' or non-space whitespace")


@dataclass(frozen=True, slots=True)
class ModelTokenSegment:
    """One segment of the global model sequence: what its rows mean."""

    identity: str
    modality: str
    role: str
    start: int
    stop: int
    grid: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_identifier("segment identity", self.identity)
        _require_identifier("segment modality", self.modality)
        _require_identifier("segment role", self.role)
        for name, value in (("start", self.start), ("stop", self.stop)):
            if type(value) is not int:
                raise TokenLayoutError(f"segment {name} must be an exact int")
        if self.start < 0:
            raise TokenLayoutError("segment start must be non-negative")
        if self.stop <= self.start:
            raise TokenLayoutError("segment stop must exceed start (half-open, non-empty)")
        if type(self.grid) is not tuple or not self.grid:
            raise TokenLayoutError("segment grid must be a non-empty tuple of exact ints")
        product = 1
        for size in self.grid:
            if type(size) is not int or size < 1:
                raise TokenLayoutError("segment grid sizes must be exact ints >= 1")
            product *= size
        if product != self.stop - self.start:
            raise TokenLayoutError("segment grid must flatten to exactly the segment's row count")

    @property
    def rows(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class ModelTokenLayout:
    """The one declared ordered global model sequence for an invocation.

    Segments are contiguous from row zero; ``padded_rows`` sit explicitly
    outside all segments as a structural tail excluded from all semantics.
    """

    segments: tuple[ModelTokenSegment, ...]
    padded_rows: int

    def __post_init__(self) -> None:
        if type(self.segments) is not tuple or not self.segments:
            raise TokenLayoutError("layout must contain at least one segment")
        if any(type(segment) is not ModelTokenSegment for segment in self.segments):
            raise TokenLayoutError("layout segments must be exact ModelTokenSegment values")
        if type(self.padded_rows) is not int or self.padded_rows < 0:
            raise TokenLayoutError("padded_rows must be a non-negative exact int")
        identities = tuple(segment.identity for segment in self.segments)
        if len(identities) != len(set(identities)):
            raise TokenLayoutError("segment identities must be unique")
        previous_stop = 0
        for segment in self.segments:
            if segment.start != previous_stop:
                raise TokenLayoutError("segments must be contiguous from row zero")
            previous_stop = segment.stop

    @property
    def valid_rows(self) -> int:
        return self.segments[-1].stop

    @property
    def total_rows(self) -> int:
        return self.valid_rows + self.padded_rows

    @property
    def modalities(self) -> tuple[str, ...]:
        seen: list[str] = []
        for segment in self.segments:
            if segment.modality not in seen:
                seen.append(segment.modality)
        return tuple(seen)

    def by_identity(self, identity: str) -> ModelTokenSegment:
        _require_identifier("segment identity", identity)
        for segment in self.segments:
            if segment.identity == identity:
                return segment
        raise KeyError(identity)

    @property
    def digest(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(b"model-token-layout.v1\n")
        for segment in self.segments:
            grid = "x".join(str(size) for size in segment.grid)
            line = (
                f"{segment.identity}|{segment.modality}|{segment.role}"
                f"|{segment.start}|{segment.stop}|{grid}\n"
            )
            hasher.update(line.encode("utf-8"))
        hasher.update(f"padded_rows={self.padded_rows}\n".encode())
        return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class TokenGridTransform:
    """A family-declared transform fact: declared source geometry to the
    global model-token rows of its one bound segment. Identity/digest fact
    only; the compiled math lives with the family's tensor kernels.

    A binding scopes to its bound segment's rows only: same-modality
    segments without a binding are context rows that no source transform
    covers. ``quantization_levels`` of ``None`` declares the unquantized
    exact path.
    """

    transform: str
    modality: str
    segment_identity: str
    source_geometry: tuple[int, ...]
    quantization_levels: int | None

    def __post_init__(self) -> None:
        _require_identifier("transform identifier", self.transform)
        _require_identifier("transform modality", self.modality)
        _require_identifier("transform segment identity", self.segment_identity)
        if type(self.source_geometry) is not tuple or not self.source_geometry:
            raise TokenLayoutError("source geometry must be a non-empty tuple of exact ints")
        if any(type(size) is not int or size < 1 for size in self.source_geometry):
            raise TokenLayoutError("source geometry sizes must be exact ints >= 1")
        if self.quantization_levels is not None and (
            type(self.quantization_levels) is not int or self.quantization_levels < 1
        ):
            raise TokenLayoutError("quantization levels must be None or an exact int >= 1")

    @property
    def digest(self) -> str:
        geometry = "x".join(str(size) for size in self.source_geometry)
        levels = "none" if self.quantization_levels is None else str(self.quantization_levels)
        line = (
            f"token-grid-transform.v1\n{self.transform}|{self.modality}"
            f"|{self.segment_identity}|{geometry}|{levels}\n"
        )
        return hashlib.sha256(line.encode("utf-8")).hexdigest()


def map_transforms(
    layout: ModelTokenLayout, transforms: tuple[TokenGridTransform, ...]
) -> dict[str, TokenGridTransform]:
    """Bind declared transforms to a layout: at most one per modality.

    Refuses duplicates, modalities the layout does not declare, and
    mismatched segment bindings. Which modalities require a transform is a
    family declaration, not a layout-wide rule: a global layout carries
    rows (text, conditioning, reference context) that no source mask
    transforms.
    """

    if type(layout) is not ModelTokenLayout:
        raise TokenLayoutError("layout must be an exact ModelTokenLayout value")
    if type(transforms) is not tuple:
        raise TokenLayoutError("transforms must be a tuple of exact TokenGridTransform values")
    if any(type(transform) is not TokenGridTransform for transform in transforms):
        raise TokenLayoutError("transforms must be a tuple of exact TokenGridTransform values")
    layout_modalities = layout.modalities
    bound: dict[str, TokenGridTransform] = {}
    for transform in transforms:
        if transform.modality not in layout_modalities:
            raise TokenLayoutError(
                f"transform modality {transform.modality!r} is not declared by the layout"
            )
        if transform.modality in bound:
            raise TokenLayoutError(
                f"exactly one transform per modality: duplicate for {transform.modality!r}"
            )
        try:
            segment = layout.by_identity(transform.segment_identity)
        except KeyError:
            raise TokenLayoutError(
                f"transform names unknown segment {transform.segment_identity!r}"
            ) from None
        if segment.modality != transform.modality:
            raise TokenLayoutError("transform modality must match its bound segment's modality")
        bound[transform.modality] = transform
    return bound


RowValue = int | float | tuple["RowValue", ...]


def _row_type_and_shape(row: RowValue) -> tuple[type, tuple[int, ...]]:
    if type(row) in (int, float):
        return type(row), ()
    if type(row) is tuple and row:
        first_type, first_shape = _row_type_and_shape(row[0])
        for element in row[1:]:
            element_type, element_shape = _row_type_and_shape(element)
            if element_type is not first_type or element_shape != first_shape:
                raise TokenLayoutError("row table values must be homogeneous")
        return first_type, (len(row), *first_shape)
    raise TokenLayoutError(
        "row table values must be exact ints, exact floats, or non-empty tuples of them"
    )


@dataclass(frozen=True, slots=True)
class TokenRowTable:
    """Dense per-row contributions or selectors over the global sequence
    (``[S_global, ...]``): one value per valid semantic row, where each row
    is a scalar or a uniformly shaped nested tuple (the trailing value
    dimensions). Padded rows never receive values. Under sequence
    partitioning, slice by the same shard ``[start, stop)`` spans as hidden
    states and RoPE.
    """

    values: tuple[RowValue, ...]

    def __post_init__(self) -> None:
        if type(self.values) is not tuple or not self.values:
            raise TokenLayoutError("row table must contain at least one value")
        first_type, first_shape = _row_type_and_shape(self.values[0])
        for row in self.values[1:]:
            row_type, row_shape = _row_type_and_shape(row)
            if row_type is not first_type or row_shape != first_shape:
                raise TokenLayoutError("row table values must be homogeneous")

    @property
    def rows(self) -> int:
        return len(self.values)

    @property
    def row_shape(self) -> tuple[int, ...]:
        """The trailing value dimensions; empty for scalar rows."""
        return _row_type_and_shape(self.values[0])[1]

    def validate_for(self, layout: ModelTokenLayout) -> None:
        if type(layout) is not ModelTokenLayout:
            raise TokenLayoutError("layout must be an exact ModelTokenLayout value")
        if self.rows != layout.valid_rows:
            raise TokenLayoutError("row table must cover exactly the layout's valid semantic rows")

    def shard_slice(self, start: int, stop: int) -> tuple[RowValue, ...]:
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop:
            raise TokenLayoutError("start and stop must be exact ints with 0 <= start < stop")
        if stop > self.rows:
            raise TokenLayoutError("shard span must lie within the table's rows")
        return self.values[start:stop]


@dataclass(frozen=True, slots=True)
class TokenRowSpan:
    """A compressed (start, stop, row) run over global rows."""

    start: int
    stop: int
    row: int

    def __post_init__(self) -> None:
        for name, value in (("start", self.start), ("stop", self.stop), ("row", self.row)):
            if type(value) is not int:
                raise TokenLayoutError(f"span {name} must be an exact int")
        if not 0 <= self.start < self.stop:
            raise TokenLayoutError("spans require 0 <= start < stop")
        if self.row < 0:
            raise TokenLayoutError("span row must be non-negative")

    @property
    def as_triple(self) -> tuple[int, int, int]:
        return (self.start, self.stop, self.row)


def validate_row_spans(layout: ModelTokenLayout, spans: tuple[TokenRowSpan, ...]) -> None:
    """Validate global row spans against a layout: ascending, non-overlapping,
    and entirely within the valid semantic rows. Padded rows sit outside all
    segments and carry no semantics, so a span reaching into the padded tail
    refuses."""

    if type(layout) is not ModelTokenLayout:
        raise TokenLayoutError("layout must be an exact ModelTokenLayout value")
    if type(spans) is not tuple or any(type(span) is not TokenRowSpan for span in spans):
        raise TokenLayoutError("spans must be a tuple of exact TokenRowSpan values")
    previous_stop = 0
    for span in spans:
        if span.start < previous_stop:
            raise TokenLayoutError("spans must be ascending and non-overlapping")
        previous_stop = span.stop
    if spans and spans[-1].stop > layout.valid_rows:
        raise TokenLayoutError(
            "spans must lie within the layout's valid semantic rows; "
            "the padded tail carries no semantics"
        )


def localize_row_spans(
    spans: tuple[TokenRowSpan, ...], start: int, stop: int
) -> tuple[TokenRowSpan, ...]:
    """Localize global row spans to a shard by half-open intersection and
    offset translation. Spans must be ascending and non-overlapping; validate
    them against their layout with ``validate_row_spans`` first."""

    if type(spans) is not tuple or any(type(span) is not TokenRowSpan for span in spans):
        raise TokenLayoutError("spans must be a tuple of exact TokenRowSpan values")
    try:
        translated = translate_segments(tuple(span.as_triple for span in spans), start, stop)
    except ValueError as error:
        raise TokenLayoutError(str(error)) from None
    return tuple(TokenRowSpan(*triple) for triple in translated)
