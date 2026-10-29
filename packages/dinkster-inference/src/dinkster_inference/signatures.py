"""Detection signatures as data.

ComfyUI decides "what model is this" imperatively: detect_unet_config
walks the state dict with an ordered if-chain, then supported_models
matches config dicts against an ordered class list, first match wins
(comfy/model_detection.py, comfy/supported_models.py @ b78cec87). The
knowledge is real - which keys exist, which shape axes carry which
architecture fact - but it is trapped in control flow.

:class:`KeySignature` lifts that knowledge into a value: keys that must
exist, keys that must not, shape constraints, and the evidence fields to
derive - all relative to declared prefix candidates (the declarative
replacement for unet_prefix_from_state_dict's counting heuristic; each
family knows where it lives in a combined checkpoint). A signature IS a
FamilyDetector, so registrations stay pure data.

Two rules ported from upstream detection behavior (@ b78cec87):

- Every key a signature dereferences is guarded - a missing key or an
  out-of-range axis is "no match", never an incidental KeyError halfway
  through extraction.
- Reads of linear-weight axis 1 (context dims and friends) inherit the
  upstream caveat that 4-bit quantized checkpoints halve that dimension;
  the quantized-checkpoint story lands with stage 4 (ROADMAP "Native
  inference" ledger).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from .families import DetectionEvidence, EvidenceValue
from .weights import WeightSource


@dataclass(frozen=True)
class ShapeIs:
    """The tensor at ``key`` must have ``shape[axis] == equals``."""

    key: str
    axis: int
    equals: int

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("constraint key must not be empty")
        if self.axis < 0 or self.equals < 0:
            raise ValueError("axis and equals must be >= 0")


@dataclass(frozen=True)
class ShapeIn:
    """The tensor at ``key`` must have ``shape[axis]`` in ``values``."""

    key: str
    axis: int
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))
        if not self.key:
            raise ValueError("constraint key must not be empty")
        if self.axis < 0 or not self.values or any(value < 0 for value in self.values):
            raise ValueError("axis must be >= 0 and values must be non-empty and >= 0")


@dataclass(frozen=True)
class RankIs:
    """The tensor at ``key`` must have exactly ``rank`` dimensions.

    (How upstream tells linear from conv projections:
    ``len(proj_in.weight.shape) == 2`` decides use_linear_in_transformer,
    comfy/model_detection.py calculate_transformer_depth @ b78cec87.)
    """

    key: str
    rank: int

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("constraint key must not be empty")
        if self.rank < 0:
            raise ValueError("rank must be >= 0")


Constraint = ShapeIs | ShapeIn | RankIs


@dataclass(frozen=True)
class DimField:
    """Derive evidence field ``name`` from ``shape[axis]`` of ``key``."""

    name: str
    key: str
    axis: int

    def __post_init__(self) -> None:
        if not self.name or not self.key:
            raise ValueError("field name and key must not be empty")
        if self.axis < 0:
            raise ValueError("axis must be >= 0")


@dataclass(frozen=True)
class KeySignature:
    """One family's checkpoint signature; implements FamilyDetector.

    ``prefixes`` are tried in order and the first that satisfies the
    whole signature wins; every key below is relative to that prefix.
    ``required_any`` groups need at least one member each (upstream's
    any_suffix_in, e.g. RMSNorm params appearing as either ``.weight``
    or ``.scale``). ``absent_toplevel`` keys are checked against the
    raw key set regardless of prefix - for top-level markers like
    SDXL's ``v_pred``/``edm_*`` sampling flags, which sit beside a
    prefixed diffusion model, not under it (comfy/supported_models.py
    SDXL.model_type @ b78cec87). The winning prefix is reported in
    evidence as ``key_prefix`` so later stages know where the
    component lives.

    Sequence fields are snapshotted to tuples at construction, so a
    caller-held list cannot mutate a validated signature afterward.
    """

    family_id: str
    prefixes: tuple[str, ...] = ("",)
    required: tuple[str, ...] = ()
    required_any: tuple[tuple[str, ...], ...] = ()
    absent: tuple[str, ...] = ()
    absent_toplevel: tuple[str, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    fields: tuple[DimField, ...] = ()

    def __post_init__(self) -> None:
        # Deep-freeze: accept any sequence, store tuples.
        object.__setattr__(self, "prefixes", tuple(self.prefixes))
        object.__setattr__(self, "required", tuple(self.required))
        object.__setattr__(self, "required_any", tuple(tuple(group) for group in self.required_any))
        object.__setattr__(self, "absent", tuple(self.absent))
        object.__setattr__(self, "absent_toplevel", tuple(self.absent_toplevel))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "fields", tuple(self.fields))
        if not self.family_id:
            raise ValueError("family_id must not be empty")
        if not self.prefixes:
            raise ValueError("at least one prefix candidate is required")
        if not (self.required or self.required_any or self.constraints):
            raise ValueError(
                "signature needs at least one positive requirement "
                "(a match-anything signature is a bug)"
            )
        if any(not group for group in self.required_any):
            raise ValueError("required_any groups must not be empty")

    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        keys = frozenset(source.keys())
        if any(key in keys for key in self.absent_toplevel):
            return None
        for prefix in self.prefixes:
            evidence = self._match(source, keys, prefix)
            if evidence is not None:
                return evidence
        return None

    def _match(
        self, source: WeightSource, keys: frozenset[str], prefix: str
    ) -> DetectionEvidence | None:
        matched: dict[str, None] = {}  # insertion-ordered de-dupe
        for suffix in self.required:
            key = prefix + suffix
            if key not in keys:
                return None
            matched[key] = None
        for group in self.required_any:
            hit = next((prefix + s for s in group if prefix + s in keys), None)
            if hit is None:
                return None
            matched[hit] = None
        for suffix in self.absent:
            if prefix + suffix in keys:
                return None
        for constraint in self.constraints:
            key = prefix + constraint.key
            if key not in keys:
                return None
            shape = source.entry(key).geometry.shape
            if isinstance(constraint, ShapeIs):
                if constraint.axis >= len(shape) or shape[constraint.axis] != constraint.equals:
                    return None
            elif isinstance(constraint, ShapeIn):
                if constraint.axis >= len(shape) or shape[constraint.axis] not in constraint.values:
                    return None
            elif len(shape) != constraint.rank:
                return None
            matched[key] = None
        fields: dict[str, EvidenceValue] = {"key_prefix": prefix}
        for dim in self.fields:
            key = prefix + dim.key
            if key not in keys:
                return None
            shape = source.entry(key).geometry.shape
            if dim.axis >= len(shape):
                return None
            fields[dim.name] = shape[dim.axis]
            matched[key] = None
        return DetectionEvidence(
            family_id=self.family_id,
            matched_keys=tuple(matched),
            fields=MappingProxyType(fields),
        )


__all__ = [
    "Constraint",
    "DimField",
    "KeySignature",
    "RankIs",
    "ShapeIn",
    "ShapeIs",
]
