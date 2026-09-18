"""Proving tests for dinkster-inference stage-1 contracts.

These pin the invariants the contracts claim: validation raises loudly,
prefix transforms match ComfyUI semantics, patch shape math is pure,
registries collide loudly and resolve aliases, and family detection
ranks by explicit specificity with ambiguity reported instead of
guessed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import pytest
from dinkster_inference import (
    FLOAT16,
    FLOAT32,
    NVFP4,
    AdapterPatch,
    ComponentWiring,
    DetectionEvidence,
    DeviceCapabilities,
    DeviceRef,
    DiffPatch,
    DType,
    EmbeddingRef,
    FamilyRegistry,
    LatentDescriptor,
    ModelFamily,
    NestedPatch,
    Parameterization,
    PatchEntry,
    PatchOffset,
    PatchSet,
    PrecisionPlan,
    PrecisionRequest,
    Registry,
    RegistryError,
    SamplingDescriptor,
    SamplingSegment,
    SetPatch,
    TensorGeometry,
    WeightedSpan,
    WeightEntry,
    WeightSource,
    calculate_shape,
    count_prefix,
    filter_prefix,
    replace_prefix,
)

# --- devices ---------------------------------------------------------------


def test_dtype_validation() -> None:
    with pytest.raises(ValueError):
        DType("", 32)
    with pytest.raises(ValueError):
        DType("bad", 0)
    with pytest.raises(ValueError):
        DType("bad", -8)


def test_device_ref_str_and_residency() -> None:
    assert str(DeviceRef("cpu")) == "cpu"
    assert str(DeviceRef("cuda", 1)) == "cuda:1"
    assert DeviceRef("cpu").residency_key() == "ram"
    assert DeviceRef("cuda", 0).residency_key() == "vram:cuda:0"
    with pytest.raises(ValueError):
        DeviceRef("")
    with pytest.raises(ValueError):
        DeviceRef("cuda", -1)


def test_precision_plan_manual_cast() -> None:
    assert not PrecisionPlan(storage=FLOAT16, compute=FLOAT16).manual_cast
    assert PrecisionPlan(storage=FLOAT16, compute=FLOAT32).manual_cast


def test_precision_request_validation() -> None:
    caps = DeviceCapabilities(
        device=DeviceRef("cuda", 0),
        compute_dtypes=frozenset({FLOAT16}),
        storage_dtypes=frozenset({FLOAT16}),
    )
    ok = PrecisionRequest(
        weights_dtype=FLOAT16,
        supported=frozenset({FLOAT16}),
        capabilities=caps,
        parameter_count=1_000,
        storage_bytes=2_000,
    )
    assert ok.parameter_count == 1_000
    with pytest.raises(ValueError):
        PrecisionRequest(
            weights_dtype=FLOAT16,
            supported=frozenset({FLOAT16}),
            capabilities=caps,
            parameter_count=-1,
            storage_bytes=0,
        )


# --- weights ---------------------------------------------------------------


def test_tensor_geometry_numel_nbytes() -> None:
    geo = TensorGeometry((2, 3, 4), FLOAT32)
    assert geo.numel == 24
    assert geo.nbytes == 96


def test_tensor_geometry_subbyte_rounds_up_total() -> None:
    # 3 elements at 4 bits = 12 bits -> 2 bytes, rounded at buffer level.
    geo = TensorGeometry((3,), NVFP4)
    assert geo.nbytes == 2


def test_tensor_geometry_rejects_negative_dims() -> None:
    with pytest.raises(ValueError):
        TensorGeometry((2, -1), FLOAT32)


def test_weight_entry_validation() -> None:
    geo = TensorGeometry((1,), FLOAT32)
    with pytest.raises(ValueError):
        WeightEntry("", geo, 0, 4)
    with pytest.raises(ValueError):
        WeightEntry("w", geo, -1, 4)


def test_prefix_transforms() -> None:
    sd = {"model.a": 1, "model.b": 2, "vae.a": 3}
    assert filter_prefix(sd, "model.") == {"a": 1, "b": 2}
    assert filter_prefix(sd, "model.", strip=False) == {"model.a": 1, "model.b": 2}
    assert filter_prefix(sd, "") == sd
    assert replace_prefix(sd, "model.", "unet.") == {"unet.a": 1, "unet.b": 2, "vae.a": 3}
    assert count_prefix(sd, "model.") == 2
    assert count_prefix(sd, "nope.") == 0


def test_filter_prefix_preserves_order() -> None:
    sd = {"p.z": 1, "p.a": 2, "p.m": 3}
    assert list(filter_prefix(sd, "p.")) == ["z", "a", "m"]


# --- patches ---------------------------------------------------------------


@dataclass(frozen=True)
class FakeTensor:
    shape: tuple[int, ...]


class FakeAdapter:
    def target_shape(self, base: tuple[int, ...]) -> tuple[int, ...]:
        return (base[0] * 2, *base[1:])

    def calculate(
        self,
        weight: FakeTensor,
        *,
        strength: float,
        function: Callable[[FakeTensor], FakeTensor] | None = None,
    ) -> FakeTensor:
        return weight


def test_patch_offset_validation() -> None:
    with pytest.raises(ValueError):
        PatchOffset(dim=-1, start=0, length=1)
    with pytest.raises(ValueError):
        PatchOffset(dim=0, start=0, length=0)


def test_calculate_shape_semantics() -> None:
    base = (4, 8)
    set_entry = PatchEntry[FakeTensor](value=SetPatch(FakeTensor((2, 2))))
    diff_pad = PatchEntry[FakeTensor](value=DiffPatch(FakeTensor((6, 8)), pad_weight=True))
    diff_plain = PatchEntry[FakeTensor](value=DiffPatch(FakeTensor((6, 8))))
    adapter = PatchEntry[FakeTensor](value=AdapterPatch(FakeAdapter()))
    offset = PatchEntry[FakeTensor](
        value=SetPatch(FakeTensor((99, 99))), offset=PatchOffset(0, 0, 2)
    )

    assert calculate_shape(base, [set_entry]) == (2, 2)
    assert calculate_shape(base, [diff_pad]) == (6, 8)
    assert calculate_shape(base, [diff_plain]) == base
    assert calculate_shape(base, [adapter]) == (8, 8)
    # Offset patches never change shape, whatever their value claims.
    assert calculate_shape(base, [offset]) == base
    # Sequential application: set to (2,2), then adapter doubles dim 0.
    assert calculate_shape(base, [set_entry, adapter]) == (4, 2)


def test_patch_set_merge_orders_and_re_revisions() -> None:
    e1 = PatchEntry[FakeTensor](value=SetPatch(FakeTensor((1,))))
    e2 = PatchEntry[FakeTensor](value=SetPatch(FakeTensor((2,))))
    a = PatchSet[FakeTensor]({"w": (e1,)})
    b = PatchSet[FakeTensor]({"w": (e2,), "v": (e2,)})
    merged = a.merge(b)
    assert merged.entries("w") == (e1, e2)
    assert merged.entries("v") == (e2,)
    assert merged.entries("missing") == ()
    assert merged.revision not in (a.revision, b.revision)


def test_patch_set_identity_is_revision_not_structure() -> None:
    e = PatchEntry[FakeTensor](value=SetPatch(FakeTensor((1,))))
    a = PatchSet[FakeTensor]({"w": (e,)})
    b = PatchSet[FakeTensor]({"w": (e,)})
    assert a.revision != b.revision
    assert a != b
    assert a == a
    assert hash(a) == hash(a.revision)


def test_patch_set_structural_digest_is_additive_not_live_identity() -> None:
    e = PatchEntry[FakeTensor](value=SetPatch(FakeTensor((1,))))
    digest = "a" * 64
    a = PatchSet[FakeTensor]({"w": (e,)}, structural_digest=digest)
    b = PatchSet[FakeTensor]({"w": (e,)}, structural_digest=digest)
    assert a.structural_digest == b.structural_digest == digest
    assert a.revision != b.revision
    assert a != b
    assert a.merge(b).structural_digest is None
    with pytest.raises(ValueError, match="lowercase sha256"):
        PatchSet[FakeTensor]({"w": (e,)}, structural_digest="not-a-digest")


def test_patch_set_snapshots_caller_dict() -> None:
    e = PatchEntry[FakeTensor](value=SetPatch(FakeTensor((1,))))
    source: dict[str, tuple[PatchEntry[FakeTensor], ...]] = {"w": (e,)}
    ps = PatchSet[FakeTensor](source)
    source["sneaky"] = (e,)
    assert list(ps.keys()) == ["w"]
    with pytest.raises(TypeError):
        ps.patches["x"] = (e,)  # type: ignore[index]


def test_nested_patch_keeps_outer_shape() -> None:
    inner = PatchEntry[FakeTensor](value=SetPatch(FakeTensor((2, 2))))
    nested = PatchEntry[FakeTensor](value=NestedPatch(FakeTensor((4, 8)), (inner,)))
    assert calculate_shape((4, 8), [nested]) == (4, 8)


def test_detection_evidence_snapshots_fields() -> None:
    fields: dict[str, str | int | float | bool] = {"width": 320}
    evidence = DetectionEvidence(family_id="dinkster.x", matched_keys=(), fields=fields)
    fields["width"] = 999
    assert evidence.fields["width"] == 320
    with pytest.raises(TypeError):
        evidence.fields["depth"] = 1  # type: ignore[index]


# --- latents ---------------------------------------------------------------


def test_latent_descriptor_validation() -> None:
    with pytest.raises(ValueError):
        LatentDescriptor(channels=0)
    with pytest.raises(ValueError):
        LatentDescriptor(channels=4, spatial_downscale=0)
    with pytest.raises(ValueError):
        LatentDescriptor(channels=4, rgb_factors=((1.0, 1.0, 1.0),) * 3)
    ok = LatentDescriptor(channels=4, rgb_factors=((0.1, 0.2, 0.3),) * 4)
    assert ok.channels == 4


# --- sampling --------------------------------------------------------------


def test_sampling_descriptor_validation() -> None:
    ok = SamplingDescriptor(Parameterization.EPS, sigma_min=0.03, sigma_max=14.6)
    assert ok.shift == 1.0
    with pytest.raises(ValueError):
        SamplingDescriptor(Parameterization.EPS, sigma_min=-0.1, sigma_max=1.0)
    with pytest.raises(ValueError):
        # sigma_min is strictly positive; the terminal 0.0 belongs to schedules.
        SamplingDescriptor(Parameterization.EPS, sigma_min=0.0, sigma_max=1.0)
    with pytest.raises(ValueError):
        SamplingDescriptor(Parameterization.EPS, sigma_min=1.0, sigma_max=1.0)
    for non_finite in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="finite"):
            SamplingDescriptor(
                Parameterization.V_PREDICTION,
                sigma_min=0.002,
                sigma_max=non_finite,
            )


def test_sampling_segment_validation() -> None:
    segment = SamplingSegment(
        steps=20, start_step=8, end_step=20, add_noise=False, return_with_leftover_noise=False
    )
    assert (segment.start_step, segment.end_step, segment.add_noise) == (8, 20, False)
    for start, end in ((-1, 10), (10, 10), (11, 10), (0, 21)):
        with pytest.raises(ValueError, match="0 <= start_step"):
            SamplingSegment(20, start, end, True, False)
    with pytest.raises(ValueError, match="positive integer"):
        SamplingSegment(0, 0, 1, True, False)
    with pytest.raises(TypeError, match="exact integers"):
        SamplingSegment(20, True, 10, True, False)
    with pytest.raises(TypeError, match="exact booleans"):
        SamplingSegment(20, 0, 10, 1, False)  # type: ignore[arg-type]


# --- text encoders ---------------------------------------------------------


def test_weighted_span_and_embedding_ref_validation() -> None:
    span = WeightedSpan(tokens=(1, 2, EmbeddingRef("thing")), weight=1.2)
    assert span.weight == 1.2
    # Negative weights are legal - ComfyUI parses (text:-1).
    assert WeightedSpan(tokens=(1,), weight=-0.5).weight == -0.5
    with pytest.raises(ValueError):
        EmbeddingRef("")


# --- registry --------------------------------------------------------------


@dataclass(frozen=True)
class Desc:
    id: str
    aliases: tuple[str, ...] = ()


def test_registry_register_get_alias() -> None:
    reg = Registry[Desc]()
    reg.register(Desc("dinkster.euler", aliases=("euler",)))
    assert reg.get("dinkster.euler") is not None
    assert reg.get("euler") is reg.get("dinkster.euler")
    assert reg.get("missing") is None
    # Aliases are vocabulary, not identity.
    assert reg.ids() == ("dinkster.euler",)
    assert len(reg) == 1


def test_registry_collisions_raise() -> None:
    reg = Registry[Desc]()
    reg.register(Desc("dinkster.euler", aliases=("euler",)))
    with pytest.raises(RegistryError):
        reg.register(Desc("dinkster.euler"))
    with pytest.raises(RegistryError):
        reg.register(Desc("other.euler", aliases=("euler",)))
    # An alias colliding with a registered id is also a collision.
    with pytest.raises(RegistryError):
        reg.register(Desc("other.thing", aliases=("dinkster.euler",)))


def test_registry_id_grammar() -> None:
    reg = Registry[Desc]()
    with pytest.raises(RegistryError):
        reg.register(Desc("notnamespaced"))
    with pytest.raises(RegistryError):
        reg.register(Desc("Dinkster.Euler"))


def test_registry_separator_equivalence() -> None:
    # dinkster.foo-bar and dinkster.foo_bar are ONE name per the workspace
    # grammar (canonical_name); the registry must not let them coexist.
    reg = Registry[Desc]()
    reg.register(Desc("dinkster.foo-bar"))
    with pytest.raises(RegistryError):
        reg.register(Desc("dinkster.foo_bar"))
    assert reg.get("dinkster.foo_bar") is reg.get("dinkster.foo-bar")
    assert reg.ids() == ("dinkster.foo-bar",)


def test_registry_intra_descriptor_duplicates() -> None:
    reg = Registry[Desc]()
    # Alias duplicating its own id.
    with pytest.raises(RegistryError):
        reg.register(Desc("dinkster.euler", aliases=("dinkster.euler",)))
    # Duplicate aliases within one descriptor (canonically equal).
    with pytest.raises(RegistryError):
        reg.register(Desc("dinkster.thing", aliases=("foo-bar", "foo_bar")))
    # Failed registration must not leave partial state behind.
    assert len(reg) == 0
    reg.register(Desc("dinkster.euler", aliases=("euler",)))
    assert reg.get("euler") is not None


def test_registry_invalid_alias_rejected() -> None:
    reg = Registry[Desc]()
    with pytest.raises(RegistryError):
        reg.register(Desc("dinkster.euler", aliases=("Bad Alias",)))


# --- families --------------------------------------------------------------


class DictSource:
    """In-memory WeightSource over a {key: shape} mapping."""

    def __init__(self, shapes: Mapping[str, tuple[int, ...]]) -> None:
        self._entries = {
            key: WeightEntry(key, TensorGeometry(shape, FLOAT16), 0, 0)
            for key, shape in shapes.items()
        }

    def keys(self) -> Sequence[str]:
        return list(self._entries)

    def entry(self, key: str) -> WeightEntry:
        return self._entries[key]

    def metadata(self) -> Mapping[str, str]:
        return {}


@dataclass(frozen=True)
class KeyDetector:
    family_id: str
    required_key: str

    def detect(self, source: WeightSource) -> DetectionEvidence | None:
        if self.required_key in source.keys():
            return DetectionEvidence(
                family_id=self.family_id,
                matched_keys=(self.required_key,),
                fields={},
            )
        return None


def _family(family_id: str, key: str, specificity: int) -> ModelFamily:
    return ModelFamily(
        id=family_id,
        display_name=family_id,
        detector=KeyDetector(family_id, key),
        specificity=specificity,
        latent=LatentDescriptor(channels=4),
        sampling=SamplingDescriptor(Parameterization.EPS, sigma_min=0.03, sigma_max=14.6),
        wiring=ComponentWiring(),
        supported_dtypes=frozenset({FLOAT16, FLOAT32}),
    )


def test_family_detection_specificity_wins() -> None:
    reg = FamilyRegistry()
    reg.register(_family("dinkster.base", "model.weight", specificity=10))
    reg.register(_family("dinkster.variant", "model.weight", specificity=20))
    result = reg.detect(DictSource({"model.weight": (1,)}))
    assert result.best is not None
    assert result.best.family_id == "dinkster.variant"
    assert [e.family_id for e in result.candidates] == ["dinkster.variant", "dinkster.base"]
    assert result.ambiguous == ()


def test_family_detection_no_match() -> None:
    reg = FamilyRegistry()
    reg.register(_family("dinkster.base", "model.weight", specificity=10))
    result = reg.detect(DictSource({"other.weight": (1,)}))
    assert result.best is None
    assert result.candidates == ()
    assert result.ambiguous == ()


def test_family_detection_ambiguity_refuses_to_guess() -> None:
    reg = FamilyRegistry()
    reg.register(_family("dinkster.a", "model.weight", specificity=10))
    reg.register(_family("dinkster.b", "model.weight", specificity=10))
    result = reg.detect(DictSource({"model.weight": (1,)}))
    assert result.best is None
    assert set(result.ambiguous) == {"dinkster.a", "dinkster.b"}
    assert len(result.candidates) == 2


def test_family_detector_claiming_wrong_family_raises() -> None:
    reg = FamilyRegistry()
    family = ModelFamily(
        id="dinkster.honest",
        display_name="honest",
        detector=KeyDetector("dinkster.imposter", "model.weight"),
        specificity=10,
        latent=LatentDescriptor(channels=4),
        sampling=SamplingDescriptor(Parameterization.EPS, sigma_min=0.03, sigma_max=14.6),
        wiring=ComponentWiring(),
        supported_dtypes=frozenset({FLOAT16}),
    )
    reg.register(family)
    with pytest.raises(ValueError):
        reg.detect(DictSource({"model.weight": (1,)}))


def test_family_registry_alias_lookup() -> None:
    reg = FamilyRegistry()
    base = _family("dinkster.sdxl", "model.weight", specificity=10)
    reg.register(
        ModelFamily(
            **{**base.__dict__, "aliases": ("sdxl",)},
        )
    )
    assert reg.get("sdxl") is reg.get("dinkster.sdxl")
    assert reg.ids() == ("dinkster.sdxl",)


def test_model_family_memory_factor_validation() -> None:
    base = _family("dinkster.x", "k", specificity=1)
    with pytest.raises(ValueError):
        ModelFamily(**{**base.__dict__, "memory_factor": 0.0})
