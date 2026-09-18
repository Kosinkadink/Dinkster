"""Torch-free scheduled request and variant ownership proofs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import pytest
from dinkster_inference import (
    EMPTY_RANGE,
    SCHEDULED_METADATA_KEYS,
    DiffPatchRef,
    EncoderStream,
    OverlayPatch,
    PatchOverlay,
    PatchTarget,
    PatchTargetComponent,
    PayloadReference,
    PercentRange,
    PostEncodeTransform,
    ScheduledEncodeRequest,
    ScheduledEncodingError,
    ScheduledPatchStack,
    ScheduledPrompt,
    ScheduledPromptRoute,
    ScheduledTransformStack,
    ScheduledVariantOwner,
    TransformTarget,
    VariantKey,
    WeightSourceRef,
    intersect_ranges,
    patch_overlay_stack_digest,
    scheduled_metadata,
)


def overlay(digit: str, *, strength: float = 1.0) -> PatchOverlay:
    return PatchOverlay.from_decoded(
        source=WeightSourceRef(
            digest="blake3:" + digit * 64,
            name=f"{digit}.safetensors",
            size=1,
        ),
        dialect="none",
        key_map="native.sd15.v1",
        strength_model=strength,
        strength_clip=1.0,
        patches=(OverlayPatch("diffusion", PatchTarget("weight"), DiffPatchRef("delta")),),
    )


def test_closed_range_algebra_covers_empty_disjoint_zero_width_and_boundaries() -> None:
    full = PercentRange(0.0, 1.0)
    left = PercentRange(0.0, 0.5)
    right = PercentRange(0.5, 1.0)
    assert intersect_ranges(full, left) == left
    assert intersect_ranges(EMPTY_RANGE, full) is EMPTY_RANGE
    assert intersect_ranges(PercentRange(0.0, 0.4), PercentRange(0.6, 1.0)) is EMPTY_RANGE
    assert intersect_ranges(left, right) == PercentRange(0.5, 0.5)
    assert intersect_ranges(PercentRange(0.2, 0.8), PercentRange(0.3, 0.7)) == PercentRange(
        0.3, 0.7
    )


def test_request_is_frozen_strict_and_reserves_the_whole_metadata_namespace() -> None:
    reference = PayloadReference("payload-1")
    nested = ["before", {"key": [1, 2], "reference": reference}]
    prompt = ScheduledPrompt(
        PercentRange(0.0, 1.0),
        (ScheduledPromptRoute(EncoderStream.CLIP_L, "cat"),),
        (("pack.example/nested", nested),),
    )
    request = ScheduledEncodeRequest((prompt,))
    nested[0] = "after"
    cast("dict[str, list[int]]", nested[1])["key"].append(3)
    frozen = dict(request.prompts[0].extension_metadata)["pack.example/nested"]
    assert isinstance(frozen, tuple)
    assert frozen[0] == "before"
    assert isinstance(frozen[1], MappingProxyType)
    assert dict(cast("Any", frozen[1])) == {"key": (1, 2), "reference": reference}
    assert cast("Any", frozen[1])["reference"] is reference
    with pytest.raises(TypeError):
        cast("Any", frozen[1])["key"] = (3,)
    assert not request.ordinary_equivalent
    with pytest.raises(ValueError, match="unique"):
        ScheduledPrompt(
            PercentRange(0.0, 1.0),
            (
                ScheduledPromptRoute(EncoderStream.CLIP_L, "a"),
                ScheduledPromptRoute(EncoderStream.CLIP_L, "b"),
            ),
        )
    with pytest.raises(ScheduledEncodingError, match="reserved namespace"):
        ScheduledPrompt(
            PercentRange(0.0, 1.0),
            (ScheduledPromptRoute(EncoderStream.CLIP_L, "cat"),),
            (("dinkster.inference/not-allowed", "value"),),
        )
    with pytest.raises(TypeError, match="non-RPC-clean"):
        ScheduledPrompt(
            PercentRange(0.0, 1.0),
            (ScheduledPromptRoute(EncoderStream.CLIP_L, "cat"),),
            cast("Any", (("pack.example/live", object()),)),
        )
    with pytest.raises(ValueError, match="pack_id/key"):
        ScheduledPrompt(
            PercentRange(0.0, 1.0),
            (ScheduledPromptRoute(EncoderStream.CLIP_L, "cat"),),
            (("malformed", "value"),),
        )


def test_transform_and_patch_descriptors_validate_identity_and_order() -> None:
    first = overlay("1")
    second = overlay("2")
    stack = ScheduledPatchStack(PercentRange(0.0, 0.5), (first, second))
    assert stack.stack_digest == patch_overlay_stack_digest((first, second))
    assert stack.stack_digest != patch_overlay_stack_digest((second, first))
    assert overlay("1", strength=0.5).structural_digest != first.structural_digest
    transform = PostEncodeTransform(
        "pack.normalize", TransformTarget.TEXT, "dinkster.sd15", (EncoderStream.CLIP_L,)
    )
    assert ScheduledTransformStack(PercentRange(0.5, 1.0), (transform,)).transforms == (transform,)
    assert (
        transform.structural_digest
        != PostEncodeTransform(
            "pack.normalize", TransformTarget.POOLED, "dinkster.sd15", (EncoderStream.CLIP_L,)
        ).structural_digest
    )
    with pytest.raises(ValueError, match="layout version"):
        PostEncodeTransform(
            "pack.normalize",
            TransformTarget.TEXT,
            "dinkster.sd15",
            (EncoderStream.CLIP_L,),
            layout_version=2,
        )


def test_reserved_metadata_is_finite_structural_and_effective_identity_rotates() -> None:
    first = overlay("1")
    second = overlay("2")
    metadata = scheduled_metadata(
        target="dinkster.sd15",
        text_overlays=(first, second),
        diffusion_overlays=(second,),
        transforms=(
            PostEncodeTransform(
                "pack.first", TransformTarget.TEXT, "dinkster.sd15", (EncoderStream.CLIP_L,)
            ),
            PostEncodeTransform(
                "pack.second", TransformTarget.POOLED, "dinkster.sd15", (EncoderStream.CLIP_L,)
            ),
        ),
    )
    values = dict(metadata)
    assert set(values) <= SCHEDULED_METADATA_KEYS
    assert values["dinkster.inference/text-overlay-digests"] == (
        first.structural_digest,
        second.structural_digest,
    )
    changed = dict(
        scheduled_metadata(
            target="dinkster.sd15",
            text_overlays=(second, first),
            diffusion_overlays=(second,),
            transforms=(
                PostEncodeTransform(
                    "pack.first",
                    TransformTarget.TEXT,
                    "dinkster.sd15",
                    (EncoderStream.CLIP_L,),
                ),
                PostEncodeTransform(
                    "pack.second",
                    TransformTarget.POOLED,
                    "dinkster.sd15",
                    (EncoderStream.CLIP_L,),
                ),
            ),
        )
    )
    assert (
        values["dinkster.inference/effective-patch-state"]
        != changed["dinkster.inference/effective-patch-state"]
    )
    assert (
        scheduled_metadata(
            target="dinkster.sd15",
            text_overlays=(),
            diffusion_overlays=(),
            transforms=(),
        )
        == ()
    )
    assert all(
        not callable(value) and not isinstance(value, (bytes, bytearray)) for _, value in metadata
    )


@dataclass
class Variant:
    name: str
    disposed: list[str]
    dispose_count: int = 0

    def dispose(self) -> None:
        self.dispose_count += 1
        self.disposed.append(self.name)


def test_variant_owner_empty_bypass_no_builder_refusal_and_default_partition_lru() -> None:
    base = object()
    no_builder: ScheduledVariantOwner[Variant] = ScheduledVariantOwner()
    assert (
        no_builder.acquire(
            base,
            base_runtime_identity="base",
            target=PatchTargetComponent.TEXT,
            overlays=(),
        )
        is base
    )
    with pytest.raises(ScheduledEncodingError, match="variant builder"):
        no_builder.acquire(
            base,
            base_runtime_identity="base",
            target=PatchTargetComponent.TEXT,
            overlays=(overlay("1"),),
        )

    disposed: list[str] = []

    def build(
        _base: object,
        target: PatchTargetComponent,
        overlays: tuple[PatchOverlay, ...],
        _cancelled: object,
    ) -> Variant:
        return Variant(f"{target.value}:{overlays[0].source.name}", disposed)

    owner = ScheduledVariantOwner(build)
    text1 = owner.acquire(
        base,
        base_runtime_identity="base",
        target=PatchTargetComponent.TEXT,
        overlays=(overlay("1"),),
    )
    diffusion1 = owner.acquire(
        base,
        base_runtime_identity="base",
        target=PatchTargetComponent.DIFFUSION,
        overlays=(overlay("1"),),
    )
    text2 = owner.acquire(
        base,
        base_runtime_identity="base",
        target=PatchTargetComponent.TEXT,
        overlays=(overlay("2"),),
    )
    assert isinstance(text1, Variant) and text1.dispose_count == 1
    assert isinstance(diffusion1, Variant) and diffusion1.dispose_count == 0
    assert isinstance(text2, Variant) and text2.dispose_count == 0
    assert disposed == ["text:1.safetensors"]
    owner.close()
    owner.end()
    assert text2.dispose_count == diffusion1.dispose_count == 1


def test_variant_owner_bounded_lru_poison_cancellation_failure_and_close_once() -> None:
    with pytest.raises(ValueError, match="capacity"):
        ScheduledVariantOwner(text_capacity=17)
    disposed: list[str] = []
    built: list[Variant] = []
    cancelled = False

    def build(
        _base: object,
        _target: PatchTargetComponent,
        overlays: tuple[PatchOverlay, ...],
        _check: object,
    ) -> Variant:
        candidate = Variant(overlays[0].source.name, disposed)
        built.append(candidate)
        return candidate

    owner = ScheduledVariantOwner(build, text_capacity=2)
    base = object()
    one = cast_variant(
        owner.acquire(
            base,
            base_runtime_identity="base",
            target=PatchTargetComponent.TEXT,
            overlays=(overlay("1"),),
        )
    )
    two = cast_variant(
        owner.acquire(
            base,
            base_runtime_identity="base",
            target=PatchTargetComponent.TEXT,
            overlays=(overlay("2"),),
        )
    )
    assert (
        owner.acquire(
            base,
            base_runtime_identity="base",
            target=PatchTargetComponent.TEXT,
            overlays=(overlay("1"),),
        )
        is one
    )
    three = cast_variant(
        owner.acquire(
            base,
            base_runtime_identity="base",
            target=PatchTargetComponent.TEXT,
            overlays=(overlay("3"),),
        )
    )
    assert two.dispose_count == 1
    key = VariantKey(
        "base",
        PatchTargetComponent.TEXT,
        patch_overlay_stack_digest((overlay("1"),)) or "",
    )
    owner.poison(key)
    assert one.dispose_count == 1
    with pytest.raises(ScheduledEncodingError, match="cancelled"):
        owner.acquire(
            base,
            base_runtime_identity="base",
            target=PatchTargetComponent.TEXT,
            overlays=(overlay("4"),),
            cancelled=lambda: True,
        )
    assert len(built) == 3
    cancelled = True
    del cancelled
    owner.close()
    assert three.dispose_count == 1


def test_candidate_cancel_after_build_disposes_and_never_caches() -> None:
    disposed: list[str] = []
    calls = 0

    def build(
        _base: object,
        _target: PatchTargetComponent,
        overlays: tuple[PatchOverlay, ...],
        check: Callable[[], bool],
    ) -> Variant:
        nonlocal calls
        calls += 1
        if overlays[0].source.name == "2.safetensors" and check():
            raise ScheduledEncodingError("builder observed cancellation")
        return Variant("candidate", disposed)

    checks = iter((False, True))
    owner = ScheduledVariantOwner(build)
    with pytest.raises(ScheduledEncodingError, match="cancelled"):
        owner.acquire(
            object(),
            base_runtime_identity="base",
            target=PatchTargetComponent.TEXT,
            overlays=(overlay("1"),),
            cancelled=lambda: next(checks),
        )
    assert disposed == ["candidate"]
    retry_owner = ScheduledVariantOwner(build)
    retry_owner.acquire(
        object(),
        base_runtime_identity="base",
        target=PatchTargetComponent.TEXT,
        overlays=(overlay("1"),),
    )
    assert calls == 2
    builder_checks = iter((False, True))
    with pytest.raises(ScheduledEncodingError, match="builder observed"):
        retry_owner.acquire(
            object(),
            base_runtime_identity="base",
            target=PatchTargetComponent.TEXT,
            overlays=(overlay("2"),),
            cancelled=lambda: next(builder_checks),
        )
    assert disposed == ["candidate", "candidate"]


def test_variant_owner_rejects_owned_alias_and_drains_throwing_disposers() -> None:
    disposed: list[str] = []

    @dataclass
    class ThrowingVariant(Variant):
        throws: bool = False

        def dispose(self) -> None:
            super().dispose()
            if self.throws:
                raise RuntimeError(f"dispose {self.name}")

    candidates = iter(
        (
            ThrowingVariant("text", disposed, throws=True),
            ThrowingVariant("diffusion", disposed),
        )
    )

    def build(
        _base: object,
        _target: PatchTargetComponent,
        _overlays: tuple[PatchOverlay, ...],
        _check: object,
    ) -> ThrowingVariant:
        return next(candidates)

    owner = ScheduledVariantOwner(build)
    base = object()
    owner.acquire(
        base,
        base_runtime_identity="base",
        target=PatchTargetComponent.TEXT,
        overlays=(overlay("1"),),
    )
    owner.acquire(
        base,
        base_runtime_identity="base",
        target=PatchTargetComponent.DIFFUSION,
        overlays=(overlay("2"),),
    )
    keys = (
        VariantKey(
            "base",
            PatchTargetComponent.TEXT,
            patch_overlay_stack_digest((overlay("1"),)) or "",
        ),
        VariantKey(
            "base",
            PatchTargetComponent.DIFFUSION,
            patch_overlay_stack_digest((overlay("2"),)) or "",
        ),
    )
    error = owner.poison_many(keys)
    assert isinstance(error, RuntimeError)
    assert disposed == ["text", "diffusion"]
    owner.close()
    assert disposed == ["text", "diffusion"]

    alias = Variant("alias", disposed)
    alias_owner = ScheduledVariantOwner(lambda *_args: alias, text_capacity=2)
    alias_owner.acquire(
        base,
        base_runtime_identity="base",
        target=PatchTargetComponent.TEXT,
        overlays=(overlay("1"),),
    )
    with pytest.raises(ScheduledEncodingError, match="newly owned"):
        alias_owner.acquire(
            base,
            base_runtime_identity="base",
            target=PatchTargetComponent.TEXT,
            overlays=(overlay("2"),),
        )
    alias_owner.close()
    assert alias.dispose_count == 1


def cast_variant(value: object) -> Variant:
    assert isinstance(value, Variant)
    return value


@pytest.mark.parametrize(
    ("encoder_id", "stream"),
    (
        ("dinkster.clip_l", EncoderStream.CLIP_L),
        ("dinkster.clip_g", EncoderStream.CLIP_G),
        ("dinkster.t5xxl", EncoderStream.T5),
        ("dinkster.ovis_qwen3_2b", EncoderStream.OVIS_QWEN3_2B),
    ),
)
def test_scheduled_stream_uses_encoder_identity(encoder_id: str, stream: EncoderStream) -> None:
    assert EncoderStream.from_encoder_id(encoder_id) is stream


def test_unknown_scheduled_encoder_reports_the_missing_capability() -> None:
    with pytest.raises(ScheduledEncodingError, match="text encoder 'test.unknown'"):
        EncoderStream.from_encoder_id("test.unknown")
