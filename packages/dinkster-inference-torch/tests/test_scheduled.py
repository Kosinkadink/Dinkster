"""Scheduled text encoding and canonical carrier proofs."""

# Runtime fixtures intentionally replace private concrete encoder members with
# recording stand-ins; production scheduled.py is checked independently.
# pyright: reportPrivateUsage=false

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    CONDITIONING_TYPE_ID,
    EMPTY_RANGE,
    FLUX_DEV,
    SD15,
    SDXL,
    AreaDescriptor,
    AreaUnits,
    Conditioning,
    DiffPatchRef,
    EncoderStream,
    InferenceTypeRegistry,
    OverlayPatch,
    PatchOverlay,
    PatchTarget,
    PatchTargetComponent,
    PercentRange,
    PostEncodeTransform,
    PromptTokenizer,
    RegionDescriptor,
    ScheduledEncodeRequest,
    ScheduledEncodingError,
    ScheduledExecution,
    ScheduledPatchStack,
    ScheduledPrompt,
    ScheduledPromptRoute,
    ScheduledTransformStack,
    ScheduledVariantOwner,
    TransformTarget,
    WeightSourceRef,
    decode_conditioning_carrier,
    encode_conditioning_carrier,
)
from dinkster_inference_torch import (
    FluxRuntime,
    SDRuntime,
    ordinary_conditioning_carrier,
    payload_binding_to_tensor,
)


class RecordingEncoder:
    def __init__(self, *, pooled: bool = True) -> None:
        self.spans: list[object] = []
        self.pooled = pooled

    def encode(self, spans: object) -> Conditioning[torch.Tensor]:
        self.spans.append(spans)
        values = cast("tuple[Any, ...]", spans)
        weight = sum(float(span.weight) for span in values) or 1.0
        embeddings = torch.full((1, 4, 2), weight, dtype=torch.float32)
        pooled = torch.full((1, 2), weight, dtype=torch.float32) if self.pooled else None
        return Conditioning(embeddings, pooled)


class RecordingOvis:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def encode(self, prompt: str) -> Conditioning[torch.Tensor]:
        self.prompts.append(prompt)
        return Conditioning(torch.arange(8, dtype=torch.float32).reshape(1, 4, 2))


def tokenizer() -> PromptTokenizer:
    return PromptTokenizer(encode_word=lambda word: (len(word),))


def overlay(digit: str) -> PatchOverlay:
    return PatchOverlay.from_decoded(
        source=WeightSourceRef(
            digest="blake3:" + digit * 64,
            name=f"{digit}.safetensors",
            size=1,
        ),
        dialect="none",
        key_map="native.sd15.v1",
        strength_model=1.0,
        strength_clip=1.0,
        patches=(OverlayPatch("diffusion", PatchTarget("weight"), DiffPatchRef("delta")),),
    )


def sd_runtime() -> SDRuntime:
    runtime = object.__new__(SDRuntime)
    raw = cast("Any", runtime)
    raw.assembled = SimpleNamespace(family=SD15)
    raw._runtime_identity = "native:test:sd"
    raw._clip_l_encoder = RecordingEncoder()
    raw._clip_g_encoder = None
    raw._tokenizer = tokenizer()
    return runtime


def flux_runtime() -> FluxRuntime:
    runtime = object.__new__(FluxRuntime)
    raw = cast("Any", runtime)
    raw.assembled = SimpleNamespace(family=FLUX_DEV)
    raw._runtime_identity = "native:test:flux"
    raw._ovis_encoder = None
    raw._clip_encoder = RecordingEncoder()
    raw._t5_encoder = RecordingEncoder(pooled=False)
    raw._clip_tokenizer = tokenizer()
    raw._t5_tokenizer = tokenizer()
    return runtime


def ovis_runtime() -> tuple[FluxRuntime, RecordingOvis]:
    runtime = object.__new__(FluxRuntime)
    raw = cast("Any", runtime)
    raw.assembled = SimpleNamespace(family=FLUX_DEV)
    raw._runtime_identity = "native:test:ovis"
    encoder = RecordingOvis()
    raw._ovis_encoder = encoder
    raw._clip_encoder = None
    raw._t5_encoder = None
    raw._clip_tokenizer = None
    raw._t5_tokenizer = None
    return runtime, encoder


def request(
    family: str,
    prompt: str,
    *,
    schedule: PercentRange | None = None,
) -> ScheduledEncodeRequest:
    if schedule is None:
        schedule = PercentRange(0.0, 1.0)
    streams = {
        "sd": (EncoderStream.CLIP_L,),
        "flux": (EncoderStream.CLIP_L, EncoderStream.T5),
        "ovis": (EncoderStream.OVIS_QWEN3_2B,),
    }[family]
    return ScheduledEncodeRequest(
        (
            ScheduledPrompt(
                schedule,
                tuple(ScheduledPromptRoute(stream, prompt) for stream in streams),
            ),
        )
    )


@pytest.mark.parametrize(
    ("runtime", "family"),
    ((sd_runtime, "sd"), (flux_runtime, "flux")),
)
def test_empty_scheduled_input_is_exact_ordinary_carrier_and_identity(
    runtime: object, family: str
) -> None:
    concrete = cast("Any", runtime)()
    identity = concrete.runtime_identity
    ordinary = ordinary_conditioning_carrier(concrete, "cat")
    types = InferenceTypeRegistry()
    scheduled = concrete.encode_text_scheduled(request(family, "cat"), type_registry=types)
    assert encode_conditioning_carrier(scheduled) == encode_conditioning_carrier(ordinary)
    assert concrete.runtime_identity == identity
    assert scheduled.conditioning.records[0].extension_metadata == ()
    assert CONDITIONING_TYPE_ID in types


def test_clip_l_and_t5_keep_existing_weighted_span_semantics_and_exact_routes() -> None:
    runtime = flux_runtime()
    carrier = runtime.encode_text_scheduled(
        ScheduledEncodeRequest(
            (
                ScheduledPrompt(
                    PercentRange(0.0, 1.0),
                    (
                        ScheduledPromptRoute(EncoderStream.CLIP_L, "(cat:2.0)"),
                        ScheduledPromptRoute(EncoderStream.T5, "(dog:3.0)"),
                    ),
                ),
            )
        ),
        type_registry=InferenceTypeRegistry(),
    )
    clip_l = cast("RecordingEncoder", cast("Any", runtime)._clip_encoder)
    t5 = cast("RecordingEncoder", cast("Any", runtime)._t5_encoder)
    assert cast("Any", clip_l.spans[0])[0].weight == 2.0
    assert cast("Any", t5.spans[0])[0].weight == 3.0
    record = carrier.conditioning.records[0]
    assert record.token_layout is not None
    assert record.token_layout.text_streams == ("clip_l", "t5")
    assert tuple(
        (segment.name, segment.stream, segment.start_token, segment.token_count)
        for segment in record.token_layout.segments
    ) == (("t5", "t5", 0, 4),)
    assert tuple(channel.value for channel, _ in record.channels) == ("text", "pooled")
    with pytest.raises(ScheduledEncodingError, match="requires scheduled routes"):
        runtime.encode_text_scheduled(request("sd", "cat"), type_registry=InferenceTypeRegistry())


def test_clip_g_keeps_existing_weighted_span_semantics() -> None:
    runtime = sd_runtime()
    cast("Any", runtime).assembled = SimpleNamespace(family=SDXL)
    clip_g = RecordingEncoder()
    cast("Any", runtime)._clip_g_encoder = clip_g
    carrier = runtime.encode_text_scheduled(
        ScheduledEncodeRequest(
            (
                ScheduledPrompt(
                    PercentRange(0.0, 1.0),
                    (
                        ScheduledPromptRoute(EncoderStream.CLIP_L, "(cat:2.0)"),
                        ScheduledPromptRoute(EncoderStream.CLIP_G, "(dog:3.0)"),
                    ),
                ),
            )
        ),
        type_registry=InferenceTypeRegistry(),
    )
    assert cast("Any", clip_g.spans[0])[0].weight == 3.0
    assert carrier.conditioning.records[0].token_layout is not None
    layout = carrier.conditioning.records[0].token_layout
    assert layout.text_streams == ("clip_l", "clip_g")
    assert tuple(
        (segment.name, segment.stream, segment.start_token, segment.token_count)
        for segment in layout.segments
    ) == (("clip_l", "clip_l", 0, 4), ("clip_g", "clip_g", 0, 4))


def test_ovis_raw_prompt_and_nonunit_weight_refusal_precede_tokenization() -> None:
    runtime, encoder = ovis_runtime()
    raw = "literal punctuation:words"
    carrier = runtime.encode_text_scheduled(
        request("ovis", raw), type_registry=InferenceTypeRegistry()
    )
    assert encoder.prompts == [raw]
    record = carrier.conditioning.records[0]
    assert record.channels[0][0].value == "text"
    assert record.token_layout is not None
    assert record.token_layout.text_streams == ("ovis_qwen3_2b",)
    assert tuple(
        (segment.name, segment.stream, segment.start_token, segment.token_count)
        for segment in record.token_layout.segments
    ) == (("ovis_qwen3_2b", "ovis_qwen3_2b", 0, 4),)
    before = encode_conditioning_carrier(carrier)
    with pytest.raises(ScheduledEncodingError, match="non-unit weights"):
        runtime.encode_text_scheduled(
            request("ovis", "(cat:1.5)"), type_registry=InferenceTypeRegistry()
        )
    assert encoder.prompts == [raw]
    assert encode_conditioning_carrier(carrier) == before


def test_transform_order_intersection_cancellation_and_no_partial_carrier() -> None:
    runtime = sd_runtime()
    order: list[str] = []

    def first(value: torch.Tensor) -> torch.Tensor:
        order.append("first")
        return value + 1

    def second(value: torch.Tensor) -> torch.Tensor:
        order.append("second")
        return value * 2

    transforms = (
        PostEncodeTransform(
            "pack.first", TransformTarget.TEXT, "dinkster.sd15", (EncoderStream.CLIP_L,)
        ),
        PostEncodeTransform(
            "pack.second", TransformTarget.TEXT, "dinkster.sd15", (EncoderStream.CLIP_L,)
        ),
    )
    scheduled = ScheduledEncodeRequest(
        (
            ScheduledPrompt(
                PercentRange(0.0, 0.5),
                (ScheduledPromptRoute(EncoderStream.CLIP_L, "cat"),),
            ),
        ),
        transform_stacks=(ScheduledTransformStack(PercentRange(0.5, 1.0), transforms),),
    )
    carrier = runtime.encode_text_scheduled(
        scheduled,
        transforms={"pack.first": first, "pack.second": second},
        type_registry=InferenceTypeRegistry(),
    )
    assert order == ["first", "second"]
    record = carrier.conditioning.records[0]
    assert record.schedule == PercentRange(0.5, 0.5)
    binding = next(item for item in carrier.bindings if item.space == "conditioning-text")
    assert torch.all(payload_binding_to_tensor(binding) == 4)
    assert dict(record.extension_metadata)["dinkster.inference/transform-ids"] == (
        "pack.first",
        "pack.second",
    )

    calls = 0

    def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 3

    with pytest.raises(ScheduledEncodingError, match="cancelled"):
        runtime.encode_text_scheduled(
            scheduled,
            transforms={"pack.first": first, "pack.second": second},
            cancelled=cancelled,
            type_registry=InferenceTypeRegistry(),
        )


def test_sd15_layout_pins_its_single_clip_l_segment() -> None:
    carrier = sd_runtime().encode_text_scheduled(
        request("sd", "cat"), type_registry=InferenceTypeRegistry()
    )
    layout = carrier.conditioning.records[0].token_layout
    assert layout is not None
    assert layout.text_streams == ("clip_l",)
    assert tuple(
        (segment.name, segment.stream, segment.start_token, segment.token_count)
        for segment in layout.segments
    ) == (("clip_l", "clip_l", 0, 4),)


def test_empty_disjoint_and_boundary_records_are_deterministic_and_round_trip() -> None:
    runtime = sd_runtime()
    scheduled = ScheduledEncodeRequest(
        (
            ScheduledPrompt(
                EMPTY_RANGE,
                (ScheduledPromptRoute(EncoderStream.CLIP_L, "never"),),
            ),
            ScheduledPrompt(
                PercentRange(0.0, 0.5),
                (ScheduledPromptRoute(EncoderStream.CLIP_L, "left"),),
            ),
            ScheduledPrompt(
                PercentRange(0.5, 1.0),
                (ScheduledPromptRoute(EncoderStream.CLIP_L, "right"),),
            ),
        )
    )
    first = runtime.encode_text_scheduled(scheduled, type_registry=InferenceTypeRegistry())
    second = runtime.encode_text_scheduled(scheduled, type_registry=InferenceTypeRegistry())
    encoded = encode_conditioning_carrier(first)
    assert encoded == encode_conditioning_carrier(second)
    assert tuple(record.schedule for record in first.conditioning.records) == (
        PercentRange(0.0, 0.5),
        PercentRange(0.5, 1.0),
    )
    assert encode_conditioning_carrier(decode_conditioning_carrier(encoded)) == encoded


def test_transform_family_and_ovis_pooled_refuse_before_encoding() -> None:
    runtime, encoder = ovis_runtime()
    pooled = ScheduledEncodeRequest(
        request("ovis", "cat").prompts,
        transform_stacks=(
            ScheduledTransformStack(
                PercentRange(0.0, 1.0),
                (
                    PostEncodeTransform(
                        "pack.pool",
                        TransformTarget.POOLED,
                        "dinkster.flux_dev",
                        (EncoderStream.OVIS_QWEN3_2B,),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ScheduledEncodingError, match="no pooled"):
        runtime.encode_text_scheduled(
            pooled,
            transforms={"pack.pool": lambda value: value},
            type_registry=InferenceTypeRegistry(),
        )
    assert encoder.prompts == []
    wrong = ScheduledEncodeRequest(
        request("sd", "cat").prompts,
        transform_stacks=(
            ScheduledTransformStack(
                PercentRange(0.0, 1.0),
                (
                    PostEncodeTransform(
                        "pack.wrong",
                        TransformTarget.TEXT,
                        "dinkster.sdxl",
                        (EncoderStream.CLIP_L, EncoderStream.CLIP_G),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ScheduledEncodingError, match="incompatible family"):
        sd_runtime().encode_text_scheduled(
            wrong,
            transforms={"pack.wrong": lambda value: value},
            type_registry=InferenceTypeRegistry(),
        )
    wrong_layout = ScheduledEncodeRequest(
        request("sd", "cat").prompts,
        transform_stacks=(
            ScheduledTransformStack(
                PercentRange(0.0, 1.0),
                (
                    PostEncodeTransform(
                        "pack.layout",
                        TransformTarget.TEXT,
                        "dinkster.sd15",
                        (EncoderStream.CLIP_G,),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ScheduledEncodingError, match="incompatible text layout"):
        sd_runtime().encode_text_scheduled(
            wrong_layout,
            transforms={"pack.layout": lambda value: value},
            type_registry=InferenceTypeRegistry(),
        )


def test_flux_transforms_bind_to_the_lane_that_produced_the_tensor() -> None:
    runtime = flux_runtime()
    prompt = ScheduledEncodeRequest(
        (
            ScheduledPrompt(
                PercentRange(0.0, 1.0),
                (
                    ScheduledPromptRoute(EncoderStream.CLIP_L, "clip"),
                    ScheduledPromptRoute(EncoderStream.T5, "t5"),
                ),
            ),
        ),
        transform_stacks=(
            ScheduledTransformStack(
                PercentRange(0.0, 1.0),
                (
                    PostEncodeTransform(
                        "pack.text",
                        TransformTarget.TEXT,
                        "dinkster.flux_dev",
                        (EncoderStream.T5,),
                    ),
                    PostEncodeTransform(
                        "pack.pooled",
                        TransformTarget.POOLED,
                        "dinkster.flux_dev",
                        (EncoderStream.CLIP_L,),
                    ),
                ),
            ),
        ),
    )
    carrier = runtime.encode_text_scheduled(
        prompt,
        transforms={"pack.text": lambda value: value + 1, "pack.pooled": lambda value: value + 2},
        type_registry=InferenceTypeRegistry(),
    )
    text = next(binding for binding in carrier.bindings if binding.space == "conditioning-text")
    pooled = next(binding for binding in carrier.bindings if binding.space == "conditioning-pooled")
    assert torch.all(payload_binding_to_tensor(text) == 2)
    assert torch.all(payload_binding_to_tensor(pooled) == 3)

    incompatible = replace(
        prompt.transform_stacks[0].transforms[0],
        text_streams=(EncoderStream.CLIP_L, EncoderStream.T5),
    )
    refused = replace(
        prompt,
        transform_stacks=(ScheduledTransformStack(PercentRange(0.0, 1.0), (incompatible,)),),
    )
    with pytest.raises(ScheduledEncodingError, match="incompatible text layout"):
        runtime.encode_text_scheduled(
            refused,
            transforms={"pack.text": lambda value: value},
            type_registry=InferenceTypeRegistry(),
        )


class VariantRuntime:
    def __init__(self, base: SDRuntime, disposed: list[str]) -> None:
        self.__dict__.update(base.__dict__)
        self.disposed = disposed

    @property
    def family(self) -> object:
        return cast("Any", self).assembled.family

    def dispose(self) -> None:
        self.disposed.append("variant")


def change_shape(value: torch.Tensor) -> torch.Tensor:
    return value[:, :-1]


def change_dtype(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float64)


def change_device(value: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(value, device="meta")


@pytest.mark.parametrize(
    ("name", "transform", "message"),
    (
        ("shape", change_shape, "changed tensor shape"),
        ("dtype", change_dtype, "changed tensor dtype"),
        ("device", change_device, "changed tensor device"),
    ),
)
def test_transform_tensor_contract_refuses_and_discards_owned_variant(
    name: str,
    transform: Any,
    message: str,
) -> None:
    runtime = sd_runtime()
    disposed: list[str] = []

    def build(
        base: object,
        _target: PatchTargetComponent,
        _overlays: tuple[PatchOverlay, ...],
        _cancelled: object,
    ) -> object:
        return VariantRuntime(cast("SDRuntime", base), disposed)

    execution = ScheduledExecution(ScheduledVariantOwner(build))
    scheduled = ScheduledEncodeRequest(
        request("sd", "cat").prompts,
        text_patches=(ScheduledPatchStack(PercentRange(0.0, 1.0), (overlay("1"),)),),
        transform_stacks=(
            ScheduledTransformStack(
                PercentRange(0.0, 1.0),
                (
                    PostEncodeTransform(
                        f"pack.{name}",
                        TransformTarget.TEXT,
                        "dinkster.sd15",
                        (EncoderStream.CLIP_L,),
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(ScheduledEncodingError, match=message):
        runtime.encode_text_scheduled(
            scheduled,
            execution=execution,
            transforms={f"pack.{name}": transform},
            type_registry=InferenceTypeRegistry(),
        )
    assert disposed == ["variant"]
    replacement = execution.variants.acquire(
        runtime,
        base_runtime_identity=runtime.runtime_identity,
        target=PatchTargetComponent.TEXT,
        overlays=(overlay("1"),),
    )
    assert isinstance(replacement, VariantRuntime)
    execution.close()
    assert disposed == ["variant", "variant"]


def test_ordinary_equivalent_cancellation_closes_existing_execution_variants() -> None:
    runtime = sd_runtime()
    disposed: list[str] = []

    def build(
        base: object,
        _target: PatchTargetComponent,
        _overlays: tuple[PatchOverlay, ...],
        _cancelled: object,
    ) -> object:
        return VariantRuntime(cast("SDRuntime", base), disposed)

    execution = ScheduledExecution(ScheduledVariantOwner(build))
    execution.variants.acquire(
        runtime,
        base_runtime_identity=runtime.runtime_identity,
        target=PatchTargetComponent.TEXT,
        overlays=(overlay("1"),),
    )
    calls = 0

    def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2

    with pytest.raises(ScheduledEncodingError, match="cancelled"):
        runtime.encode_text_scheduled(
            request("sd", "cat"),
            execution=execution,
            cancelled=cancelled,
            type_registry=InferenceTypeRegistry(),
        )
    assert disposed == ["variant"]
    with pytest.raises(ScheduledEncodingError, match="closed"):
        execution.variants.acquire(
            runtime,
            base_runtime_identity=runtime.runtime_identity,
            target=PatchTargetComponent.TEXT,
            overlays=(overlay("2"),),
        )


def test_patch_builder_preflight_metadata_preservation_and_failure_cleanup() -> None:
    runtime = sd_runtime()
    encoder = cast("RecordingEncoder", cast("Any", runtime)._clip_l_encoder)
    patched = ScheduledEncodeRequest(
        (
            ScheduledPrompt(
                PercentRange(0.0, 1.0),
                (ScheduledPromptRoute(EncoderStream.CLIP_L, "cat"),),
                (("pack.example/note", "kept"),),
            ),
        ),
        text_patches=(ScheduledPatchStack(PercentRange(0.25, 0.75), (overlay("1"),)),),
    )
    with pytest.raises(ScheduledEncodingError, match="execution owner"):
        runtime.encode_text_scheduled(patched, type_registry=InferenceTypeRegistry())
    assert encoder.spans == []

    disposed: list[str] = []

    def build(
        base: object,
        target: PatchTargetComponent,
        _overlays: tuple[PatchOverlay, ...],
        _cancelled: object,
    ) -> object:
        assert target is PatchTargetComponent.TEXT
        return VariantRuntime(cast("SDRuntime", base), disposed)

    execution = ScheduledExecution(ScheduledVariantOwner(build))
    carrier = runtime.encode_text_scheduled(
        patched, execution=execution, type_registry=InferenceTypeRegistry()
    )
    record = carrier.conditioning.records[0]
    assert record.schedule == PercentRange(0.25, 0.75)
    metadata = dict(record.extension_metadata)
    assert metadata["pack.example/note"] == "kept"
    assert metadata["dinkster.inference/text-overlay-digests"] == (overlay("1").structural_digest,)
    assert record.clone().extension_metadata == record.extension_metadata
    combined = carrier.conditioning.combine(carrier.conditioning)
    assert tuple(item.extension_metadata for item in combined.records) == (
        record.extension_metadata,
        record.extension_metadata,
    )
    region = RegionDescriptor(area=AreaDescriptor(1, 1, 0, 0, AreaUnits.LATENT_CELLS))
    assert (
        carrier.conditioning.for_region(region).records[0].extension_metadata
        == record.extension_metadata
    )

    failing = ScheduledEncodeRequest(
        patched.prompts,
        text_patches=patched.text_patches,
        transform_stacks=(
            ScheduledTransformStack(
                PercentRange(0.0, 1.0),
                (
                    PostEncodeTransform(
                        "pack.fail",
                        TransformTarget.TEXT,
                        "dinkster.sd15",
                        (EncoderStream.CLIP_L,),
                    ),
                ),
            ),
        ),
    )

    def fail(_value: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("transform failed")

    with pytest.raises(RuntimeError, match="transform failed"):
        runtime.encode_text_scheduled(
            failing,
            execution=execution,
            transforms={"pack.fail": fail},
            type_registry=InferenceTypeRegistry(),
        )
    assert disposed == ["variant"]
    execution.close()
    assert disposed == ["variant"]


@pytest.mark.parametrize("factory", (sd_runtime, flux_runtime))
def test_scheduled_encoding_uses_declared_encoders_not_family_id(factory: Any) -> None:
    from dinkster_inference_torch.scheduled import _expected_streams

    runtime = factory()
    expected = runtime.encode_text("cat")
    runtime.assembled.family = replace(runtime.family, id="test.synthetic-family")
    routes = tuple(ScheduledPromptRoute(stream, "cat") for stream in _expected_streams(runtime))
    result = runtime.encode_text_scheduled(
        ScheduledEncodeRequest((ScheduledPrompt(PercentRange(0.0, 1.0), routes),)),
        type_registry=InferenceTypeRegistry(),
    )
    from dinkster_inference_torch import materialize_basic_conditioning

    actual = materialize_basic_conditioning(result, device="cpu")
    assert torch.equal(actual.embeddings, expected.embeddings)
    if expected.pooled is not None:
        assert actual.pooled is not None
        assert torch.equal(actual.pooled, expected.pooled)
