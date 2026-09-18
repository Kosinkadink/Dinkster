"""Synthetic ACE text math and real ordered handle reconstruction, not pretrained parity."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import (
    LUMINA2,
    AreaDescriptor,
    AreaUnits,
    ConditioningCarrier,
    ConditioningChannel,
    ConditioningRecord,
    ConditioningSet,
    ConditioningWireError,
    ConditionScaleVector,
    MaskDescriptor,
    PayloadDescriptor,
    PayloadReference,
    PercentRange,
    TokenLayoutDescriptor,
    TokenSegmentDescriptor,
    component_catalog,
    load_safetensors_header,
    make_conditioning_carrier,
    text_recipes,
)
from dinkster_inference import ace15_text as planning
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.component_registry import ComponentRegistry
from dinkster_inference.qwen_text import (
    ACE15_QWEN3_06B_CONFIG,
    ACE15_QWEN3_2B_CONFIG,
    QwenTextConfig,
)
from dinkster_inference.registry import Registry
from dinkster_inference.text_recipes import (
    TextEncodingProfile,
    TextRecipeBinding,
    TextRecipeComponent,
    TextRecipeDescriptor,
)
from dinkster_inference_torch import ace15_text as implementation
from dinkster_inference_torch.ace15_text import (
    AUDIO_START,
    EOS,
    PAD,
    ACE15Generation,
    ACE15TextRuntime,
    ACE15Tokens,
    ace15_attention_mask,
    ace15_prompts,
    compose_ace15_conditioning,
    generate_audio_codes,
    materialize_ace15_conditioning,
    tokenize_ace15_prompt,
)
from dinkster_inference_torch.payloads import payload_binding_to_tensor, tensor_to_payload_binding
from dinkster_inference_torch.quant_linear import Fp8Linear
from dinkster_inference_torch.qwen_text import (
    QwenBlock,
    QwenTextModel,
    _QwenFixedKV,  # pyright: ignore[reportPrivateUsage]
)
from dinkster_inference_torch.text_recipes import LoadedTextRecipe
from dinkster_native import native_arm
from safetensors.torch import save_file


def tiny_config(lm: bool = False) -> QwenTextConfig:
    return replace(
        ACE15_QWEN3_2B_CONFIG if lm else ACE15_QWEN3_06B_CONFIG,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_head_dim=4,
        max_position_embeddings=1024,
    )


def tiny_model(lm: bool = False) -> QwenTextModel:
    model = QwenTextModel(tiny_config(lm))
    rng = torch.Generator().manual_seed(1261)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=rng) * 0.1)
            if "norm" in name:
                parameter.add_(1.0)
    return model


def runtime(conditioner: QwenTextModel, lm: QwenTextModel) -> ACE15TextRuntime:
    components = tuple(
        TextRecipeComponent(
            index,
            role,
            ComponentPlan(role, Path("unused"), model.config, {}, {}, {}),
            TextEncodingProfile(planning.ACE15_TOKENIZER_PROFILE),
        )
        for index, (role, model) in enumerate(
            zip(planning.ACE15_TEXT_ROLES, (conditioner, lm), strict=True)
        )
    )
    binding = TextRecipeBinding(
        "dinkster.text_ace15",
        "dinkster.ace_step_1_5",
        components,
        "dinkster_inference_torch.ace15_text:assemble_ace15_text_recipe",
        "dinkster_inference_torch.ace15_text:ACE15TextRuntime",
        "dinkster_inference_torch.ace15_text:compose_ace15_conditioning",
        planning.ACE15_TEXT_ROLES,
    )
    return ACE15TextRuntime(
        LoadedTextRecipe(
            binding,
            torch.nn.ModuleDict(
                dict(zip(planning.ACE15_TEXT_ROLES, (conditioner, lm), strict=True))
            ),
            (),
        )
    )


def test_templates_metadata_tokenizer_and_duration() -> None:
    prompts, options = ace15_prompts(
        "  (piano:2.0)  ",
        lyrics=" hi \n",
        language="en",
        duration="1.1 seconds",
        bpm="123",
        timesignature="4/4",
        keyscale="C major",
        caption_negative="  noise ",
        lyrics_negative="bye",
        bpm_negative="unspecified",
        duration_negative=3.5,
        max_tokens=30,
    )
    assert options.min_tokens == 10 and options.max_tokens == 30
    assert prompts == {
        "lm_prompt": (
            "<|im_start|>system\n# Instruction\nGenerate audio semantic tokens based on the given "
            "conditions:\n\n<|im_end|>\n<|im_start|>user\n# Caption\n(piano:2.0)\n\n"
            "# Lyric\nhi\n<|im_end|>\n<|im_start|>assistant\n<think>\nbpm: 123\n"
            "duration: 2\nkeyscale: C major\ntimesignature: 4\n</think>\n\n<|im_end|>\n"
        ),
        "lm_prompt_negative": (
            "<|im_start|>system\n# Instruction\nGenerate audio semantic tokens based on the given "
            "conditions:\n\n<|im_end|>\n<|im_start|>user\n# Caption\nnoise\n\n# Lyric\n"
            "bye\n<|im_end|>\n<|im_start|>assistant\n<think>\nduration: 3.5\n</think>\n\n"
            "<|im_end|>\n"
        ),
        "lyrics": "# Languages\nen\n\n# Lyric\n hi \n<|endoftext|><|endoftext|>",
        "qwen3_06b": (
            "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n"
            "# Caption\n(piano:2.0)\n\n# Metas\n- bpm: 123\n- timesignature: 4\n"
            "- keyscale: C major\n- duration: 2 seconds\n<|endoftext|>\n<|endoftext|>"
        ),
    }
    tokens = tokenize_ace15_prompt("piano", duration=0)
    assert tokens.lm_prompt[:3] == (151644, 8948, 198)
    assert tokens.lm_prompt[-2:] == (151645, 198)
    assert tokens.lyrics[-2:] == (PAD, PAD)
    assert tokens.qwen3_06b[-3:] == (PAD, 198, PAD)
    assert tokens.generation.min_tokens == 0
    assert tokenize_ace15_prompt("piano").generation.min_tokens == 600
    assert tokenize_ace15_prompt("(piano:2.0)", duration=0).qwen3_06b != tokens.qwen3_06b
    padded = tokenize_ace15_prompt("piano", min_length=200)
    assert len(padded.lyrics) == len(padded.qwen3_06b) == 200

    recorded: list[str] = []

    class InputSensitiveTokenizer:
        def encode(self, text: str) -> list[int]:
            recorded.append(text)
            return [len(text)]

    weighted_prompts, _ = ace15_prompts("(piano:2.0)", duration=0)
    weighted = tokenize_ace15_prompt(
        "(piano:2.0)", duration=0, tokenizer=cast(Any, InputSensitiveTokenizer())
    )
    assert recorded == list(weighted_prompts.values())
    assert weighted.qwen3_06b == (len(weighted_prompts["qwen3_06b"]),)


@pytest.mark.parametrize(
    "row,mask",
    [
        ((PAD, PAD, 1, 2), (0, 0, 1, 1)),
        ((1, PAD, 2, PAD), (1, 0, 0, 0)),
        ((PAD, PAD), (0, 0)),
    ],
)
def test_padding_mask(row: tuple[int, ...], mask: tuple[int, ...]) -> None:
    assert ace15_attention_mask(row) == mask


@pytest.mark.parametrize("fixed", [False, True])
@torch.inference_mode()
def test_incremental_mask_matches_full_forward(fixed: bool) -> None:
    model = tiny_model(True)
    ids = torch.tensor([[1, 2, 3], [PAD, PAD, 4]])
    mask = torch.tensor([[1, 1, 1], [0, 0, 1]])
    cache = model.allocate_causal_cache(2, 5, dtype=torch.float32)
    if fixed:
        cache = tuple(
            _QwenFixedKV(
                torch.empty(2, 5, 1, 4),
                torch.empty(2, 5, 1, 4),
                torch.empty(2, dtype=torch.int64),
                torch.zeros(2, dtype=torch.int32),
            )
            for _ in model.layers
        )
    actual, _ = model.forward_causal(ids, cache, attention_mask=mask)
    torch.testing.assert_close(actual, model(ids, mask), rtol=0, atol=0)
    next_ids = torch.tensor([[6], [7]])
    mask = torch.cat((mask, torch.ones((2, 1), dtype=torch.long)), dim=1)
    actual, _ = model.forward_causal(next_ids, cache, cache_position=3, attention_mask=mask)
    expected = model(torch.cat((ids, next_ids), dim=1), mask)[:, -1:]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="full cached span"):
        model.forward_causal(next_ids, cache, cache_position=3, attention_mask=mask[:, :-1])


@pytest.mark.parametrize(
    "temperature,top_k,top_p,min_p", [(0.0, 0, 1.0, 0.0), (0.85, 8, 0.9, 0.1), (1.0, 0, 0.2, 0.9)]
)
@pytest.mark.parametrize("cfg", [1.0, 2.0])
@torch.inference_mode()
def test_seeded_code_generation(
    temperature: float, top_k: int, top_p: float, min_p: float, cfg: float
) -> None:
    model = tiny_model(True)
    options = ACE15Generation(
        min_tokens=5,
        max_tokens=5,
        seed=123,
        cfg_scale=cfg,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
    )
    before = torch.random.get_rng_state()
    first = generate_audio_codes(model, (1, 2, 3), (4,), options)
    assert torch.equal(first, generate_audio_codes(model, (1, 2, 3), (4,), options))
    assert torch.equal(before, torch.random.get_rng_state())
    assert first.shape == (1, 5)
    assert torch.all((first >= 0) & (first < 64000))
    assert (
        not torch.equal(first, generate_audio_codes(model, (1, 2, 3), (4, 5, 6, 7), options))
        or cfg == 1.0
    )


@torch.inference_mode()
def test_seed_changes_unconstrained_sampling() -> None:
    model = tiny_model(True)
    options = ACE15Generation(min_tokens=5, max_tokens=5, top_p=None, top_k=None, min_p=None)
    first = generate_audio_codes(model, (1, 2, 3), (4,), options)
    assert not torch.equal(
        first, generate_audio_codes(model, (1, 2, 3), (4,), replace(options, seed=1))
    )


@pytest.mark.parametrize(
    "top_k,top_p,min_p,expected_support",
    [
        (3, None, None, 3),
        (None, 1e-9, None, 1),
        (None, None, 1.0, 1),
    ],
)
@torch.inference_mode()
def test_generation_filters_sampling_support(
    monkeypatch: pytest.MonkeyPatch,
    top_k: int | None,
    top_p: float | None,
    min_p: float | None,
    expected_support: int,
) -> None:
    probabilities: list[torch.Tensor] = []

    def choose_maximum(values: torch.Tensor, _count: int, **_kwargs: object) -> torch.Tensor:
        probabilities.append(values.clone())
        return values.argmax(dim=-1, keepdim=True)

    monkeypatch.setattr(torch, "multinomial", choose_maximum)
    options = ACE15Generation(
        min_tokens=1,
        max_tokens=1,
        cfg_scale=1,
        temperature=1,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
    )
    generate_audio_codes(tiny_model(True), (1, 2, 3), (), options)
    assert len(probabilities) == 1
    assert torch.count_nonzero(probabilities[0]).item() == expected_support


@torch.inference_mode()
def test_eos_strict_minimum_and_greedy_code_offset() -> None:
    model = tiny_model(True)
    for parameter in model.parameters():
        parameter.zero_()
    model.embed_tokens.weight.fill_(1)
    model.embed_tokens.weight[AUDIO_START].fill_(2)
    model.embed_tokens.weight[EOS].fill_(100)
    model.norm.weight.fill_(1)
    codes = generate_audio_codes(
        model, (1,), (), ACE15Generation(min_tokens=1, max_tokens=7, cfg_scale=1, temperature=0)
    )
    assert codes.tolist() == [[0, 0]]
    empty = generate_audio_codes(model, (1,), (), ACE15Generation(max_tokens=0, cfg_scale=1))
    assert empty.shape == (1, 0)


def test_assembly_consumes_role_sources_and_hardens_quantized_qwen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conditioner, lm = tiny_model(), tiny_model(True)
    quant = Fp8Linear(2, 2, compute_dtype=torch.float16)
    conditioner.add_module("quant_probe", quant)
    recipe = runtime(conditioner, lm).loaded.binding
    conditioner_source, lm_source = object(), object()
    conditioner_file, lm_file = object(), object()
    loaded: list[tuple[str, object, object]] = []

    def load(
        plan: ComponentPlan[Any],
        _builder: object,
        *,
        source: object,
        source_file: object,
        **_kwargs: object,
    ) -> torch.nn.Module:
        loaded.append((plan.component, source, source_file))
        return conditioner if plan.component == planning.ACE15_TEXT_ROLES[0] else lm

    monkeypatch.setattr(implementation, "_load_component", load)
    result = implementation.assemble_ace15_text_recipe(
        recipe,
        compute_dtype=torch.float16,
        sources=cast(Any, (conditioner_source, lm_source)),
        source_files=cast(Any, (conditioner_file, lm_file)),
    )
    assert loaded == [
        (planning.ACE15_TEXT_ROLES[0], conditioner_source, conditioner_file),
        (planning.ACE15_TEXT_ROLES[1], lm_source, lm_file),
    ]
    assert result.module[planning.ACE15_TEXT_ROLES[0]] is conditioner
    assert result.module[planning.ACE15_TEXT_ROLES[1]] is lm
    assert quant.compute_dtype == torch.float32
    assert quant.full_precision_matmul


@torch.inference_mode()
def test_carrier_final_conditioner_and_unnormalized_layer_zero() -> None:
    conditioner, lm = tiny_model(), tiny_model(True)
    encoder = runtime(conditioner, lm)
    with pytest.raises(ValueError, match="does not support textual inversion"):
        ACE15TextRuntime(encoder.loaded, embedding_lookups={"qwen3_06b_ace15": cast(Any, object())})
    tokens = ACE15Tokens(
        (1, 2),
        (3,),
        (4, 5, PAD, PAD),
        (6, 7, PAD),
        ACE15Generation(min_tokens=2, max_tokens=9, temperature=0),
    )
    captured: list[torch.Tensor] = []
    hook = conditioner.layers[0].register_forward_hook(
        lambda module, args, output: captured.append(output.clone())
    )
    try:
        carrier = encoder.encode_tokens(tokens)
    finally:
        hook.remove()
    payloads = {item.space: payload_binding_to_tensor(item) for item in carrier.bindings}
    assert torch.equal(payloads["conditioning-lyrics"], captured[1])
    assert not torch.equal(payloads["conditioning-lyrics"], conditioner.norm(captured[1]))
    assert torch.equal(
        payloads["conditioning-text"],
        conditioner(
            torch.tensor([tokens.qwen3_06b]), torch.tensor([ace15_attention_mask(tokens.qwen3_06b)])
        ),
    )
    assert payloads["audio-codes"].shape == (1, 2)
    assert list(dict(carrier.conditioning.records[0].channels)) == [ConditioningChannel.TEXT]
    assert dict(carrier.conditioning.records[0].extension_metadata) == {
        "dinkster.ace15/" + name: PayloadReference(
            next(item.reference_id for item in carrier.bindings if item.space == space)
        )
        for name, space in (
            ("conditioning_lyrics", "conditioning-lyrics"),
            ("audio_codes", "audio-codes"),
        )
    }
    assert conditioner.config.output_hidden_layer is None
    no_codes = encoder.encode_tokens(
        replace(tokens, generation=replace(tokens.generation, generate_audio_codes=False))
    )
    assert "dinkster.ace15/audio_codes" not in dict(
        no_codes.conditioning.records[0].extension_metadata
    )


@pytest.mark.parametrize("with_audio_codes", [False, True])
def test_materialize_ace15_conditioning_round_trip(with_audio_codes: bool) -> None:
    sequence = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    lyrics = torch.arange(16, dtype=torch.bfloat16).reshape(1, 2, 8)
    audio_codes = torch.tensor([[1, 5, 9]], dtype=torch.int64) if with_audio_codes else None
    carrier = compose_ace15_conditioning(sequence, lyrics, audio_codes)

    actual = materialize_ace15_conditioning(carrier, device="cpu")
    encoder = runtime(tiny_model(), tiny_model(True))
    via_runtime = encoder.materialize_conditioning(carrier, device="cpu")

    for materialized in (actual, via_runtime):
        assert torch.equal(materialized.sequence, sequence)
        assert torch.equal(materialized.lyrics, lyrics)
        if audio_codes is None:
            assert materialized.audio_codes is None
        else:
            assert materialized.audio_codes is not None
            assert torch.equal(materialized.audio_codes, audio_codes)


def ace_carrier_with_spaces(
    text_space: str = "conditioning-text",
    lyrics_space: str = "conditioning-lyrics",
    audio_space: str = "audio-codes",
) -> ConditioningCarrier:
    text = tensor_to_payload_binding("text", torch.ones((1, 3, 8)), space=text_space)
    lyrics = tensor_to_payload_binding("lyrics", torch.ones((1, 2, 8)), space=lyrics_space)
    audio = tensor_to_payload_binding(
        "audio", torch.ones((1, 4), dtype=torch.int64), space=audio_space
    )
    record = ConditioningRecord(
        channels=(
            (
                ConditioningChannel.TEXT,
                PayloadDescriptor(PayloadReference("text"), text.shape, text.dtype, text.space),
            ),
        ),
        extension_metadata=(
            ("dinkster.ace15/conditioning_lyrics", PayloadReference("lyrics")),
            ("dinkster.ace15/audio_codes", PayloadReference("audio")),
        ),
    )
    return make_conditioning_carrier(ConditioningSet((record,)), (text, lyrics, audio))


@pytest.mark.parametrize(
    "spaces,expected",
    [
        (("wrong-text", "conditioning-lyrics", "audio-codes"), "conditioning-text"),
        (("conditioning-text", "wrong-lyrics", "audio-codes"), "conditioning-lyrics"),
        (("conditioning-text", "conditioning-lyrics", "wrong-audio"), "audio-codes"),
    ],
)
def test_materialize_ace15_conditioning_refuses_each_wrong_payload_space(
    spaces: tuple[str, str, str], expected: str
) -> None:
    carrier = ace_carrier_with_spaces(*spaces)
    with pytest.raises(ValueError, match=expected):
        materialize_ace15_conditioning(carrier, device="cpu")


@pytest.mark.parametrize("code", ["duplicate-content-id", "unbound-reference"])
def test_materialize_ace15_conditioning_preserves_wire_error_codes(code: str) -> None:
    carrier = compose_ace15_conditioning(
        torch.ones((1, 3, 8)), torch.ones((1, 2, 8)), torch.ones((1, 4), dtype=torch.int64)
    )
    if code == "duplicate-content-id":
        bindings = tuple(
            sorted((*carrier.bindings, carrier.bindings[0]), key=lambda item: item.reference_id)
        )
    else:
        bindings = carrier.bindings[1:]
    with pytest.raises(ConditioningWireError) as caught:
        materialize_ace15_conditioning(replace(carrier, bindings=bindings), device="cpu")
    assert caught.value.code == code


def test_materialize_ace15_conditioning_refuses_noncanonical_shapes() -> None:
    carrier = compose_ace15_conditioning(
        torch.ones((1, 3, 8)), torch.ones((1, 2, 8)), torch.ones((1, 4), dtype=torch.int64)
    )
    record = carrier.conditioning.records[0]
    text_channel = record.channels[0]
    lyrics_binding = next(item for item in carrier.bindings if item.space == "conditioning-lyrics")
    without_lyrics = tuple(item for item in carrier.bindings if item is not lyrics_binding)
    metadata_without_lyrics = tuple(
        item
        for item in record.extension_metadata
        if item[0] != "dinkster.ace15/conditioning_lyrics"
    )
    mask_binding = tensor_to_payload_binding("mask", torch.ones((1, 1)), space="mask")
    scale_binding = tensor_to_payload_binding("scale", torch.ones((1,)), space="scale")
    scale_descriptor = PayloadDescriptor(
        PayloadReference(scale_binding.reference_id),
        scale_binding.shape,
        scale_binding.dtype,
        scale_binding.space,
    )
    wrong_lyrics = tensor_to_payload_binding(
        "wrong-lyrics", torch.ones((1, 2, 8)), space="wrong-lyrics"
    )

    malformed = (
        replace(carrier, conditioning=ConditioningSet((record, record))),
        replace(
            carrier,
            conditioning=ConditioningSet(
                (replace(record, area=AreaDescriptor(1, 1, 0, 0, AreaUnits.LATENT_CELLS)),)
            ),
        ),
        replace(
            carrier,
            conditioning=ConditioningSet(
                (
                    replace(
                        record,
                        mask=MaskDescriptor(PayloadReference(mask_binding.reference_id)),
                    ),
                )
            ),
            bindings=(*carrier.bindings, mask_binding),
        ),
        replace(
            carrier,
            conditioning=ConditioningSet(
                (replace(record, scale_vector=ConditionScaleVector(scale_descriptor)),)
            ),
            bindings=(*carrier.bindings, scale_binding),
        ),
        replace(
            carrier,
            conditioning=ConditioningSet(
                (
                    replace(
                        record,
                        token_layout=TokenLayoutDescriptor(
                            "dinkster.ace_step_1_5",
                            1,
                            ("text",),
                            (TokenSegmentDescriptor("prompt", "text", 0, 1),),
                        ),
                    ),
                )
            ),
        ),
        replace(
            carrier,
            conditioning=ConditioningSet((replace(record, schedule=PercentRange(0.25, 1.0)),)),
        ),
        replace(
            carrier,
            conditioning=ConditioningSet(
                (
                    replace(
                        record,
                        channels=(*record.channels, (ConditioningChannel.POOLED, text_channel[1])),
                    ),
                )
            ),
        ),
        replace(
            carrier,
            conditioning=ConditioningSet(
                (
                    replace(
                        record,
                        extension_metadata=(
                            *record.extension_metadata,
                            ("dinkster.ace15/unknown", text_channel[1].reference),
                        ),
                    ),
                )
            ),
        ),
        replace(
            carrier,
            conditioning=ConditioningSet(
                (
                    replace(
                        record,
                        extension_metadata=metadata_without_lyrics,
                    ),
                )
            ),
            bindings=without_lyrics,
        ),
        replace(
            carrier,
            conditioning=ConditioningSet(
                (
                    replace(
                        record,
                        extension_metadata=(
                            *metadata_without_lyrics,
                            ("dinkster.ace15/conditioning_lyrics", "not-a-payload"),
                        ),
                    ),
                )
            ),
            bindings=without_lyrics,
        ),
        replace(
            carrier,
            conditioning=ConditioningSet(
                (
                    replace(
                        record,
                        extension_metadata=(
                            *metadata_without_lyrics,
                            (
                                "dinkster.ace15/conditioning_lyrics",
                                PayloadReference(wrong_lyrics.reference_id),
                            ),
                        ),
                    ),
                )
            ),
            bindings=(*without_lyrics, wrong_lyrics),
        ),
        replace(carrier, bindings=carrier.bindings[:-1]),
        replace(carrier, bindings=(*carrier.bindings, carrier.bindings[0])),
        replace(
            carrier,
            conditioning=ConditioningSet(
                (
                    replace(
                        record,
                        channels=(
                            (
                                ConditioningChannel.TEXT,
                                replace(text_channel[1], shape=(1, 999, 8)),
                            ),
                        ),
                    ),
                )
            ),
        ),
    )
    for value in malformed:
        with pytest.raises(ValueError):
            materialize_ace15_conditioning(value, device="cpu")


class Resolver:
    def __init__(self, path: Path) -> None:
        self.path = path

    def resolve(self, digest: str) -> Path:
        return self.path


@pytest.mark.parametrize("reverse", [False, True])
@torch.inference_mode()
def test_real_handle_assembly_rebuild_and_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reverse: bool
) -> None:
    import dinkster_inference as inference

    models = (tiny_model(), tiny_model(True))
    # Only geometry admission is reduced; planning, quant splitting, loading,
    # identity, residency, runtime and overlay reconstruction are production code.
    monkeypatch.setattr(planning, "ACE15_QWEN3_06B_CONFIG", models[0].config)
    monkeypatch.setattr(planning, "ACE15_QWEN3_2B_CONFIG", models[1].config)
    import dinkster_inference.qwen_text as configs

    monkeypatch.setattr(
        configs, "_KNOWN_QWEN_TEXT_CONFIGS", tuple(model.config for model in models)
    )
    registry = ComponentRegistry()
    registry.register(
        planning.ace15_component_descriptor(replace(LUMINA2, id="dinkster.ace_step_1_5"))
    )
    recipes: Registry[TextRecipeDescriptor] = Registry()
    recipes.register(
        TextRecipeDescriptor("dinkster.text_ace15", planning.bind_ace15_text_recipe, ("ace",))
    )
    monkeypatch.setattr(component_catalog, "default_component_registry", lambda: registry)
    monkeypatch.setattr(text_recipes, "default_text_recipe_registry", lambda: recipes)
    monkeypatch.setenv("DINKSTER_AIMDO_ARM", "off")
    assets: list[AssetRef] = []
    for index, model in enumerate(models):
        path = tmp_path / f"text-{index}.safetensors"
        save_file({"model." + key: value for key, value in model.state_dict().items()}, path)
        assets.append(
            AssetRef(
                digest=digest_file(path),
                name=path.name,
                size=path.stat().st_size,
                resolver=Resolver(path),
            )
        )
    if reverse:
        assets.reverse()
    binding = planning.bind_ace15_text_recipe(
        tuple(
            registry.detect(
                load_safetensors_header(
                    asset.local_path(), asset_digest=asset.digest, asset_size=asset.size
                ),
                asset.local_path(),
            )
            for asset in assets
        )
    )
    arm: Any = native_arm
    recipe = binding.recipe(
        tuple(arm._weight_source_ref(inference, asset) for asset in assets), "float32"
    )
    handle = arm.build_text_recipe_handle(
        tuple(assets), "ace", recipe.runtime_identity, compute_dtype="float32", load_device="cpu"
    )
    try:
        assert handle.recipe == recipe
        with handle.stage():
            original = handle.runtime.encode_text(
                "piano", lyrics="sing", duration=1, generate_audio_codes=False
            )
        rebuilt = handle.rebuild()
        try:
            assert rebuilt.recipe == recipe
            with rebuilt.stage():
                actual = rebuilt.runtime.encode_text(
                    "piano", lyrics="sing", duration=1, generate_audio_codes=False
                )
            for expected, found in zip(original.bindings, actual.bindings, strict=True):
                assert torch.equal(
                    payload_binding_to_tensor(expected), payload_binding_to_tensor(found)
                )
        finally:
            rebuilt.terminal_release()
        patch_path = tmp_path / "patch.safetensors"
        delta = torch.full((8, 8), 0.125)
        save_file({"delta": delta}, patch_path)
        patch = AssetRef(
            digest=digest_file(patch_path),
            name=patch_path.name,
            size=patch_path.stat().st_size,
            resolver=Resolver(patch_path),
        )
        overlay = inference.PatchOverlay.from_decoded(
            source=arm._weight_source_ref(inference, patch),
            dialect="none",
            key_map="native.dinkster.test.v1",
            strength_model=0.0,
            strength_clip=1.0,
            patches=(
                inference.OverlayPatch(
                    planning.ACE15_TEXT_ROLES[0],
                    inference.PatchTarget("layers.0.self_attn.q_proj.weight"),
                    inference.DiffPatchRef("delta"),
                ),
            ),
        )
        patched = handle.clone((overlay,), source_resolvers={patch.digest: patch.resolver})
        try:
            assert patched.resource_identity != handle.resource_identity
            with patched.stage():
                model = cast(QwenTextModel, patched.module[planning.ACE15_TEXT_ROLES[0]])
                linear = cast(QwenBlock, model.layers[0]).self_attn.q_proj
                assert linear is not None
                actual = linear(torch.eye(8))
            expected = (models[0].state_dict()["layers.0.self_attn.q_proj.weight"] + delta).T
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            patched.terminal_release()
        with pytest.raises(RuntimeError, match="identity differs"):
            arm.build_text_recipe_handle(
                tuple(assets), "ace", "wrong", compute_dtype="float32", load_device="cpu"
            )
    finally:
        handle.terminal_release()
