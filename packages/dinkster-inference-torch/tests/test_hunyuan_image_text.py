"""Hunyuan Image prompt, capture, and conditioning policy."""

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference import Conditioning, PayloadReference
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.prompt_tokens import TokenizerProfile
from dinkster_inference.qwen_bpe import load_qwen_bpe
from dinkster_inference.qwen_image_text import QWEN_IMAGE_TEXT_CONFIG
from dinkster_inference.t5_text import BYT5_SMALL_GLYPH_CONFIG
from dinkster_inference.text_recipes import (
    TextEncodingProfile,
    TextRecipeBinding,
    TextRecipeComponent,
)
from dinkster_inference_torch import hunyuan_image_text as implementation
from dinkster_inference_torch.hunyuan_image_text import (
    BYT5_CONDITIONING_METADATA,
    ByT5Tokenizer,
    HunyuanImageTextRuntime,
    HunyuanQwenEncoder,
    HunyuanQwenEncoding,
    compose_hunyuan_image_conditioning,
    extract_byt5_prompt,
)
from dinkster_inference_torch.operations import INITLESS
from dinkster_inference_torch.payloads import payload_binding_to_tensor
from dinkster_inference_torch.quant_linear import Fp8Linear
from dinkster_inference_torch.qwen_image_text import (
    QwenImageLanguageModel,
    QwenImageTextModel,
    QwenImageVisionTransformer,
)
from dinkster_inference_torch.t5_text import T5TextModel
from dinkster_inference_torch.text_recipes import LoadedTextRecipe


def reduced_qwen() -> QwenImageTextModel:
    language = QwenImageLanguageModel.reduced(
        vocab_size=QWEN_IMAGE_TEXT_CONFIG.vocab_size,
        hidden_size=12,
        intermediate_size=24,
        num_layers=4,
        num_heads=2,
        num_kv_heads=1,
        rope_dims=(1, 1, 1),
        operations=INITLESS,
    )
    vision = QwenImageVisionTransformer.reduced(
        hidden_size=12,
        output_size=12,
        intermediate_size=24,
        num_heads=2,
        num_layers=1,
        patch=(1, 2, 2),
        spatial_merge_size=1,
        window_size=2,
        full_attention_blocks=(0,),
        operations=INITLESS,
    )
    model = QwenImageTextModel(language=language, visual=vision)
    generator = torch.Generator().manual_seed(1261)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
            if "norm.weight" in name:
                parameter.add_(1)
    return model


def reduced_byt5() -> T5TextModel:
    config = replace(
        BYT5_SMALL_GLYPH_CONFIG,
        d_model=12,
        d_ff=24,
        d_kv=2,
        num_heads=2,
        num_layers=2,
    )
    model = T5TextModel(config)
    generator = torch.Generator().manual_seed(1262)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.1)
    return model


def binding() -> TextRecipeBinding:
    qwen = ComponentPlan("qwen25_vl", Path("qwen"), QWEN_IMAGE_TEXT_CONFIG, {}, {}, {})
    byt5 = ComponentPlan("byt5_small", Path("byt5"), BYT5_SMALL_GLYPH_CONFIG, {}, {}, {})
    profile = TextEncodingProfile(
        TokenizerProfile(99999999, None, 1, 0, False, min_length=1),
        attention_masked=True,
        zero_out_masked=True,
    )
    return TextRecipeBinding(
        "dinkster.text_hunyuan_image",
        "dinkster.hunyuan_image",
        (
            TextRecipeComponent(0, "qwen25_vl", qwen, None),
            TextRecipeComponent(1, "byt5_small", byt5, profile),
        ),
        "loader",
        "runtime",
        "dinkster_inference_torch.hunyuan_image_text:compose_hunyuan_image_conditioning",
        ("qwen25_vl", "byt5_small"),
    )


def test_quote_extraction_preserves_upstream_class_order() -> None:
    text = 'first \u2018second\u2019 then "third" and \u201cfourth\u201d'
    assert extract_byt5_prompt(text) == ('Text "third". Text "second". Text "fourth". ')
    assert extract_byt5_prompt("no glyph quote") is None


def test_byt5_tokenizer_uses_utf8_bytes_and_added_tokens() -> None:
    tokenizer = ByT5Tokenizer()
    assert tokenizer.encode("A") == [68]
    assert tokenizer.encode("\u4e2d") == [231, 187, 176]
    assert tokenizer.encode("<extra_id_10>") == [269]
    assert tokenizer.encode("a</s>b") == [100, 1, 101]


def test_qwen_template_crop_and_minus_three_unnormalized_capture() -> None:
    model = reduced_qwen()
    encoder = HunyuanQwenEncoder(model)
    tokens = encoder.tokenize("cat")
    ids = [token.unit for token in tokens]
    assert ids[-5:] == [151644, 872, 198, 4616, 151645]
    captures: list[torch.Tensor] = []

    def capture(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captures.append(output.clone())

    hook = model.model.layers[-3].register_forward_hook(capture)
    with torch.no_grad():
        encoded = encoder.encode("cat")
    hook.remove()
    torch.testing.assert_close(encoded.embeddings, captures[0][:, -2:], rtol=0, atol=0)
    assert model.model.norm is not None
    assert not torch.equal(encoded.embeddings, model.model.norm(encoded.embeddings))
    assert encoded.attention_mask.tolist() == [[1, 1]]
    weighted_syntax = encoder.tokenize("(cat:1.5)")
    literal_ids = load_qwen_bpe().encode("(cat:1.5)")
    assert [token.unit for token in weighted_syntax[-len(literal_ids) - 1 : -1]] == literal_ids
    assert all(token.weight == 1.0 for token in weighted_syntax)
    preformatted = encoder.tokenize("<|im_start|>user\ncat<|im_end|>")
    assert preformatted[0].unit == 151644


def test_qwen_long_llama_header_uses_fixed_upstream_crop() -> None:
    model = reduced_qwen()
    encoder = HunyuanQwenEncoder(model)
    text = "<|start_header_id|>" + "one two three four five six seven eight " * 4
    tokens = encoder.tokenize(text)
    assert tokens[0].unit == 27
    assert len(tokens) > 36
    captures: list[torch.Tensor] = []

    def capture(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captures.append(output.clone())

    hook = model.model.layers[-3].register_forward_hook(capture)
    with torch.no_grad():
        encoded = encoder.encode(text)
    hook.remove()
    torch.testing.assert_close(encoded.embeddings, captures[0][:, 36:], rtol=0, atol=0)
    assert encoded.attention_mask.shape == (1, len(tokens) - 36)


def test_qwen_textual_inversion_rows_are_consumed() -> None:
    model = reduced_qwen()
    vectors = torch.ones(2, 12)
    encoder = HunyuanQwenEncoder(model, lambda name: vectors if name == "style" else None)
    zero_encoder = HunyuanQwenEncoder(
        model, lambda name: torch.zeros_like(vectors) if name == "style" else None
    )
    with torch.no_grad():
        embedded = encoder.encode("embedding:style cat")
        zeroed = zero_encoder.encode("embedding:style cat")
    assert embedded.embeddings.shape == zeroed.embeddings.shape
    assert not torch.equal(embedded.embeddings, zeroed.embeddings)


def test_optional_byt5_conditioning_is_an_extension_reference() -> None:
    qwen = HunyuanQwenEncoding(torch.randn(1, 2, 8), torch.ones(1, 2, dtype=torch.long))
    without = compose_hunyuan_image_conditioning(qwen, None)
    assert not without.conditioning.records[0].extension_metadata
    glyph = Conditioning(torch.randn(1, 5, 1472), None)
    with_glyph = compose_hunyuan_image_conditioning(qwen, glyph)
    metadata = dict(with_glyph.conditioning.records[0].extension_metadata)
    assert set(metadata) == {BYT5_CONDITIONING_METADATA}
    reference = metadata[BYT5_CONDITIONING_METADATA]
    assert isinstance(reference, PayloadReference)
    bindings = {item.reference_id: payload_binding_to_tensor(item) for item in with_glyph.bindings}
    torch.testing.assert_close(bindings[reference.id], glyph.embeddings, rtol=0, atol=0)


def test_runtime_emits_byt5_only_for_captured_quotes() -> None:
    loaded = LoadedTextRecipe(
        binding(),
        torch.nn.ModuleDict({"qwen25_vl": reduced_qwen(), "byt5_small": reduced_byt5()}),
        (),
    )
    runtime = HunyuanImageTextRuntime(loaded)
    with torch.no_grad():
        plain = runtime.encode_text("plain prompt")
        quoted = runtime.encode_text('paint "glyph"')
    assert not plain.conditioning.records[0].extension_metadata
    assert dict(quoted.conditioning.records[0].extension_metadata).keys() == {
        BYT5_CONDITIONING_METADATA
    }


def test_assembly_consumes_both_roles_and_hardens_quantized_qwen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    qwen = reduced_qwen()
    quant = Fp8Linear(2, 2, compute_dtype=torch.float16)
    qwen.add_module("quant_probe", quant)
    byt5 = reduced_byt5()
    loaded_configs: list[object] = []

    def load(plan: ComponentPlan[Any], _builder: object, **_kwargs: object) -> torch.nn.Module:
        loaded_configs.append(plan.config)
        return qwen if plan.component == "qwen25_vl" else byt5

    monkeypatch.setattr(implementation, "_load_component", load)
    result = implementation.assemble_hunyuan_image_text(
        binding(),
        compute_dtype=torch.float16,
        sources=cast(Any, (object(), object())),
        source_files=cast(Any, (object(), object())),
    )
    assert loaded_configs == [QWEN_IMAGE_TEXT_CONFIG, BYT5_SMALL_GLYPH_CONFIG]
    assert result.module["qwen25_vl"] is qwen
    assert result.module["byt5_small"] is byt5
    assert quant.compute_dtype == torch.float32
    assert quant.full_precision_matmul


def test_byt5_model_geometry_matches_exact_small_config() -> None:
    model = T5TextModel(replace(BYT5_SMALL_GLYPH_CONFIG, d_model=12, d_ff=24, num_layers=2))
    assert model.config.num_heads == 6
    assert model.config.d_kv == 64
