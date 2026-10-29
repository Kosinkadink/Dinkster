"""NewBie Jina math and Gemma embedding injection."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from dinkster_inference import ConditioningChannel
from dinkster_inference.assembly import ComponentPlan
from dinkster_inference.gemma_text import GemmaTextConfig
from dinkster_inference.jina_clip_text import (
    JINA_CLIP_V2_CONFIG,
    JinaClipTextConfig,
    jina_clip_text_layout,
)
from dinkster_inference.text_recipes import TextRecipeBinding, TextRecipeComponent
from dinkster_inference_torch import newbie_text as implementation
from dinkster_inference_torch.gemma_text import GemmaTextModel
from dinkster_inference_torch.jina_clip_text import JinaClipTextModel
from dinkster_inference_torch.newbie_text import (
    LoadedNewBieText,
    NewBieSentencePiece,
    NewBieTextRuntime,
)
from dinkster_inference_torch.payloads import payload_binding_to_tensor
from dinkster_inference_torch.quant_linear import Fp8Linear


def tiny_jina() -> JinaClipTextConfig:
    return replace(
        JINA_CLIP_V2_CONFIG,
        vocab_size=11,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
    )


def tiny_gemma() -> GemmaTextConfig:
    return GemmaTextConfig(
        architecture="gemma3_newbie_4b",
        vocab_size=12,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rms_norm_eps=1e-6,
        rope_theta_global=10000.0,
        rope_theta_local=10000.0,
        rope_scale_global=1.0,
        rope_scale_local=1.0,
        sliding_window=8,
        sliding_pattern=(True, False),
        prompt_template="{}",
        min_tokens=1,
        pad_token_id=0,
        bos_token_id=2,
        end_of_turn_token_id=3,
        image_soft_token_id=4,
    )


def initialize(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("bias"):
                parameter.zero_()
            elif "norm" in name:
                parameter.fill_(1.0)
            else:
                parameter.uniform_(-0.05, 0.05)


class FakeSentencePiece:
    facts = (262144, 2, 1, 0)

    def __init__(self, **kwargs: object) -> None:
        assert kwargs == {"model_proto": b"model", "add_bos": False, "add_eos": False}

    def get_piece_size(self) -> int:
        return self.facts[0]

    def bos_id(self) -> int:
        return self.facts[1]

    def eos_id(self) -> int:
        return self.facts[2]

    def pad_id(self) -> int:
        return self.facts[3]

    def encode(self, text: str, *, out_type: type[int], add_bos: bool, add_eos: bool) -> list[int]:
        assert out_type is int and not add_bos and not add_eos
        return [len(text)]


def binding() -> TextRecipeBinding:
    gemma = ComponentPlan("gemma", Path("gemma"), tiny_gemma(), {}, {}, {})
    jina = ComponentPlan("jina", Path("jina"), tiny_jina(), {}, {}, {})
    return TextRecipeBinding(
        "dinkster.text_newbie",
        "dinkster.newbie",
        (
            TextRecipeComponent(0, "gemma", gemma, None),
            TextRecipeComponent(1, "jina", jina, None),
        ),
        "loader",
        "runtime",
        "dinkster_inference_torch.newbie_text:compose_newbie_conditioning",
        ("gemma", "jina"),
    )


def test_full_jina_state_layout_matches_detector() -> None:
    with torch.device("meta"):
        model = JinaClipTextModel(JINA_CLIP_V2_CONFIG)
    assert {
        key: tuple(value.shape) for key, value in model.state_dict().items()
    } == jina_clip_text_layout()


def test_sentencepiece_contract_and_added_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str) -> object:
        assert name == "sentencepiece"
        return SimpleNamespace(SentencePieceProcessor=FakeSentencePiece)

    monkeypatch.setattr(
        implementation.importlib,
        "import_module",
        fake_import,
    )
    tokenizer = NewBieSentencePiece(
        b"model",
        size=262144,
        special_ids=(2, 1, 0),
        added_tokens={"<image_soft_token>": 262144},
    )
    assert tokenizer.encode("a<image_soft_token>bc") == [1, 262144, 2]
    monkeypatch.setattr(FakeSentencePiece, "facts", (262144, 2, 1, 7))
    with pytest.raises(ValueError, match="wrong vocabulary or special ids"):
        NewBieSentencePiece(b"model", size=262144, special_ids=(2, 1, 0))


def test_jina_masked_pooling_uses_only_attended_tokens() -> None:
    torch.manual_seed(4)
    model = JinaClipTextModel(tiny_jina())
    initialize(model)
    ids = torch.tensor([[0, 3, 2, 1]])
    mask = torch.tensor([[1, 1, 1, 0]])
    sequence, pooled = model(ids, mask)
    torch.testing.assert_close(pooled, sequence[:, :3].mean(dim=1), rtol=0, atol=0)
    unmasked = model(ids)[1]
    assert not torch.equal(pooled, unmasked)
    with pytest.raises(ValueError, match="matching"):
        model(ids, mask[:, :2])


def test_gemma_preembedded_input_is_consumed_without_default_drift() -> None:
    torch.manual_seed(8)
    model = GemmaTextModel(tiny_gemma())
    initialize(model)
    ids = torch.tensor([[2, 5, 6]])
    mask = torch.ones_like(ids)
    embeds = model.embed_tokens(ids)
    torch.testing.assert_close(model(ids, mask), model(ids, mask, embeds=embeds), rtol=0, atol=0)
    changed = embeds.clone()
    changed[:, 1] += 1
    assert not torch.equal(model(ids, mask), model(ids, mask, embeds=changed))
    with pytest.raises(ValueError, match="preembedded"):
        model(ids, mask, embeds=embeds[:, :2])


def test_assembly_reads_both_tokenizers_and_hardens_quantized_gemma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemma = GemmaTextModel(tiny_gemma())
    quant = Fp8Linear(2, 2, compute_dtype=torch.float16)
    gemma.add_module("quant_probe", quant)
    jina = JinaClipTextModel(tiny_jina())
    loaded: list[tuple[str, object, object]] = []
    gemma_file = object()
    jina_file = object()

    class Source:
        def __init__(self, role: str, source_file: object) -> None:
            self.role = role
            self.source_file = source_file

        def read_uint8_configuration_from_file(
            self, source_file: object, key: str, *, limit: int
        ) -> bytes:
            assert source_file is self.source_file
            assert key == "spiece_model"
            assert limit == implementation.TOKENIZER_BYTE_CAP
            return f"{self.role}-tokenizer".encode()

    gemma_source = Source("gemma", gemma_file)
    jina_source = Source("jina", jina_file)

    def load(
        plan: ComponentPlan[Any],
        _builder: object,
        *,
        source: object,
        source_file: object,
        **_kwargs: object,
    ) -> torch.nn.Module:
        loaded.append((plan.component, source, source_file))
        return gemma if plan.component == "gemma" else jina

    monkeypatch.setattr(implementation, "_load_component", load)
    result = implementation.assemble_newbie_text(
        binding(),
        compute_dtype=torch.float16,
        sources=cast(Any, (gemma_source, jina_source)),
        source_files=cast(Any, (gemma_file, jina_file)),
    )
    assert loaded == [
        ("gemma", gemma_source, gemma_file),
        ("jina", jina_source, jina_file),
    ]
    assert result.tokenizers == (b"gemma-tokenizer", b"jina-tokenizer")
    assert quant.compute_dtype == torch.float32
    assert quant.full_precision_matmul


def test_runtime_composes_gemma_sequence_and_jina_masked_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemma = GemmaTextModel(tiny_gemma())
    jina = JinaClipTextModel(tiny_jina())
    initialize(gemma)
    initialize(jina)
    encoded_text: list[str] = []

    class Tokenizer:
        def __init__(self, _model: bytes, **_facts: object) -> None:
            pass

        def encode(self, text: str) -> list[int]:
            encoded_text.append(text)
            return [5, 6] if text == "(prompt:2.0)" else [7, 8]

    monkeypatch.setattr(implementation, "NewBieSentencePiece", Tokenizer)
    runtime = NewBieTextRuntime(
        LoadedNewBieText(
            binding(),
            torch.nn.ModuleDict({"gemma": gemma, "jina": jina}),
            (),
            (b"gemma", b"jina"),
        )
    )
    with torch.no_grad():
        carrier = runtime.encode_text("(prompt:2.0)")
        gemma_ids = torch.tensor([[2, 5, 6]])
        expected_text = gemma(gemma_ids, torch.ones_like(gemma_ids))[:, -2].float()
        jina_ids = torch.tensor([[0, 7, 8, 2]])
        expected_pooled = jina(jina_ids, torch.ones_like(jina_ids))[1].float()
    assert encoded_text == ["(prompt:2.0)", "prompt"]
    record = carrier.conditioning.records[0]
    bindings = {item.reference_id: payload_binding_to_tensor(item) for item in carrier.bindings}
    channels = dict(record.channels)
    torch.testing.assert_close(
        bindings[channels[ConditioningChannel.TEXT].reference.id], expected_text
    )
    torch.testing.assert_close(
        bindings[channels[ConditioningChannel.POOLED].reference.id], expected_pooled
    )


def test_runtime_consumes_role_isolated_textual_inversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gemma = GemmaTextModel(tiny_gemma())
    jina = JinaClipTextModel(tiny_jina())
    initialize(gemma)
    initialize(jina)

    class Tokenizer:
        def __init__(self, _model: bytes, **_facts: object) -> None:
            pass

        def encode(self, text: str) -> list[int]:
            return [5] if text else []

    def encoded(runtime: NewBieTextRuntime) -> tuple[torch.Tensor, torch.Tensor]:
        carrier = runtime.encode_text("embedding:style cat")
        record = carrier.conditioning.records[0]
        payloads = {item.reference_id: payload_binding_to_tensor(item) for item in carrier.bindings}
        channels = dict(record.channels)
        return (
            payloads[channels[ConditioningChannel.TEXT].reference.id],
            payloads[channels[ConditioningChannel.POOLED].reference.id],
        )

    monkeypatch.setattr(implementation, "NewBieSentencePiece", Tokenizer)
    monkeypatch.setattr(implementation, "GEMMA3_NEWBIE_4B_CONFIG", tiny_gemma())
    loaded = LoadedNewBieText(
        binding(),
        torch.nn.ModuleDict({"gemma": gemma, "jina": jina}),
        (),
        (b"gemma", b"jina"),
    )
    vectors = {
        "gemma": torch.ones(2, tiny_gemma().hidden_size),
        "jina": torch.ones(2, tiny_jina().hidden_size),
    }
    runtime = NewBieTextRuntime(
        loaded,
        embedding_lookups={role: lambda name, role=role: vectors[role] for role in vectors},
    )
    zero_runtime = NewBieTextRuntime(
        loaded,
        embedding_lookups={
            role: lambda name, role=role: torch.zeros_like(vectors[role]) for role in vectors
        },
    )
    with torch.no_grad():
        embedded = encoded(runtime)
        zeroed = encoded(zero_runtime)
    assert embedded[0].shape == zeroed[0].shape
    assert embedded[1].shape == zeroed[1].shape
    assert not torch.equal(embedded[0], zeroed[0])
    assert not torch.equal(embedded[1], zeroed[1])
