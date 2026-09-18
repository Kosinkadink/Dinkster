"""Gemma checkpoint tokenizer contract and official-model goldens.

The executed-reference cases live in the root golden
(tests/goldens/ltx_gemma_text_goldens.json); replaying them needs the
official SentencePiece model extracted from the packaged LTX-2 text
encoder, supplied through DINKSTER_LTX_GEMMA_SPIECE_MODEL and verified
against the golden's recorded byte size and sha256.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from dinkster_inference import tokenize_ltx_gemma_prompt
from dinkster_inference_torch.gemma_tokenizer import (
    GemmaJsonTokenizer,
    GemmaSentencePieceTokenizer,
    GemmaTokenizerError,
)
from golden_files import load_platform_golden

GOLDENS = load_platform_golden(
    Path(__file__).parents[3] / "tests" / "goldens" / "ltx_gemma_text_goldens.json"
)


class FakeProcessor:
    constructor_kwargs: dict[str, object] = {}
    encode_calls: list[tuple[str, type[int], bool, bool]] = []
    facts = (262144, 0, 1, 2, 3)

    def __init__(self, **kwargs: object) -> None:
        type(self).constructor_kwargs = kwargs

    def get_piece_size(self) -> int:
        return self.facts[0]

    def pad_id(self) -> int:
        return self.facts[1]

    def eos_id(self) -> int:
        return self.facts[2]

    def bos_id(self) -> int:
        return self.facts[3]

    def unk_id(self) -> int:
        return self.facts[4]

    def encode(self, text: str, *, out_type: type[int], add_bos: bool, add_eos: bool) -> list[int]:
        type(self).encode_calls.append((text, out_type, add_bos, add_eos))
        return [17, 23]


def _patch_sentencepiece(monkeypatch: pytest.MonkeyPatch) -> None:
    from dinkster_inference_torch import gemma_tokenizer

    def fake_import(name: str) -> object:
        assert name == "sentencepiece"
        return SimpleNamespace(SentencePieceProcessor=FakeProcessor)

    monkeypatch.setattr(gemma_tokenizer.importlib, "import_module", fake_import)


def test_checkpoint_tokenizer_uses_sentencepiece_without_inline_specials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeProcessor.encode_calls = []
    _patch_sentencepiece(monkeypatch)
    tokenizer = GemmaSentencePieceTokenizer(b"model")

    assert FakeProcessor.constructor_kwargs == {
        "model_proto": b"model",
        "add_bos": False,
        "add_eos": False,
    }
    assert tokenizer.encode("prompt") == [17, 23]
    assert FakeProcessor.encode_calls == [("prompt", int, False, False)]


@pytest.mark.parametrize(
    "model,facts,message",
    (
        (b"", FakeProcessor.facts, "nonempty bytes"),
        (b"wrong", (262143, 0, 1, 2, 3), "262144 pieces"),
        (b"wrong", (262144, 1, 0, 2, 3), "pad=0"),
    ),
)
def test_checkpoint_tokenizer_rejects_invalid_payload_or_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
    model: bytes,
    facts: tuple[int, int, int, int, int],
    message: str,
) -> None:
    monkeypatch.setattr(FakeProcessor, "facts", facts)
    _patch_sentencepiece(monkeypatch)
    with pytest.raises(GemmaTokenizerError, match=message):
        GemmaSentencePieceTokenizer(model)


def _official_model() -> bytes:
    configured = os.environ.get("DINKSTER_LTX_GEMMA_SPIECE_MODEL")
    if configured is None:
        pytest.skip("DINKSTER_LTX_GEMMA_SPIECE_MODEL is not set")
    data = Path(configured).read_bytes()
    recorded = GOLDENS["real_checkpoint"]["spiece_model"]
    assert len(data) == recorded["byte_size"]
    assert hashlib.sha256(data).hexdigest() == recorded["sha256"]
    return data


def test_official_gemma_prompt_cases_match_executed_reference() -> None:
    tokenizer = GemmaSentencePieceTokenizer(_official_model())
    for case in GOLDENS["tokenizer"]:
        got = tokenize_ltx_gemma_prompt(
            str(case["text"]),
            encode=tokenizer.encode,
            apply_template=not case["skip_template"],
        )
        assert list(got.ids) == case["ids"]
        assert list(got.attention_mask) == case["attention_mask"]
        assert list(got.word_ids) == case["word_ids"]


def test_official_gemma_corpus_matches_executed_reference() -> None:
    tokenizer = GemmaSentencePieceTokenizer(_official_model())
    outputs = [
        [
            list(
                tokenize_ltx_gemma_prompt(
                    str(text), encode=tokenizer.encode, apply_template=apply_template
                ).ids
            )
            for apply_template in (False, True)
        ]
        for text in GOLDENS["tokenizer_corpus"]["prompts"]
    ]
    canonical = json.dumps(outputs, separators=(",", ":")).encode("ascii")
    assert hashlib.sha256(canonical).hexdigest() == GOLDENS["tokenizer_corpus"]["token_ids_sha256"]


class FakeJsonProcessor:
    facts: tuple[int, int | None, int | None, int | None, int | None] = (
        262144,
        0,
        1,
        2,
        3,
    )
    source = ""
    encode_calls: list[tuple[str, bool]] = []

    @classmethod
    def from_str(cls, source: str) -> FakeJsonProcessor:
        cls.source = source
        return cls()

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        assert with_added_tokens
        return self.facts[0]

    def token_to_id(self, token: str) -> int | None:
        ids: dict[str, int | None] = dict(
            zip(("<pad>", "<eos>", "<bos>", "<unk>"), self.facts[1:], strict=True)
        )
        return ids[token]

    def encode(self, text: str, add_special_tokens: bool = True) -> SimpleNamespace:
        self.encode_calls.append((text, add_special_tokens))
        return SimpleNamespace(ids=[9307])


def _patch_json_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    from dinkster_inference_torch import gemma_tokenizer

    def fake_import(name: str) -> object:
        assert name == "tokenizers"
        return SimpleNamespace(Tokenizer=FakeJsonProcessor)

    monkeypatch.setattr(
        gemma_tokenizer.importlib,
        "import_module",
        fake_import,
    )


def test_json_tokenizer_uses_embedded_json_without_special_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeJsonProcessor.facts = (262144, 0, 1, 2, 3)
    FakeJsonProcessor.encode_calls = []
    _patch_json_tokenizer(monkeypatch)

    tokenizer = GemmaJsonTokenizer(b'{"version":"1.0"}')

    assert FakeJsonProcessor.source == '{"version":"1.0"}'
    assert list(tokenizer.encode("cat")) == [9307]
    assert FakeJsonProcessor.encode_calls == [("cat", False)]
    with pytest.raises(TypeError, match="must be a string"):
        tokenizer.encode(1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("payload", "facts", "message"),
    (
        (b"", (262144, 0, 1, 2, 3), "nonempty bytes"),
        (b"\xff", (262144, 0, 1, 2, 3), "invalid Gemma tokenizer JSON"),
        (b"{}", (262143, 0, 1, 2, 3), "262144 tokens"),
        (b"{}", (262144, 1, 0, 2, 3), "pad=0"),
    ),
)
def test_json_tokenizer_rejects_invalid_payload_or_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    facts: tuple[int, int | None, int | None, int | None, int | None],
    message: str,
) -> None:
    FakeJsonProcessor.facts = facts
    _patch_json_tokenizer(monkeypatch)
    with pytest.raises(GemmaTokenizerError, match=message):
        GemmaJsonTokenizer(payload)


def test_official_gemma4_json_tokenizer_cases() -> None:
    configured = os.environ.get("DINKSTER_LTX_GEMMA4_TOKENIZER_JSON")
    if configured is None:
        pytest.skip("DINKSTER_LTX_GEMMA4_TOKENIZER_JSON is not set")
    payload = Path(configured).read_bytes()
    assert len(payload) == 32169626
    assert hashlib.sha256(payload).hexdigest() == (
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
    )
    tokenizer = GemmaJsonTokenizer(payload)
    assert list(tokenizer.encode("cat")) == [9307]
    assert list(tokenizer.encode("a photo of a cat")) == [236746, 4429, 529, 496, 5866]
