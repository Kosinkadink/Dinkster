"""UMT5 checkpoint tokenizer contract and official-model goldens."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from dinkster_inference import (
    UMT5_SPIECE_SHA256,
    UMT5_XXL_WAN_PROFILE,
    load_umt5_spiece,
    pack_spans,
    tokenize_prompt,
)
from dinkster_inference_torch.umt5_tokenizer import (
    Umt5SentencePieceTokenizer,
    Umt5TokenizerError,
)

GOLDENS = json.loads(
    (Path(__file__).parent / "goldens" / "umt5_tokenizer_goldens.json").read_text()
)


class FakeProcessor:
    constructor_kwargs: dict[str, object] = {}
    encode_calls: list[tuple[str, type[int], bool, bool]] = []
    facts = (256000, 0, 1, 2, 3)

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


def test_checkpoint_tokenizer_uses_sentencepiece_without_inline_specials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dinkster_inference_torch import umt5_tokenizer

    FakeProcessor.encode_calls = []

    def fake_import(name: str) -> object:
        assert name == "sentencepiece"
        return SimpleNamespace(SentencePieceProcessor=FakeProcessor)

    monkeypatch.setattr(
        umt5_tokenizer.importlib,
        "import_module",
        fake_import,
    )
    tokenizer = Umt5SentencePieceTokenizer(b"model")

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
        (b"wrong", (255999, 0, 1, 2, 3), "256000 pieces"),
        (b"wrong", (256000, 1, 0, 2, 3), "pad=0"),
    ),
)
def test_checkpoint_tokenizer_rejects_invalid_payload_or_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
    model: bytes,
    facts: tuple[int, int, int, int, int],
    message: str,
) -> None:
    from dinkster_inference_torch import umt5_tokenizer

    monkeypatch.setattr(FakeProcessor, "facts", facts)

    def fake_import(name: str) -> object:
        assert name == "sentencepiece"
        return SimpleNamespace(SentencePieceProcessor=FakeProcessor)

    monkeypatch.setattr(
        umt5_tokenizer.importlib,
        "import_module",
        fake_import,
    )
    with pytest.raises(Umt5TokenizerError, match=message):
        Umt5SentencePieceTokenizer(model)


def _official_model() -> bytes:
    configured = os.environ.get("DINKSTER_UMT5_SPIECE_MODEL")
    if configured is None:
        pytest.skip("DINKSTER_UMT5_SPIECE_MODEL is not set")
    data = Path(configured).read_bytes()
    assert len(data) == GOLDENS["byte_size"]
    assert hashlib.sha256(data).hexdigest() == GOLDENS["sha256"]
    return data


def test_official_umt5_sentencepiece_goldens() -> None:
    tokenizer = Umt5SentencePieceTokenizer(_official_model())
    for case in GOLDENS["cases"]:
        assert list(tokenizer.encode(case["text"])) == case["tokens"]


def test_vendored_umt5_sentencepiece_matches_official_goldens() -> None:
    assert UMT5_SPIECE_SHA256 == GOLDENS["sha256"]
    data = load_umt5_spiece()
    assert len(data) == GOLDENS["byte_size"]
    assert hashlib.sha256(data).hexdigest() == GOLDENS["sha256"]
    tokenizer = Umt5SentencePieceTokenizer(data)
    for case in GOLDENS["cases"]:
        assert list(tokenizer.encode(case["text"])) == case["tokens"]


def test_official_umt5_prompt_weights_and_minimum_padding() -> None:
    tokenizer = Umt5SentencePieceTokenizer(_official_model())
    prompt = tokenize_prompt("(fox:1.5)", encode_word=tokenizer.encode)
    chunks = pack_spans(prompt.spans, UMT5_XXL_WAN_PROFILE)

    assert len(chunks) == 1
    assert len(chunks[0]) == 512
    encoded = list(tokenizer.encode("fox"))
    assert [token.unit for token in chunks[0][: len(encoded) + 1]] == [*encoded, 1]
    assert [token.weight for token in chunks[0][: len(encoded)]] == [1.5] * len(encoded)
    assert all(token.unit == 0 for token in chunks[0][len(encoded) + 1 :])
