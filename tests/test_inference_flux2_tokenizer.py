"""Flux2 prompt tokenization pinned to the executed ComfyUI reference.

Golden data comes from tools/gen_flux2_text_goldens.py: the Mistral3
tekken route and the Klein Qwen3 route at b78cec87, under their fixed
templates and padding/mask policies. The converter fingerprints pin
the tekken conversion itself (id assignment and merge derivation), and
the replay cases pin the end-to-end prompt policies.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from dinkster_inference import (
    TEKKEN_BOS,
    TEKKEN_FLUX2_SHA256,
    Flux2DevPromptTokens,
    Flux2KleinPromptTokens,
    TekkenBpe,
    load_flux2_tekken_bpe,
    tokenize_flux2_dev_prompt,
    tokenize_flux2_klein_prompt,
)

GOLDENS = json.loads((Path(__file__).parent / "goldens" / "flux2_text_goldens.json").read_text())


def canonical_sha256(payload: object) -> str:
    canonical = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def test_goldens_pin_the_vendored_blob() -> None:
    assert GOLDENS["reference"]["tekken_sha256"] == TEKKEN_FLUX2_SHA256
    # The generator asserts KleinTokenizer8B produces identical ids, so
    # one policy covers both Klein towers.
    assert GOLDENS["klein_8b_identical"] is True


def test_converter_vocabulary_fingerprint_matches_reference_conversion() -> None:
    bpe = load_flux2_tekken_bpe()
    vocab: dict[str, int] = {
        **bpe._special_ids,  # pyright: ignore[reportPrivateUsage]
        **bpe._vocab,  # pyright: ignore[reportPrivateUsage]
    }
    converter = GOLDENS["converter"]
    assert len(vocab) == converter["vocab_size"]
    assert canonical_sha256(sorted(vocab.items())) == converter["vocab_sha256"]


def test_converter_merge_fingerprint_matches_reference_conversion() -> None:
    ranks = load_flux2_tekken_bpe()._ranks  # pyright: ignore[reportPrivateUsage]
    merges = [list(pair) for pair, _ in sorted(ranks.items(), key=lambda item: item[1])]
    converter = GOLDENS["converter"]
    assert len(merges) == converter["merges_count"]
    assert canonical_sha256(merges) == converter["merges_sha256"]


@pytest.mark.parametrize("case", GOLDENS["dev_tokenizer"], ids=lambda case: case["text"])
def test_dev_prompt_matches_executed_reference(case: dict[str, Any]) -> None:
    row = tokenize_flux2_dev_prompt(case["text"], tokenizer=load_flux2_tekken_bpe())
    assert list(row.ids) == case["ids"]
    assert list(row.attention_mask) == case["attention_mask"]
    assert all(weight == 1.0 for weight in case["weights"])


@pytest.mark.parametrize("case", GOLDENS["klein_tokenizer"], ids=lambda case: case["text"])
def test_klein_prompt_matches_executed_reference(case: dict[str, Any]) -> None:
    row = tokenize_flux2_klein_prompt(case["text"])
    assert list(row.ids) == case["ids"]
    assert list(row.attention_mask) == case["attention_mask"]
    assert all(weight == 1.0 for weight in case["weights"])


def test_dev_corpus_matches_executed_reference() -> None:
    bpe = load_flux2_tekken_bpe()
    outputs = [
        list(tokenize_flux2_dev_prompt(text, tokenizer=bpe).ids)
        for text in GOLDENS["dev_corpus"]["prompts"]
    ]
    assert canonical_sha256(outputs) == GOLDENS["dev_corpus"]["token_ids_sha256"]


def test_klein_corpus_matches_executed_reference() -> None:
    outputs = [
        list(tokenize_flux2_klein_prompt(text).ids) for text in GOLDENS["klein_corpus"]["prompts"]
    ]
    assert canonical_sha256(outputs) == GOLDENS["klein_corpus"]["token_ids_sha256"]


def test_vendored_tokenizer_is_deterministic() -> None:
    bpe = load_flux2_tekken_bpe()
    text = "determinism 12345 \U0001f600 [INST]"
    assert bpe.encode(text) == bpe.encode(text)


def tiny_blob(
    *,
    special_tokens: list[dict[str, object]] | None = None,
    vocab_tokens: list[bytes] | None = None,
    num_special: int = 2,
    vocab_size: int = 6,
) -> bytes:
    if special_tokens is None:
        special_tokens = [{"rank": 0, "token_str": "<pad>"}, {"rank": 1, "token_str": "<s>"}]
    if vocab_tokens is None:
        vocab_tokens = [b"a", b"b", b"ab", b"c"]
    vocab = [
        {"rank": rank, "token_bytes": base64.b64encode(token).decode("ascii")}
        for rank, token in enumerate(vocab_tokens)
    ]
    return json.dumps(
        {
            "config": {
                "default_num_special_tokens": num_special,
                "default_vocab_size": vocab_size,
            },
            "special_tokens": special_tokens,
            "vocab": vocab,
        }
    ).encode("ascii")


def test_tiny_conversion_maps_ids_and_extracts_specials() -> None:
    bpe = TekkenBpe.from_tekken_bytes(tiny_blob())
    # Regular ids are rank + num_special; whole-piece hit bypasses BPE.
    assert bpe.encode("ab") == [4]
    # Non-vocabulary word runs the derived merges: (a, b) then + c.
    assert bpe.encode("abc") == [4, 5]
    # Specials are extracted from raw text before pretokenization.
    assert bpe.encode("<pad><s>ab") == [0, 1, 4]


def test_tiny_conversion_drops_ranks_beyond_the_kept_vocabulary() -> None:
    bpe = TekkenBpe.from_tekken_bytes(tiny_blob(vocab_tokens=[b"a", b"b", b"ab", b"c", b"d"]))
    with pytest.raises(ValueError, match="no BPE token"):
        bpe.encode("d")


def test_out_of_order_special_ranks_refuse() -> None:
    specials = [{"rank": 1, "token_str": "<s>"}, {"rank": 0, "token_str": "<pad>"}]
    with pytest.raises(ValueError, match="special_tokens must be ordered"):
        TekkenBpe.from_tekken_bytes(tiny_blob(special_tokens=specials))


def test_special_without_token_str_refuses() -> None:
    specials = [
        {"rank": 0, "token_bytes": base64.b64encode(b"<pad>").decode("ascii")},
        {"rank": 1, "token_str": "<s>"},
    ]
    with pytest.raises(ValueError, match="without token_str"):
        TekkenBpe.from_tekken_bytes(tiny_blob(special_tokens=specials))


def test_wrong_special_count_refuses() -> None:
    with pytest.raises(ValueError, match="declares 3 special tokens"):
        TekkenBpe.from_tekken_bytes(tiny_blob(num_special=3, vocab_size=7))


def test_out_of_order_vocab_ranks_refuse() -> None:
    blob = json.loads(tiny_blob())
    blob["vocab"][0]["rank"], blob["vocab"][1]["rank"] = 1, 0
    with pytest.raises(ValueError, match="vocab must be ordered"):
        TekkenBpe.from_tekken_bytes(json.dumps(blob).encode("ascii"))


def test_short_vocab_refuses() -> None:
    with pytest.raises(ValueError, match="kept tokens"):
        TekkenBpe.from_tekken_bytes(tiny_blob(vocab_size=7))


def test_dev_rows_reject_broken_invariants() -> None:
    with pytest.raises(ValueError, match="non-empty and equal"):
        Flux2DevPromptTokens((), ())
    with pytest.raises(ValueError, match="non-empty and equal"):
        Flux2DevPromptTokens((TEKKEN_BOS,), (1, 1))
    with pytest.raises(ValueError, match="start with <s>"):
        Flux2DevPromptTokens((5,), (1,))
    with pytest.raises(ValueError, match="fully attended"):
        Flux2DevPromptTokens((TEKKEN_BOS, 5), (1, 0))


def test_klein_rows_reject_broken_invariants() -> None:
    with pytest.raises(ValueError, match="equal length"):
        Flux2KleinPromptTokens((151643,) * 512, (0,) * 511)
    with pytest.raises(ValueError, match="at least 512"):
        Flux2KleinPromptTokens((151643,) * 511, (0,) * 511)
