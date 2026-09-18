"""Generate Anima Qwen3-0.6B text goldens from pinned ComfyUI.

This runs the reference Anima dual tokenizer and Qwen3-0.6B text tower
at 82f839f5. Both token streams are captured: the Qwen row that feeds
the text tower (weights forced to 1.0 by the reference) and the T5 row
whose ids and emphasis weights ride to the diffusion model's LLM
adapter. The Qwen attention mask is produced by executing the real
``SDClipModel.process_tokens`` against a stub transformer, so the
masked-pad policy in the golden is reference code, not a re-port. The
full architecture is constructed on the meta device for its strict
key/shape layout. A tiny geometry runs the same reference transformer
math with deterministic hash-filled weights, keeping the golden small
enough to replay on CPU.

Run from the Dinkster root with a sibling ``ComfyUI`` checkout detached at
the pinned commit. The interpreter needs torch and transformers (the
reference tokenizers are Qwen2Tokenizer and T5TokenizerFast).
transformers must be < 5: from 5.x, ``from_pretrained`` rebuilds the
vendored T5 pre_tokenizer from tokenizer_config.json into
``Sequence[WhitespaceSplit, Metaspace(prepend_scheme="always")]``,
which prefixes text adjacent to inline special tokens. Dinkster and the
Flux T5 goldens (tests/goldens/t5_tokenizer_goldens.json) pin the
tokenizer.json's own ``Metaspace(prepend_scheme="first")`` semantics,
so generation refuses when the loaded pipeline does not match the
vendored file.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = (REPO.parent / "ComfyUI").resolve()
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"

sys.path.insert(0, str(REPO / "packages" / "dinkster-inference-torch" / "tests"))
sys.path.insert(0, str(COMFY_ROOT))

# comfy.model_management probes CUDA at import; tokenization runs on
# CPU either way. The reference only reads argv when args parsing is
# explicitly enabled.
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import tokenizers  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from clip_fill import fill_state_dict  # noqa: E402
from comfy import ops, sd1_clip  # noqa: E402
from comfy.text_encoders import anima, llama  # noqa: E402

OUT = REPO / "tests" / "goldens" / "anima_text_goldens.json"

#: The vendored t5_tokenizer/tokenizer.json pre_tokenizer. transformers
#: 5.x ``from_pretrained`` rebuilds it into ``Sequence[WhitespaceSplit,
#: Metaspace(always)]``, which tokenizes text adjacent to inline
#: special tokens differently from the pinned Flux T5 goldens.
VENDORED_T5_PRE_TOKENIZER = {
    "type": "Metaspace",
    "replacement": "\u2581",
    "prepend_scheme": "first",
    "split": True,
}


def require_vendored_t5_pipeline(tokenizer: anima.AnimaTokenizer) -> None:
    backend = json.loads(tokenizer.t5xxl.tokenizer.backend_tokenizer.to_str())
    if backend["pre_tokenizer"] != VENDORED_T5_PRE_TOKENIZER:
        raise SystemExit(
            "transformers rebuilt the T5 pre_tokenizer as "
            f"{backend['pre_tokenizer']!r}; the goldens pin the vendored "
            "Metaspace(prepend_scheme='first') pipeline - use transformers < 5 "
            f"(found {transformers.__version__})"
        )


QWEN_PAD = 151643

PROMPTS = (
    "cat",
    "(cat:1.2)",
    "(cat:2.0)",
    "((cat))",
    "(a (b:2.0) c:0.5)",
    "(neg:-1.0)",
    "(unbalanced",
    r"\(cat:1.2\)",
    "a embedding:foo",
    "",
    "  cafe\u0301 \u4e2d  ",
    "don't stop...\nnext",
    "<|im_start|>literal",
    "<|endoftext|> masked tail",
    "tail then <|endoftext|> more text",
    "</s> t5 special",
    "emoji \U0001f600 and symbols +/-=",
    "1234567890 42nd 3.14159",
    "'s 't 're 've 'm 'll 'd",
    "\u3000ideographic space\u3000wide",
)

CORPUS_ATOMS = (
    "a",
    "Z",
    "cafe\u0301",
    "caf\u00e9",
    "\u4e2d\u6587",
    "\u0661\u0662",
    "\u2163",
    "\u017f",
    "'S",
    "'re",
    "...",
    "+/-=",
    "\U0001f600",
    "\U0001f469\u200d\U0001f4bb",
    r"\(x\)",
    "embedding:item",
    "(pair:1.5)",
    "</s>",
    "<|im_start|>",
    "<|endoftext|>",
)
CORPUS_SEPARATORS = ("", " ", "  ", "\n", "\r\n", "\t", "\u3000")
CORPUS_PROMPTS = tuple(
    CORPUS_ATOMS[index % len(CORPUS_ATOMS)]
    + CORPUS_SEPARATORS[index % len(CORPUS_SEPARATORS)]
    + CORPUS_ATOMS[(index * 7 + 3) % len(CORPUS_ATOMS)]
    for index in range(256)
)

TINY_CONFIG = {
    "architecture": "anima_qwen3_06b",
    "vocab_size": 64,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 32,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000.0,
    "qkv_bias": False,
    "qk_norm": True,
    "prompt_template": "{}",
    "min_tokens": 1,
    "pad_token_id": 0,
    "zero_masked": False,
    "attention_head_dim": 8,
    "layer_norm_hidden_state": False,
}


def git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def enc(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "data": tensor.float().flatten().tolist(),
    }


def full_layout() -> list[tuple[str, list[int]]]:
    model = llama.Qwen3_06B({}, torch.float32, "meta", ops.disable_weight_init)
    return sorted(
        (key.removeprefix("model."), list(value.shape)) for key, value in model.state_dict().items()
    )


class _EmbeddingStub:
    def __call__(self, ids: torch.Tensor, out_dtype: object = None) -> torch.Tensor:
        return torch.zeros((*ids.shape, 4), dtype=torch.float32)


class _TransformerStub:
    def get_input_embeddings(self) -> _EmbeddingStub:
        return _EmbeddingStub()


class _MaskProbe:
    """Carries exactly the state process_tokens reads, so the golden
    masks come from executing the reference masking code itself."""

    special_tokens = {"pad": QWEN_PAD}
    transformer = _TransformerStub()


def reference_mask(ids: list[int]) -> list[int]:
    masks = sd1_clip.SDClipModel.process_tokens(_MaskProbe(), [list(ids)], "cpu")[1]
    return [int(value) for value in masks[0]]


def _single_chunk(chunks: list[list[tuple]], stream: str) -> list[tuple]:
    if len(chunks) != 1:
        raise AssertionError(f"Anima {stream} produced {len(chunks)} chunks")
    return chunks[0]


def tokenizer_goldens() -> list[dict[str, object]]:
    tokenizer = anima.AnimaTokenizer()
    cases = []
    for text in PROMPTS:
        streams = tokenizer.tokenize_with_weights(text, return_word_ids=True)
        qwen = _single_chunk(streams["qwen3_06b"], "qwen3_06b")
        t5 = _single_chunk(streams["t5xxl"], "t5xxl")
        qwen_ids = [int(token) for token, _, _ in qwen]
        cases.append(
            {
                "text": text,
                "qwen_ids": qwen_ids,
                "qwen_weights": [float(weight) for _, weight, _ in qwen],
                "qwen_word_ids": [int(word_id) for _, _, word_id in qwen],
                "qwen_attention_mask": reference_mask(qwen_ids),
                "t5xxl_ids": [int(token) for token, _, _ in t5],
                "t5xxl_weights": [float(weight) for _, weight, _ in t5],
                "t5xxl_word_ids": [int(word_id) for _, _, word_id in t5],
            }
        )
    return cases


def tokenizer_corpus_golden() -> dict[str, object]:
    tokenizer = anima.AnimaTokenizer()
    outputs = []
    for text in CORPUS_PROMPTS:
        streams = tokenizer.tokenize_with_weights(text, return_word_ids=True)
        outputs.append(
            {
                "qwen_ids": [
                    [int(token) for token, _, _ in chunk] for chunk in streams["qwen3_06b"]
                ],
                "t5xxl_ids": [[int(token) for token, _, _ in chunk] for chunk in streams["t5xxl"]],
                "t5xxl_weights": [
                    [float(weight) for _, weight, _ in chunk] for chunk in streams["t5xxl"]
                ],
            }
        )
    canonical = json.dumps(outputs, separators=(",", ":")).encode("ascii")
    return {
        "prompts": list(CORPUS_PROMPTS),
        "streams_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def tiny_model_golden() -> dict[str, object]:
    config = SimpleNamespace(
        **{
            key: value
            for key, value in TINY_CONFIG.items()
            if key
            not in {
                "architecture",
                "prompt_template",
                "min_tokens",
                "pad_token_id",
                "zero_masked",
                "qk_norm",
                "attention_head_dim",
                "layer_norm_hidden_state",
            }
        },
        transformer_type="llama",
        head_dim=TINY_CONFIG["attention_head_dim"],
        rms_norm_add=False,
        mlp_activation="silu",
        rope_dims=None,
        rope_scale=None,
        q_norm="gemma3",
        k_norm="gemma3",
        final_norm=True,
        lm_head=False,
    )
    model = llama.Llama2_(
        config,
        device="cpu",
        dtype=torch.float32,
        ops=ops.disable_weight_init,
    )
    entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    model.load_state_dict(fill_state_dict(entries), strict=True)
    ids = torch.tensor([[3, 5, 7, 11, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
    with torch.no_grad():
        output = model(ids, attention_mask=mask)[0]
    return {
        "config": TINY_CONFIG,
        "state_dict": entries,
        "ids": ids.tolist(),
        "attention_mask": mask.tolist(),
        "output": enc(output),
    }


def main() -> None:
    commit = git_head(COMFY_ROOT)
    if commit != REFERENCE_COMMIT:
        raise SystemExit(f"{COMFY_ROOT} is at {commit}; expected {REFERENCE_COMMIT}")
    module_file = Path(anima.__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(f"Anima reference imported from {module_file}")
    require_vendored_t5_pipeline(anima.AnimaTokenizer())
    payload = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
        },
        "layout": full_layout(),
        "tokenizer": tokenizer_goldens(),
        "tokenizer_corpus": tokenizer_corpus_golden(),
        "model": tiny_model_golden(),
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=True) + "\n", newline="\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
