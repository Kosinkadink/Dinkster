"""Generate Ovis Qwen3-2B text goldens from pinned ComfyUI.

This runs the reference Ovis tokenizer and Llama text tower at
947c2749. The full architecture is constructed on the meta device for
its strict key/shape layout. A tiny geometry runs the same reference
transformer math with deterministic hash-filled weights, keeping the
golden small enough to replay on CPU.

Run from the Dinkster root with a sibling ``ComfyUI`` checkout detached at
the pinned commit. ``PYTHONPATH`` must include comfy-aimdo when the
reference interpreter does not already provide it.
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

import torch  # noqa: E402
from clip_fill import fill_state_dict  # noqa: E402
from comfy import ops  # noqa: E402
from comfy.text_encoders import llama, ovis  # noqa: E402

OUT = REPO / "tests" / "goldens" / "ovis_text_goldens.json"

PROMPTS = (
    "cat",
    "(cat:2.0)",
    r"\(cat:1.2\)",
    "a embedding:foo",
    "",
    "  cafe\u0301 \u4e2d  ",
    "don't stop...\nnext",
    "<|im_start|>literal",
    "<|endoftext|> masked tail",
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
    "architecture": "ovis_qwen3_2b",
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
    "slice_marker_id": 1,
    "slice_marker_suffix_id": 2,
    "zero_masked": True,
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
    model = llama.Ovis25_2B({}, torch.float32, "meta", ops.disable_weight_init)
    return sorted(
        (key.removeprefix("model."), list(value.shape)) for key, value in model.state_dict().items()
    )


def tokenizer_goldens() -> list[dict[str, object]]:
    tokenizer = ovis.OvisTokenizer()
    cases = []
    for text in PROMPTS:
        chunks = tokenizer.tokenize_with_weights(text, return_word_ids=True)["qwen3_2b"]
        if len(chunks) != 1:
            raise AssertionError(f"Ovis produced {len(chunks)} chunks")
        ids = [int(token) for token, _, _ in chunks[0]]
        weights = [float(weight) for _, weight, _ in chunks[0]]
        word_ids = [int(word_id) for _, _, word_id in chunks[0]]
        first_pad = ids.index(151643) if 151643 in ids else len(ids)
        cases.append(
            {
                "text": text,
                "ids": ids,
                "weights": weights,
                "word_ids": word_ids,
                "attention_mask": [1] * first_pad + [0] * (len(ids) - first_pad),
            }
        )
    return cases


def tokenizer_corpus_golden() -> dict[str, object]:
    tokenizer = ovis.OvisTokenizer()
    outputs = []
    for text in CORPUS_PROMPTS:
        chunks = tokenizer.tokenize_with_weights(text, return_word_ids=True)["qwen3_2b"]
        outputs.append([[int(token) for token, _, _ in chunk] for chunk in chunks])
    canonical = json.dumps(outputs, separators=(",", ":")).encode("ascii")
    return {
        "prompts": list(CORPUS_PROMPTS),
        "token_ids_sha256": hashlib.sha256(canonical).hexdigest(),
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
                "slice_marker_id",
                "slice_marker_suffix_id",
                "zero_masked",
                "qk_norm",
            }
        },
        transformer_type="llama",
        head_dim=4,
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
    module_file = Path(ovis.__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(f"Ovis reference imported from {module_file}")
    payload = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
        },
        "layout": full_layout(),
        "tokenizer": tokenizer_goldens(),
        "tokenizer_corpus": tokenizer_corpus_golden(),
        "model": tiny_model_golden(),
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=True) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
