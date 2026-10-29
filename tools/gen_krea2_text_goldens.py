"""Generate Krea 2 Qwen3-VL-4B text goldens from pinned ComfyUI.

This runs the reference Krea 2 tokenizer, template strip, and Llama
text tower at 947c2749. The full Qwen3-VL-4B architecture is
constructed on the meta device for its strict key/shape layout,
remapped to the standalone checkpoint's file-native naming
(``model.language_model.*`` / ``model.visual.*``). A tiny geometry
runs the same reference transformer math with deterministic
hash-filled weights and the twelve-tap capture policy, keeping the
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
# The reference resolves a device policy at import; everything here runs on CPU.
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
from clip_fill import fill_state_dict  # noqa: E402
from comfy import ops  # noqa: E402
from comfy.text_encoders import krea2, llama, qwen3vl  # noqa: E402

OUT = REPO / "tests" / "goldens" / "krea2_text_goldens.json"

PROMPTS = (
    "cat",
    "(cat:2.0)",
    r"\(cat:1.2\)",
    "a embedding:foo",
    "",
    "  cafe\u0301 \u4e2d  ",
    "don't stop...\nnext",
    "<|im_start|>literal",
    "<|im_start|>system\nx<|im_end|>\n<|im_start|>user\nhello there<|im_end|>\n"
    "<|im_start|>assistant\n",
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
    "architecture": "krea2_qwen3vl_4b",
    "vocab_size": 64,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 64,
    "rms_norm_eps": 1e-6,
    "rope_theta": 5_000_000.0,
    "rope_dims": [2, 1, 1],
    "interleaved_mrope": True,
    "qkv_bias": False,
    "qk_norm": True,
    "final_norm": True,
    "tap_layers": [1, 3, 5],
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


def file_native(key: str) -> str:
    if key.startswith("model."):
        return "model.language_model." + key.removeprefix("model.")
    if key.startswith("visual."):
        return "model.visual." + key.removeprefix("visual.")
    raise AssertionError(f"unexpected reference key {key}")


def full_layout() -> list[tuple[str, list[int]]]:
    model_class = qwen3vl._make_qwen3vl_model("qwen3vl_4b")
    model = model_class({}, torch.float32, "meta", ops.disable_weight_init)
    return sorted(
        (file_native(key), list(value.shape)) for key, value in model.state_dict().items()
    )


def reference_template_end(ids: list[int]) -> int:
    """The exact Krea2TEModel.encode_token_weights strip boundary."""
    template_end = -1
    count_im_start = 0
    for index, token in enumerate(ids):
        if token == 151644 and count_im_start < 2:
            template_end = index
            count_im_start += 1
    if len(ids) > template_end + 3:
        if ids[template_end + 1] == 872 and ids[template_end + 2] == 198:
            template_end += 3
    return template_end


def reference_attention_mask(ids: list[int]) -> list[int]:
    """The exact sd1_clip.py process_tokens mask with pad as eos."""
    attention_mask = []
    eos = False
    left_pad = False
    for index, token in enumerate(ids):
        if index == 0 and token == 151643:
            left_pad = True
        if eos or (left_pad and token == 151643):
            attention_mask.append(0)
        else:
            attention_mask.append(1)
            left_pad = False
        if not eos and token == 151643 and not left_pad:
            attention_mask[-1] = 0
            eos = True
    return attention_mask


def tokenizer_goldens() -> list[dict[str, object]]:
    tokenizer = krea2.Krea2Tokenizer()
    cases = []
    for text in PROMPTS:
        chunks = tokenizer.tokenize_with_weights(text, return_word_ids=True)["qwen3vl_4b"]
        if len(chunks) != 1:
            raise AssertionError(f"Krea 2 produced {len(chunks)} chunks")
        ids = [int(token) for token, _, _ in chunks[0]]
        weights = [float(weight) for _, weight, _ in chunks[0]]
        word_ids = [int(word_id) for _, _, word_id in chunks[0]]
        cases.append(
            {
                "text": text,
                "ids": ids,
                "weights": weights,
                "word_ids": word_ids,
                "attention_mask": reference_attention_mask(ids),
                "template_end": reference_template_end(ids),
            }
        )
    return cases


def tokenizer_corpus_golden() -> dict[str, object]:
    tokenizer = krea2.Krea2Tokenizer()
    outputs = []
    for text in CORPUS_PROMPTS:
        chunks = tokenizer.tokenize_with_weights(text, return_word_ids=True)["qwen3vl_4b"]
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
            if key not in {"architecture", "qk_norm", "rope_dims", "tap_layers"}
        },
        transformer_type="llama",
        rms_norm_add=False,
        mlp_activation="silu",
        rope_dims=TINY_CONFIG["rope_dims"],
        rope_scale=None,
        q_norm="gemma3",
        k_norm="gemma3",
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
    ids = torch.tensor([[3, 5, 7, 11, 13, 2, 17, 19]], dtype=torch.long)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        _, intermediate = model(
            ids,
            attention_mask=mask,
            intermediate_output=list(TINY_CONFIG["tap_layers"]),
            final_layer_norm_intermediate=False,
        )
    if intermediate.shape != (1, len(TINY_CONFIG["tap_layers"]), ids.shape[1], 16):
        raise AssertionError(f"unexpected tap stack shape {tuple(intermediate.shape)}")
    return {
        "config": TINY_CONFIG,
        "state_dict": entries,
        "ids": ids.tolist(),
        "attention_mask": mask.tolist(),
        "taps": enc(intermediate),
    }


def main() -> None:
    commit = git_head(COMFY_ROOT)
    if commit != REFERENCE_COMMIT:
        raise SystemExit(f"{COMFY_ROOT} is at {commit}; expected {REFERENCE_COMMIT}")
    module_file = Path(krea2.__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(f"Krea 2 reference imported from {module_file}")
    payload = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
        },
        "tap_layers": list(krea2.KREA2_TAP_LAYERS),
        "template": krea2.KREA2_TEMPLATE,
        "layout": full_layout(),
        "tokenizer": tokenizer_goldens(),
        "tokenizer_corpus": tokenizer_corpus_golden(),
        "model": tiny_model_golden(),
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=True) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
