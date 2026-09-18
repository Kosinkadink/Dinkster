"""Generate Flux2 text-encoder tokenizer goldens from pinned ComfyUI.

Runs the REFERENCE Flux2 prompt tokenizers at the pinned commit: the
Mistral3 tekken route (ComfyUI's own byte-level BPE built by
``from_tekken_json`` over the tekken_model blob) and the Klein Qwen3
route, both under their fixed templates and padding/mask policies.
dinkster_inference.tekken_bpe and the Flux2 Klein policy in qwen_bpe are
pinned against these outputs; the oracle is the executed reference,
never a re-derivation. Reduced-geometry towers also run the reference
transformer with the Flux2 multi-layer capture policy so the Dinkster
stacked forward replays against executed reference math.

The tekken blob is read from the vendored copy in
packages/dinkster-inference/src/dinkster_inference/data (the exact bytes of
the tekken_model tensor in the official mistral_3_small_flux2_bf16
checkpoint), verified against the recorded sha256.

Run from the Dinkster root with a sibling ``ComfyUI`` checkout detached at
the pinned commit, under a torch interpreter. ``PYTHONPATH`` must
include comfy-aimdo when the reference interpreter does not already
provide it.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = (REPO.parent / "ComfyUI").resolve()
#: The audited reference baseline. Generation REFUSES on any other
#: commit so the recorded provenance is a guarantee, not a claim.
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"

# The torch test helpers provide the deterministic hash-fill shared by
# every executed-reference tower golden.
sys.path.insert(0, str(REPO / "packages" / "dinkster-inference-torch" / "tests"))
# In front of any ambient PYTHONPATH: the reference must come from the
# pinned sibling checkout, not an installed or stray comfy package.
sys.path.insert(0, str(COMFY_ROOT))

# comfy.model_management probes CUDA at import; tokenization runs on
# CPU either way. The reference only reads argv when args parsing is
# explicitly enabled.
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402
from clip_fill import fill_state_dict  # noqa: E402
from comfy import ops  # noqa: E402
from comfy.text_encoders import flux, llama  # noqa: E402

TEKKEN_GZ = (
    REPO
    / "packages"
    / "dinkster-inference"
    / "src"
    / "dinkster_inference"
    / "data"
    / "flux2_tekken.json.gz"
)
TEKKEN_SHA256 = "6e2501687ccd0e1f30f36319eaf2b46958b897811e246cd8eb5d385b9e3de7d1"

OUT = REPO / "tests" / "goldens" / "flux2_text_goldens.json"

DEV_PAD = 0
KLEIN_PAD = 151643

#: One shared prompt list: dev treats the Qwen specials as plain text
#: and Klein treats the tekken specials as plain text, so every case
#: exercises both the matching and the non-matching special extractor.
PROMPTS = (
    "cat",
    "Hello, world!",
    "(cat:2.0)",
    r"\(cat:1.2\)",
    "a embedding:foo",
    "",
    "  cafe\u0301 \u4e2d  ",
    "don't stop...\nnext",
    "[INST]literal[/INST]",
    "[SYSTEM_PROMPT]masked tail",
    "<s> and </s> literals",
    "<|im_start|>literal",
    "<|endoftext|> masked tail",
    "emoji \U0001f600 and symbols +/-=",
    "1234567890 42nd 3.14159",
    "123456 12 1 007",
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
    "[INST]",
    "[SYSTEM_PROMPT]",
    "1234567890",
    "42nd",
    "123456",
    "007",
)
CORPUS_SEPARATORS = ("", " ", "  ", "\n", "\r\n", "\t", "\u3000")
CORPUS_PROMPTS = tuple(
    CORPUS_ATOMS[index % len(CORPUS_ATOMS)]
    + CORPUS_SEPARATORS[index % len(CORPUS_SEPARATORS)]
    + CORPUS_ATOMS[(index * 7 + 3) % len(CORPUS_ATOMS)]
    for index in range(256)
)


#: Reduced geometries that run the exact reference transformer math on
#: CPU with deterministic hash-filled weights. Each case pins one Flux2
#: capture policy: the full Mistral3 tower (interior captures under a
#: live final norm), the layer-pruned tower (a capture at the layer
#: count itself with no final norm - the reference's post-loop append),
#: and the Klein Qwen3 tower (gemma3 q/k norms).
TINY_GEOMETRY = {
    "vocab_size": 64,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "max_position_embeddings": 32,
    "qkv_bias": False,
    "min_tokens": 1,
    "zero_masked": False,
    "output_hidden_layers": [0, 1, 2],
    "layer_norm_hidden_state": False,
}

TINY_MODEL_CONFIGS: dict[str, dict[str, object]] = {
    "mistral3_full": {
        **TINY_GEOMETRY,
        "architecture": "mistral3_24b",
        "num_hidden_layers": 4,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1e9,
        "qk_norm": False,
        "prompt_template": "[INST]{}[/INST]",
        "pad_token_id": 11,
        "final_norm": True,
    },
    "mistral3_pruned": {
        **TINY_GEOMETRY,
        "architecture": "mistral3_24b_pruned",
        "num_hidden_layers": 3,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1e9,
        "qk_norm": False,
        "prompt_template": "[INST]{}[/INST]",
        "pad_token_id": 11,
        "final_norm": False,
    },
    "klein": {
        **TINY_GEOMETRY,
        "architecture": "klein_qwen3_4b",
        "num_hidden_layers": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1e6,
        "qk_norm": True,
        "prompt_template": (
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ),
        "pad_token_id": 63,
        "final_norm": True,
    },
}


def git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def tekken_bytes() -> bytes:
    raw = gzip.decompress(TEKKEN_GZ.read_bytes())
    digest = hashlib.sha256(raw).hexdigest()
    if digest != TEKKEN_SHA256:
        raise SystemExit(f"{TEKKEN_GZ} decompresses to sha256 {digest}; expected {TEKKEN_SHA256}")
    return raw


def one_chunk(tokenizer: object, text: str, key: str) -> list[tuple[int, float, int]]:
    chunks = tokenizer.tokenize_with_weights(text, return_word_ids=True)[key]  # type: ignore[attr-defined]
    if len(chunks) != 1:
        raise AssertionError(f"{key} produced {len(chunks)} chunks for {text!r}")
    return chunks[0]


def mask_for(ids: list[int], pad: int) -> list[int]:
    # Mirrors sd1_clip.SDClipModel.process_tokens: attended until the
    # first model-side pad id, masked from it onward. The dev pad id 0
    # is never produced, so dev masks are all ones.
    first_pad = ids.index(pad) if pad in ids else len(ids)
    return [1] * first_pad + [0] * (len(ids) - first_pad)


def tokenizer_cases(tokenizer: object, key: str, pad: int) -> list[dict[str, object]]:
    cases = []
    for text in PROMPTS:
        chunk = one_chunk(tokenizer, text, key)
        ids = [int(token) for token, _, _ in chunk]
        cases.append(
            {
                "text": text,
                "ids": ids,
                "weights": [float(weight) for _, weight, _ in chunk],
                "word_ids": [int(word_id) for _, _, word_id in chunk],
                "attention_mask": mask_for(ids, pad),
            }
        )
    return cases


def corpus_golden(tokenizer: object, key: str) -> dict[str, object]:
    outputs = []
    for text in CORPUS_PROMPTS:
        outputs.append([int(token) for token, _, _ in one_chunk(tokenizer, text, key)])
    canonical = json.dumps(outputs, separators=(",", ":")).encode("ascii")
    return {
        "prompts": list(CORPUS_PROMPTS),
        "token_ids_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def converter_fingerprints(blob: bytes) -> dict[str, object]:
    """Pin the reference conversion itself, not just its outputs.

    The sha256 of the converted tokenizer's full vocabulary mapping and
    ordered merge list fixes the id assignment (specials at their tekken
    ranks, blob rank + the special offset) and the tiktoken-style merge
    derivation that dinkster_inference.tekken_bpe reimplements. The
    reference tokenizer's own BPE encoder has no whole-piece vocabulary
    bypass; end-to-end behavior is pinned by the corpus goldens.
    """
    converted = flux.load_mistral_tokenizer(blob)["tokenizer_object"]
    vocab = {str(token): int(index) for token, index in converted.get_vocab().items()}
    ranks = converted._merges  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
    merges = [list(pair) for pair, _ in sorted(ranks.items(), key=lambda item: item[1])]
    vocab_canonical = json.dumps(sorted(vocab.items()), separators=(",", ":"), ensure_ascii=True)
    merges_canonical = json.dumps(merges, separators=(",", ":"), ensure_ascii=True)
    return {
        "vocab_size": len(vocab),
        "vocab_sha256": hashlib.sha256(vocab_canonical.encode("ascii")).hexdigest(),
        "merges_count": len(merges),
        "merges_sha256": hashlib.sha256(merges_canonical.encode("ascii")).hexdigest(),
    }


def enc(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "data": tensor.float().flatten().tolist(),
    }


def tiny_model_golden(config: dict[str, object]) -> dict[str, object]:
    qk = "gemma3" if config["qk_norm"] else None
    hidden = int(config["hidden_size"])  # type: ignore[arg-type]
    heads = int(config["num_attention_heads"])  # type: ignore[arg-type]
    reference_config = SimpleNamespace(
        vocab_size=config["vocab_size"],
        hidden_size=hidden,
        intermediate_size=config["intermediate_size"],
        num_hidden_layers=config["num_hidden_layers"],
        num_attention_heads=heads,
        num_key_value_heads=config["num_key_value_heads"],
        max_position_embeddings=config["max_position_embeddings"],
        rms_norm_eps=config["rms_norm_eps"],
        rope_theta=config["rope_theta"],
        transformer_type="llama",
        head_dim=hidden // heads,
        rms_norm_add=False,
        mlp_activation="silu",
        qkv_bias=config["qkv_bias"],
        rope_dims=None,
        rope_scale=None,
        q_norm=qk,
        k_norm=qk,
        final_norm=config["final_norm"],
        lm_head=False,
    )
    model = llama.Llama2_(
        reference_config,
        device="cpu",
        dtype=torch.float32,
        ops=ops.disable_weight_init,
    )
    entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    model.load_state_dict(fill_state_dict(entries), strict=True)
    # The reference appends a capture BEFORE running layer i, so the
    # Dinkster after-layer indices map to reference taps shifted by one;
    # the pruned tower's tap at the layer count itself lands on the
    # reference's post-loop append (no final norm there).
    captures_config = config["output_hidden_layers"]
    assert isinstance(captures_config, list)
    taps = [index + 1 for index in captures_config]
    ids = torch.tensor([[3, 5, 7, 11, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.long)
    with torch.no_grad():
        captures = model(
            ids,
            attention_mask=mask,
            intermediate_output=taps,
            final_layer_norm_intermediate=False,
        )[1]
    expected = (1, len(taps), ids.shape[1], hidden)
    if tuple(captures.shape) != expected:
        raise AssertionError(f"reference captures shape {tuple(captures.shape)}; want {expected}")
    return {
        "config": config,
        "reference_intermediate_output": taps,
        "state_dict": entries,
        "ids": ids.tolist(),
        "attention_mask": mask.tolist(),
        "captures": enc(captures),
    }


def main() -> None:
    commit = git_head(COMFY_ROOT)
    if commit != REFERENCE_COMMIT:
        raise SystemExit(f"{COMFY_ROOT} is at {commit}; expected {REFERENCE_COMMIT}")
    module_file = Path(flux.__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(f"Flux reference imported from {module_file}")

    blob = tekken_bytes()
    dev = flux.Flux2Tokenizer(tokenizer_data={"tekken_model": blob})
    klein = flux.KleinTokenizer()
    klein_8b = flux.KleinTokenizer8B()

    dev_cases = tokenizer_cases(dev, "mistral3_24b", DEV_PAD)
    for case in dev_cases:
        ids = case["ids"]
        assert isinstance(ids, list)
        if DEV_PAD in ids:
            raise AssertionError(f"dev produced the pad id for {case['text']!r}")

    klein_cases = tokenizer_cases(klein, "qwen3_4b", KLEIN_PAD)
    for case, text in zip(klein_cases, PROMPTS, strict=True):
        ids_8b = [int(token) for token, _, _ in one_chunk(klein_8b, text, "qwen3_8b")]
        if ids_8b != case["ids"]:
            raise AssertionError(f"KleinTokenizer8B diverged from KleinTokenizer on {text!r}")

    payload = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
            "tekken_sha256": TEKKEN_SHA256,
        },
        "converter": converter_fingerprints(blob),
        "dev_tokenizer": dev_cases,
        "dev_corpus": corpus_golden(dev, "mistral3_24b"),
        "klein_tokenizer": klein_cases,
        "klein_corpus": corpus_golden(klein, "qwen3_4b"),
        "klein_8b_identical": True,
        "models": {name: tiny_model_golden(config) for name, config in TINY_MODEL_CONFIGS.items()},
    }
    OUT.write_text(json.dumps(payload, indent=1, ensure_ascii=True) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
