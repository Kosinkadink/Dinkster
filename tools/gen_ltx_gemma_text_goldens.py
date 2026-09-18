"""Generate LTX-2 Gemma 3 and Gemma 4 12B text goldens from pinned ComfyUI.

This runs the reference LTXAV Gemma tokenizer, attention-mask packer,
Llama gemma3 text tower, and both LTX text projections at b78cec87.
The full 12B architecture is constructed on the meta device for its
strict key/shape layout and cross-checked against the real packaged
checkpoint header. A tiny geometry runs the same reference transformer
math with deterministic hash-filled weights, keeping the golden small
enough to replay on CPU, and the reference encode_token_weights is
executed over that tiny stack for both projection kinds.

Run from the Dinkster root with a sibling ``ComfyUI`` checkout detached at
the pinned commit. ``PYTHONPATH`` must include comfy-aimdo when the
reference interpreter does not already provide it. Three artifact
inputs are required, each verified against its pinned sha256:

    DINKSTER_LTX_GEMMA_SPIECE_MODEL   SentencePiece model extracted from the
                                   Comfy-Org packaged text encoder
    DINKSTER_LTX_GEMMA_HEADER_JSON    safetensors header JSON of that file
    DINKSTER_LTX2_19B_HEADER_JSON     safetensors header JSON of the combined
                                   Lightricks ltx-2-19b-dev checkpoint

The dual-projection layout has no reachable public checkpoint header;
its listing is derived from the reference module structure and marked
as such in the payload.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = (REPO.parent / "ComfyUI").resolve()
#: The audited reference baseline. Generation REFUSES on any other
#: commit so the recorded provenance is a guarantee, not a claim.
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"

sys.path.insert(0, str(REPO / "packages" / "dinkster-inference-torch" / "tests"))
sys.path.insert(0, str(COMFY_ROOT))
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402
import torch  # noqa: E402
from clip_fill import fill_state_dict, fill_value  # noqa: E402

comfy.options.enable_args_parsing()

from comfy import model_management, ops, sd1_clip  # noqa: E402
from comfy.ldm.modules import attention as _attention  # noqa: E402
from comfy.text_encoders import gemma4, llama, lt  # noqa: E402
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402

# Force the pytorch SDPA backend (the one Dinkster ports). On this CPU-only
# interpreter optimized_attention_for_device would pick the manual
# attention_basic kernel, whose softmax rounding drifts from SDPA by
# ~1e-6 per block. llama.py bound the selector as a module global at
# import time, so that name is what must be patched.
llama.optimized_attention_for_device = lambda device, mask=False, small_input=False: (
    _attention.attention_pytorch
)
gemma4.optimized_attention_for_device = lambda device, mask=False, small_input=False: (
    _attention.attention_pytorch
)
ATTENTION_BACKEND = "attention_pytorch"

OUT = platform_golden_path(
    REPO / "tests" / "goldens" / "ltx_gemma_text_goldens.json", torch.__version__
)
GEMMA4_OUT = platform_golden_path(
    REPO / "tests" / "goldens" / "ltx_gemma4_text_goldens.json", torch.__version__
)

SPIECE_SHA256 = "1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c"
GEMMA_HEADER_SHA256 = "0d95607532a27e4a1d6e097b3ee000e1bab040230312488ac43ee66883a8030c"
LTX2_HEADER_SHA256 = "92f323a94dff45d5cb985917692535a1dc8af5126b9aca7e7d162e535e552fd0"

#: Provenance of the packaged text encoder the header and SentencePiece
#: model were extracted from (sha256 is the upload's x-linked-etag).
GEMMA_CHECKPOINT_URL = (
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/"
    "text_encoders/gemma_3_12B_it.safetensors"
)
GEMMA_CHECKPOINT_BYTES = 24379468890
GEMMA_CHECKPOINT_SHA256 = "56eaa964a0d9325d2dc9ecaf7759bfaf0fac78ae36c789bed6e03e275a3729ec"

PROMPTS = (
    "",
    "cat",
    "a photo of a cat",
    "  cafe\u0301 \u4e2d  ",
    "line one\nline two",
    r"\(cat\) and (dog:1.2)",
    "a embedding:foo bar",
    "one embedding:two embedding:three",
    "hello <end_of_turn> world",
    "<image_soft_token> literal soft token",
    "<start_of_turn>user\nalready templated<end_of_turn>\n",
    "long prompt " + "wander through the luminous canyon " * 220,
)

CORPUS_ATOMS = (
    "a",
    "Z",
    "cafe\u0301",
    "caf\u00e9",
    "\u4e2d\u6587",
    "\U0001f600",
    "'re",
    "...",
    "+/-=",
    r"\(x\)",
    "embedding:item",
    "<end_of_turn>",
    "<image_soft_token>",
    "<start_of_turn>",
    "(weight:1.5)",
    "\u3000wide",
)
CORPUS_SEPARATORS = ("", " ", "  ", "\n", "\r\n", "\t", "\u3000")
CORPUS_PROMPTS = tuple(
    CORPUS_ATOMS[index % len(CORPUS_ATOMS)]
    + CORPUS_SEPARATORS[index % len(CORPUS_SEPARATORS)]
    + CORPUS_ATOMS[(index * 7 + 3) % len(CORPUS_ATOMS)]
    for index in range(128)
)

#: Tiny geometry in Dinkster GemmaTextConfig field names; the reference
#: config is derived from it below. Token ids stay inside vocab 64.
TINY_CONFIG = {
    "architecture": "gemma3_ltx_12b",
    "vocab_size": 64,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "rms_norm_eps": 1e-6,
    "rope_theta_global": 1_000_000.0,
    "rope_theta_local": 10_000.0,
    "rope_scale_global": 8.0,
    "rope_scale_local": 1.0,
    "sliding_window": 8,
    "sliding_pattern": [True, True, True, True, True, False],
    "prompt_template": "{}",
    "min_tokens": 12,
    "pad_token_id": 0,
    "bos_token_id": 2,
    "end_of_turn_token_id": 3,
    "image_soft_token_id": 5,
}

TINY_STACK_DEPTH = TINY_CONFIG["num_hidden_layers"] + 1
TINY_FEATURES = TINY_CONFIG["hidden_size"] * TINY_STACK_DEPTH

TINY_GEMMA4_CONFIG = {
    "architecture": "gemma4_ltx_12b",
    "vocab_size": 64,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 4,
    "rms_norm_eps": 1e-6,
    "rope_theta_global": 1_000_000.0,
    "rope_theta_local": 10_000.0,
    "rope_scale_global": 1.0,
    "rope_scale_local": 1.0,
    "sliding_window": 8,
    "sliding_pattern": [True, True, True, True, True, False],
    "prompt_template": "{}",
    "min_tokens": 12,
    "pad_token_id": 0,
    "bos_token_id": 2,
    "end_of_turn_token_id": 1,
    "image_soft_token_id": 0,
    "global_head_dim": 8,
    "num_global_key_value_heads": 1,
    "global_k_eq_v": True,
    "rms_norm_add": False,
    "value_rms_norm": True,
    "layer_scalar": True,
    "global_partial_rotary_factor": 0.25,
    "attention_scale": 1.0,
}


def required_input(name: str, sha256: str) -> Path:
    configured = os.environ.get(name)
    if configured is None:
        raise SystemExit(f"{name} is not set")
    path = Path(configured)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise SystemExit(f"{name} {path} has sha256 {digest}, expected {sha256}")
    return path


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


def enc_probes(tensor: torch.Tensor) -> dict[str, object]:
    flattened = tensor.float().flatten()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "sample_stride": 7,
        "samples": flattened[::7].tolist(),
    }


def full_layout() -> dict[str, object]:
    model = llama.Gemma3_12B({}, torch.float32, "meta", ops.disable_weight_init)
    state = model.state_dict()
    siblings: dict[str, int] = {}
    for key in state:
        if not key.startswith("model."):
            prefix = key.split(".", 1)[0]
            siblings[prefix] = siblings.get(prefix, 0) + 1
    return {
        "model": sorted(
            (key.removeprefix("model."), list(value.shape))
            for key, value in state.items()
            if key.startswith("model.")
        ),
        "sibling_key_counts": dict(sorted(siblings.items())),
    }


def real_checkpoint(header_path: Path, spiece_path: Path, meta_model: list) -> dict[str, object]:
    header = json.loads(header_path.read_text())
    model_entries = sorted(
        (key.removeprefix("model."), list(entry["shape"]))
        for key, entry in header.items()
        if key.startswith("model.")
    )
    if model_entries != meta_model:
        raise SystemExit("packaged checkpoint model.* subtree does not match the meta layout")
    siblings: dict[str, int] = {}
    for key in header:
        if key != "__metadata__" and not key.startswith("model."):
            prefix = key.split(".", 1)[0]
            siblings[prefix] = siblings.get(prefix, 0) + 1
    spiece_bytes = spiece_path.read_bytes()
    return {
        "url": GEMMA_CHECKPOINT_URL,
        "byte_size": GEMMA_CHECKPOINT_BYTES,
        "sha256": GEMMA_CHECKPOINT_SHA256,
        "header_sha256": GEMMA_HEADER_SHA256,
        "model_subtree_matches_meta_layout": True,
        "model_dtypes": sorted(
            {entry["dtype"] for key, entry in header.items() if key.startswith("model.")}
        ),
        "sibling_key_counts": dict(sorted(siblings.items())),
        "spiece_model": {
            "header_key": "spiece_model",
            "byte_size": len(spiece_bytes),
            "sha256": SPIECE_SHA256,
        },
    }


def ltx2_19b_dev(header_path: Path) -> dict[str, object]:
    header = json.loads(header_path.read_text())
    projection = sorted(
        (key, list(entry["shape"]))
        for key, entry in header.items()
        if key.startswith("text_embedding_projection.")
    )
    absent = (
        "model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.2.attn1.to_q.bias"
    )
    trigger = (
        "model.diffusion_model.audio_embeddings_connector.transformer_1d_blocks.0.attn1.to_q.bias"
    )
    connector_keys = sorted(key for key in header if "embeddings_connector" in key)
    return {
        "source": "Lightricks ltx-2-19b-dev.safetensors full header",
        "header_sha256": LTX2_HEADER_SHA256,
        "text_embedding_projection": projection,
        "legacy_connector": {
            "absent_probe_key": absent,
            "absent_probe_present": absent in header,
            "trigger_key": trigger,
            "trigger_shape": list(header[trigger]["shape"]) if trigger in header else None,
            "connector_key_count": len(connector_keys),
            "fires_reference_compat_mode": absent not in header
            and trigger in header
            and header[trigger]["shape"][0] == 3840,
        },
    }


class _ZeroEmbedding:
    """Mask computation never reads embedding values."""

    def __call__(self, ids: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.zeros(ids.shape + (4,), dtype=out_dtype)


def executed_mask(ids: list[int]) -> list[int]:
    stub = SimpleNamespace(
        special_tokens={"start": 2, "pad": 0},
        transformer=SimpleNamespace(get_input_embeddings=lambda: _ZeroEmbedding()),
    )
    _, attention_mask, _, _ = sd1_clip.SDClipModel.process_tokens(stub, [ids], torch.device("cpu"))
    return attention_mask[0].tolist()


def tokenizer_rows(
    tokenizer: lt.Gemma3_12BTokenizer, text: str, skip_template: bool
) -> tuple[list[int], list[float], list[int]]:
    rows = tokenizer.tokenize_with_weights(text, return_word_ids=True, skip_template=skip_template)
    if len(rows) != 1:
        raise AssertionError(f"Gemma produced {len(rows)} rows for {text!r}")
    ids = [int(token) for token, _, _ in rows[0]]
    weights = [float(weight) for _, weight, _ in rows[0]]
    word_ids = [int(word_id) for _, _, word_id in rows[0]]
    return ids, weights, word_ids


def tokenizer_goldens(spiece: bytes) -> list[dict[str, object]]:
    tokenizer = lt.Gemma3_12BTokenizer(tokenizer_data={"spiece_model": spiece})
    cases = []
    for text in PROMPTS:
        for skip_template in (True, False):
            ids, weights, word_ids = tokenizer_rows(tokenizer, text, skip_template)
            if set(weights) != {1.0}:
                raise AssertionError("Gemma prompt weights must be disabled")
            cases.append(
                {
                    "text": text,
                    "skip_template": skip_template,
                    "ids": ids,
                    "word_ids": word_ids,
                    "attention_mask": executed_mask(ids),
                }
            )
    return cases


def tokenizer_corpus_golden(spiece: bytes) -> dict[str, object]:
    tokenizer = lt.Gemma3_12BTokenizer(tokenizer_data={"spiece_model": spiece})
    outputs = []
    for text in CORPUS_PROMPTS:
        outputs.append(
            [tokenizer_rows(tokenizer, text, skip_template)[0] for skip_template in (True, False)]
        )
    canonical = json.dumps(outputs, separators=(",", ":")).encode("ascii")
    return {
        "prompts": list(CORPUS_PROMPTS),
        "token_ids_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def reference_tiny_model() -> torch.nn.Module:
    config = SimpleNamespace(
        vocab_size=TINY_CONFIG["vocab_size"],
        hidden_size=TINY_CONFIG["hidden_size"],
        intermediate_size=TINY_CONFIG["intermediate_size"],
        num_hidden_layers=TINY_CONFIG["num_hidden_layers"],
        num_attention_heads=TINY_CONFIG["num_attention_heads"],
        num_key_value_heads=TINY_CONFIG["num_key_value_heads"],
        max_position_embeddings=128,
        rms_norm_eps=TINY_CONFIG["rms_norm_eps"],
        rope_theta=[TINY_CONFIG["rope_theta_global"], TINY_CONFIG["rope_theta_local"]],
        rope_scale=[TINY_CONFIG["rope_scale_global"], TINY_CONFIG["rope_scale_local"]],
        transformer_type="gemma3",
        head_dim=TINY_CONFIG["head_dim"],
        rms_norm_add=True,
        mlp_activation="gelu_pytorch_tanh",
        qkv_bias=False,
        rope_dims=None,
        q_norm="gemma3",
        k_norm="gemma3",
        sliding_attention=[
            TINY_CONFIG["sliding_window"] if sliding else False
            for sliding in TINY_CONFIG["sliding_pattern"]
        ],
        final_norm=True,
        lm_head=False,
    )
    model = llama.Llama2_(config, device="cpu", dtype=torch.float32, ops=ops.disable_weight_init)
    entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    model.load_state_dict(fill_state_dict(entries), strict=True)
    return model


def reference_tiny_gemma4_model() -> torch.nn.Module:
    config = SimpleNamespace(
        vocab_size=TINY_GEMMA4_CONFIG["vocab_size"],
        hidden_size=TINY_GEMMA4_CONFIG["hidden_size"],
        intermediate_size=TINY_GEMMA4_CONFIG["intermediate_size"],
        num_hidden_layers=TINY_GEMMA4_CONFIG["num_hidden_layers"],
        num_attention_heads=TINY_GEMMA4_CONFIG["num_attention_heads"],
        num_key_value_heads=TINY_GEMMA4_CONFIG["num_key_value_heads"],
        num_global_key_value_heads=TINY_GEMMA4_CONFIG["num_global_key_value_heads"],
        max_position_embeddings=128,
        rms_norm_eps=TINY_GEMMA4_CONFIG["rms_norm_eps"],
        rope_theta=[
            TINY_GEMMA4_CONFIG["rope_theta_global"],
            TINY_GEMMA4_CONFIG["rope_theta_local"],
        ],
        head_dim=TINY_GEMMA4_CONFIG["head_dim"],
        global_head_dim=TINY_GEMMA4_CONFIG["global_head_dim"],
        attention_k_eq_v=True,
        rms_norm_add=False,
        mlp_activation="gelu_pytorch_tanh",
        qkv_bias=False,
        q_norm="gemma3",
        k_norm="gemma3",
        sliding_attention=[
            TINY_GEMMA4_CONFIG["sliding_window"] if sliding else False
            for sliding in TINY_GEMMA4_CONFIG["sliding_pattern"]
        ],
        partial_rotary_factor=TINY_GEMMA4_CONFIG["global_partial_rotary_factor"],
        final_norm=True,
        lm_head=False,
        vision_bidirectional=False,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        use_double_wide_mlp=False,
        suppress_tokens=[],
    )
    model = gemma4.Gemma4Transformer(
        config, device="cpu", dtype=torch.float32, ops=ops.disable_weight_init
    )
    entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    model.load_state_dict(fill_state_dict(entries), strict=True)
    return model


#: Sequence 12 exceeds the tiny sliding window 8, exercising the
#: sliding-mask path; sequence 6 stays below it.
TINY_CASES = (
    {
        "ids": [[0, 0, 0, 2, 5, 7, 11, 3, 9, 4, 6, 1]],
        "attention_mask": [[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
    },
    {
        "ids": [[2, 5, 7, 11, 3, 9]],
        "attention_mask": [[1, 1, 1, 1, 1, 1]],
    },
)


def tiny_model_golden() -> dict[str, object]:
    model = reference_tiny_model()
    entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    cases = []
    for case in TINY_CASES:
        ids = torch.tensor(case["ids"], dtype=torch.long)
        mask = torch.tensor(case["attention_mask"], dtype=torch.long)
        with torch.no_grad():
            _, intermediate = model(
                ids,
                attention_mask=mask,
                intermediate_output="all",
                final_layer_norm_intermediate=False,
            )
        cases.append(
            {
                "ids": case["ids"],
                "attention_mask": case["attention_mask"],
                "stack": enc(intermediate),
            }
        )
    return {"config": TINY_CONFIG, "state_dict": entries, "cases": cases}


def tiny_gemma4_model_golden() -> dict[str, object]:
    model = reference_tiny_gemma4_model()
    entries = sorted((key, list(value.shape)) for key, value in model.state_dict().items())
    cases = []
    for case in TINY_CASES:
        ids = torch.tensor(case["ids"], dtype=torch.long)
        mask = torch.tensor(case["attention_mask"], dtype=torch.long)
        with torch.no_grad():
            _, intermediate = model(
                ids,
                attention_mask=mask,
                intermediate_output="all",
                final_layer_norm_intermediate=False,
            )
        cases.append(
            {
                "ids": case["ids"],
                "attention_mask": case["attention_mask"],
                "stack": enc_probes(intermediate),
            }
        )
    return {"config": TINY_GEMMA4_CONFIG, "state_dict": entries, "cases": cases}


def encoder_goldens(tiny: dict[str, object]) -> dict[str, object]:
    case = tiny["cases"][0]
    stack = torch.tensor(case["stack"]["data"], dtype=torch.float32).reshape(case["stack"]["shape"])
    mask = torch.tensor(case["attention_mask"], dtype=torch.long)

    single = torch.nn.Linear(TINY_FEATURES, 24, bias=False)
    with torch.no_grad():
        single.weight.copy_(fill_value("text_embedding_projection.weight", (24, TINY_FEATURES)))
    dual = lt.DualLinearProjection(
        TINY_FEATURES, 20, 10, dtype=torch.float32, device="cpu", operations=ops.disable_weight_init
    )
    dual_entries = [
        (f"text_embedding_projection.{key}", list(value.shape))
        for key, value in dual.state_dict().items()
    ]
    dual.load_state_dict(
        {
            key.removeprefix("text_embedding_projection."): fill_value(key, shape)
            for key, shape in dual_entries
        },
        strict=True,
    )

    original = model_management.should_use_bf16
    model_management.should_use_bf16 = lambda *args, **kwargs: False
    try:
        outputs = {}
        for kind, projection in (("single_linear", single), ("dual_linear", dual)):
            stub = SimpleNamespace(
                gemma3_12b=SimpleNamespace(
                    encode_token_weights=lambda pairs: (
                        stack.clone(),
                        None,
                        {"attention_mask": mask.clone()},
                    )
                ),
                text_encoder_key="gemma3_12b",
                text_projection_type=kind,
                execution_device=torch.device("cpu"),
                text_embedding_projection=projection,
                compat_mode=False,
            )
            with torch.no_grad():
                out, pooled, extra = lt.LTXAVTEModel.encode_token_weights(stub, {"gemma3_12b": []})
            if pooled is not None or extra != {"unprocessed_ltxav_embeds": True}:
                raise AssertionError("unexpected reference encoder extras")
            outputs[kind] = enc(out)
    finally:
        model_management.should_use_bf16 = original

    return {
        "attended_tokens": int(mask.sum().item()),
        "projection_fill": {
            "single": [["text_embedding_projection.weight", [24, TINY_FEATURES]]],
            "dual": dual_entries,
        },
        "single_linear": outputs["single_linear"],
        "dual_linear": outputs["dual_linear"],
    }


def dual_projection_layout() -> dict[str, object]:
    """Structure-derived: no public checkpoint header with the dual
    projection was reachable; the listing enumerates the reference
    DualLinearProjection(3840 * 49, 4096, 2048) module state."""
    dual = lt.DualLinearProjection(
        3840 * 49,
        4096,
        2048,
        dtype=torch.float32,
        device="meta",
        operations=ops.disable_weight_init,
    )
    return {
        "structure_derived": True,
        "layout": sorted((key, list(value.shape)) for key, value in dual.state_dict().items()),
    }


def main() -> None:
    commit = git_head(COMFY_ROOT)
    if commit != REFERENCE_COMMIT:
        raise SystemExit(f"{COMFY_ROOT} is at {commit}; expected {REFERENCE_COMMIT}")
    for module in (gemma4, lt, llama, sd1_clip):
        module_file = Path(module.__file__ or "").resolve()
        if not module_file.is_relative_to(COMFY_ROOT):
            raise SystemExit(f"reference module imported from {module_file}")

    spiece_path = required_input("DINKSTER_LTX_GEMMA_SPIECE_MODEL", SPIECE_SHA256)
    gemma_header_path = required_input("DINKSTER_LTX_GEMMA_HEADER_JSON", GEMMA_HEADER_SHA256)
    ltx2_header_path = required_input("DINKSTER_LTX2_19B_HEADER_JSON", LTX2_HEADER_SHA256)
    spiece = spiece_path.read_bytes()

    layout = full_layout()
    tiny = tiny_model_golden()
    payload = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
            "attention_backend": ATTENTION_BACKEND,
            **tuple_provenance(torch.__version__),
        },
        "layout": layout,
        "real_checkpoint": real_checkpoint(gemma_header_path, spiece_path, layout["model"]),
        "ltx2_19b_dev": ltx2_19b_dev(ltx2_header_path),
        "dual_projection": dual_projection_layout(),
        "tokenizer": tokenizer_goldens(spiece),
        "tokenizer_corpus": tokenizer_corpus_golden(spiece),
        "model": tiny,
        "encoder": encoder_goldens(tiny),
    }
    OUT.write_text(
        json.dumps(payload, ensure_ascii=True, indent=None, separators=(",", ":")) + "\n"
    )
    print(f"wrote {OUT}", file=sys.stderr)
    GEMMA4_OUT.write_text(
        json.dumps(
            {
                "reference": payload["reference"],
                "model": tiny_gemma4_model_golden(),
            },
            ensure_ascii=True,
            indent=None,
            separators=(",", ":"),
        )
        + "\n"
    )
    print(f"wrote {GEMMA4_OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
