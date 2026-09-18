"""Generate T5 text-model goldens from the ComfyUI reference.

Runs the REFERENCE T5 stack @ the audited baseline - the architecture
(comfy/text_encoders/t5.py T5), the encode policy (comfy/sd1_clip.py
SDClipModel.encode_token_weights with Flux's T5XXLModel and Wan's
UMT5XXlModel knobs), and the
Flux two-model composition (comfy/text_encoders/flux.py
FluxClipModel.encode_token_weights) - and writes
packages/dinkster-inference-torch/tests/goldens/t5_text_goldens.json.
dinkster_inference_torch.t5_text is pinned against these outputs; the
oracle is the executed reference, never a re-derivation.

Payloads:

- "layouts": the sorted (key, shape) state-dict listing of the
  FULL-SIZE T5-XXL model, built from the reference's own config JSON
  on the meta device (weights never materialize). Pins torch-free
  detection and layout generation.
- "buckets": the reference _relative_position_bucket values for
  relative positions -200..200 (crossing the max_distance=128 clamp
  on both sides), pinning the Mesh-TF bucket math.
- "cases": tiny architectures executed with deterministic hash-filled
  weights (clip_fill.py, shared with the replay tests). T5 reads its
  vocabulary from the config, so tiny cases shrink every dimension
  but keep the real special token ids (EOS 1, pad 0). Chunks are
  stored as literal (unit, weight) position lists; textual-inversion
  rows fill from the shared ``embedding:{name}`` pseudo-key.
- "umt5": a tiny per-block-bias architecture executed with Wan's
  attention-mask and zero-out-masked policy.
- "flux": FluxClipModel composition over the tiny T5 and a tiny
  CLIP-L (T5 sequence conditioning + CLIP-L raw pooled).

Usage (a torch interpreter; comfy.ops imports comfy_aimdo, so the
sibling comfy-aimdo checkout must be on PYTHONPATH when the venv
lacks it):

    PYTHONPATH=../comfy-aimdo /path/to/torch-venv/bin/python \\
        tools/gen_t5_text_goldens.py

Run from the Dinkster repo root with ../ComfyUI at the recorded baseline
commit (recorded in the output for provenance).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", REPO.parent / "ComfyUI")).resolve()
#: The audited reference baseline. Generation REFUSES on any other
#: commit so the recorded provenance is a guarantee, not a claim.
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"

# In front of any ambient PYTHONPATH: the reference must come from the
# pinned sibling checkout, not an installed or stray comfy package.
sys.path.insert(0, str(REPO / "packages" / "dinkster-inference-torch" / "tests"))
sys.path.insert(0, str(COMFY_ROOT))
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options  # noqa: E402
import torch  # noqa: E402
from clip_fill import embedding_vectors, fill_state_dict  # noqa: E402

comfy.options.enable_args_parsing()

from comfy import ops, sd1_clip  # noqa: E402
from comfy.text_encoders import flux as flux_te  # noqa: E402
from comfy.text_encoders import t5  # noqa: E402

OUT = REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "t5_text_goldens.json"

EOS, PAD = 1, 0
T5_SPECIALS = {"end": EOS, "pad": PAD}

#: Tiny classic-T5 geometry: every dimension shrunk, the layout rules
#: intact (gated GELU-tanh FF, block-0-only relative bias).
TINY_T5 = {
    "d_model": 48,
    "d_ff": 96,
    "d_kv": 12,
    "num_heads": 4,
    "num_layers": 3,
    "vocab_size": 512,
    "dense_act_fn": "gelu_pytorch_tanh",
    "is_gated_act": True,
    "model_type": "t5",
}
#: The non-gated relu variant (old T5 feed-forward): wi instead of
#: wi_0/wi_1.
TINY_T5_RELU = {
    **TINY_T5,
    "dense_act_fn": "relu",
    "is_gated_act": False,
}

TINY_UMT5 = {
    **TINY_T5,
    "model_type": "umt5",
}

#: Tiny CLIP-L for the Flux composition golden (mirrors
#: gen_clip_text_goldens.py TINY_L).
TINY_L = {
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 128,
    "hidden_act": "quick_gelu",
    "max_position_embeddings": 77,
    "eos_token_id": 49407,
}
L_BOS, L_EOS = 49406, 49407


def t5_chunk(body: list, length: int) -> list:
    """One T5 fixture chunk: body + EOS + pad fill to ``length`` (no
    BOS - the Flux T5 profile has no start token), each position a
    [unit, weight] pair."""
    positions = [*body, [EOS, 1.0]]
    positions += [[PAD, 1.0]] * (length - len(positions))
    return positions


def l_chunk(body: list) -> list:
    positions = [[L_BOS, 1.0], *body, [L_EOS, 1.0]]
    positions += [[L_EOS, 1.0]] * (77 - len(positions))
    return positions


def emb(name: str, row: int, weight: float) -> list:
    return [{"embedding": name, "row": row}, weight]


CASES = {
    "t5_unweighted": {
        "config": TINY_T5,
        "embeddings": {},
        "chunks": [
            t5_chunk([[100, 1.0], [200, 1.0], [300, 1.0]], 16),
        ],
    },
    "t5_weighted": {
        "config": TINY_T5,
        "embeddings": {},
        "chunks": [
            t5_chunk([[100, 1.0], [200, 1.3], [300, 0.7]], 16),
        ],
    },
    "t5_embedding": {
        "config": TINY_T5,
        "embeddings": {"twin": 2},
        "chunks": [
            t5_chunk(
                [[100, 1.0], emb("twin", 0, 1.2), emb("twin", 1, 1.2)],
                16,
            ),
        ],
    },
    "t5_long": {
        # 160 positions: relative offsets beyond max_distance=128
        # exercise the bucket clamp inside a real forward.
        "config": TINY_T5,
        "embeddings": {},
        "chunks": [
            t5_chunk([[i % 500, 1.0] for i in range(40)], 160),
        ],
    },
    "t5_relu": {
        "config": TINY_T5_RELU,
        "embeddings": {},
        "chunks": [
            t5_chunk([[100, 1.0], [200, 1.1]], 16),
        ],
    },
}

UMT5_CASE = {
    "config": TINY_UMT5,
    "embeddings": {},
    "chunks": [
        t5_chunk([[100, 1.0], [200, 1.3], [300, 0.7]], 16),
    ],
}


def enc(x: torch.Tensor) -> dict:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).removeprefix("torch."),
        "data": x.to(torch.float32).flatten().tolist(),
    }


def build_t5_reference(
    spec: dict, *, attention_masks: bool = False
) -> tuple[sd1_clip.SDClipModel, list]:
    """T5XXLModel knobs over a tiny config, optionally using Wan's masks."""
    model = sd1_clip.SDClipModel(
        device="cpu",
        dtype=torch.float32,
        layer="last",
        layer_idx=None,
        textmodel_json_config=dict(spec["config"]),
        special_tokens=dict(T5_SPECIALS),
        model_class=t5.T5,
        enable_attention_masks=attention_masks,
        zero_out_masked=attention_masks,
        return_attention_masks=False,
        model_options={"custom_operations": ops.disable_weight_init},
    )
    entries = sorted(
        (key, list(value.shape)) for key, value in model.transformer.state_dict().items()
    )
    model.transformer.load_state_dict(fill_state_dict(entries), strict=True)
    return model, entries


def build_l_reference() -> sd1_clip.SDClipModel:
    model = sd1_clip.SDClipModel(
        device="cpu",
        dtype=torch.float32,
        layer="last",
        layer_idx=None,
        textmodel_json_config=dict(TINY_L),
        special_tokens={"start": L_BOS, "end": L_EOS, "pad": L_EOS},
        layer_norm_hidden_state=True,
        return_projected_pooled=False,
        model_options={"custom_operations": ops.disable_weight_init},
    )
    entries = sorted(
        (key, list(value.shape)) for key, value in model.transformer.state_dict().items()
    )
    model.transformer.load_state_dict(fill_state_dict(entries), strict=True)
    return model


def reference_pairs(spec: dict, dim: int) -> list[list[tuple]]:
    pairs = []
    for stored in spec["chunks"]:
        section = []
        for unit, weight in stored:
            if isinstance(unit, dict):
                rows = spec["embeddings"][unit["embedding"]]
                vectors = embedding_vectors(unit["embedding"], rows, dim)
                section.append((vectors[unit["row"]], weight))
            else:
                section.append((unit, weight))
        pairs.append(section)
    return pairs


def full_layout(config_name: str) -> list:
    config = json.loads((COMFY_ROOT / "comfy" / "text_encoders" / config_name).read_text())
    model = t5.T5(config, None, "meta", ops.disable_weight_init)
    return sorted((key, list(value.shape)) for key, value in model.state_dict().items())


def bucket_table() -> dict:
    positions = torch.arange(-200, 201, dtype=torch.long)
    buckets = t5.T5Attention._relative_position_bucket(
        positions, bidirectional=True, num_buckets=32, max_distance=128
    )
    return {
        "first_position": -200,
        "num_buckets": 32,
        "max_distance": 128,
        "buckets": buckets.tolist(),
    }


def main() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=COMFY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if commit != REFERENCE_COMMIT:
        raise SystemExit(
            f"{COMFY_ROOT} is at {commit}; goldens must be generated"
            f" from the audited baseline {REFERENCE_COMMIT}"
        )
    module_file = Path(t5.__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(
            "comfy.text_encoders.t5 was imported from"
            f" {module_file}, not the pinned checkout {COMFY_ROOT}"
        )

    payload: dict = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
        },
        "layouts": {
            "t5_xxl": full_layout("t5_config_xxl.json"),
            "umt5_xxl": full_layout("umt5_config_xxl.json"),
        },
        "buckets": bucket_table(),
        "cases": {},
    }

    models: dict[str, sd1_clip.SDClipModel] = {}
    for name, spec in sorted(CASES.items()):
        model, entries = build_t5_reference(spec)
        models[name] = model
        with torch.no_grad():
            cond, pooled = model.encode_token_weights(
                reference_pairs(spec, spec["config"]["d_model"])
            )
        assert pooled is None, "T5 must expose no pooled output"
        payload["cases"][name] = {
            **{k: v for k, v in spec.items() if k != "config"},
            "config": spec["config"],
            "state_dict": entries,
            "cond": enc(cond),
        }

    umt5, entries = build_t5_reference(UMT5_CASE, attention_masks=True)
    with torch.no_grad():
        cond, pooled = umt5.encode_token_weights(reference_pairs(UMT5_CASE, TINY_UMT5["d_model"]))
    assert pooled is None, "UMT5 must expose no pooled output"
    payload["umt5"] = {
        **{k: v for k, v in UMT5_CASE.items() if k != "config"},
        "config": UMT5_CASE["config"],
        "state_dict": entries,
        "cond": enc(cond),
    }

    # Flux composition, executed from the reference class method over
    # the tiny towers (its logic reads only clip_l/t5xxl).
    clip_l = build_l_reference()
    towers = SimpleNamespace(clip_l=clip_l, t5xxl=models["t5_weighted"])
    l_chunks = [l_chunk([[1000, 1.0], [2000, 1.2]])]
    with torch.no_grad():
        cond, pooled = flux_te.FluxClipModel.encode_token_weights(
            towers,
            {
                "l": [[(unit, weight) for unit, weight in stored] for stored in l_chunks],
                "t5xxl": reference_pairs(CASES["t5_weighted"], TINY_T5["d_model"]),
            },
        )
    payload["flux"] = {
        "t5_case": "t5_weighted",
        "l_config": TINY_L,
        "l_chunks": l_chunks,
        "cond": enc(cond),
        "pooled": enc(pooled),
    }

    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
