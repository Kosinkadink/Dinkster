"""Generate CLIP text-model goldens from the ComfyUI reference.

Runs the REFERENCE text-model stack @ the audited baseline - the
architecture (comfy/clip_model.py CLIPTextModel), the weighted-chunk
encoding policy (comfy/sd1_clip.py SDClipModel.encode_token_weights),
and the SDXL two-tower composition (comfy/sdxl_clip.py
SDXLClipModel.encode_token_weights) - and writes
packages/dinkster-inference-torch/tests/goldens/clip_text_goldens.json.
dinkster_inference_torch.clip_text is pinned against these outputs; the
oracle is the executed reference, never a re-derivation.

Two payloads:

- "layouts": the sorted (key, shape) state-dict listings of the
  FULL-SIZE CLIP-L and CLIP-G text models, built from the reference's
  own config JSONs on the meta device (weights never materialize).
  These pin torch-free detection and layout generation.
- "cases": tiny architectures executed with deterministic hash-filled
  weights (clip_fill.py, shared with the replay tests). The reference
  hardcodes the 49408-token vocabulary in CLIPEmbeddings, so tiny
  cases shrink every other dimension but keep the real special token
  ids. Chunks are stored as literal (unit, weight) position lists -
  the fixture both sides consume; textual-inversion rows fill from
  the shared ``embedding:{name}`` pseudo-key.

Usage (a torch interpreter; comfy.ops imports comfy_aimdo, so the
sibling comfy-aimdo checkout must be on PYTHONPATH when the venv
lacks it):

    PYTHONPATH=../comfy-aimdo /path/to/torch-venv/bin/python \\
        tools/gen_clip_text_goldens.py

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

import torch  # noqa: E402
from clip_fill import embedding_vectors, fill_state_dict  # noqa: E402
from comfy.cli_args import args as _comfy_args  # noqa: E402

_comfy_args.cpu = True

from comfy import clip_model, ops, sd1_clip, sdxl_clip  # noqa: E402

OUT = (
    REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "clip_text_goldens.json"
)

BOS, EOS = 49406, 49407
CHUNK_LEN = 77

TINY_L = {
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 128,
    "hidden_act": "quick_gelu",
    "max_position_embeddings": 77,
    "eos_token_id": EOS,
}
TINY_G = {
    "hidden_size": 80,
    "num_hidden_layers": 6,
    "num_attention_heads": 5,
    "intermediate_size": 160,
    "hidden_act": "gelu",
    "max_position_embeddings": 77,
    "eos_token_id": EOS,
}

L_SPECIALS = {"start": BOS, "end": EOS, "pad": EOS}
G_SPECIALS = {"start": BOS, "end": EOS, "pad": 0}


def chunk(body: list, pad: int) -> list:
    """One fixture chunk: BOS + body + EOS + pad fill, each position a
    [unit, weight] pair. Units are ints or {"embedding", "row"}."""
    positions = [[BOS, 1.0], *body, [EOS, 1.0]]
    positions += [[pad, 1.0]] * (CHUNK_LEN - len(positions))
    return positions


def emb(name: str, row: int, weight: float) -> list:
    return [{"embedding": name, "row": row}, weight]


CASES = {
    "l_weighted": {
        "config": TINY_L,
        "special_tokens": L_SPECIALS,
        "layer": "last",
        "layer_idx": None,
        "layer_norm_hidden_state": True,
        "return_projected_pooled": False,
        "embeddings": {},
        "chunks": [
            chunk(
                [[1000, 1.0], [2000, 1.2], [3000, 1.2], [4000, 0.8]],
                L_SPECIALS["pad"],
            ),
            chunk([[5000, 0.5], [6000, 1.0]], L_SPECIALS["pad"]),
        ],
    },
    "l_unweighted": {
        "config": TINY_L,
        "special_tokens": L_SPECIALS,
        "layer": "last",
        "layer_idx": None,
        "layer_norm_hidden_state": True,
        "return_projected_pooled": False,
        "embeddings": {},
        "chunks": [
            chunk([[1000, 1.0], [2000, 1.0]], L_SPECIALS["pad"]),
        ],
    },
    "l_clip_skip_2": {
        "config": TINY_L,
        "special_tokens": L_SPECIALS,
        "layer": "hidden",
        "layer_idx": -2,
        "layer_norm_hidden_state": True,
        "return_projected_pooled": False,
        "embeddings": {},
        "chunks": [
            chunk([[1000, 1.0], [2000, 1.0]], L_SPECIALS["pad"]),
        ],
    },
    "l_empty_prompt": {
        "config": TINY_L,
        "special_tokens": L_SPECIALS,
        "layer": "last",
        "layer_idx": None,
        "layer_norm_hidden_state": True,
        "return_projected_pooled": False,
        "embeddings": {},
        "chunks": [chunk([], L_SPECIALS["pad"])],
    },
    "l_embedding": {
        "config": TINY_L,
        "special_tokens": L_SPECIALS,
        "layer": "last",
        "layer_idx": None,
        "layer_norm_hidden_state": True,
        "return_projected_pooled": False,
        "embeddings": {"twin": 2},
        "chunks": [
            chunk(
                [
                    [1000, 1.0],
                    emb("twin", 0, 1.3),
                    emb("twin", 1, 1.3),
                    [2000, 0.9],
                ],
                L_SPECIALS["pad"],
            ),
        ],
    },
    "g_hidden": {
        "config": TINY_G,
        "special_tokens": G_SPECIALS,
        "layer": "hidden",
        "layer_idx": -2,
        "layer_norm_hidden_state": False,
        "return_projected_pooled": True,
        "embeddings": {},
        "chunks": [
            chunk([[700, 1.0], [800, 1.4]], G_SPECIALS["pad"]),
        ],
    },
    "sdxl_l": {
        "config": TINY_L,
        "special_tokens": L_SPECIALS,
        "layer": "hidden",
        "layer_idx": -2,
        "layer_norm_hidden_state": False,
        "return_projected_pooled": True,
        "embeddings": {},
        "chunks": [
            chunk([[1000, 1.0], [2000, 1.2]], L_SPECIALS["pad"]),
            chunk([[3000, 0.7]], L_SPECIALS["pad"]),
        ],
    },
    "sdxl_g": {
        "config": TINY_G,
        "special_tokens": G_SPECIALS,
        "layer": "hidden",
        "layer_idx": -2,
        "layer_norm_hidden_state": False,
        "return_projected_pooled": True,
        "embeddings": {},
        "chunks": [
            chunk([[700, 1.0], [800, 1.4]], G_SPECIALS["pad"]),
        ],
    },
    "sdxl_override_l": {
        "config": TINY_L,
        "special_tokens": L_SPECIALS,
        "layer": "hidden",
        "layer_idx": -3,
        "layer_norm_hidden_state": False,
        "return_projected_pooled": True,
        "embeddings": {},
        "chunks": [
            chunk([[1100, 1.0], [2100, 1.1]], L_SPECIALS["pad"]),
        ],
    },
    "sdxl_override_g": {
        "config": TINY_G,
        "special_tokens": G_SPECIALS,
        "layer": "hidden",
        "layer_idx": -3,
        "layer_norm_hidden_state": False,
        "return_projected_pooled": True,
        "embeddings": {},
        "chunks": [
            chunk([[900, 1.0], [1000, 1.1]], G_SPECIALS["pad"]),
        ],
    },
}


def enc(x: torch.Tensor) -> dict:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).removeprefix("torch."),
        "data": x.to(torch.float32).flatten().tolist(),
    }


def build_reference(spec: dict) -> tuple[sd1_clip.SDClipModel, list]:
    model = sd1_clip.SDClipModel(
        device="cpu",
        dtype=torch.float32,
        layer=spec["layer"],
        layer_idx=spec["layer_idx"],
        textmodel_json_config=dict(spec["config"]),
        special_tokens=dict(spec["special_tokens"]),
        layer_norm_hidden_state=spec["layer_norm_hidden_state"],
        return_projected_pooled=spec["return_projected_pooled"],
        model_options={"custom_operations": ops.disable_weight_init},
    )
    entries = sorted(
        (key, list(value.shape)) for key, value in model.transformer.state_dict().items()
    )
    model.transformer.load_state_dict(fill_state_dict(entries), strict=True)
    return model, entries


def reference_pairs(spec: dict) -> list[list[tuple]]:
    dim = spec["config"]["hidden_size"]
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
    config = json.loads((COMFY_ROOT / "comfy" / config_name).read_text())
    model = clip_model.CLIPTextModel(config, None, "meta", ops.disable_weight_init)
    return sorted((key, list(value.shape)) for key, value in model.state_dict().items())


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
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=COMFY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip():
        raise SystemExit(f"{COMFY_ROOT} must be clean before golden generation")
    module_file = Path(sd1_clip.__file__ or "").resolve()
    if not module_file.is_relative_to(COMFY_ROOT):
        raise SystemExit(
            f"comfy.sd1_clip was imported from {module_file}, not the pinned checkout {COMFY_ROOT}"
        )

    payload: dict = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
        },
        "layouts": {
            "clip_l": full_layout("sd1_clip_config.json"),
            "clip_g": full_layout("clip_config_bigg.json"),
        },
        "cases": {},
    }

    models: dict[str, sd1_clip.SDClipModel] = {}
    for name, spec in sorted(CASES.items()):
        model, entries = build_reference(spec)
        models[name] = model
        with torch.no_grad():
            cond, pooled = model.encode_token_weights(reference_pairs(spec))
        payload["cases"][name] = {
            **{k: v for k, v in spec.items() if k != "config"},
            "config": spec["config"],
            "state_dict": entries,
            "cond": enc(cond),
            "pooled": enc(pooled),
        }

    # SDXL composition, executed from the reference class method over
    # the two tiny towers (its logic reads only clip_l/clip_g).
    towers = SimpleNamespace(clip_l=models["sdxl_l"], clip_g=models["sdxl_g"])
    with torch.no_grad():
        cond, pooled = sdxl_clip.SDXLClipModel.encode_token_weights(
            towers,
            {
                "l": reference_pairs(CASES["sdxl_l"]),
                "g": reference_pairs(CASES["sdxl_g"]),
            },
        )
    payload["sdxl"] = {
        "l_case": "sdxl_l",
        "g_case": "sdxl_g",
        "cond": enc(cond),
        "pooled": enc(pooled),
    }

    override_towers = SimpleNamespace(
        clip_l=models["sdxl_override_l"],
        clip_g=models["sdxl_override_g"],
    )
    with torch.no_grad():
        cond, pooled = sdxl_clip.SDXLClipModel.encode_token_weights(
            override_towers,
            {
                "l": reference_pairs(CASES["sdxl_override_l"]),
                "g": reference_pairs(CASES["sdxl_override_g"]),
            },
        )
    payload["sdxl_override"] = {
        "l_case": "sdxl_override_l",
        "g_case": "sdxl_override_g",
        "layer": -3,
        "cond": enc(cond),
        "pooled": enc(pooled),
    }

    mixed_layer = -5
    for tower in (override_towers.clip_l, override_towers.clip_g):
        tower.set_clip_options({"layer": mixed_layer})
    with torch.no_grad():
        cond, pooled = sdxl_clip.SDXLClipModel.encode_token_weights(
            override_towers,
            {
                "l": reference_pairs(CASES["sdxl_override_l"]),
                "g": reference_pairs(CASES["sdxl_override_g"]),
            },
        )
    payload["sdxl_mixed_depth_override"] = {
        "l_case": "sdxl_override_l",
        "g_case": "sdxl_override_g",
        "layer": mixed_layer,
        "cond": enc(cond),
        "pooled": enc(pooled),
    }

    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
