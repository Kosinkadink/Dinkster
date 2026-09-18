"""Generate LoRA-decode goldens from the ComfyUI reference checkout.

Runs the REFERENCE implementations (comfy.lora_convert.convert_lora,
comfy.lora.load_lora, comfy.lora.model_lora_keys_clip @ the audited
baseline) on synthetic key sets and writes
tests/goldens/lora_goldens.json. Dinkster's torch-free dialect decoder is
pinned against these classifications - the oracle is the reference
code itself, never a re-derivation.

Usage (needs a torch interpreter; the workspace root venv is
deliberately torch-free):

    PYTHONPATH=../ComfyUI:../comfy-aimdo:../comfy-kitchen \
        /path/to/torch-venv/bin/python tools/gen_lora_goldens.py

Run from the Dinkster repo root with ../ComfyUI at the recorded baseline
commit (recorded in the output for provenance).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import comfy.lora as clora  # noqa: E402
import comfy.lora_convert as cconvert  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "goldens" / "lora_goldens.json"


def t(*shape: int) -> torch.Tensor:
    return torch.zeros(shape, dtype=torch.float32)


def scalar(v: float) -> torch.Tensor:
    return torch.tensor(v, dtype=torch.float32)


# ---------------------------------------------------------------- convert


CONVERT_CASES: dict[str, dict[str, torch.Tensor]] = {
    "bfl_flux_control": {
        "img_in.lora_A.weight": t(4, 64),
        "img_in.lora_B.weight": t(3072, 4),
        "img_in.lora_B.bias": t(3072),
        "single_blocks.0.norm.key_norm.scale": t(128),
        "double_blocks.1.img_attn.qkv.lora_A.weight": t(4, 3072),
        "double_blocks.1.img_attn.qkv.lora_B.weight": t(9216, 4),
    },
    "wan_fun": {
        "lora_unet__blocks_0_cross_attn_k.lora_down.weight": t(4, 5120),
        "lora_unet__blocks_0_cross_attn_k.lora_up.weight": t(5120, 4),
        "lora_unet__blocks_0_cross_attn_k.alpha": scalar(4.0),
        "some_other.key": t(2, 2),
    },
    "uso": {
        "single_blocks.37.processor.qkv_lora.up.weight": t(9216, 4),
        "single_blocks.37.processor.qkv_lora.down.weight": t(4, 3072),
        "double_blocks.18.processor.qkv_lora2.up.weight": t(9216, 4),
        "double_blocks.18.processor.qkv_lora2.down.weight": t(4, 3072),
        "double_blocks.18.processor.proj_lora1.up.weight": t(3072, 4),
        "double_blocks.18.processor.proj_lora1.down.weight": t(4, 3072),
    },
    "passthrough": {
        "lora_unet_input_blocks_1_0_emb_layers_1.lora_up.weight": t(320, 4),
        "lora_unet_input_blocks_1_0_emb_layers_1.lora_down.weight": t(4, 1280),
        "lora_unet_input_blocks_1_0_emb_layers_1.alpha": scalar(4.0),
    },
}


def gen_convert() -> list[dict[str, object]]:
    cases = []
    for name, sd in CONVERT_CASES.items():
        out = cconvert.convert_lora(dict(sd))
        values = {k: v.tolist() for k, v in out.items() if k.endswith(".reshape_weight")}
        cases.append(
            {
                "name": name,
                "input_shapes": {k: list(v.shape) for k, v in sd.items()},
                "output_keys": sorted(out.keys()),
                "reshape_values": values,
            }
        )
    return cases


# ----------------------------------------------------------------- decode


def lora_case(
    name: str,
    lora: dict[str, torch.Tensor],
    to_load: dict[str, str],
) -> dict[str, object]:
    patch_dict = clora.load_lora(dict(lora), dict(to_load), log_missing=False)

    # Leftovers: reconstruct via the reference's own bookkeeping by
    # re-running with logging captured.
    import logging

    records: list[str] = []

    class Cap(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record.getMessage())

    handler = Cap()
    logging.getLogger().addHandler(handler)
    try:
        clora.load_lora(dict(lora), dict(to_load), log_missing=True)
    finally:
        logging.getLogger().removeHandler(handler)
    prefix = "lora key not loaded: "
    leftovers = sorted(r[len(prefix) :] for r in records if r.startswith(prefix))

    patches: dict[str, dict[str, object]] = {}
    for key, value in patch_dict.items():
        target = key if isinstance(key, str) else list(key)
        if isinstance(value, tuple):
            kind = value[0]
            slots: list[object] = [
                f"tensor:{list(v.shape)}" if isinstance(v, torch.Tensor) else v for v in value[1]
            ]
        else:
            kind = type(value).__name__
            slots = []
            for v in value.weights:
                if isinstance(v, torch.Tensor):
                    slots.append(f"tensor:{list(v.shape)}")
                elif v is None:
                    slots.append(None)
                else:
                    slots.append(v)
        patches[json.dumps(target)] = {"kind": kind, "slots": slots}

    return {
        "name": name,
        "lora_shapes": {k: list(v.shape) for k, v in lora.items()},
        "to_load": to_load,
        "patches": patches,
        "leftovers": leftovers,
    }


def gen_decode() -> list[dict[str, object]]:
    W = "diffusion_model.blk.attn.weight"
    cases = [
        lora_case(
            "kohya_locon_alpha_dora",
            {
                "stem.lora_up.weight": t(320, 4),
                "stem.lora_down.weight": t(4, 320),
                "stem.lora_mid.weight": t(4, 4, 3, 3),
                "stem.alpha": scalar(4.0),
                "stem.dora_scale": t(320, 1),
            },
            {"stem": W},
        ),
        lora_case(
            "diffusers_underscore",
            {
                "stem_lora.up.weight": t(320, 4),
                "stem_lora.down.weight": t(4, 320),
            },
            {"stem": W},
        ),
        lora_case(
            "peft",
            {
                "stem.lora_B.weight": t(320, 4),
                "stem.lora_A.weight": t(4, 320),
            },
            {"stem": W},
        ),
        lora_case(
            "diffusers3",
            {
                "stem.lora.up.weight": t(320, 4),
                "stem.lora.down.weight": t(4, 320),
            },
            {"stem": W},
        ),
        lora_case(
            "mochi",
            {"stem.lora_B": t(320, 4), "stem.lora_A": t(4, 320)},
            {"stem": W},
        ),
        lora_case(
            "transformers",
            {
                "stem.lora_linear_layer.up.weight": t(320, 4),
                "stem.lora_linear_layer.down.weight": t(4, 320),
            },
            {"stem": W},
        ),
        lora_case(
            "peft_qwen",
            {
                "stem.lora_B.default.weight": t(320, 4),
                "stem.lora_A.default.weight": t(4, 320),
            },
            {"stem": W},
        ),
        lora_case(
            "kohya_reshape",
            {
                "stem.lora_up.weight": t(320, 4),
                "stem.lora_down.weight": t(4, 320),
                "stem.reshape_weight": torch.tensor([320, 320]),
            },
            {"stem": W},
        ),
        lora_case(
            "loha_plain",
            {
                "stem.hada_w1_a": t(320, 4),
                "stem.hada_w1_b": t(4, 320),
                "stem.hada_w2_a": t(320, 4),
                "stem.hada_w2_b": t(4, 320),
                "stem.alpha": scalar(4.0),
            },
            {"stem": W},
        ),
        lora_case(
            "loha_tucker",
            {
                "stem.hada_w1_a": t(320, 4),
                "stem.hada_w1_b": t(4, 320),
                "stem.hada_w2_a": t(320, 4),
                "stem.hada_w2_b": t(4, 320),
                "stem.hada_t1": t(4, 4, 3, 3),
                "stem.hada_t2": t(4, 4, 3, 3),
            },
            {"stem": W},
        ),
        lora_case(
            "lokr_direct",
            {"stem.lokr_w1": t(10, 10), "stem.lokr_w2": t(32, 32)},
            {"stem": W},
        ),
        lora_case(
            "lokr_decomposed_t2",
            {
                "stem.lokr_w1_a": t(10, 2),
                "stem.lokr_w1_b": t(2, 10),
                "stem.lokr_w2_a": t(32, 2),
                "stem.lokr_w2_b": t(2, 32),
                "stem.lokr_t2": t(2, 2, 3, 3),
            },
            {"stem": W},
        ),
        lora_case(
            "lokr_mixed",
            {
                "stem.lokr_w1": t(10, 10),
                "stem.lokr_w2_a": t(32, 2),
                "stem.lokr_w2_b": t(2, 32),
            },
            {"stem": W},
        ),
        lora_case(
            "glora",
            {
                "stem.a1.weight": t(320, 4),
                "stem.a2.weight": t(4, 320),
                "stem.b1.weight": t(320, 4),
                "stem.b2.weight": t(4, 320),
                "stem.alpha": scalar(4.0),
            },
            {"stem": W},
        ),
        lora_case(
            "oft_rescale",
            {"stem.oft_blocks": t(4, 80, 80), "stem.rescale": t(320, 1)},
            {"stem": W},
        ),
        lora_case(
            "boft",
            {"stem.oft_blocks": t(2, 4, 80, 80)},
            {"stem": W},
        ),
        lora_case(
            "oft_blocks_bad_rank",
            {"stem.oft_blocks": t(80, 80)},
            {"stem": W},
        ),
        lora_case(
            "norms",
            {"stem.w_norm": t(320), "stem.b_norm": t(320)},
            {"stem": W},
        ),
        lora_case(
            "diff_pair",
            {"stem.diff": t(320, 320), "stem.diff_b": t(320)},
            {"stem": W},
        ),
        lora_case(
            "set_weight",
            {"stem.set_weight": t(320, 320)},
            {"stem": W},
        ),
        lora_case(
            "adapter_then_diff_overwrites",
            {
                "stem.lora_up.weight": t(320, 4),
                "stem.lora_down.weight": t(4, 320),
                "stem.diff": t(320, 320),
            },
            {"stem": W},
        ),
        lora_case(
            "alpha_only_consumed",
            {"stem.alpha": scalar(4.0), "unrelated.key": t(2, 2)},
            {"stem": W},
        ),
        lora_case(
            "two_stems",
            {
                "a.lora_up.weight": t(320, 4),
                "a.lora_down.weight": t(4, 320),
                "b.diff_b": t(640),
            },
            {
                "a": "diffusion_model.a.weight",
                "b": "diffusion_model.b.weight",
            },
        ),
    ]
    return cases


# ----------------------------------------------------------- clip key maps


def sd15_clip_keys() -> list[str]:
    keys = ["clip_l.transformer.text_model.embeddings.token_embedding.weight"]
    for b in range(12):
        for c in clora.LORA_CLIP_MAP:
            keys.append(f"clip_l.transformer.text_model.encoder.layers.{b}.{c}.weight")
    keys.append("clip_l.transformer.text_projection.weight")
    return keys


def sdxl_clip_keys() -> list[str]:
    keys = sd15_clip_keys()
    for b in range(32):
        for c in clora.LORA_CLIP_MAP:
            keys.append(f"clip_g.transformer.text_model.encoder.layers.{b}.{c}.weight")
    keys.append("clip_g.transformer.text_projection.weight")
    return keys


def refiner_clip_keys() -> list[str]:
    keys = []
    for b in range(32):
        for c in clora.LORA_CLIP_MAP:
            keys.append(f"clip_g.transformer.text_model.encoder.layers.{b}.{c}.weight")
    keys.append("clip_g.transformer.text_projection.weight")
    return keys


def sd3_clip_keys() -> list[str]:
    keys = sdxl_clip_keys()
    keys.append("t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.q.weight")
    return keys


def t5_only_keys() -> list[str]:
    return ["t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.q.weight"]


def hydit_keys() -> list[str]:
    return ["hydit_clip.transformer.bert.encoder.layer.0.attention.self.query.weight"]


def single_llama_keys() -> list[str]:
    return [
        "llama.transformer.model.layers.0.self_attn.q_proj.weight",
        "llama.transformer.model.layers.0.mlp.gate_proj.weight",
    ]


CLIP_CASES = {
    "sd15": sd15_clip_keys,
    "sdxl": sdxl_clip_keys,
    "sdxl_refiner": refiner_clip_keys,
    "sd3": sd3_clip_keys,
    "t5_only": t5_only_keys,
    "hydit": hydit_keys,
    "single_llama": single_llama_keys,
}


def gen_clip_maps() -> list[dict[str, object]]:
    cases = []
    for name, keys_fn in CLIP_CASES.items():
        keys = keys_fn()
        model = SimpleNamespace(state_dict=lambda keys=keys: dict.fromkeys(keys))
        key_map = clora.model_lora_keys_clip(model, {})
        cases.append({"name": name, "model_keys": keys, "key_map": key_map})
    return cases


def main() -> None:
    commit = subprocess.check_output(
        ["git", "-C", str(REPO.parent / "ComfyUI"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    out = {
        "_meta": {
            "reference_commit": commit,
            "torch": torch.__version__,
            "generator": "tools/gen_lora_goldens.py",
        },
        "convert": gen_convert(),
        "decode": gen_decode(),
        "clip_key_maps": gen_clip_maps(),
    }
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
