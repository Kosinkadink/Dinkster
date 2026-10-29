"""Generate TripoSplat goldens from ComfyUI.

Runs the REFERENCE TripoSplat stack (comfy/ldm/triposplat/model.py
flow denoiser, comfy/ldm/triposplat/vae.py octree gaussian decoder,
and comfy/image_encoders/dino3.py DINOv3 encoder @ the audited
baseline) and writes
packages/dinkster-inference-torch/tests/goldens/triposplat_goldens.json.
dinkster_inference.triposplat / dinkster_inference.dinov3 and
dinkster_inference_torch.triposplat_model / triposplat_decoder / dinov3
are pinned against these outputs; the oracle is the executed
reference, never a re-derivation.

Payload:

- "layouts": the sorted (key, shape) state-dict listings of the
  FULL-SIZE flow model, octree gaussian decoder, and DINOv3 ViT-H/16+
  encoder at the reference defaults (weights never materialize).
- "cases": tiny architectures executed with deterministic
  hash-filled weights (unet_fill.py) covering the flow model with and
  without a reference-image latent, the octree probability decoder's
  logits, a seeded octree descent, the elastic gaussian decoder's
  features and offsets, a seeded end-to-end splat decode into
  render-ready tensors, the DINOv3 encoder, and its preprocessing.

The octree and elastic decoders are instantiated STANDALONE (not
under the combined OctreeGaussianDecoder) so their state-dict keys
carry no ``octree.`` / ``gs.`` prefix; the hash fill keys on the key
name, and the replay tests instantiate the Dinkster submodules standalone
the same way.

Determinism: attention is forced to the pytorch SDPA backend in both
the triposplat module (whose ``attention`` helper the octree decoder
shares) and the DINOv3 module. Every random draw in the octree
descent goes through an explicitly seeded CPU generator.

Usage (a torch interpreter; comfy.ops imports comfy_aimdo, so the
sibling comfy-aimdo checkout must be on PYTHONPATH when the venv
lacks it):

    /path/to/torch-venv/bin/python tools/gen_triposplat_goldens.py

Run from the Dinkster repo root with ../ComfyUI at the recorded baseline
commit (recorded in the output for provenance).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COMFY_ROOT = (REPO.parent / "ComfyUI").resolve()
#: The audited reference baseline. Generation REFUSES on any other
#: commit so the recorded provenance is a guarantee, not a claim.
REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"

# In front of any ambient PYTHONPATH: the reference must come from the
# pinned sibling checkout, not an installed or stray comfy package.
sys.path.insert(0, str(REPO / "packages" / "dinkster-inference-torch" / "tests"))
sys.path.insert(0, str(COMFY_ROOT))

# comfy.model_management probes CUDA at import; goldens execute on
# CPU float32 either way, so force ComfyUI's CPU state and keep the
# generator runnable from a CPU-only torch venv.
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import torch  # noqa: E402

#: Bit-stability holds only for one interpreter: generation REFUSES
#: any other torch build so a regeneration cannot silently rotate the
#: payload hash through CPU-kernel drift.
GENERATOR_TORCH = "2.13.0+cpu"
if torch.__version__ != GENERATOR_TORCH:
    raise SystemExit(
        f"goldens are pinned to torch {GENERATOR_TORCH}; this interpreter has"
        f" {torch.__version__}. Regenerating on another build rotates the payload"
        " hash - update the pin deliberately and re-prove bit-stability."
    )

from comfy import ops  # noqa: E402
from comfy.clip_model import clip_preprocess  # noqa: E402
from comfy.image_encoders import dino3 as _dino3  # noqa: E402
from comfy.ldm.modules import attention as _attention  # noqa: E402
from comfy.ldm.triposplat import gaussian as _gaussian  # noqa: E402
from comfy.ldm.triposplat import model as _triposplat_model  # noqa: E402
from comfy.ldm.triposplat import vae as _triposplat_vae  # noqa: E402
from unet_fill import fill_state_dict, hashed_input  # noqa: E402

# Force the pytorch SDPA backend (the one Dinkster ports). The flow
# model's attention helper reads optimized_attention from its module
# globals at call time; the octree decoder imports that helper, so one
# patch covers both. DINOv3 resolves its backend per device at call
# time; pin it the same way.
_triposplat_model.optimized_attention = _attention.attention_pytorch
_dino3.optimized_attention_for_device = lambda device, mask=False: _attention.attention_pytorch
assert _triposplat_model.attention.__globals__["optimized_attention"] is (
    _attention.attention_pytorch
), "the triposplat modules are not calling the forced pytorch SDPA backend"
assert _triposplat_vae.attention is _triposplat_model.attention
ATTENTION_BACKEND = "attention_pytorch"

OUT = (
    REPO / "packages" / "dinkster-inference-torch" / "tests" / "goldens" / "triposplat_goldens.json"
)

#: Tiny flow model: head_dim 12 splits into rope axes 4 / 4 / 4, the
#: smallest even three-way split; repo_hidden_size falls out of the
#: reference's int(model_channels * 0.125) = 3.
_TINY_DIT = {
    "q_token_length": 32,
    "in_channels": 4,
    "out_channels": 4,
    "model_channels": 24,
    "cond_channels": 16,
    "cond2_channels": 8,
    "num_blocks": 2,
    "num_refiner_blocks": 1,
    "num_heads": None,
    "num_head_channels": 12,
    "cam_channels": 5,
    "mlp_ratio": 4,
    "share_mod": True,
    "qk_rms_norm": True,
}

#: Tiny octree probability decoder.
_TINY_OCTREE = {
    "model_channels": 32,
    "cond_channels": 6,
    "num_blocks": 2,
    "num_heads": 2,
    "num_head_channels": 16,
    "mlp_ratio": 4.0,
    "share_mod": True,
}

#: Tiny elastic gaussian decoder: 8 gaussians per point pack into
#: 120 feature channels.
_TINY_ELASTIC_REPRESENTATION = {
    "lr": {"_xyz": 1.0, "_features_dc": 1.0, "_opacity": 1.0, "_scaling": 1.0, "_rotation": 0.1},
    "perturb_offset": True,
    "perturbe_size": 1.5,
    "offset_scale": 0.05,
    "num_gaussians": 8,
    "filter_kernel_size_3d": 0.0009,
    "scaling_bias": 0.004,
    "opacity_bias": 0.1,
    "scaling_activation": "softplus",
}
_TINY_ELASTIC = {
    "in_channels": 3,
    "model_channels": 32,
    "cond_channels": 6,
    "num_blocks": 2,
    "num_heads": 2,
    "num_head_channels": 16,
    "mlp_ratio": 4.0,
}

#: Tiny DINOv3: hidden 32 over 2 heads gives head_dim 16 (divisible
#: by four for the patch rope); image 16 at patch 4 gives 16 patch
#: tokens behind the class token and 2 registers.
_TINY_DINOV3 = {
    "model_type": "dinov3",
    "num_hidden_layers": 2,
    "hidden_size": 32,
    "num_attention_heads": 2,
    "num_register_tokens": 2,
    "intermediate_size": 64,
    "layer_norm_eps": 1e-5,
    "num_channels": 3,
    "patch_size": 4,
    "rope_theta": 100.0,
    "use_gated_mlp": True,
    "gated_mlp_act": "silu",
    "image_size": 16,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
}


def enc(x: torch.Tensor) -> dict:
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype).removeprefix("torch."),
        "data": x.to(torch.float32).flatten().tolist(),
    }


def entries_of(module: torch.nn.Module) -> list[tuple[str, list[int]]]:
    return sorted((key, list(value.shape)) for key, value in module.state_dict().items())


def hash_fill(module: torch.nn.Module) -> list[tuple[str, list[int]]]:
    entries = entries_of(module)
    module.load_state_dict(fill_state_dict(entries), strict=True)
    return entries


def coords_input(key: str, shape: tuple[int, ...]) -> torch.Tensor:
    """Deterministic octree-style coordinates inside the unit cube."""
    return hashed_input(key, shape) * 0.25 + 0.5


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
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=COMFY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            f"{COMFY_ROOT} has uncommitted changes; a clean checkout"
            f" of {REFERENCE_COMMIT} is required:\n{dirty}"
        )
    for module in (_triposplat_model, _triposplat_vae, _gaussian, _dino3):
        module_file = Path(module.__file__ or "").resolve()
        if not module_file.is_relative_to(COMFY_ROOT):
            raise SystemExit(
                f"the reference module {module.__name__} was imported from"
                f" {module_file}, not the pinned checkout {COMFY_ROOT}"
            )

    op_kwargs = {"dtype": torch.float32, "operations": ops.disable_weight_init}
    payload: dict = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
            "attention": ATTENTION_BACKEND,
        },
        "layouts": {},
        "cases": {},
    }

    payload["layouts"]["triposplat_dit"] = entries_of(
        _triposplat_model.LatentSeqMMFlowModel(device="meta", **op_kwargs)
    )
    payload["layouts"]["octree_gaussian_decoder"] = entries_of(
        _triposplat_vae.OctreeGaussianDecoder(device="meta", **op_kwargs)
    )
    payload["layouts"]["dinov3_vith"] = entries_of(
        _dino3.DINOv3ViTModel(
            _dino3.DINOV3_VITH_CONFIG, torch.float32, "meta", ops.disable_weight_init
        )
    )

    # Flow model: one case with the reference-image latent leg, one
    # without. Hooks pin the first refiner blocks and the first joint
    # block alongside the final outputs.
    for name, use_reference in (("dit_reference_latent", True), ("dit_context_only", False)):
        model = _triposplat_model.LatentSeqMMFlowModel(device="cpu", **op_kwargs, **_TINY_DIT)
        entries = hash_fill(model)
        batch, rows = 2, 10
        latent = hashed_input(f"{name}:latent", (batch, _TINY_DIT["q_token_length"], 4))
        camera = hashed_input(f"{name}:camera", (batch, 1, _TINY_DIT["cam_channels"]))
        timesteps = torch.linspace(0.05, 0.95, batch, dtype=torch.float32)
        context = hashed_input(f"{name}:context", (batch, rows, _TINY_DIT["cond_channels"]))
        reference_latent = None
        if use_reference:
            reference_latent = hashed_input(
                f"{name}:reference_latent", (batch, _TINY_DIT["cond2_channels"], 2, 2)
            )
        intermediates: dict[str, object] = {}
        hooks = [
            model.noise_refiner[0].register_forward_hook(
                lambda _m, _i, output, record=intermediates: record.update(
                    noise_refiner=enc(output)
                )
            ),
            model.context_refiner[0].register_forward_hook(
                lambda _m, _i, output, record=intermediates: record.update(
                    context_refiner=enc(output)
                )
            ),
            model.blocks[0].register_forward_hook(
                lambda _m, _i, output, record=intermediates: record.update(block=enc(output))
            ),
        ]
        try:
            with torch.no_grad():
                out_latent, out_camera = model._forward(
                    [latent, camera],
                    timesteps,
                    context=context,
                    ref_latents=None if reference_latent is None else [reference_latent],
                )
        finally:
            for hook in hooks:
                hook.remove()
        payload["cases"][name] = {
            "config": _TINY_DIT,
            "state_dict": entries,
            "batch": batch,
            "context_rows": rows,
            "timesteps": timesteps.tolist(),
            "use_reference_latent": use_reference,
            "intermediates": intermediates,
            "latent_output": enc(out_latent),
            "camera_output": enc(out_camera),
        }

    # Octree probability decoder: raw logits, then a seeded descent.
    octree = _triposplat_vae.OctreeProbabilityFixedlenDecoder(
        device="cpu", **op_kwargs, **_TINY_OCTREE
    )
    octree_entries = hash_fill(octree)
    points = coords_input("octree_logits:points", (2, 5, 3))
    levels = torch.tensor([2, 4], dtype=torch.long)
    cond = hashed_input("octree_logits:cond", (2, 7, _TINY_OCTREE["cond_channels"]))
    with torch.no_grad():
        logits = octree(points, levels, cond)["logits"]
    payload["cases"]["octree_logits"] = {
        "config": _TINY_OCTREE,
        "state_dict": octree_entries,
        "levels": levels.tolist(),
        "cond_rows": 7,
        "point_rows": 5,
        "batch": 2,
        "logits": enc(logits),
    }

    sample_cond = hashed_input("octree_sample:cond", (2, 7, _TINY_OCTREE["cond_channels"]))
    sample_seed, sample_points, sample_level = 7, 16, 4
    with torch.no_grad():
        sampled = _triposplat_vae.OctreeProbabilityFixedlenDecoder.sample(
            octree,
            sample_cond,
            num_points=sample_points,
            level=sample_level,
            temperature=1.0,
            generator=torch.Generator().manual_seed(sample_seed),
        )
    payload["cases"]["octree_sample"] = {
        "config": _TINY_OCTREE,
        "state_dict": octree_entries,
        "cond_rows": 7,
        "batch": 2,
        "seed": sample_seed,
        "num_points": sample_points,
        "level": sample_level,
        "points": enc(sampled["points"]),
        "log_probs": enc(sampled["log_probs"]),
    }

    # Elastic gaussian decoder: packed features and the offset port.
    elastic = _triposplat_vae.ElasticGaussianFixedlenDecoder(
        device="cpu",
        representation_config=_TINY_ELASTIC_REPRESENTATION,
        **op_kwargs,
        **_TINY_ELASTIC,
    )
    elastic_entries = hash_fill(elastic)
    elastic_points = coords_input("elastic_features:points", (2, 6, 3))
    elastic_cond = hashed_input("elastic_features:cond", (2, 7, _TINY_ELASTIC["cond_channels"]))
    with torch.no_grad():
        features = elastic(x={"points": elastic_points}, cond=elastic_cond)["features"]
        offsets = elastic._get_offset(features)
    payload["cases"]["elastic_features"] = {
        "config": _TINY_ELASTIC,
        "representation": _TINY_ELASTIC_REPRESENTATION,
        "state_dict": elastic_entries,
        "cond_rows": 7,
        "point_rows": 6,
        "batch": 2,
        "features": enc(features),
        "offsets": enc(offsets),
    }

    # Seeded end-to-end decode: octree descent, elastic features, and
    # activation into render-ready splat tensors.
    decode_cond = hashed_input("splat_decode:cond", (2, 7, _TINY_OCTREE["cond_channels"]))
    decode_seed, decode_gaussians, decode_level = 42, 64, 3
    with torch.no_grad():
        generator = torch.Generator().manual_seed(decode_seed)
        tokens = max(1, decode_gaussians // _TINY_ELASTIC_REPRESENTATION["num_gaussians"])
        points_pred = _triposplat_vae.OctreeProbabilityFixedlenDecoder.sample(
            octree,
            decode_cond,
            num_points=tokens,
            level=decode_level,
            temperature=1.0,
            generator=generator,
        )
        pred = elastic(x=points_pred, cond=decode_cond)
        models = _gaussian.build_gaussian_models(elastic, points_pred, pred)
    splats = []
    for gaussian_model in models:
        positions, scales, rotations, opacities, sh = gaussian_model.render_tensors()
        splats.append(
            {
                "positions": enc(positions),
                "scales": enc(scales),
                "rotations": enc(rotations),
                "opacities": enc(opacities),
                "sh": enc(sh),
            }
        )
    payload["cases"]["splat_decode"] = {
        "octree_config": _TINY_OCTREE,
        "octree_state_dict": octree_entries,
        "elastic_config": _TINY_ELASTIC,
        "representation": _TINY_ELASTIC_REPRESENTATION,
        "elastic_state_dict": elastic_entries,
        "cond_rows": 7,
        "batch": 2,
        "seed": decode_seed,
        "num_gaussians": decode_gaussians,
        "level": decode_level,
        "splats": splats,
    }

    # DINOv3: the encoded sequence with its pooled class token, and
    # the preprocessing leg over a non-square image (resize + crop).
    vit = _dino3.DINOv3ViTModel(_TINY_DINOV3, torch.float32, "cpu", ops.disable_weight_init)
    vit_entries = hash_fill(vit)
    pixel_values = hashed_input("dinov3_encode:pixels", (2, 3, 16, 16))
    with torch.no_grad():
        sequence, _, pooled, _ = vit(pixel_values)
    payload["cases"]["dinov3_encode"] = {
        "config": _TINY_DINOV3,
        "state_dict": vit_entries,
        "batch": 2,
        "sequence": enc(sequence),
        "pooled": enc(pooled),
    }

    image = (hashed_input("dinov3_preprocess:image", (1, 20, 24, 3)) + 1.0).clamp(0, 2) * 0.5
    with torch.no_grad():
        preprocessed = clip_preprocess(
            image,
            size=_TINY_DINOV3["image_size"],
            mean=_TINY_DINOV3["image_mean"],
            std=_TINY_DINOV3["image_std"],
            crop=True,
        )
    payload["cases"]["dinov3_preprocess"] = {
        "config": _TINY_DINOV3,
        "image_shape": list(image.shape),
        "output": enc(preprocessed),
    }

    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
