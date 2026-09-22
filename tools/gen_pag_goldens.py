"""Generate PerturbedAttentionGuidance goldens from the ComfyUI reference.

Usage:
    COMFYUI_REFERENCE=/path/to/ComfyUI-goldenref \
      /path/to/torch-venv/bin/python tools/gen_pag_goldens.py

The reference may instead be supplied with ``--comfy-root``. The checkout
must be clean and exactly at the audited commit. Every case drives real
reference code end to end: the node's own
``comfy_extras.nodes_pag.PerturbedAttentionGuidance.execute`` registers the
post-CFG callback on a cloned ModelPatcher, and ``comfy_sample.sample``
runs ``comfy.samplers.sampling_function`` -> ``calc_cond_batch`` -> the
tiny UNet forward, with the callback's auxiliary conditional pass
evaluating the model under the reference ``attn1`` middle-block-0
replacement. This script never re-implements the PAG math; it only
observes the reference passes through a ``calc_cond_batch`` wrapper and
records what they produced.

Each case uses the tiny SD15-style UNet behind a CPU ModelPatcher (the
CheckpointLoader product shape) with deterministic weights from
``unet_fill.fill_state_dict`` and hashed inputs. Cases record the exact
sigmas, the first model input/timestep seam, the plain and perturbed
predictions, and the baseline and PAG outputs, and prove at generation
time that the active run differs from the baseline while the scale 0
run is bit-identical to it (the reference callback returns the CFG
result unchanged, without running its auxiliary pass, at scale 0).

The executed values drift by ULPs across CPU microarchitectures, so the
fixture records the mint host CPU (pin_cpu) and the replay suite skips on
other hosts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_platform import platform_golden_path, tuple_provenance  # noqa: E402

REFERENCE_COMMIT = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "packages/dinkster-inference-torch/tests/goldens/pag_goldens.json"


def reference_root() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path)
    args, remaining = parser.parse_known_args()
    sys.argv[1:] = remaining
    value = args.comfy_root or os.environ.get("COMFYUI_REFERENCE")
    if value is None:
        raise SystemExit("pass --comfy-root or set COMFYUI_REFERENCE")
    return Path(value).resolve()


COMFY_ROOT = reference_root()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=COMFY_ROOT, text=True, capture_output=True, check=True
    ).stdout.strip()


commit = git_output("rev-parse", "HEAD")
dirty = git_output("status", "--porcelain")
if commit != REFERENCE_COMMIT:
    raise SystemExit(f"reference is at {commit}; required {REFERENCE_COMMIT}")
if dirty:
    raise SystemExit(f"reference checkout must be clean:\n{dirty}")

sys.path.insert(0, str(REPO / "packages" / "dinkster-inference-torch" / "tests"))
sys.path.insert(0, str(COMFY_ROOT))

import torch  # noqa: E402
from comfy.cli_args import args  # noqa: E402

args.cpu = True

from comfy import (  # noqa: E402
    model_base,
    model_patcher,
    ops,
    samplers,
    supported_models,
)
from comfy import sample as comfy_sample  # noqa: E402
from comfy.ldm.modules import attention as _attention  # noqa: E402
from comfy_extras import nodes_pag  # noqa: E402
from unet_fill import fill_state_dict, hashed_input  # noqa: E402

# The reference CrossAttention calls the ambient optimized_attention,
# selected per environment. Force the pytorch SDPA backend Dinkster ports.
_attention.optimized_attention = _attention.attention_pytorch
_attention.optimized_attention_masked = _attention.attention_pytorch
ATTENTION_BACKEND = "attention_pytorch"

CPU = torch.device("cpu")

#: Tiny SD1-style geometry: one transformer block in each input stage and
#: the middle block, so the PAG middle-block-0 attn1 replacement has a
#: real site to perturb. The keys ride supported_models.SD15 ->
#: model_base.BaseModel like a CheckpointLoader product.
TINY_SD1 = {
    "use_checkpoint": False,
    "image_size": 32,
    "use_spatial_transformer": True,
    "legacy": False,
    "use_temporal_attention": False,
    "use_temporal_resblock": False,
    "in_channels": 4,
    "out_channels": 4,
    "model_channels": 32,
    "num_res_blocks": [1, 1],
    "channel_mult": [1, 2],
    "transformer_depth": [1, 1],
    "transformer_depth_output": [1, 1, 1, 1],
    "transformer_depth_middle": 1,
    "context_dim": 16,
    "use_linear_in_transformer": False,
    "adm_in_channels": None,
    "num_heads": 8,
    "num_head_channels": -1,
}

STEPS = 1
SAMPLER = "euler"
SCHEDULER = "normal"
DENOISE = 1.0
SEED = 113
CFG = 7.5
HEIGHT, WIDTH, CONTEXT_LEN = 16, 16, 7


def build_diffusion() -> model_patcher.ModelPatcher:
    arch = dict(TINY_SD1)
    arch["dtype"] = torch.float32
    config = supported_models.SD15(arch)
    # BASE.__init__ overlays the class's unet_extra_config head facts over
    # the passed dict; the tiny geometry's 8 heads already match SD15's,
    # but keep them pinned explicitly like the other pipeline generators.
    config.unet_config["num_heads"] = arch["num_heads"]
    config.unet_config["num_head_channels"] = arch["num_head_channels"]
    config.custom_operations = ops.disable_weight_init
    model = model_base.BaseModel(config, device=CPU)
    entries = sorted(
        (key, list(value.shape)) for key, value in model.diffusion_model.state_dict().items()
    )
    model.diffusion_model.load_state_dict(fill_state_dict(entries), strict=True)
    return model_patcher.ModelPatcher(model, load_device=CPU, offload_device=CPU)


class _CondBatchObserver:
    """Records every reference calc_cond_batch result by cond count.

    The post-CFG callback the node registers resolves
    ``comfy.samplers.calc_cond_batch`` at call time, so wrapping the
    module attribute observes both the main cond+uncond pass and the
    callback's single-conditional PAG pass without touching their code.
    Each record keeps the batched model input and timestep so the
    fixture pins the exact seam the model forward saw.
    """

    def __init__(self) -> None:
        self.records: list[tuple[int, list[torch.Tensor], torch.Tensor, torch.Tensor]] = []
        self._original = samplers.calc_cond_batch

    def install(self) -> None:
        samplers.calc_cond_batch = self._observe  # type: ignore[assignment]

    def uninstall(self) -> None:
        samplers.calc_cond_batch = self._original  # type: ignore[assignment]

    def _observe(self, model, conds, x_in, timestep, model_options):
        out = self._original(model, conds, x_in, timestep, model_options)
        self.records.append(
            (
                len(conds),
                [tensor.detach().clone() for tensor in out],
                x_in.detach().clone(),
                timestep.detach().clone(),
            )
        )
        return out

    def by_cond_count(
        self, count: int
    ) -> list[tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]]:
        return [
            (tensors, x_in, timestep)
            for size, tensors, x_in, timestep in self.records
            if size == count
        ]


def enc(value: torch.Tensor) -> dict[str, object]:
    return {"shape": list(value.shape), "dtype": "float32", "data": value.flatten().tolist()}


def run_case(name: str, pag_scale: float) -> dict[str, Any]:
    latent = hashed_input(f"{name}:latent", (1, TINY_SD1["in_channels"], HEIGHT, WIDTH))
    positive_context = hashed_input(f"{name}:positive", (1, CONTEXT_LEN, TINY_SD1["context_dim"]))
    negative_context = hashed_input(f"{name}:negative", (1, CONTEXT_LEN, TINY_SD1["context_dim"]))
    positive = [[positive_context, {}]]
    negative = [[negative_context, {}]]
    noise = comfy_sample.prepare_noise(latent, SEED)

    def sample(patcher: model_patcher.ModelPatcher, observer: _CondBatchObserver) -> torch.Tensor:
        observer.install()
        try:
            with torch.no_grad():
                return comfy_sample.sample(
                    patcher,
                    noise,
                    STEPS,
                    CFG,
                    SAMPLER,
                    SCHEDULER,
                    positive,
                    negative,
                    latent,
                    denoise=DENOISE,
                    disable_pbar=True,
                    seed=SEED,
                )
        finally:
            observer.uninstall()

    base = build_diffusion()
    result = nodes_pag.PerturbedAttentionGuidance.execute(base, pag_scale)
    patched = result.result[0]
    if len(patched.model_options.get("sampler_post_cfg_function", [])) != 1:
        raise SystemExit(f"case {name}: node did not register exactly one post-CFG callback")

    baseline_observer = _CondBatchObserver()
    baseline = sample(base, baseline_observer)
    if len(baseline_observer.by_cond_count(2)) != 1:
        raise SystemExit(
            f"case {name}: expected one baseline cond+uncond pass, saw {baseline_observer.records}"
        )

    active_observer = _CondBatchObserver()
    active = sample(patched, active_observer)

    main_passes = active_observer.by_cond_count(2)
    auxiliary_passes = active_observer.by_cond_count(1)
    if len(main_passes) != 1 or len(main_passes[0][0]) != 2:
        raise SystemExit(
            f"case {name}: expected one cond+uncond pass, saw {active_observer.records}"
        )
    (cond_pred, uncond_pred), first_input, first_timestep = main_passes[0]

    case: dict[str, Any] = {
        "params": {
            "pag_scale": pag_scale,
            "cfg": CFG,
            "seed": SEED,
            "steps": STEPS,
            "sampler": SAMPLER,
            "scheduler": SCHEDULER,
            "denoise": DENOISE,
        },
        # The exact schedule the run walked (KSampler.set_steps: scheduler
        # + discard-penultimate + denoise trim), so replay reconstructs
        # the sigmas without a reference checkout.
        "sigmas": [
            float(sigma)
            for sigma in samplers.KSampler(
                base,
                steps=STEPS,
                device=CPU,
                sampler=SAMPLER,
                scheduler=SCHEDULER,
                denoise=DENOISE,
            ).sigmas
        ],
        "config": TINY_SD1,
        "state_dict": sorted(
            (key, list(value.shape))
            for key, value in base.model.diffusion_model.state_dict().items()
        ),
        "latent_image": enc(latent),
        "positive_context": enc(positive_context),
        "negative_context": enc(negative_context),
        "first_pass_input": enc(first_input),
        "first_pass_timestep": enc(first_timestep),
        "prediction_cond": enc(cond_pred),
        "prediction_uncond": enc(uncond_pred),
        "output_baseline": enc(baseline),
        "output": enc(active),
    }

    if pag_scale == 0:
        # The reference callback returns the CFG result unchanged at
        # scale 0 and never runs the auxiliary pass.
        if auxiliary_passes:
            raise SystemExit(f"case {name}: scale 0 still ran the PAG auxiliary pass")
    else:
        if len(auxiliary_passes) != 1 or len(auxiliary_passes[0][0]) != 1:
            raise SystemExit(
                f"case {name}: expected exactly one PAG auxiliary pass,"
                f" saw {active_observer.records}"
            )
        (pag_pred,), pag_input, pag_timestep = auxiliary_passes[0]
        # The auxiliary pass reruns the conditional under the reference
        # attn1 middle-block-0 replacement; if the replacement did not
        # engage, the pass would reproduce the conditional prediction
        # bit for bit.
        if torch.equal(cond_pred, pag_pred):
            raise SystemExit(f"case {name}: the PAG auxiliary pass did not perturb attention")
        case["pag_pass_input"] = enc(pag_input)
        case["pag_pass_timestep"] = enc(pag_timestep)
        case["prediction_pag"] = enc(pag_pred)

        if torch.equal(baseline, active):
            raise SystemExit(f"case {name}: the PAG output did not change the sample")

    if pag_scale == 0:
        zero_observer = _CondBatchObserver()
        zero = sample(patched, zero_observer)
        if zero_observer.by_cond_count(1):
            raise SystemExit(f"case {name}: scale 0 rerun still ran the PAG auxiliary pass")
        if not torch.equal(zero, baseline):
            raise SystemExit(f"case {name}: scale 0 output is not bit-identical to the baseline")
        case["output_scale_zero"] = enc(zero)

    return case


def dependency_provenance() -> dict[str, str]:
    """Versions (or source paths) of the ComfyUI companion packages the
    run actually imported: the pinned reference's ops and attention layers
    pull these in, so their identities belong in the fixture provenance."""
    deps: dict[str, str] = {}
    for distribution, module in (
        ("comfy-aimdo", "comfy_aimdo"),
        ("comfy-kitchen", "comfy_kitchen"),
    ):
        imported = importlib.import_module(module)
        try:
            deps[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            deps[distribution] = str(Path(imported.__file__ or "").resolve())
    return deps


def main() -> None:
    path = Path(nodes_pag.__file__ or "").resolve()
    if not path.is_relative_to(COMFY_ROOT):
        raise SystemExit(f"{nodes_pag.__name__} was imported from {path}, not {COMFY_ROOT}")

    cases = {
        "pag_default_scale": run_case("pag_default_scale", 3.0),
        "pag_three_quarter_scale": run_case("pag_three_quarter_scale", 0.75),
        "pag_scale_zero": run_case("pag_scale_zero", 0.0),
    }

    payload: dict[str, Any] = {
        "reference": {
            "repo": "ComfyUI",
            "commit": commit,
            "torch": torch.__version__,
            "attention": ATTENTION_BACKEND,
        },
        "dependencies": dependency_provenance(),
        "cases": cases,
    }
    provenance = tuple_provenance(str(torch.__version__), pin_cpu=True)
    if provenance:
        payload["_meta"] = provenance
    out = platform_golden_path(OUT, str(torch.__version__))
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    out.write_bytes(content)
    print(f"{out}: sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    main()
