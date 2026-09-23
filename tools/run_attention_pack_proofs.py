"""Run real-image CUDA proofs for the issue 284 attention packs.

The workload and thresholds are fixed in attention_pack_proof_config.json.
This runner verifies every artifact before loading it, executes each pack from
its own directory, saves the evaluated images, and writes the measured metrics
and environment facts as JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import runpy
from pathlib import Path
from typing import Any, cast

import torch
from dinkster_inference import (
    DualSamplingGuidance,
    GuidanceCondition,
    GuidanceRole,
    SamplingGuidance,
    builtin_schedulers,
    load_safetensors_header,
    sampling_execution_context,
    sampling_sigmas,
    use_sampling_environment,
)
from dinkster_inference_torch import (
    FluxDenoiser,
    FluxRuntime,
    GuidanceExecutor,
    GuidanceRegistry,
    SDDenoiser,
    SDRuntime,
    load_runtime,
)
from dinkster_inference_torch.attention_extensions import AttentionExecution
from PIL import Image
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPooling

REPO = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).with_name("attention_pack_proof_config.json")
PACKS = REPO / "tests" / "fixtures"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def proof_pack(name: str) -> Any:
    module = next((PACKS / name).glob("*.py"))
    return runpy.run_path(str(module))["make"]


def executor(owner: str, contribution: Any) -> GuidanceExecutor:
    guidance = () if contribution.guidance is None else ((owner, contribution.guidance),)
    return GuidanceExecutor(
        GuidanceRegistry(
            guidance,
            attention_contributions=((owner, contribution.attention),),
        )
    )


def configured_runtime(runtime: Any, owner: str, contribution: Any) -> Any:
    return type(runtime)(
        runtime.assembled,
        runtime_identity=f"{runtime.runtime_identity}:{owner}",
        guidance_executor=executor(owner, contribution),
    )


def move_text(runtime: SDRuntime | FluxRuntime, device: str) -> None:
    assembled = runtime.assembled
    for name in ("clip_l", "clip_g", "t5xxl"):
        module = getattr(assembled, name, None)
        if module is not None:
            module.to(device)


def sample(
    runtime: SDRuntime | FluxRuntime,
    latent: torch.Tensor,
    positive: Any,
    negative: Any,
    *,
    scale: float,
    config: dict[str, Any],
    owner: str | None = None,
    seed: int | None = None,
    middle: Any | None = None,
) -> torch.Tensor:
    environment = () if owner is None else (owner,)
    guidance = (
        SamplingGuidance(negative, scale)
        if middle is None
        else DualSamplingGuidance(middle, negative, 1.0, scale)
    )
    with use_sampling_environment(environment, lambda: False), torch.inference_mode():
        return runtime.sample(
            latent,
            cond=positive,
            cfg=guidance,
            sampler_id=config["workload"]["sampler"],
            scheduler_id="dinkster.normal",
            steps=config["workload"]["steps"],
            seed=config["workload"]["seed"] if seed is None else seed,
            device="cuda:0",
        )


def capture_reference_state(
    runtime: SDRuntime | FluxRuntime,
    reference_latent: torch.Tensor,
    positive: Any,
    negative: Any,
    config: dict[str, Any],
) -> dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]:
    contribution = proof_pack("reference-attention-pack")(
        strength=config["reference_attention"]["strength"]
    )
    registry = GuidanceRegistry(
        attention_contributions=(("proof_reference", contribution.attention),)
    )
    schedule = sampling_sigmas(
        next(item for item in builtin_schedulers() if item.id == "dinkster.normal"),
        runtime.sampling_sigma_space(),
        config["workload"]["steps"],
    )
    lanes = (
        GuidanceCondition("positive", GuidanceRole.CONDITIONAL, positive),
        GuidanceCondition("negative", GuidanceRole.UNCONDITIONAL, negative),
    )
    with use_sampling_environment(("proof_reference",), lambda: False):
        execution = sampling_execution_context(schedule, config["workload"]["seed"])
    active = AttentionExecution(registry.attention_extensions, execution, lanes, 1)
    sigma = float(schedule[0])
    if isinstance(runtime, SDRuntime):
        evaluator = SDDenoiser(
            runtime.assembled.diffusion,
            runtime.sampling_sigma_space(),
            compute_dtype=torch.float16,
        )
        conditions = (
            evaluator.prepare_conditioning(positive, lane_id="positive"),
            evaluator.prepare_conditioning(negative, lane_id="negative"),
        )
    else:
        evaluator = FluxDenoiser(runtime.assembled.diffusion, compute_dtype=torch.bfloat16)
        conditions = (
            evaluator.prepare_conditioning(positive),
            evaluator.prepare_conditioning(negative),
        )
    with torch.inference_mode():
        evaluator.evaluate_conditioning_batch(reference_latent, sigma, conditions, active)
    state = execution.extension_state["proof_reference"].get("reference")
    if type(state) is not dict or not state:
        raise RuntimeError("reference pack captured no attention state")
    return cast("dict[tuple[str, str, str], tuple[torch.Tensor, torch.Tensor]]", state)


def decode(runtime: SDRuntime | FluxRuntime, latent: torch.Tensor) -> torch.Tensor:
    runtime.assembled.diffusion.to("cpu")
    torch.cuda.empty_cache()
    runtime.assembled.vae.to("cuda:0")
    with torch.inference_mode():
        image = runtime.decode_latent(latent.to("cuda:0")).float().cpu()
    runtime.assembled.vae.to("cpu")
    torch.cuda.empty_cache()
    return image.clamp(0, 1)


def as_pil(image: torch.Tensor) -> Image.Image:
    value = image[0]
    if value.shape[0] in (3, 4):
        value = value[:3].permute(1, 2, 0)
    return Image.fromarray((value.clamp(0, 1) * 255).round().to(torch.uint8).numpy())


def clip_feature_tensor(value: object, label: str) -> torch.Tensor:
    if isinstance(value, BaseModelOutputWithPooling):
        value = value.pooler_output
    if type(value) is not torch.Tensor:
        raise RuntimeError(f"CLIP {label} features did not contain a tensor")
    return value


def clip_metrics(
    images: dict[str, torch.Tensor],
    config: dict[str, Any],
    clip_root: Path,
) -> dict[str, float]:
    model = CLIPModel.from_pretrained(clip_root, local_files_only=True).to("cuda:0").eval()
    tokenizer = CLIPTokenizer.from_pretrained(clip_root, local_files_only=True)
    processor = CLIPImageProcessor.from_pretrained(clip_root, local_files_only=True)
    coupled = as_pil(images["coupled"])
    width, height = coupled.size
    inputs = [
        coupled.crop((0, 0, width, height // 2)),
        coupled.crop((0, height // 2, width, height)),
        as_pil(images["reference"]),
        as_pil(images["baseline"]),
        as_pil(images["injected"]),
    ]
    pixels = processor(images=inputs, return_tensors="pt")["pixel_values"].to("cuda:0")
    prompts = [
        config["attention_couple"]["top_prompt"],
        config["attention_couple"]["bottom_prompt"],
    ]
    tokens = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        image_features = clip_feature_tensor(model.get_image_features(pixel_values=pixels), "image")
        text_features = clip_feature_tensor(model.get_text_features(**tokens), "text")
    image_features = torch.nn.functional.normalize(image_features, dim=-1)
    text_features = torch.nn.functional.normalize(text_features, dim=-1)
    scores = image_features[:2] @ text_features.T
    reference = image_features[2]
    baseline_similarity = float(reference @ image_features[3])
    injected_similarity = float(reference @ image_features[4])
    result = {
        "top_own": float(scores[0, 0]),
        "top_other": float(scores[0, 1]),
        "top_margin": float(scores[0, 0] - scores[0, 1]),
        "bottom_own": float(scores[1, 1]),
        "bottom_other": float(scores[1, 0]),
        "bottom_margin": float(scores[1, 1] - scores[1, 0]),
        "reference_baseline_cosine": baseline_similarity,
        "reference_injected_cosine": injected_similarity,
        "reference_cosine_gain": injected_similarity - baseline_similarity,
    }
    margin = config["attention_couple"]["minimum_clip_cosine_margin"]
    gain = config["reference_attention"]["minimum_clip_image_cosine_gain"]
    if result["top_margin"] < margin or result["bottom_margin"] < margin:
        raise AssertionError(f"attention-couple CLIP margins failed: {result}")
    if result["reference_cosine_gain"] < gain:
        raise AssertionError(f"reference-attention CLIP gain failed: {result}")
    model.to("cpu")
    torch.cuda.empty_cache()
    return result


def run_family(
    family: str,
    artifact: Path,
    config: dict[str, Any],
    clip_root: Path,
    output: Path,
) -> dict[str, Any]:
    source = load_safetensors_header(artifact)
    runtime = load_runtime(source, fp8_matmul=family == "flux")
    if family == "sd15" and not isinstance(runtime, SDRuntime):
        raise RuntimeError("SD 1.5 artifact did not load an SDRuntime")
    if family == "flux" and not isinstance(runtime, FluxRuntime):
        raise RuntimeError("Flux artifact did not load a FluxRuntime")
    runtime = cast("SDRuntime | FluxRuntime", runtime)
    move_text(runtime, "cuda:0")
    couple = config["attention_couple"]
    reference_config = config["reference_attention"]
    with torch.inference_mode():
        top = runtime.encode_text(couple["top_prompt"])
        bottom = runtime.encode_text(couple["bottom_prompt"])
        target = runtime.encode_text(reference_config["prompt"])
        reference_prompt = runtime.encode_text(reference_config["reference_prompt"])
        empty = runtime.encode_text("")
    move_text(runtime, "cpu")
    torch.cuda.empty_cache()
    runtime.assembled.diffusion.to("cuda:0")
    height = config["workload"]["height"]
    width = config["workload"]["width"]
    channels = runtime.family.single_stream_latent().channels
    downscale = runtime.family.single_stream_latent().spatial_downscale
    latent = torch.zeros(1, channels, height // downscale, width // downscale)

    reference_latent = sample(
        runtime,
        latent,
        reference_prompt,
        empty,
        scale=7.0,
        config=config,
        seed=config["workload"]["seed"] + reference_config["reference_seed_offset"],
    )
    baseline_latent = sample(runtime, latent, target, empty, scale=7.0, config=config)
    captured = capture_reference_state(runtime, reference_latent, target, empty, config)
    reference_contribution = proof_pack("reference-attention-pack")(
        strength=reference_config["strength"],
        reference_state=captured,
    )
    captured.clear()
    reference_runtime = configured_runtime(runtime, "proof_reference", reference_contribution)
    injected_latent = sample(
        reference_runtime,
        latent,
        target,
        empty,
        scale=7.0,
        config=config,
        owner="proof_reference",
    )
    couple_contribution = proof_pack("attention-couple-pack")(
        split=0.5,
        attention_strength=couple["attention_strength"],
    )
    couple_runtime = configured_runtime(runtime, "proof_couple", couple_contribution)
    coupled_latent = sample(
        couple_runtime,
        latent,
        top,
        empty,
        scale=7.0,
        config=config,
        owner="proof_couple",
        middle=bottom,
    )
    images = {
        "reference": decode(runtime, reference_latent),
        "baseline": decode(runtime, baseline_latent),
        "injected": decode(runtime, injected_latent),
        "coupled": decode(runtime, coupled_latent),
    }
    family_output = output / family
    family_output.mkdir(parents=True, exist_ok=True)
    image_hashes = {}
    for name, image in images.items():
        path = family_output / f"{name}.png"
        as_pil(image).save(path)
        image_hashes[name] = sha256(path)
    return {
        "images": image_hashes,
        "metrics": clip_metrics(images, config, clip_root),
        "runtime_identity": runtime.runtime_identity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd15", type=Path, required=True)
    parser.add_argument("--flux", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("run with exactly one visible CUDA device")
    config = json.loads(CONFIG_PATH.read_text())
    paths = {"sd15": args.sd15, "flux": args.flux, "clip_metric": args.clip / "model.safetensors"}
    artifacts = {}
    for name, path in paths.items():
        actual = sha256(path)
        expected = config["artifacts"][name]["sha256"]
        if actual != expected:
            raise SystemExit(f"{name} sha256 {actual} != {expected}")
        artifacts[name] = {"path": str(path.resolve()), "sha256": actual}
    clip_support = {}
    for filename, expected in config["artifacts"]["clip_metric"]["support_files"].items():
        path = args.clip / filename
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"clip_metric {filename} sha256 {actual} != {expected}")
        clip_support[filename] = {"path": str(path.resolve()), "sha256": actual}
    artifacts["clip_metric"]["support_files"] = clip_support
    result = {
        "artifacts": artifacts,
        "config": config,
        "environment": {
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "families": {},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for family, artifact in (("sd15", args.sd15), ("flux", args.flux)):
        result["families"][family] = run_family(family, artifact, config, args.clip, args.output)
        (args.output / "results.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
