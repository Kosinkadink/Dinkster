"""Run matched official Ideogram 4 inference in clean ComfyUI and Dinkster processes."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ctypes
import fcntl
import gc
import hashlib
import json
import logging
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

COMFYUI_COMMIT = "1af040bf022569d7a890241c8dd79b296cda483f"
TEMPLATES_COMMIT = "8417f4f2a8380556070721d0ff1da8285d6e5438"
GPU_INDEX = 0
GPU_UUID = "GPU-666d1242-9c20-341c-73ea-e63770947451"
GPU_CLAIM = Path.home() / "gpu-claims/gpu0.lock"
THREAD_ID = "T-01a0650a-94c3-700e-96d0-c62b00ed07c6"
WIDTH = 1024
HEIGHT = 1024
STEPS = 20
MU = 0.5
STD = 1.75
CFG = 7.0
OVERRIDE_CFG = 3.0
OVERRIDE_START = 0.7
OVERRIDE_END = 1.0
SEED = 1085
PROMPT = "A red panda reading a book in a quiet library, detailed editorial illustration"
PHASES = ("cold", "warm-1", "warm-2", "warm-3")
WORKFLOWS = {
    "fp8": (
        "templates/image_ideogram4_t2i.json",
        "87dba0e3002b76990836a1ebbe43e0885175d2bcdbcd41e854d5bcd2209ea2dd",
    ),
    "int8": (
        "templates/image_ideogram4_t2i_int8.json",
        "a85346f5ad1ab07dcaf8bbd99f6618a26dbd46c11fb13128680822fe9dd880c4",
    ),
}


@dataclass(frozen=True)
class Artifact:
    role: str
    relative_path: str
    size: int
    sha256: str
    blake3: str
    url: str


SHARED = {
    "text": Artifact(
        "text",
        "text_encoders/qwen3vl_8b_fp8_scaled.safetensors",
        10_588_637_512,
        "4ba424cf62e51392e4d1a39933e803706f4e823c1065f36aaf149c6453f66bcd",
        "b82a81d829c1d8db687c8e9f79fc07e7211f5ac59ed05672423c1cedc8db0924",
        "https://huggingface.co/Comfy-Org/Qwen3-VL/resolve/5529a3c630b649351fb72d8c251577b5962371d8/text_encoders/qwen3vl_8b_fp8_scaled.safetensors",
    ),
    "vae": Artifact(
        "vae",
        "vae/flux2-vae.safetensors",
        336_213_556,
        "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5",
        "fcb1d172993424c66d325d139863ccbaadf64a920073b2d005d73a31fa5a851d",
        "https://huggingface.co/Comfy-Org/flux2-dev/resolve/ab9055628ea245000e610f2aa2c96f4746093546/split_files/vae/flux2-vae.safetensors",
    ),
}
VARIANTS = {
    "fp8": {
        "conditional": Artifact(
            "conditional",
            "diffusion_models/ideogram4_fp8_scaled.safetensors",
            9_280_741_285,
            "49a946f1b0f8bcf5eab7d3b1ecc7b453c104e034cb1b592032745692724bd306",
            "dadac522cf4fd911d25c70401df4e1233805887481e21be0e19839458fb69eba",
            "https://huggingface.co/Comfy-Org/Ideogram-4/resolve/bbee2ab2b14b2b5223448d12d6e31e5f9cec0546/diffusion_models/ideogram4_fp8_scaled.safetensors",
        ),
        "unconditional": Artifact(
            "unconditional",
            "diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
            9_280_741_293,
            "9b359007dae162cca7591d00868feea733eb7c56e56e3a214a4d5a9a2a07cd60",
            "1e26b2ecf7cb7ab57495faff8534875be41a44d6fc2ffe37f934dd65a460958d",
            "https://huggingface.co/Comfy-Org/Ideogram-4/resolve/bbee2ab2b14b2b5223448d12d6e31e5f9cec0546/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors",
        ),
    },
    "int8": {
        "conditional": Artifact(
            "conditional",
            "diffusion_models/ideogram4_int8_convrot.safetensors",
            9_583_465_712,
            "a9164002943463b4c7b2abd88c82a488c088acc35762651e4d8604d6ce4a163d",
            "e3cf071faafcf04192a66fa0324948589e60325b54a7935206f4e30b8ef257ce",
            "https://huggingface.co/Comfy-Org/Ideogram-4/resolve/bbee2ab2b14b2b5223448d12d6e31e5f9cec0546/diffusion_models/ideogram4_int8_convrot.safetensors",
        ),
        "unconditional": Artifact(
            "unconditional",
            "diffusion_models/ideogram4_unconditional_int8_convrot.safetensors",
            9_583_465_712,
            "cd03ed94f244c9cb705e7d30ca0f40b5f5b004bb20674117adff88d16416c23d",
            "eb50781b817ef134e114ffa55def137206c56ce32846f12e43397f0ac2a8e909",
            "https://huggingface.co/Comfy-Org/Ideogram-4/resolve/bbee2ab2b14b2b5223448d12d6e31e5f9cec0546/diffusion_models/ideogram4_unconditional_int8_convrot.safetensors",
        ),
    },
}


class ParityError(RuntimeError):
    pass


class _VaeFallbackCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "retrying with tiled VAE" in record.getMessage():
            self.count += 1


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="ascii")
    temporary.replace(path)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments], capture_output=True, text=True, check=False
    )
    if result.returncode:
        raise ParityError(result.stderr.strip())
    return result.stdout.strip()


def _verify_source(root: Path, commit: str, name: str) -> dict[str, str]:
    if _git(root, "rev-parse", "HEAD") != commit:
        raise ParityError(f"{name} is not at pinned commit {commit}")
    if _git(root, "status", "--porcelain"):
        raise ParityError(f"{name} checkout is not clean")
    return {"commit": commit, "remote": _git(root, "remote", "get-url", "origin")}


def _verify_dinkster_source(root: Path, commit: str) -> dict[str, str]:
    head = _git(root, "rev-parse", "HEAD")
    if head != commit:
        raise ParityError(f"Dinkster is not at requested commit {commit}")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise ParityError("Dinkster checkout is not clean")
    return {
        "commit": head,
        "remote": _git(root, "remote", "get-url", "origin"),
    }


def _full_commit(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 40 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError("commit must be a full 40-character hexadecimal object ID")
    return normalized


def _hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _blake3(path: Path) -> str:
    from blake3 import blake3

    digest = blake3()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifacts(root: Path, variant: str) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for artifact in (*VARIANTS[variant].values(), *SHARED.values()):
        path = root / artifact.relative_path
        if not path.is_file() or path.stat().st_size != artifact.size:
            raise ParityError(f"{artifact.role} artifact size mismatch")
        sha256 = _hash(path, "sha256")
        if sha256 != artifact.sha256:
            raise ParityError(f"{artifact.role} artifact SHA-256 mismatch")
        blake3 = _blake3(path)
        if blake3 != artifact.blake3:
            raise ParityError(f"{artifact.role} artifact BLAKE3 mismatch")
        records[artifact.role] = {
            "path": str(path.resolve()),
            "bytes": artifact.size,
            "sha256": sha256,
            "blake3": blake3,
            "url": artifact.url,
        }
    return records


def _verify_workflows(root: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for variant, (relative_path, expected_sha256) in WORKFLOWS.items():
        path = root / relative_path
        if not path.is_file():
            raise ParityError(f"{variant} official workflow is missing")
        sha256 = _hash(path, "sha256")
        if sha256 != expected_sha256:
            raise ParityError(f"{variant} official workflow SHA-256 mismatch")
        records[variant] = {"path": relative_path, "sha256": sha256}
    return records


def _add_dinkster_sources(root: Path) -> None:
    sys.path.insert(0, str(root / "src"))
    for source in sorted((root / "packages").glob("*/src"), reverse=True):
        sys.path.insert(0, str(source))


def _save(path: Path, tensor: Any) -> str:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tensor.detach().float().cpu().contiguous().numpy(), allow_pickle=False)
    return str(path.resolve())


def _rss_peak() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _rss_current() -> int:
    for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise ParityError("cannot read current resident memory")


def _trim_host_allocator() -> None:
    libc = ctypes.CDLL(None)
    malloc_trim = libc.malloc_trim
    malloc_trim.argtypes = [ctypes.c_size_t]
    malloc_trim.restype = ctypes.c_int
    malloc_trim(0)


def _memory_snapshot(torch: Any) -> dict[str, int]:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "host_bytes": _rss_current(),
    }


def _cleanup_memory(torch: Any) -> dict[str, int]:
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
        _trim_host_allocator()
    return _memory_snapshot(torch)


def _dtype_name(value: object) -> str:
    return str(value).removeprefix("torch.")


@dataclass
class _AttentionCalls:
    count: int = 0


@contextlib.contextmanager
def _observe_sdpa(torch: Any) -> Any:
    calls = _AttentionCalls()
    original = torch.nn.functional.scaled_dot_product_attention

    def observed(*args: object, **kwargs: object) -> object:
        calls.count += 1
        return original(*args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = observed
    try:
        yield calls
    finally:
        torch.nn.functional.scaled_dot_product_attention = original


def _observed_sdpa(calls: int, lane: str) -> str:
    if calls <= 0:
        raise ParityError(f"{lane} did not execute PyTorch scaled-dot-product attention")
    return "sdpa"


def _node_result(value: Any) -> tuple[Any, ...]:
    result = getattr(value, "result", value)
    if not isinstance(result, tuple):
        raise ParityError("ComfyUI node returned an unexpected result")
    return result


def _capture_calls(module: Any, output: Path, lane: str) -> tuple[dict[str, str], Any]:
    captured: dict[str, str] = {}
    calls = 0

    def hook(
        _module: object,
        arguments: tuple[object, ...],
        keywords: dict[str, object],
        result: object,
    ) -> None:
        nonlocal calls
        if calls:
            return
        calls += 1
        values = {
            "input": arguments[0],
            "timestep": arguments[1],
            "context": arguments[2] if len(arguments) > 2 else keywords.get("context"),
            "output": result,
        }
        for name, value in values.items():
            if value is not None:
                key = f"{lane}_denoiser_{name}"
                captured[key] = _save(output / f"{key}.npy", value)

    return captured, module.register_forward_hook(hook, with_kwargs=True)


def _configure_comfyui(root: Path, artifacts: Path) -> dict[str, Any]:
    sys.path.insert(0, str(root))
    from comfy.cli_args import args as comfy_args

    comfy_args.normalvram = True
    comfy_args.highvram = False
    comfy_args.lowvram = False
    comfy_args.novram = False
    comfy_args.gpu_only = False
    comfy_args.bf16_unet = True
    comfy_args.fp16_unet = False
    comfy_args.fp32_unet = False
    comfy_args.bf16_text_enc = False
    comfy_args.fp16_text_enc = False
    comfy_args.fp32_text_enc = True
    comfy_args.bf16_vae = True
    comfy_args.fp16_vae = False
    comfy_args.fp32_vae = False
    comfy_args.use_pytorch_cross_attention = True
    comfy_args.disable_cuda_malloc = True
    comfy_args.cuda_malloc = False
    import folder_paths
    import nodes
    from comfy_extras.nodes_custom_sampler import (
        CFGOverride,
        DualModelGuider,
        KSamplerSelect,
        RandomNoise,
        SamplerCustomAdvanced,
    )
    from comfy_extras.nodes_flux import EmptyFlux2LatentImage
    from comfy_extras.nodes_ideogram4 import Ideogram4Scheduler

    folder_paths.add_model_folder_path(
        "diffusion_models", str(artifacts / "diffusion_models"), is_default=True
    )
    folder_paths.add_model_folder_path(
        "text_encoders", str(artifacts / "text_encoders"), is_default=True
    )
    folder_paths.add_model_folder_path("vae", str(artifacts / "vae"), is_default=True)
    return {
        "args": comfy_args,
        "nodes": nodes,
        "cfg_override": CFGOverride,
        "dual_guider": DualModelGuider,
        "sampler": KSamplerSelect,
        "noise": RandomNoise,
        "sample": SamplerCustomAdvanced,
        "empty": EmptyFlux2LatentImage,
        "scheduler": Ideogram4Scheduler,
    }


def _run_comfyui(args: argparse.Namespace) -> dict[str, object]:
    boundary = _configure_comfyui(args.root, args.artifacts)
    import comfy.model_management as model_management
    import torch

    names = VARIANTS[args.variant]
    cond_name = Path(names["conditional"].relative_path).name
    uncond_name = Path(names["unconditional"].relative_path).name
    text_name = Path(SHARED["text"].relative_path).name
    vae_name = Path(SHARED["vae"].relative_path).name
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    baseline = _memory_snapshot(torch)
    conditional = boundary["nodes"].UNETLoader().load_unet(cond_name, "default")[0]
    unconditional = boundary["nodes"].UNETLoader().load_unet(uncond_name, "default")[0]
    clip = boundary["nodes"].CLIPLoader().load_clip(text_name, "ideogram4", "default")[0]
    vae = boundary["nodes"].VAELoader().load_vae(vae_name)[0]
    with _observe_sdpa(torch) as text_attention:
        positive = boundary["nodes"].CLIPTextEncode().encode(clip, PROMPT)[0]
    sampled_model = _node_result(
        boundary["cfg_override"].execute(conditional, OVERRIDE_CFG, OVERRIDE_START, OVERRIDE_END)
    )[0]
    guider = _node_result(
        boundary["dual_guider"].execute(
            sampled_model,
            positive,
            CFG,
            model_negative=unconditional,
            negative=None,
        )
    )[0]
    noise = _node_result(boundary["noise"].execute(SEED))[0]
    sampler = _node_result(boundary["sampler"].execute("euler"))[0]
    sigmas = _node_result(boundary["scheduler"].execute(STEPS, WIDTH, HEIGHT, MU, STD))[0]
    latent = _node_result(boundary["empty"].execute(WIDTH, HEIGHT, 1))[0]
    cond_tensor = positive[0][0]
    initial_noise = noise.generate_noise(latent)
    intermediates = {
        "conditioning": _save(output / "conditioning.npy", cond_tensor),
        "sigmas": _save(output / "sigmas.npy", sigmas),
        "noise": _save(output / "noise.npy", initial_noise),
    }
    cond_captured, cond_hook = _capture_calls(
        conditional.model.diffusion_model, output, "conditional"
    )
    uncond_captured, uncond_hook = _capture_calls(
        unconditional.model.diffusion_model, output, "unconditional"
    )
    phases: dict[str, dict[str, object]] = {}
    diffusion_attention_calls = 0
    fallback_counter = _VaeFallbackCounter()
    logging.getLogger().addHandler(fallback_counter)
    try:
        for phase in PHASES:
            fallbacks_before = fallback_counter.count
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            with _observe_sdpa(torch) as diffusion_attention:
                sampled = _node_result(
                    boundary["sample"].execute(noise, guider, sampler, sigmas, latent)
                )[0]
            diffusion_attention_calls += diffusion_attention.count
            torch.cuda.synchronize()
            sampled_ns = time.perf_counter_ns()
            image = boundary["nodes"].VAEDecode().decode(vae, sampled)[0]
            torch.cuda.synchronize()
            completed_ns = time.perf_counter_ns()
            phases[phase] = {
                "latency_ns": completed_ns - started,
                "sampling_latency_ns": sampled_ns - started,
                "vae_latency_ns": completed_ns - sampled_ns,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "peak_host_bytes": _rss_peak(),
                "latent": _save(output / f"{phase}-latent.npy", sampled["samples"]),
                "image": _save(output / f"{phase}-image.npy", image),
                "vae_fallbacks": fallback_counter.count - fallbacks_before,
            }
            del sampled, image
    finally:
        logging.getLogger().removeHandler(fallback_counter)
        cond_hook.remove()
        uncond_hook.remove()
    intermediates.update(cond_captured)
    intermediates.update(uncond_captured)
    diffusion_dtype = _dtype_name(conditional.model.get_dtype_inference())
    unconditional_dtype = _dtype_name(unconditional.model.get_dtype_inference())
    if unconditional_dtype != diffusion_dtype:
        raise ParityError("ComfyUI conditional and unconditional diffusion dtypes differ")
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "dtypes": {
            "diffusion": diffusion_dtype,
            "text": _dtype_name(clip.patcher.get_model_object("manual_cast_dtype")),
            "vae": _dtype_name(vae.vae_dtype),
        },
        "attention": {
            "diffusion": _observed_sdpa(diffusion_attention_calls, "ComfyUI diffusion"),
            "text": _observed_sdpa(text_attention.count, "ComfyUI text encoding"),
        },
        "attention_calls": {
            "diffusion": diffusion_attention_calls,
            "text": text_attention.count,
        },
    }
    conditional = unconditional = clip = vae = positive = sampled_model = guider = None
    noise = sampler = sigmas = latent = cond_tensor = initial_noise = None
    model_management.unload_all_models()
    model_management.cleanup_models()
    cleanup = _cleanup_memory(torch)
    return {
        "engine": "comfyui",
        "variant": args.variant,
        "intermediates": intermediates,
        "phases": phases,
        "runtime": runtime,
        "baseline": baseline,
        "cleanup": cleanup,
    }


@dataclass(frozen=True)
class _Proof:
    artifact: Artifact
    path: Path
    verification: Any

    @property
    def digest(self) -> str:
        return "blake3:" + self.artifact.blake3


@dataclass(frozen=True)
class _Resolver:
    proofs: dict[str, _Proof]

    def resolve(self, digest: str) -> Path | None:
        proof = self.proofs.get(digest)
        return None if proof is None else proof.path

    def resolve_asset(self, digest: str) -> object | None:
        proof = self.proofs.get(digest)
        if proof is None:
            return None
        from dinkster_assets.model import AssetResolution

        return AssetResolution(proof.path, proof.verification)


def _dinkster_value(type_id: str, payload: object, fingerprint: str) -> object:
    from dinkster_values import Value, ValueMeta
    from dinkster_values.model import PyObjPayload

    return Value(type_id, fingerprint, ValueMeta(), PyObjPayload(payload))


def _dinkster_asset_value(reference: object) -> object:
    from dinkster_assets import AssetRef
    from dinkster_values import Value, ValueMeta
    from dinkster_values.model import PyObjPayload

    assert isinstance(reference, AssetRef)
    return Value(
        "dinkster.asset",
        reference.digest,
        ValueMeta({"digest": reference.digest, "name": reference.name}),
        PyObjPayload(None),
    )


def _dinkster_selection(policy: Any, node: str, inputs: dict[str, object]) -> Any:
    selection = asyncio.run(
        policy.select(
            node,
            inputs,
            ("compat", {"compat": "compat-tag", "compat@native": "native-default"}),
        )
    )
    if selection is None or selection.target != "compat@native":
        raise ParityError(f"native dispatch refused {node}")
    return selection


def _dinkster_context(selection: Any) -> Any:
    from dinkster_workers.execution import ExecutionContext

    return ExecutionContext(
        arm=selection.target,
        expected_execution_identity=selection.cache_tag,
        fp8_matmul=selection.fp8_matmul,
        diffusion_dtype=selection.diffusion_dtype,
        text_dtype=selection.text_dtype,
        vae_dtype=selection.vae_dtype,
        attention_policy=selection.attention_policy,
        attention_route_token=selection.attention_route_token,
    )


def _run_dinkster(args: argparse.Namespace) -> dict[str, object]:
    _add_dinkster_sources(args.root)
    import torch
    from dinkster_assets import AssetRef
    from dinkster_assets.integrity import verification_record
    from dinkster_compat_comfy.native_arm import (
        GenerationCFGOverride,
        GenerationClipTextEncode,
        GenerationDualModelGuider,
        GenerationEmptyFlux2LatentImage,
        GenerationIdeogram4Scheduler,
        GenerationKSamplerSelect,
        GenerationRandomNoise,
        GenerationSamplerCustomAdvanced,
        GenerationVAEDecode,
        NativeLoadClip,
        NativeLoadDiffusionModel,
        NativeLoadVae,
    )
    from dinkster_compat_comfy.native_residency import observe_native_stages
    from dinkster_inference import split_component_conditioning
    from dinkster_inference_torch import materialize_ideogram4_conditioning, prepare_noise
    from dinkster_workers.execution import use_execution_context

    from dinkster.native_policy import NativeDispatchPolicy

    artifacts = {**VARIANTS[args.variant], **SHARED}
    proofs: dict[str, _Proof] = {}
    for artifact in artifacts.values():
        path = args.artifacts / artifact.relative_path
        verification = verification_record("blake3:" + artifact.blake3, path.stat())
        if verification is None:
            raise ParityError(f"cannot bind {artifact.role} artifact descriptor")
        proof = _Proof(artifact, path, verification)
        proofs[proof.digest] = proof
    resolver = _Resolver(proofs)
    refs = {
        role: AssetRef(
            proof.digest,
            proof.path.name,
            proof.artifact.size,
            resolver=resolver,
        )
        for role, proof in (
            (artifact.role, proofs["blake3:" + artifact.blake3]) for artifact in artifacts.values()
        )
    }
    policy = NativeDispatchPolicy(
        lambda digest: None if digest not in proofs else proofs[digest].path,
        lambda diagnostic: (_ for _ in ()).throw(ParityError(repr(diagnostic))),
    )
    asset_values = {role: _dinkster_asset_value(reference) for role, reference in refs.items()}
    string_default = _dinkster_value("dinkster.string", "default", "default")
    selections = {
        "conditional": _dinkster_selection(
            policy,
            "dinkster.load_diffusion_model",
            {"diffusion_model": asset_values["conditional"], "weight_dtype": string_default},
        ),
        "unconditional": _dinkster_selection(
            policy,
            "dinkster.load_diffusion_model",
            {"diffusion_model": asset_values["unconditional"], "weight_dtype": string_default},
        ),
        "text": _dinkster_selection(
            policy,
            "dinkster.load_clip",
            {
                "text_encoder": asset_values["text"],
                "type": _dinkster_value("dinkster.string", "ideogram4", "ideogram4"),
                "device": string_default,
            },
        ),
        "vae": _dinkster_selection(policy, "dinkster.load_vae", {"vae": asset_values["vae"]}),
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    baseline = _memory_snapshot(torch)
    with use_execution_context(_dinkster_context(selections["conditional"])):
        conditional = NativeLoadDiffusionModel.execute(
            diffusion_model=refs["conditional"], weight_dtype="default"
        )["model"]
    with use_execution_context(_dinkster_context(selections["unconditional"])):
        unconditional = NativeLoadDiffusionModel.execute(
            diffusion_model=refs["unconditional"], weight_dtype="default"
        )["model"]
    with use_execution_context(_dinkster_context(selections["text"])):
        clip = NativeLoadClip.execute(
            text_encoder=refs["text"], type="ideogram4", device="default"
        )["clip"]
    with use_execution_context(_dinkster_context(selections["vae"])):
        vae = NativeLoadVae.execute(vae=refs["vae"])["vae"]
    with _observe_sdpa(torch) as text_attention:
        positive = GenerationClipTextEncode.execute(text=PROMPT, clip=clip)["conditioning"]
    sampled_model = GenerationCFGOverride.execute(
        model=conditional,
        cfg=OVERRIDE_CFG,
        start_percent=OVERRIDE_START,
        end_percent=OVERRIDE_END,
    )["model"]
    guider = GenerationDualModelGuider.execute(
        model=sampled_model,
        model_negative=unconditional,
        positive=positive,
        negative=None,
        cfg=CFG,
    )["guider"]
    noise = GenerationRandomNoise.execute(noise_seed=SEED)["noise"]
    sampler = GenerationKSamplerSelect.execute(sampler_name="euler")["sampler"]
    sigmas = GenerationIdeogram4Scheduler.execute(
        steps=STEPS, width=WIDTH, height=HEIGHT, mu=MU, std=STD
    )["sigmas"]
    latent = GenerationEmptyFlux2LatentImage.execute(width=WIDTH, height=HEIGHT, batch_size=1)[
        "latent"
    ]
    carrier, _binding = split_component_conditioning(positive)
    conditioning = materialize_ideogram4_conditioning(carrier, device="cpu")
    initial_noise = prepare_noise(latent["samples"], SEED)
    intermediates = {
        "conditioning": _save(output / "conditioning.npy", conditioning.embeddings),
        "sigmas": _save(output / "sigmas.npy", torch.tensor(sigmas.values)),
        "noise": _save(output / "noise.npy", initial_noise),
    }
    cond_captured, cond_hook = _capture_calls(
        conditional.runtime.assembled.diffusion, output, "conditional"
    )
    uncond_captured, uncond_hook = _capture_calls(
        unconditional.runtime.assembled.diffusion, output, "unconditional"
    )
    phases: dict[str, dict[str, object]] = {}
    diffusion_attention_calls = 0
    fallback_counter = _VaeFallbackCounter()
    logging.getLogger().addHandler(fallback_counter)
    try:
        for phase in PHASES:
            fallbacks_before = fallback_counter.count
            native_events: list[Any] = []
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            with observe_native_stages(native_events.append, invocation_id=phase):
                with _observe_sdpa(torch) as diffusion_attention:
                    sampled = GenerationSamplerCustomAdvanced.execute(
                        noise=noise,
                        guider=guider,
                        sampler=sampler,
                        sigmas=sigmas,
                        latent_image=latent,
                    )["output"]
                diffusion_attention_calls += diffusion_attention.count
                torch.cuda.synchronize()
                sampled_ns = time.perf_counter_ns()
                image = GenerationVAEDecode.execute(samples=sampled, vae=vae)["image"]
                torch.cuda.synchronize()
            completed_ns = time.perf_counter_ns()
            phases[phase] = {
                "latency_ns": completed_ns - started,
                "sampling_latency_ns": sampled_ns - started,
                "vae_latency_ns": completed_ns - sampled_ns,
                "native_spans": [
                    {
                        "span_id": event.span_id,
                        "parent_span_id": event.parent_span_id,
                        "phase": event.phase,
                        "stage": event.stage,
                        "operation": event.operation,
                        "monotonic_ns": event.monotonic_ns,
                        "component_role": event.component_role,
                        "device": event.device,
                    }
                    for event in native_events
                ],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "peak_host_bytes": _rss_peak(),
                "latent": _save(output / f"{phase}-latent.npy", sampled["samples"]),
                "image": _save(output / f"{phase}-image.npy", image),
                "vae_fallbacks": fallback_counter.count - fallbacks_before,
            }
            del sampled, image
    finally:
        logging.getLogger().removeHandler(fallback_counter)
        cond_hook.remove()
        uncond_hook.remove()
    intermediates.update(cond_captured)
    intermediates.update(uncond_captured)
    diffusion_dtype = _dtype_name(conditional.runtime.assembled.compute_dtype("diffusion"))
    unconditional_dtype = _dtype_name(unconditional.runtime.assembled.compute_dtype("diffusion"))
    if unconditional_dtype != diffusion_dtype:
        raise ParityError("Dinkster conditional and unconditional diffusion dtypes differ")
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "dtypes": {
            "diffusion": diffusion_dtype,
            "text": clip.recipe.knobs.text_dtype,
            "vae": vae.recipe.knobs.vae_dtype,
        },
        "attention": {
            "diffusion": _observed_sdpa(diffusion_attention_calls, "Dinkster diffusion"),
            "text": _observed_sdpa(text_attention.count, "Dinkster text encoding"),
        },
        "attention_calls": {
            "diffusion": diffusion_attention_calls,
            "text": text_attention.count,
        },
    }
    handles = (clip, vae, conditional, unconditional)
    conditional = unconditional = clip = vae = positive = sampled_model = guider = None
    noise = sampler = sigmas = latent = carrier = conditioning = initial_noise = _binding = None
    for handle in handles:
        if not handle.released:
            handle.terminal_release()
    handles = ()
    handle = None
    cleanup = _cleanup_memory(torch)
    return {
        "engine": "dinkster",
        "variant": args.variant,
        "intermediates": intermediates,
        "phases": phases,
        "dispatch": {
            role: {"target": selection.target, "identity": selection.cache_tag}
            for role, selection in selections.items()
        },
        "runtime": runtime,
        "baseline": baseline,
        "cleanup": cleanup,
    }


def _array_metrics(left_path: str, right_path: str) -> dict[str, object]:
    import numpy as np

    left = np.load(left_path, allow_pickle=False).astype(np.float64, copy=False)
    right = np.load(right_path, allow_pickle=False).astype(np.float64, copy=False)
    if left.shape != right.shape:
        raise ParityError(f"array shapes differ: {left.shape} != {right.shape}")
    delta = left - right
    denominator = np.linalg.norm(left.ravel()) * np.linalg.norm(right.ravel())
    cosine = (
        1.0
        if denominator == 0 and np.array_equal(left, right)
        else float(np.dot(left.ravel(), right.ravel()) / denominator)
    )
    return {
        "shape": list(left.shape),
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "cosine": cosine,
        "bit_equal": bool(np.array_equal(left, right)),
    }


def _performance(record: dict[str, object]) -> dict[str, object]:
    phases = record["phases"]
    assert isinstance(phases, dict)
    diagnostics = record["diagnostics"]
    assert isinstance(diagnostics, dict)
    baseline = record["baseline"]
    assert isinstance(baseline, dict)
    cleanup = record["cleanup"]
    assert isinstance(cleanup, dict)
    warm = [int(phases[name]["latency_ns"]) for name in PHASES[1:]]
    # Engine baselines contain different eager global state; total cleanup memory is gated.
    retained = {
        name: max(0, int(cleanup[name]) - int(baseline[name]))
        for name in ("allocated_bytes", "reserved_bytes", "host_bytes")
    }
    vae_fallbacks = sum(int(phases[name]["vae_fallbacks"]) for name in PHASES)
    if vae_fallbacks != int(diagnostics["vae_fallback_warnings"]):
        raise ParityError("in-process and subprocess VAE fallback counts differ")
    if vae_fallbacks != int(diagnostics["recovered_vae_ooms"]):
        raise ParityError("VAE fallback did not report a recovered out-of-memory event")
    return {
        "cold_ns": int(phases["cold"]["latency_ns"]),
        "warm_median_ns": int(statistics.median(warm)),
        "warm_p95_ns": max(warm),
        "warm_max_ns": max(warm),
        "peak_allocated_bytes": max(int(phases[name]["peak_allocated_bytes"]) for name in PHASES),
        "peak_reserved_bytes": max(int(phases[name]["peak_reserved_bytes"]) for name in PHASES),
        "peak_host_bytes": max(int(phases[name]["peak_host_bytes"]) for name in PHASES),
        "vae_fallbacks": vae_fallbacks,
        "recovered_vae_ooms": int(diagnostics["recovered_vae_ooms"]),
        "unrecovered_ooms": int(diagnostics["unrecovered_ooms"]),
        "allocator_oom_warnings": int(diagnostics["allocator_oom_warnings"]),
        "residual_allocated_bytes": int(cleanup["allocated_bytes"]),
        "residual_reserved_bytes": int(cleanup["reserved_bytes"]),
        "residual_host_bytes": int(cleanup["host_bytes"]),
        "retained_allocated_bytes": retained["allocated_bytes"],
        "retained_reserved_bytes": retained["reserved_bytes"],
        "retained_host_bytes": retained["host_bytes"],
    }


def _compare(comfy: dict[str, object], dinkster: dict[str, object]) -> dict[str, object]:
    exact_intermediates = (
        "conditioning",
        "noise",
        "sigmas",
        "conditional_denoiser_input",
        "conditional_denoiser_timestep",
        "conditional_denoiser_context",
        "conditional_denoiser_output",
        "unconditional_denoiser_input",
        "unconditional_denoiser_timestep",
        "unconditional_denoiser_output",
    )
    comfy_intermediates = comfy["intermediates"]
    dinkster_intermediates = dinkster["intermediates"]
    assert isinstance(comfy_intermediates, dict) and isinstance(dinkster_intermediates, dict)
    arrays: dict[str, object] = {}
    for key in exact_intermediates:
        metrics = _array_metrics(comfy_intermediates[key], dinkster_intermediates[key])
        metrics["pass"] = metrics["bit_equal"]
        arrays[key] = metrics
    for phase in PHASES:
        for name in ("latent", "image"):
            key = f"{phase}_{name}"
            metrics = _array_metrics(comfy["phases"][phase][name], dinkster["phases"][phase][name])
            metrics["pass"] = (
                metrics["bit_equal"]
                if name == "latent"
                else metrics["max_abs"] <= 0.05 and metrics["cosine"] >= 0.99999
            )
            arrays[key] = metrics
    comfy_perf = _performance(comfy)
    dinkster_perf = _performance(dinkster)
    comfy_runtime = comfy["runtime"]
    dinkster_runtime = dinkster["runtime"]
    assert isinstance(comfy_runtime, dict) and isinstance(dinkster_runtime, dict)
    attention_observed = all(
        int(runtime["attention_calls"][lane]) > 0
        for runtime in (comfy_runtime, dinkster_runtime)
        for lane in ("diffusion", "text")
    )
    performance = {
        "comfyui": comfy_perf,
        "dinkster": dinkster_perf,
        "pass": all(
            int(dinkster_perf[key]) <= int(comfy_perf[key])
            for key in (
                "cold_ns",
                "warm_median_ns",
                "warm_p95_ns",
                "warm_max_ns",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "peak_host_bytes",
                "vae_fallbacks",
                "recovered_vae_ooms",
                "unrecovered_ooms",
                "allocator_oom_warnings",
                "residual_allocated_bytes",
                "residual_reserved_bytes",
                "residual_host_bytes",
            )
        ),
    }
    runtime = {
        "comfyui": comfy_runtime,
        "dinkster": dinkster_runtime,
        "pass": comfy_runtime == dinkster_runtime
        and attention_observed
        and comfy_runtime["dtypes"]
        == {"diffusion": "bfloat16", "text": "float32", "vae": "bfloat16"}
        and comfy_runtime["attention"] == {"diffusion": "sdpa", "text": "sdpa"},
    }
    passed = (
        all(bool(value["pass"]) for value in arrays.values())
        and performance["pass"]
        and runtime["pass"]
    )
    return {"arrays": arrays, "performance": performance, "runtime": runtime, "pass": passed}


def _engine(args: argparse.Namespace) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(GPU_INDEX):
        raise ParityError("engine requires physical GPU 0 visibility")
    record = _run_comfyui(args) if args.engine == "comfyui" else _run_dinkster(args)
    _write_json(args.output / "record.json", record)
    return 0


def _driver(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=False)
    claims = GPU_CLAIM.open("a+", encoding="ascii")
    try:
        try:
            fcntl.flock(claims, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ParityError(f"GPU 0 claim is occupied: {GPU_CLAIM}") from error
        claims.seek(0)
        claims.truncate()
        claims.write(
            f"thread={THREAD_ID}\n"
            "purpose=Ideogram4 parity\n"
            f"timestamp={datetime.now(UTC).isoformat()}\n"
        )
        claims.flush()
        source = {
            "comfyui": _verify_source(args.comfyui_root, COMFYUI_COMMIT, "ComfyUI"),
            "templates": {
                **_verify_source(args.templates_root, TEMPLATES_COMMIT, "workflow templates"),
                "workflows": _verify_workflows(args.templates_root),
            },
            "dinkster": _verify_dinkster_source(args.dinkster_root, args.dinkster_commit),
        }
        results: dict[str, object] = {}
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
        for variant in ("fp8", "int8"):
            artifacts = _verify_artifacts(args.artifacts, variant)
            records: dict[str, object] = {}
            engines = (
                ("comfyui", args.comfyui_root, args.comfyui_python),
                ("dinkster", args.dinkster_root, args.dinkster_python),
            )
            for engine, root, interpreter in engines:
                output = args.output / variant / engine
                command = [
                    str(interpreter),
                    str(Path(__file__).resolve()),
                    "engine",
                    "--engine",
                    engine,
                    "--variant",
                    variant,
                    "--root",
                    str(root),
                    "--artifacts",
                    str(args.artifacts),
                    "--output",
                    str(output),
                ]
                completed = subprocess.run(
                    command,
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=7200,
                )
                output.mkdir(parents=True, exist_ok=True)
                (output / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
                (output / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
                if completed.returncode:
                    raise ParityError(
                        f"{variant} {engine} failed with {completed.returncode}: "
                        f"{completed.stderr[-2000:]}"
                    )
                record = json.loads((output / "record.json").read_text())
                fallback_lines = [
                    line
                    for line in completed.stderr.splitlines()
                    if "retrying with tiled VAE" in line
                ]
                record["diagnostics"] = {
                    "allocator_oom_warnings": completed.stderr.count(
                        "memory allocation failed with OOM"
                    ),
                    "vae_fallback_warnings": len(fallback_lines),
                    "recovered_vae_ooms": sum(
                        "out of memory" in line.lower() for line in fallback_lines
                    ),
                    "unrecovered_ooms": 0,
                }
                records[engine] = record
            comparison = _compare(records["comfyui"], records["dinkster"])
            results[variant] = {
                "artifacts": artifacts,
                "records": records,
                "comparison": comparison,
            }
        receipt = {
            "schema": "dinkster.ideogram4.parity.v1",
            "timestamp": datetime.now(UTC).isoformat(),
            "gpu": {"index": GPU_INDEX, "uuid": GPU_UUID},
            "source": source,
            "workload": {
                "prompt": PROMPT,
                "seed": SEED,
                "width": WIDTH,
                "height": HEIGHT,
                "steps": STEPS,
                "sampler": "euler",
                "scheduler": {"name": "Ideogram4Scheduler", "mu": MU, "std": STD},
                "guidance": {
                    "name": "DualModelGuider",
                    "cfg": CFG,
                    "unconditional": "image-only",
                    "override": {
                        "cfg": OVERRIDE_CFG,
                        "start": OVERRIDE_START,
                        "end": OVERRIDE_END,
                    },
                },
                "dtypes": {"diffusion": "bfloat16", "text": "float32", "vae": "bfloat16"},
                "attention": "sdpa",
            },
            "variants": results,
            "pass": all(results[name]["comparison"]["pass"] for name in results),
        }
        _write_json(args.output / "receipt.json", receipt)
        if not receipt["pass"]:
            raise ParityError("Ideogram 4 parity or non-regression gate failed")
        return 0
    finally:
        with contextlib.suppress(OSError):
            claims.seek(0)
            claims.truncate()
            claims.flush()
            fcntl.flock(claims, fcntl.LOCK_UN)
        claims.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    engine = subparsers.add_parser("engine")
    engine.add_argument("--engine", choices=("comfyui", "dinkster"), required=True)
    engine.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    engine.add_argument("--root", type=Path, required=True)
    engine.add_argument("--artifacts", type=Path, required=True)
    engine.add_argument("--output", type=Path, required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--comfyui-python", type=Path, required=True)
    run.add_argument("--dinkster-python", type=Path, required=True)
    run.add_argument("--comfyui-root", type=Path, required=True)
    run.add_argument("--templates-root", type=Path, required=True)
    run.add_argument("--dinkster-root", type=Path, required=True)
    run.add_argument("--dinkster-commit", type=_full_commit, required=True)
    run.add_argument("--artifacts", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return _engine(args) if args.command == "engine" else _driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
