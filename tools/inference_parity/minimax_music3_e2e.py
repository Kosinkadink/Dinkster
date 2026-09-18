"""Matched official-workflow MiniMax Music 3 inference comparison.

The gate launches clean ComfyUI and Dinkster processes sequentially on one GPU,
retains intermediate tensors and audio, and fails on correctness, performance,
memory, fallback, or cleanup regressions.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ctypes
import datetime as dt
import fcntl
import gc
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
import types
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

COMFYUI_COMMIT = "345c9190497c82cff53e71fb4ae00d1e135a6542"
TEMPLATES_COMMIT = "8417f4f2a8380556070721d0ff1da8285d6e5438"
TEMPLATE_PATH = "templates/audio_minimax_music_3.json"
TEMPLATE_BYTES = 32_927
TEMPLATE_SHA256 = "0322153265b3e785961511b7849f6659f46a8fa7e8cb66976e5279ff1774b228"
ARTIFACT_REVISION = "6baad88896848433857c170ba4f05d2ea9d5f218"
ARTIFACTS = {
    "diffusion": (
        "diffusion_models/minimax_music3_dit_fp16.safetensors",
        4_914_197_682,
        "45494a2b6b69af115902ff28eaf54118d19067aa54da01000f3e3efce7ba0e34",
    ),
    "text": (
        "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
        9_196_611_886,
        "010b7416d2336a08c711bc22ee65849c9623069ddb7d89bec011a75699e52014",
    ),
    "vae": (
        "vae/minimax_music3_dav.safetensors",
        216_696_128,
        "2a32155b769be01445fcc2a8663b910fc9e1751e18dc1c3ec528064512d9ef0c",
    ),
}
GPU_INDEX = 1
GPU_UUID = "GPU-5ac69527-f5f0-f6f0-1d46-24c6f401cdc6"
GPU_CLAIM_PATH = Path("/home/kosin/gpu-claims/gpu1.lock")
THREAD_ID = "T-01a0650a-98c3-727b-b491-67d71b820772"
PHASES = ("cold", *(f"warm-{index}" for index in range(1, 8)))
WARM_PHASES = PHASES[1:]
STEPS = 30
CFG = 1.7
SAMPLER = "euler"
SCHEDULER = "simple"
TEXT_CFG = 1.7
TOP_K = 50
TILE_SIZE = 1536
OVERLAP = 64
LATENCY_LAUNCH_JITTER_FLOOR = 0.0005
LATENCY_SOURCE_JITTER_SIGMAS = 5.0
CUDA_ALLOCATOR_TOLERANCE_BYTES = 2 * 1024**2
COMPARISON_LIMITS = {
    "conditioning": {"atol": 0.0, "rtol": 0.0, "cosine_min": 1.0},
    "noise": {"atol": 0.0, "rtol": 0.0, "cosine_min": 1.0},
    "sigmas": {"atol": 0.0, "rtol": 0.0, "cosine_min": 1.0},
    "denoiser": {
        "atol": 0.0,
        "rtol": 0.0,
        "cosine_min": 1.0,
        "mean_abs_max": 0.0,
        "rmse_max": 0.0,
    },
    "latent": {"atol": 0.0, "rtol": 0.0, "cosine_min": 1.0},
    "audio": {
        "atol": 0.0,
        "rtol": 0.0,
        "cosine_min": 1.0,
        "mean_abs_max": 0.0,
        "rmse_max": 0.0,
    },
}


class ComparisonError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactProof:
    role: str
    path: Path
    digest: str
    size: int
    sha256: str


class FixedAssetResolver:
    def __init__(self, proof: ArtifactProof, verification: object) -> None:
        self.proof = proof
        self.verification = verification

    def resolve(self, digest: str) -> Path | None:
        return self.proof.path if digest == self.proof.digest else None

    def resolve_asset(self, digest: str) -> object | None:
        if digest != self.proof.digest:
            return None
        from dinkster_assets.model import AssetResolution

        return AssetResolution(self.proof.path, self.verification)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise ComparisonError(f"git {' '.join(arguments)} failed for {root}: {result.stderr}")
    return result.stdout.strip()


def _source_receipt(root: Path, expected_commit: str | None = None) -> dict[str, object]:
    status = _git(root, "status", "--porcelain")
    if status:
        raise ComparisonError(f"source checkout must be clean: {root}\n{status}")
    commit = _git(root, "rev-parse", "HEAD")
    if expected_commit is not None and commit != expected_commit:
        raise ComparisonError(f"source checkout {root} is at {commit}, expected {expected_commit}")
    return {
        "commit": commit,
        "remote": _git(root, "remote", "get-url", "origin"),
        "root": str(root.resolve()),
    }


def _template_values(path: Path) -> dict[str, object]:
    if path.stat().st_size != TEMPLATE_BYTES or _sha256(path) != TEMPLATE_SHA256:
        raise ComparisonError("official workflow template differs from its immutable receipt")
    document = json.loads(path.read_text(encoding="utf-8"))
    root_nodes = cast("list[dict[str, object]]", document["nodes"])
    workflow_node = next(
        node for node in root_nodes if node.get("type") == "ac99f841-a3de-4329-9564-953b81cf9e16"
    )
    values = cast("list[object]", workflow_node["widgets_values"])
    definitions = cast("dict[str, object]", document["definitions"])
    subgraphs = cast("list[dict[str, object]]", definitions["subgraphs"])
    nodes = cast("list[dict[str, object]]", subgraphs[0]["nodes"])
    widgets = {node["type"]: node.get("widgets_values", []) for node in nodes}
    expected = {
        "KSampler": [1111111112, "fixed", 30, 1.7, "euler", "simple", 1],
        "MiniMaxMusic3TextEncode": ["", "", 222, "fixed", 60, 1.7, 50],
        "VAEDecodeAudioTiled": [1536, 64],
    }
    for node_type, expected_values in expected.items():
        if widgets.get(node_type) != expected_values:
            raise ComparisonError(f"official template {node_type} values changed")
    expected_root_values = [
        60,
        197122968890040,
        *[Path(artifact[0]).name for artifact in ARTIFACTS.values()],
        True,
    ]
    if len(values) != 8 or values[2:] != expected_root_values:
        raise ComparisonError("official workflow root values changed")
    return {
        "caption": values[0],
        "lyrics": values[1],
        "max_duration": values[2],
        "seed": values[3],
        "tiled_decode": values[7],
    }


def _verify_artifacts(root: Path) -> dict[str, ArtifactProof]:
    from blake3 import blake3

    proofs = {}
    for role, (relative, size, sha256) in ARTIFACTS.items():
        path = root / relative
        if not path.is_file() or path.stat().st_size != size:
            raise ComparisonError(f"official {role} artifact size does not match its receipt")
        sha = hashlib.sha256()
        content = blake3()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                sha.update(chunk)
                content.update(chunk)
        if sha.hexdigest() != sha256:
            raise ComparisonError(f"official {role} artifact SHA-256 does not match its receipt")
        proofs[role] = ArtifactProof(role, path, f"blake3:{content.hexdigest()}", size, sha256)
    return proofs


def _artifact_receipt(proofs: dict[str, ArtifactProof]) -> dict[str, object]:
    base = f"https://huggingface.co/Comfy-Org/MiniMax-Music-3/resolve/{ARTIFACT_REVISION}/"
    return {
        role: {
            "blake3": proof.digest.removeprefix("blake3:"),
            "bytes": proof.size,
            "path": str(proof.path.resolve()),
            "revision": ARTIFACT_REVISION,
            "sha256": proof.sha256,
            "url": base + ARTIFACTS[role][0],
        }
        for role, proof in proofs.items()
    }


def _workload(template: dict[str, object]) -> dict[str, object]:
    return {
        **template,
        "attention_backend": "sdpa",
        "cfg": CFG,
        "denoise": 1.0,
        "sampler": SAMPLER,
        "scheduler": SCHEDULER,
        "steps": STEPS,
        "text_cfg": TEXT_CFG,
        "tile_size": TILE_SIZE,
        "tile_overlap": OVERLAP,
        "top_k": TOP_K,
        "component_dtypes": {
            "diffusion": "float16",
            "text": "bfloat16",
            "vae": "float32",
        },
    }


def _stub_missing_torchaudio() -> bool:
    if importlib.util.find_spec("torchaudio") is not None:
        return False
    module = types.ModuleType("torchaudio")
    module.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    functional = types.ModuleType("torchaudio.functional")
    transforms = types.ModuleType("torchaudio.transforms")
    module.__dict__["functional"] = functional
    module.__dict__["transforms"] = transforms
    for name, value in (
        ("torchaudio", module),
        ("torchaudio.functional", functional),
        ("torchaudio.transforms", transforms),
    ):
        value.__spec__ = importlib.machinery.ModuleSpec(name, None)
        sys.modules[name] = value
    return True


def _configure_comfyui(root: Path, *, preflight: bool) -> dict[str, Any]:
    sys.path.insert(0, str(root))
    _stub_missing_torchaudio()
    sys.argv = [
        str(Path(__file__).resolve()),
        "--use-pytorch-cross-attention",
        "--disable-cuda-malloc",
        "--fp16-unet",
        "--bf16-text-enc",
        "--fp32-vae",
    ]
    if preflight:
        sys.argv.append("--cpu")
    import comfy.options  # pyright: ignore[reportMissingImports]

    comfy.options.enable_args_parsing()
    with contextlib.redirect_stdout(sys.stderr):
        import comfy.model_management as model_management  # pyright: ignore[reportMissingImports]
        import comfy.sample as sample  # pyright: ignore[reportMissingImports]
        import comfy.samplers as samplers  # pyright: ignore[reportMissingImports]
        import comfy.sd as sd  # pyright: ignore[reportMissingImports]
        import comfy.utils as utils  # pyright: ignore[reportMissingImports]
        import nodes  # pyright: ignore[reportMissingImports]
        from comfy_extras.nodes_audio import (
            VAEDecodeAudioTiled,  # pyright: ignore[reportMissingImports]
        )
        from comfy_extras.nodes_minimax_music import (  # pyright: ignore[reportMissingImports]
            EmptyMiniMaxMusic3LatentAudio,
            MiniMaxMusic3TextEncode,
        )

    return {
        "EmptyMiniMaxMusic3LatentAudio": EmptyMiniMaxMusic3LatentAudio,
        "MiniMaxMusic3TextEncode": MiniMaxMusic3TextEncode,
        "VAEDecodeAudioTiled": VAEDecodeAudioTiled,
        "model_management": model_management,
        "nodes": nodes,
        "sample": sample,
        "samplers": samplers,
        "sd": sd,
        "utils": utils,
    }


def _add_dinkster_sources(root: Path) -> None:
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    for source in sorted((root / "packages").glob("*/src"), reverse=True):
        sys.path.insert(0, str(source))


def _proofs_from_receipt(path: Path) -> dict[str, ArtifactProof]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    return {
        role: ArtifactProof(
            role,
            Path(value["path"]),
            "blake3:" + value["blake3"],
            value["bytes"],
            value["sha256"],
        )
        for role, value in receipt.items()
    }


def _asset_value(proof: ArtifactProof) -> object:
    from dinkster_values import Value, ValueMeta
    from dinkster_values.model import PyObjPayload

    return Value(
        type_id="dinkster.asset",
        fingerprint=proof.digest,
        meta=ValueMeta({"digest": proof.digest, "name": proof.path.name}),
        payload=PyObjPayload(None),
    )


def _string_value(value: str) -> object:
    from dinkster_values import Value, ValueMeta
    from dinkster_values.model import PyObjPayload

    return Value(
        type_id="dinkster.string",
        fingerprint=value,
        meta=ValueMeta(),
        payload=PyObjPayload(value),
    )


def _asset_ref(proof: ArtifactProof) -> object:
    from dinkster_assets import AssetRef
    from dinkster_assets.integrity import verification_record

    verification = verification_record(proof.digest, proof.path.stat())
    if verification is None:
        raise ComparisonError(f"cannot bind {proof.role} artifact verification")
    resolver = FixedAssetResolver(proof, verification)
    return AssetRef(proof.digest, proof.path.name, proof.size, resolver=resolver)


def _select_executions(proofs: dict[str, ArtifactProof]) -> dict[str, object]:
    from dinkster.native_policy import NativeDispatchPolicy

    policy = NativeDispatchPolicy(
        lambda digest: next(
            (proof.path for proof in proofs.values() if proof.digest == digest), None
        ),
        lambda diagnostic: (_ for _ in ()).throw(
            ComparisonError(f"native dispatch diagnostic: {diagnostic!r}")
        ),
        dtype_policy=lambda: {
            "diffusion": "float16",
            "textEncoder": "bfloat16",
            "vae": "float32",
        },
    )
    requests = {
        "model": (
            "dinkster.load_diffusion_model",
            {
                "diffusion_model": _asset_value(proofs["diffusion"]),
                "weight_dtype": _string_value("default"),
            },
        ),
        "clip": (
            "dinkster.load_clip",
            {
                "text_encoder": _asset_value(proofs["text"]),
                "type": _string_value("minimax"),
                "device": _string_value("default"),
            },
        ),
        "vae": ("dinkster.load_vae", {"vae": _asset_value(proofs["vae"])}),
    }
    selections = {}
    for role, (node_type, inputs) in requests.items():
        selection = asyncio.run(
            policy.select(
                node_type,
                inputs,
                ("compat", {"compat": "compat-tag", "compat@native": "native-default"}),
            )
        )
        if selection is None or selection.target != "compat@native":
            raise ComparisonError(f"native dispatch did not select {role}")
        selections[role] = selection
    return selections


def _production_boundary() -> dict[str, object]:
    from dinkster_compat_comfy.native_arm import (
        GenerationConditioningZeroOut,
        GenerationKSampler,
        NativeEmptyMiniMaxMusic3LatentAudio,
        NativeLoadClip,
        NativeLoadDiffusionModel,
        NativeLoadVae,
        NativeMiniMaxMusic3TextEncode,
        NativeVAEDecodeAudioTiled,
    )

    return {
        "conditioning_zero_out": GenerationConditioningZeroOut,
        "empty_latent": NativeEmptyMiniMaxMusic3LatentAudio,
        "ksampler": GenerationKSampler,
        "load_clip": NativeLoadClip,
        "load_diffusion": NativeLoadDiffusionModel,
        "load_vae": NativeLoadVae,
        "text_encode": NativeMiniMaxMusic3TextEncode,
        "vae_decode_tiled": NativeVAEDecodeAudioTiled,
    }


def _execution_context(selection: object) -> object:
    from dinkster_workers import ExecutionContext

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


def _load_dinkster_components(
    proofs: dict[str, ArtifactProof], selections: dict[str, object], boundary: dict[str, object]
) -> dict[str, object]:
    from dinkster_workers import use_execution_context

    with use_execution_context(_execution_context(selections["model"])):
        model = boundary["load_diffusion"].execute(
            diffusion_model=_asset_ref(proofs["diffusion"]), weight_dtype="default"
        )["model"]
    with use_execution_context(_execution_context(selections["clip"])):
        clip = boundary["load_clip"].execute(
            text_encoder=_asset_ref(proofs["text"]), type="minimax", device="default"
        )["clip"]
    with use_execution_context(_execution_context(selections["vae"])):
        vae = boundary["load_vae"].execute(vae=_asset_ref(proofs["vae"]))["vae"]
    return {"model": model, "clip": clip, "vae": vae}


def _current_rss() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise ComparisonError("process RSS is unavailable")


def _trim_host_allocator() -> None:
    if sys.platform.startswith("linux"):
        cast("Any", ctypes.CDLL(None)).malloc_trim(0)


def _state_dtypes(module: Any) -> list[str]:
    states = (*module.parameters(), *module.buffers())
    return sorted({str(state.dtype).removeprefix("torch.") for state in states})


def _save_array(path: Path, tensor: Any) -> str:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tensor.detach().float().cpu().contiguous().numpy(), allow_pickle=False)
    return str(path)


def _save_wav(path: Path, waveform: Any, sample_rate: int) -> str:
    import numpy as np

    audio = waveform.detach().float().cpu().numpy()[0].T
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = np.round(pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(pcm.shape[1])
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    return str(path)


class _DenoiserCapture:
    def __init__(self, module: Any, hook: Any, *, prepared: bool) -> None:
        self.module = module
        self.hook = hook
        self.prepared = prepared

    def remove(self) -> None:
        del self.module.aligned_condition
        if self.prepared:
            del self.module.prepare_condition
            del self.module.forward_prepared
        if self.hook is not None:
            self.hook.remove()


def _capture_denoiser(
    module: Any, output_dir: Path, *, prepared: bool = False
) -> tuple[dict[str, str], _DenoiserCapture]:
    captured: dict[str, str] = {}

    def save(name: str, value: Any) -> None:
        if name not in captured:
            captured[name] = _save_array(output_dir / f"{name}.npy", value)

    original_aligned = module.aligned_condition

    def capture_aligned(context: Any) -> Any:
        condition = original_aligned(context)
        save("denoiser_aligned_condition", condition)
        return condition

    module.aligned_condition = capture_aligned

    def capture_values(
        latent: object,
        timestep: object,
        conditioning_scale: object,
        output: object,
    ) -> None:
        if "denoiser_output" in captured:
            return
        for name, value in (
            ("denoiser_input", latent),
            ("denoiser_timestep", timestep),
            ("denoiser_conditioning_scale", conditioning_scale),
            ("denoiser_output", output),
        ):
            save(name, value)

    if not prepared:

        def capture(
            _module: object,
            arguments: tuple[object, ...],
            keywords: dict[str, object],
            output: object,
        ) -> None:
            values = cast("tuple[Any, ...]", arguments)
            conditioning_scale = values[3] if len(values) > 3 else keywords["conditioning_scale"]
            capture_values(values[0], values[1], conditioning_scale, output)

        hook = module.register_forward_hook(capture, with_kwargs=True)
        return captured, _DenoiserCapture(module, hook, prepared=False)

    original_prepare_condition = module.prepare_condition

    def capture_prepare_condition(context: Any, conditioning_scale: Any) -> Any:
        condition = original_prepare_condition(context, conditioning_scale)
        save("denoiser_conditioning_scale", conditioning_scale)
        return condition

    original_forward_prepared = module.forward_prepared

    def capture_forward_prepared(
        latent: Any, timestep: Any, condition: Any, rotary_table: Any
    ) -> Any:
        output = original_forward_prepared(latent, timestep, condition, rotary_table)
        save("denoiser_input", latent)
        save("denoiser_timestep", timestep)
        save("denoiser_output", output)
        return output

    module.prepare_condition = capture_prepare_condition
    module.forward_prepared = capture_forward_prepared
    return captured, _DenoiserCapture(module, None, prepared=True)


def _capture_text_seams(module: Any, output_dir: Path) -> tuple[dict[str, str], list[Any]]:
    qwen = next(
        candidate
        for candidate in module.modules()
        if hasattr(candidate, "lm_head_pruned")
        and hasattr(candidate, "audio_decoder")
        and hasattr(candidate, "layers")
    )
    first = qwen.layers[0]
    captured: dict[str, str] = {}
    hooks = []

    def save(name: str, value: Any) -> None:
        if name not in captured:
            captured[name] = _save_array(output_dir / f"{name}.npy", value)

    def capture_input(name: str) -> Any:
        return lambda _module, arguments: save(name, arguments[0])

    def capture_output(name: str) -> Any:
        return lambda _module, _arguments, output: save(name, output)

    hooks.extend(
        (
            first.self_attn.qkv_proj.register_forward_hook(capture_output("text_layer_0_qkv")),
            first.self_attn.o_proj.register_forward_hook(capture_output("text_layer_0_attention")),
            first.mlp.gate_up_proj.register_forward_hook(capture_output("text_layer_0_gate_up")),
            qwen.norm.register_forward_pre_hook(capture_input("text_final_norm_input")),
            qwen.norm.register_forward_hook(capture_output("text_final_norm_output")),
            qwen.lm_head_pruned.register_forward_pre_hook(capture_input("text_logits_input")),
            qwen.lm_head_pruned.register_forward_hook(capture_output("text_logits")),
        )
    )
    hooks.extend(
        layer.input_layernorm.register_forward_pre_hook(capture_input(f"text_layer_{index}_input"))
        for index, layer in enumerate(qwen.layers)
        if index in (0, 18, 35)
    )
    return captured, hooks


def _source_conditioning_tensor(conditioning: object) -> Any:
    rows = cast("list[list[object]]", conditioning)
    if len(rows) != 1:
        raise ComparisonError("source conditioning must contain one row")
    return rows[0][0]


def _source_conditioning_scale(conditioning: object) -> Any:
    rows = cast("list[list[object]]", conditioning)
    if len(rows) != 1:
        raise ComparisonError("source conditioning must contain one row")
    metadata = cast("dict[str, object]", rows[0][1])
    scale = metadata.get("conditioning_scale")
    if scale is None:
        raise ComparisonError("source conditioning has no condition scale")
    return scale


def _engine_comfyui(args: argparse.Namespace) -> int:
    boundary = _configure_comfyui(args.root, preflight=False)
    import torch  # pyright: ignore[reportMissingImports]

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(GPU_INDEX):
        raise ComparisonError("ComfyUI engine requires physical GPU 1 visibility")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ComparisonError("ComfyUI engine requires exactly one visible CUDA device")
    proofs = _proofs_from_receipt(args.artifacts_receipt)
    template = cast("dict[str, object]", json.loads(args.workload.read_text(encoding="utf-8")))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model = clip = vae = None
    hook = None
    captured: dict[str, str] = {}
    text_captured: dict[str, str] = {}
    text_hooks: list[Any] = []
    torch.cuda.set_device(0)
    print(
        json.dumps(
            {
                "status": "ready",
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "python": sys.executable,
                "torch": torch.__version__,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        for line in sys.stdin:
            phase = json.loads(line)["phase"]
            torch.cuda.reset_peak_memory_stats()
            setup_ns = 0
            if model is None:
                started = time.perf_counter_ns()
                with contextlib.redirect_stdout(sys.stderr):
                    model = boundary["sd"].load_diffusion_model(str(proofs["diffusion"].path))
                    clip = boundary["sd"].load_clip(
                        [str(proofs["text"].path)],
                        embedding_directory=[],
                        clip_type=boundary["sd"].CLIPType.MINIMAX,
                    )
                    vae_state = boundary["utils"].load_torch_file(str(proofs["vae"].path))
                    vae = boundary["sd"].VAE(sd=vae_state)
                torch.cuda.synchronize()
                setup_ns = time.perf_counter_ns() - started
                captured, hook = _capture_denoiser(model.model.diffusion_model, output_dir)
                text_captured, text_hooks = _capture_text_seams(clip.cond_stage_model, output_dir)
            conditioning_started = time.perf_counter_ns()
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                positive_result = boundary["MiniMaxMusic3TextEncode"].execute(
                    clip,
                    template["caption"],
                    template["lyrics"],
                    template["seed"],
                    template["max_duration"],
                    TEXT_CFG,
                    TOP_K,
                )
                positive = positive_result[0]
                negative = boundary["nodes"].ConditioningZeroOut().zero_out(positive)[0]
                seconds = positive_result[1]
                latent = boundary["EmptyMiniMaxMusic3LatentAudio"].execute(seconds, 1)[0]
            torch.cuda.synchronize()
            conditioning_ns = time.perf_counter_ns() - conditioning_started
            sigmas = boundary["samplers"].calculate_sigmas(
                model.get_model_object("model_sampling"), SCHEDULER, STEPS
            )
            noise = boundary["sample"].prepare_noise(latent["samples"], template["seed"])
            sampling_started = time.perf_counter_ns()
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                sampled = (
                    boundary["nodes"]
                    .KSampler()
                    .sample(
                        model,
                        template["seed"],
                        STEPS,
                        CFG,
                        SAMPLER,
                        SCHEDULER,
                        positive,
                        negative,
                        latent,
                        1.0,
                    )[0]
                )
            torch.cuda.synchronize()
            sampling_ns = time.perf_counter_ns() - sampling_started
            decode_started = time.perf_counter_ns()
            with contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
                audio = boundary["VAEDecodeAudioTiled"].execute(vae, sampled, TILE_SIZE, OVERLAP)[0]
            torch.cuda.synchronize()
            decode_ns = time.perf_counter_ns() - decode_started
            waveform = audio["waveform"]
            outputs = {
                "audio": _save_array(output_dir / f"{phase}-audio.npy", waveform),
                "latent": _save_array(output_dir / f"{phase}-latent.npy", sampled["samples"]),
                "wav": _save_wav(
                    output_dir / f"{phase}.wav", waveform, cast("int", audio["sample_rate"])
                ),
            }
            intermediates = {**captured, **text_captured}
            if phase == "cold":
                intermediates.update(
                    {
                        "negative_conditioning": _save_array(
                            output_dir / "negative-conditioning.npy",
                            _source_conditioning_tensor(negative),
                        ),
                        "negative_conditioning_scale": _save_array(
                            output_dir / "negative-conditioning-scale.npy",
                            _source_conditioning_scale(negative),
                        ),
                        "noise": _save_array(output_dir / "noise.npy", noise),
                        "positive_conditioning": _save_array(
                            output_dir / "positive-conditioning.npy",
                            _source_conditioning_tensor(positive),
                        ),
                        "positive_conditioning_scale": _save_array(
                            output_dir / "positive-conditioning-scale.npy",
                            _source_conditioning_scale(positive),
                        ),
                        "sigmas": _save_array(output_dir / "sigmas.npy", sigmas),
                    }
                )
            del positive_result, positive, negative, latent, sampled, audio, waveform
            gc.collect()
            reply = {
                "attention_backend": "sdpa",
                "audio_quality": _audio_quality_from_path(outputs["audio"]),
                "cold_setup_ns": setup_ns,
                "conditioning_ns": conditioning_ns,
                "decode_fallback_count": 0,
                "decode_ns": decode_ns,
                "generation_ns": conditioning_ns + sampling_ns + decode_ns,
                "intermediates": intermediates,
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                "oom": False,
                "outputs": outputs,
                "phase": phase,
                "residual_allocated_bytes": torch.cuda.memory_allocated(),
                "residual_reserved_bytes": torch.cuda.memory_reserved(),
                "rss_after_bytes": _current_rss(),
                "runtime": {
                    "compute_dtypes": {
                        "diffusion": str(model.model.get_dtype_inference()).removeprefix("torch."),
                        "text": "bfloat16",
                        "vae": str(vae.vae_dtype).removeprefix("torch."),
                    },
                    "diffusion": type(model.model.diffusion_model).__name__,
                    "executable": sys.executable,
                    "python": platform.python_version(),
                    "storage_dtypes": {
                        "diffusion": _state_dtypes(model.model.diffusion_model),
                        "text": _state_dtypes(clip.cond_stage_model),
                        "vae": _state_dtypes(vae.first_stage_model),
                    },
                    "text": type(clip.cond_stage_model).__name__,
                    "torch": torch.__version__,
                    "vae": type(vae.first_stage_model).__name__,
                },
                "sampling_ns": sampling_ns,
            }
            print(json.dumps(reply, sort_keys=True), flush=True)
    finally:
        if hook is not None:
            hook.remove()
        for text_hook in text_hooks:
            text_hook.remove()
        hook = None
        text_hooks.clear()
        text_hook = None
        model = clip = vae = None
        gc.collect()
        boundary["model_management"].unload_all_models()
        boundary["model_management"].cleanup_models()
        torch.cuda.empty_cache()
        _trim_host_allocator()
        _write_json(
            output_dir / "cleanup.json",
            {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "rss_before_exit_bytes": _current_rss(),
            },
        )
    return 0


def _audio_quality_from_path(path: str) -> dict[str, object]:
    import numpy as np

    value = np.load(path, allow_pickle=False)
    return {
        "clipped_fraction": float(np.mean(np.abs(value) >= 1.0)),
        "finite": bool(np.isfinite(value).all()),
        "peak": float(np.max(np.abs(value))),
        "rms": float(np.sqrt(np.mean(value**2))),
        "standard_deviation": float(np.std(value)),
    }


def _engine_dinkster(args: argparse.Namespace) -> int:
    _add_dinkster_sources(args.root)
    import torch  # pyright: ignore[reportMissingImports]
    from dinkster_inference import split_component_conditioning
    from dinkster_inference_torch import materialize_minimax_music3_conditioning, prepare_noise

    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(GPU_INDEX):
        raise ComparisonError("Dinkster engine requires physical GPU 1 visibility")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ComparisonError("Dinkster engine requires exactly one visible CUDA device")
    proofs = _proofs_from_receipt(args.artifacts_receipt)
    template = cast("dict[str, object]", json.loads(args.workload.read_text(encoding="utf-8")))
    selections = _select_executions(proofs)
    boundary = _production_boundary()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = model = clip = vae = None
    hook = None
    captured: dict[str, str] = {}
    text_captured: dict[str, str] = {}
    text_hooks: list[Any] = []
    torch.cuda.set_device(0)
    print(
        json.dumps(
            {
                "status": "ready",
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "python": sys.executable,
                "torch": torch.__version__,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        for line in sys.stdin:
            phase = json.loads(line)["phase"]
            torch.cuda.reset_peak_memory_stats()
            setup_ns = 0
            if loaded is None:
                started = time.perf_counter_ns()
                loaded = _load_dinkster_components(proofs, selections, boundary)
                model, clip, vae = loaded["model"], loaded["clip"], loaded["vae"]
                torch.cuda.synchronize()
                setup_ns = time.perf_counter_ns() - started
                captured, hook = _capture_denoiser(
                    model.runtime.assembled.diffusion, output_dir, prepared=True
                )
                text_captured, text_hooks = _capture_text_seams(clip.component, output_dir)
            conditioning_started = time.perf_counter_ns()
            text_result = boundary["text_encode"].execute(
                clip=clip,
                caption=template["caption"],
                lyrics=template["lyrics"],
                seed=template["seed"],
                max_duration=template["max_duration"],
                cfg_scale=TEXT_CFG,
                top_k=TOP_K,
            )
            positive = text_result["conditioning"]
            negative = boundary["conditioning_zero_out"].execute(conditioning=positive)[
                "conditioning"
            ]
            seconds = text_result["seconds"]
            latent = boundary["empty_latent"].execute(seconds=seconds, batch_size=1)["latent"]
            torch.cuda.synchronize()
            conditioning_ns = time.perf_counter_ns() - conditioning_started
            if phase == "cold":
                positive_carrier, _binding = split_component_conditioning(positive)
                negative_carrier, _negative_binding = split_component_conditioning(negative)
                positive_value = materialize_minimax_music3_conditioning(
                    positive_carrier, device="cpu"
                )
                negative_value = materialize_minimax_music3_conditioning(
                    negative_carrier, device="cpu"
                )
            sigmas = model.runtime.custom_sampling_sigmas(SCHEDULER, STEPS, 1.0)
            noise = prepare_noise(latent["samples"], cast("int", template["seed"]))
            sampling_started = time.perf_counter_ns()
            sampled = boundary["ksampler"].execute(
                model=model,
                seed=template["seed"],
                steps=STEPS,
                cfg=CFG,
                sampler_name=SAMPLER,
                scheduler=SCHEDULER,
                positive=positive,
                negative=negative,
                latent_image=latent,
                denoise=1.0,
            )["latent"]
            torch.cuda.synchronize()
            sampling_ns = time.perf_counter_ns() - sampling_started
            decode_started = time.perf_counter_ns()
            audio = boundary["vae_decode_tiled"].execute(
                samples=sampled, vae=vae, tile_size=TILE_SIZE, overlap=OVERLAP
            )["audio"]
            torch.cuda.synchronize()
            decode_ns = time.perf_counter_ns() - decode_started
            waveform = audio["waveform"]
            outputs = {
                "audio": _save_array(output_dir / f"{phase}-audio.npy", waveform),
                "latent": _save_array(output_dir / f"{phase}-latent.npy", sampled["samples"]),
                "wav": _save_wav(
                    output_dir / f"{phase}.wav", waveform, cast("int", audio["sample_rate"])
                ),
            }
            intermediates = {**captured, **text_captured}
            if phase == "cold":
                intermediates.update(
                    {
                        "negative_conditioning": _save_array(
                            output_dir / "negative-conditioning.npy", negative_value.embeddings
                        ),
                        "negative_conditioning_scale": _save_array(
                            output_dir / "negative-conditioning-scale.npy",
                            negative_value.conditioning_scale.reshape(-1, 1, 1),
                        ),
                        "noise": _save_array(output_dir / "noise.npy", noise),
                        "positive_conditioning": _save_array(
                            output_dir / "positive-conditioning.npy", positive_value.embeddings
                        ),
                        "positive_conditioning_scale": _save_array(
                            output_dir / "positive-conditioning-scale.npy",
                            positive_value.conditioning_scale.reshape(-1, 1, 1),
                        ),
                        "sigmas": _save_array(
                            output_dir / "sigmas.npy", torch.tensor(sigmas, dtype=torch.float32)
                        ),
                    }
                )
                del positive_carrier, negative_carrier, positive_value, negative_value
            del text_result, positive, negative, latent, sampled, audio, waveform
            gc.collect()
            reply = {
                "attention_backend": "sdpa",
                "audio_quality": _audio_quality_from_path(outputs["audio"]),
                "cold_setup_ns": setup_ns,
                "conditioning_ns": conditioning_ns,
                "decode_fallback_count": 0,
                "decode_ns": decode_ns,
                "generation_ns": conditioning_ns + sampling_ns + decode_ns,
                "intermediates": intermediates,
                "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
                "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
                "oom": False,
                "outputs": outputs,
                "phase": phase,
                "residual_allocated_bytes": torch.cuda.memory_allocated(),
                "residual_reserved_bytes": torch.cuda.memory_reserved(),
                "rss_after_bytes": _current_rss(),
                "runtime": {
                    "compute_dtypes": {
                        "diffusion": str(
                            model.runtime.assembled.compute_dtype("diffusion")
                        ).removeprefix("torch."),
                        "text": str(clip.recipe.knobs.text_dtype),
                        "vae": str(vae.recipe.knobs.vae_dtype),
                    },
                    "diffusion": type(model.runtime.assembled.diffusion).__name__,
                    "executable": sys.executable,
                    "family": model.runtime.family.id,
                    "python": platform.python_version(),
                    "storage_dtypes": {
                        "diffusion": _state_dtypes(model.runtime.assembled.diffusion),
                        "text": _state_dtypes(clip.component),
                        "vae": _state_dtypes(vae.component),
                    },
                    "text": type(clip.component).__name__,
                    "torch": torch.__version__,
                    "vae": type(vae.component).__name__,
                },
                "sampling_ns": sampling_ns,
            }
            print(json.dumps(reply, sort_keys=True), flush=True)
    finally:
        if hook is not None:
            hook.remove()
        for text_hook in text_hooks:
            text_hook.remove()
        hook = None
        text_hooks.clear()
        text_hook = None
        handles = () if loaded is None else (loaded["clip"], loaded["vae"], loaded["model"])
        loaded = model = clip = vae = None
        for handle in handles:
            if not handle.released:
                handle.terminal_release()
        handles = ()
        handle = None
        gc.collect()
        torch.cuda.empty_cache()
        _trim_host_allocator()
        _write_json(
            output_dir / "cleanup.json",
            {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "rss_before_exit_bytes": _current_rss(),
            },
        )
    return 0


def _tree_pids(pid: int) -> set[int]:
    parents = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            parents[int(entry.name)] = int((entry / "stat").read_text().split()[3])
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    members = {pid}
    changed = True
    while changed:
        changed = False
        for child, parent in parents.items():
            if parent in members and child not in members:
                members.add(child)
                changed = True
    return members


def _tree_rss(pid: int) -> int:
    total = 0
    for member in _tree_pids(pid):
        try:
            for line in Path(f"/proc/{member}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return total


def _gpu_process_memory(pids: set[int]) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={GPU_INDEX}",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        return 0
    total = 0
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 2 and values[0].isdigit() and int(values[0]) in pids:
            total += int(values[1]) * 1024 * 1024
    return total


def _sample_process(pid: int, stop: threading.Event, peaks: dict[str, int]) -> None:
    while not stop.wait(0.1):
        pids = _tree_pids(pid)
        peaks["rss"] = max(peaks["rss"], _tree_rss(pid))
        peaks["vram"] = max(peaks["vram"], _gpu_process_memory(pids))


def _read_line(pipe: Any, timeout: float) -> str:
    lines: list[str] = []
    reader = threading.Thread(target=lambda: lines.append(pipe.readline()), daemon=True)
    reader.start()
    reader.join(timeout=timeout)
    if reader.is_alive():
        raise ComparisonError("engine reply timed out")
    if not lines or not lines[0]:
        raise ComparisonError("engine closed its reply stream")
    return lines[0]


def _engine_command(args: argparse.Namespace, engine: str, output_dir: Path) -> list[str]:
    python = args.comfyui_python if engine == "comfyui" else args.dinkster_python
    return [
        str(python),
        str(Path(__file__).resolve()),
        "engine",
        "--engine",
        engine,
        "--root",
        str(args.comfyui_root if engine == "comfyui" else args.dinkster_root),
        "--artifacts-receipt",
        str(args.artifacts_receipt),
        "--workload",
        str(args.workload),
        "--output-dir",
        str(output_dir),
    ]


def _run_process(args: argparse.Namespace, engine: str, output_dir: Path) -> dict[str, object]:
    process_dir = output_dir / engine
    process_dir.mkdir(parents=True, exist_ok=False)
    command = _engine_command(args, engine, process_dir)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
    if args.dependency_root is not None:
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(args.dependency_root), environment.get("PYTHONPATH", "")))
        )
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        start_new_session=True,
        env=environment,
    )
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    stderr_chunks: list[str] = []
    stderr_reader = threading.Thread(
        target=lambda: stderr_chunks.append(process.stderr.read()), daemon=True
    )
    stderr_reader.start()
    replies = {}
    ready: dict[str, object] = {}
    try:
        ready = json.loads(_read_line(process.stdout, 300.0))
        if ready.get("status") != "ready":
            raise ComparisonError(f"{engine} did not reach its launch boundary")
        for phase in PHASES:
            peaks = {"rss": 0, "vram": 0}
            stop = threading.Event()
            sampler = threading.Thread(
                target=_sample_process, args=(process.pid, stop, peaks), daemon=True
            )
            started = time.perf_counter_ns()
            sampler.start()
            try:
                process.stdin.write(json.dumps({"phase": phase}) + "\n")
                process.stdin.flush()
                reply = json.loads(_read_line(process.stdout, 7200.0))
                ended = time.perf_counter_ns()
            finally:
                stop.set()
                sampler.join(timeout=10)
            if reply.get("phase") != phase:
                raise ComparisonError(f"{engine} phase protocol mismatch")
            reply["external_request_ns"] = ended - started
            reply["peak_process_rss_bytes"] = peaks["rss"]
            reply["peak_process_vram_bytes"] = peaks["vram"]
            reply["residual_process_vram_bytes"] = _gpu_process_memory(_tree_pids(process.pid))
            replies[phase] = reply
            _write_json(process_dir / "record.partial.json", {"ready": ready, "replies": replies})
        process.stdin.close()
        process.wait(timeout=300)
        stderr_reader.join(timeout=10)
        if process.returncode:
            raise ComparisonError(f"{engine} exited with {process.returncode}")
    except BaseException as error:
        _write_json(
            process_dir / "failure.json",
            {
                "error": {"message": str(error), "type": type(error).__name__},
                "replies": replies,
                "traceback": traceback.format_exc(),
            },
        )
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)
        stderr_reader.join(timeout=10)
        raise
    finally:
        (process_dir / "stderr.txt").write_text("".join(stderr_chunks), encoding="utf-8")
    record = {
        "cleanup": json.loads((process_dir / "cleanup.json").read_text(encoding="utf-8")),
        "command": command,
        "engine": engine,
        "ready": ready,
        "replies": replies,
    }
    _write_json(process_dir / "record.json", record)
    return record


def _array_metrics(left_path: str, right_path: str, limits: dict[str, float]) -> dict[str, object]:
    import numpy as np

    left = np.load(left_path, mmap_mode="r", allow_pickle=False)
    right = np.load(right_path, mmap_mode="r", allow_pickle=False)
    if left.shape != right.shape:
        return {"left_shape": list(left.shape), "pass": False, "right_shape": list(right.shape)}
    flat_left = left.reshape(-1)
    flat_right = right.reshape(-1)
    maximum = total_abs = total_square = dot = norm_left = norm_right = 0.0
    close = True
    exact = True
    count = flat_left.size
    for start in range(0, count, 1_000_000):
        lvalue = np.asarray(flat_left[start : start + 1_000_000], dtype=np.float64)
        rvalue = np.asarray(flat_right[start : start + 1_000_000], dtype=np.float64)
        difference = np.abs(lvalue - rvalue)
        maximum = max(maximum, float(difference.max(initial=0.0)))
        total_abs += float(difference.sum())
        total_square += float(np.square(lvalue - rvalue).sum())
        dot += float(np.dot(lvalue, rvalue))
        norm_left += float(np.dot(lvalue, lvalue))
        norm_right += float(np.dot(rvalue, rvalue))
        exact = exact and bool(np.array_equal(lvalue, rvalue))
        close = close and bool(
            np.allclose(lvalue, rvalue, atol=limits["atol"], rtol=limits["rtol"])
        )
    mean_abs = total_abs / max(1, count)
    rmse = math.sqrt(total_square / max(1, count))
    denominator = math.sqrt(norm_left * norm_right)
    cosine = 1.0 if exact or (denominator == 0.0 and total_square == 0.0) else dot / denominator
    passed = close and cosine >= limits["cosine_min"]
    if "mean_abs_max" in limits:
        passed = passed and mean_abs <= limits["mean_abs_max"]
    if "rmse_max" in limits:
        passed = passed and rmse <= limits["rmse_max"]
    return {
        "cosine": cosine,
        "exact": exact,
        "limits": limits,
        "max_abs": maximum,
        "mean_abs": mean_abs,
        "pass": passed,
        "rmse": rmse,
        "shape": list(left.shape),
    }


def _summary(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    p95 = ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]
    return {
        "max": ordered[-1],
        "median": float(statistics.median(ordered)),
        "p95": p95,
        "standard_deviation": statistics.stdev(ordered),
    }


def _compare_records(
    records: dict[str, dict[str, object]],
    sources: dict[str, object],
    artifacts: dict[str, object],
    workload: dict[str, object],
) -> dict[str, object]:
    comfy = cast("dict[str, Any]", records["comfyui"])
    dinkster = cast("dict[str, Any]", records["dinkster"])
    comparisons = []
    names = (
        ("positive_conditioning", "conditioning"),
        ("negative_conditioning", "conditioning"),
        ("positive_conditioning_scale", "noise"),
        ("negative_conditioning_scale", "noise"),
        *((f"text_layer_{index}_input", "conditioning") for index in (0, 18, 35)),
        ("text_layer_0_qkv", "conditioning"),
        ("text_layer_0_attention", "conditioning"),
        ("text_layer_0_gate_up", "conditioning"),
        ("text_final_norm_input", "conditioning"),
        ("text_final_norm_output", "conditioning"),
        ("text_logits_input", "conditioning"),
        ("text_logits", "conditioning"),
        ("sigmas", "sigmas"),
        ("noise", "noise"),
        ("denoiser_input", "noise"),
        ("denoiser_timestep", "noise"),
        ("denoiser_conditioning_scale", "noise"),
        ("denoiser_aligned_condition", "conditioning"),
        ("denoiser_output", "denoiser"),
    )
    for name, limits in names:
        comparisons.append(
            {
                "name": name,
                **_array_metrics(
                    comfy["replies"]["cold"]["intermediates"][name],
                    dinkster["replies"]["cold"]["intermediates"][name],
                    COMPARISON_LIMITS[limits],
                ),
            }
        )
    for phase in PHASES:
        for name in ("latent", "audio"):
            comparisons.append(
                {
                    "name": f"{phase}-{name}",
                    **_array_metrics(
                        comfy["replies"][phase]["outputs"][name],
                        dinkster["replies"][phase]["outputs"][name],
                        COMPARISON_LIMITS[name],
                    ),
                }
            )
    source_repeatability = []
    for phase in PHASES[1:]:
        for name in ("latent", "audio"):
            source_repeatability.append(
                {
                    "name": f"cold-vs-{phase}-{name}",
                    **_array_metrics(
                        comfy["replies"]["cold"]["outputs"][name],
                        comfy["replies"][phase]["outputs"][name],
                        COMPARISON_LIMITS[name],
                    ),
                }
            )
    for record in (comfy, dinkster):
        for phase in PHASES:
            reply = record["replies"][phase]
            if reply["oom"] or reply["decode_fallback_count"]:
                raise ComparisonError(f"{record['engine']} used a fallback or raised OOM")
            if not reply["audio_quality"]["finite"]:
                raise ComparisonError(f"{record['engine']} produced non-finite audio")
    latency = {}
    for name, key in (
        ("generation", "generation_ns"),
        ("conditioning", "conditioning_ns"),
        ("sampling", "sampling_ns"),
        ("decode", "decode_ns"),
        ("external_request", "external_request_ns"),
    ):
        baseline = _summary([comfy["replies"][phase][key] for phase in WARM_PHASES])
        candidate = _summary([dinkster["replies"][phase][key] for phase in WARM_PHASES])
        source_jitter_tolerance = max(
            baseline["median"] * LATENCY_LAUNCH_JITTER_FLOOR,
            baseline["standard_deviation"] * LATENCY_SOURCE_JITTER_SIGMAS,
        )
        latency[name] = {
            "comfyui_ns": baseline,
            "dinkster_ns": candidate,
            "source_jitter_tolerance_ns": source_jitter_tolerance,
            "pass": all(
                candidate[metric] <= baseline[metric] + source_jitter_tolerance
                for metric in ("median", "p95", "max")
            ),
        }
    cold = {}
    for key in ("cold_setup_ns", "generation_ns", "external_request_ns"):
        cold[key] = {
            "comfyui": comfy["replies"]["cold"][key],
            "dinkster": dinkster["replies"]["cold"][key],
            "pass": dinkster["replies"]["cold"][key] <= comfy["replies"]["cold"][key],
        }
    memory = {}
    for key in (
        "max_memory_allocated_bytes",
        "max_memory_reserved_bytes",
        "peak_process_rss_bytes",
        "peak_process_vram_bytes",
        "residual_allocated_bytes",
        "residual_reserved_bytes",
        "rss_after_bytes",
        "residual_process_vram_bytes",
    ):
        baseline = max(comfy["replies"][phase][key] for phase in PHASES)
        candidate = max(dinkster["replies"][phase][key] for phase in PHASES)
        tolerance = CUDA_ALLOCATOR_TOLERANCE_BYTES if key == "max_memory_allocated_bytes" else 0
        memory[key] = {
            "comfyui": baseline,
            "dinkster": candidate,
            "tolerance_bytes": tolerance,
            "pass": candidate <= baseline + tolerance,
        }
    cleanup = {}
    for key in ("allocated_bytes", "reserved_bytes", "rss_before_exit_bytes"):
        baseline = comfy["cleanup"][key]
        candidate = dinkster["cleanup"][key]
        tolerance = CUDA_ALLOCATOR_TOLERANCE_BYTES if key == "reserved_bytes" else 0
        cleanup[key] = {
            "comfyui": baseline,
            "dinkster": candidate,
            "tolerance_bytes": tolerance,
            "pass": candidate <= baseline + tolerance,
        }
    repeatability_pass = all(item["pass"] for item in source_repeatability)
    correctness_pass = repeatability_pass and all(item["pass"] for item in comparisons)
    performance_pass = all(item["pass"] for item in latency.values()) and all(
        item["pass"] for item in cold.values()
    )
    memory_pass = all(item["pass"] for item in memory.values())
    cleanup_pass = all(item["pass"] for item in cleanup.values())
    return {
        "artifacts": artifacts,
        "cleanup": {
            "metrics": cleanup,
            "pass": cleanup_pass,
            "tolerance_basis": (
                "CUDA reserved memory is compared within one 2 MiB allocator segment."
            ),
        },
        "correctness": {
            "comparisons": comparisons,
            "pass": correctness_pass,
            "source_repeatability": source_repeatability,
            "tolerance_basis": (
                "ComfyUI cold and repeated warm outputs were bit-exact, so every captured "
                "cross-engine seam and final output is required to be bit-exact."
            ),
        },
        "fallback_oom": {"pass": True, "result": "No fallback or OOM occurred."},
        "memory": {
            "metrics": memory,
            "pass": memory_pass,
            "tolerance_basis": (
                "Peak allocated CUDA memory is compared within one 2 MiB allocator segment; "
                "reserved and process-level memory remain strict."
            ),
        },
        "overall_pass": correctness_pass and performance_pass and memory_pass and cleanup_pass,
        "performance": {
            "cold": cold,
            "pass": performance_pass,
            "tolerance_basis": (
                "Warm latency uses the larger of a 0.05% launch floor and five source "
                "sample standard deviations measured across seven warm runs; cold latency "
                "remains strict."
            ),
            "warm": latency,
        },
        "schema": "dinkster.minimax_music3.matched-e2e.v1",
        "sources": sources,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "workload": workload,
    }


def _gpu_identity() -> dict[str, object]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={GPU_INDEX}",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    values = [value.strip() for value in result.stdout.strip().split(",")]
    if result.returncode or len(values) != 5 or values[2] != GPU_UUID:
        raise ComparisonError(f"physical GPU 1 identity mismatch: {values}")
    return {
        "driver": values[4],
        "index": int(values[0]),
        "memory_total_mib": int(values[3]),
        "name": values[1],
        "uuid": values[2],
    }


def _gpu_claim() -> dict[str, str]:
    metadata = GPU_CLAIM_PATH.read_text(encoding="utf-8").strip()
    fields_present = (
        f"thread={THREAD_ID}" in metadata and "purpose=" in metadata and "utc=" in metadata
    )
    if not fields_present:
        raise ComparisonError("GPU 1 claim metadata is missing or belongs to another thread")
    with GPU_CLAIM_PATH.open("r", encoding="utf-8") as claim:
        try:
            fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(claim, fcntl.LOCK_UN)
            raise ComparisonError("GPU 1 claim lock is not held by the launch shell")
    return {"metadata": metadata, "path": str(GPU_CLAIM_PATH)}


def _common_receipt(args: argparse.Namespace) -> tuple[dict[str, object], dict[str, object]]:
    sources = {
        "comfyui": _source_receipt(args.comfyui_root, COMFYUI_COMMIT),
        "dinkster": _source_receipt(args.dinkster_root),
        "workflow_templates": _source_receipt(args.templates_root, TEMPLATES_COMMIT),
    }
    template_path = args.templates_root / TEMPLATE_PATH
    template = _template_values(template_path)
    sources["workflow_templates"]["template"] = {
        "bytes": TEMPLATE_BYTES,
        "path": TEMPLATE_PATH,
        "sha256": TEMPLATE_SHA256,
    }
    proofs = _verify_artifacts(args.models_root)
    artifacts = _artifact_receipt(proofs)
    _write_json(args.artifacts_receipt, artifacts)
    _write_json(args.workload, _workload(template))
    return sources, artifacts


def _run_preflight(args: argparse.Namespace) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        raise ComparisonError("preflight requires CUDA_VISIBLE_DEVICES to be empty")
    sources, artifacts = _common_receipt(args)
    engines = {}
    for engine in ("comfyui", "dinkster"):
        root = args.comfyui_root if engine == "comfyui" else args.dinkster_root
        python = args.comfyui_python if engine == "comfyui" else args.dinkster_python
        command = [
            str(python),
            str(Path(__file__).resolve()),
            "engine",
            "--engine",
            engine,
            "--root",
            str(root),
            "--artifacts-receipt",
            str(args.artifacts_receipt),
            "--workload",
            str(args.workload),
            "--output-dir",
            str(args.output_dir / f"preflight-{engine}"),
            "--preflight",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        if args.dependency_root is not None:
            environment["PYTHONPATH"] = os.pathsep.join(
                filter(None, (str(args.dependency_root), environment.get("PYTHONPATH", "")))
            )
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=600, env=environment
        )
        if result.returncode:
            raise ComparisonError(f"{engine} preflight failed: {result.stderr}")
        engines[engine] = json.loads(result.stdout)
    receipt = {
        "artifacts": artifacts,
        "engines": engines,
        "schema": "dinkster.minimax_music3.matched-e2e.preflight.v1",
        "sources": sources,
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "workload": json.loads(args.workload.read_text(encoding="utf-8")),
    }
    _write_json(args.output_dir / "preflight.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


def _engine_preflight(args: argparse.Namespace) -> int:
    if args.engine == "comfyui":
        boundary = _configure_comfyui(args.root, preflight=True)
        import torch  # pyright: ignore[reportMissingImports]

        receipt = {
            "cuda_initialized": torch.cuda.is_initialized(),
            "nodes": sorted(key for key in boundary if key[0].isupper()),
            "torch": torch.__version__,
        }
    else:
        _add_dinkster_sources(args.root)
        proofs = _proofs_from_receipt(args.artifacts_receipt)
        selections = _select_executions(proofs)
        boundary = _production_boundary()
        receipt = {
            "cuda_initialized": False,
            "imported_torch": any(
                name == "torch" or name.startswith("torch.") for name in sys.modules
            ),
            "nodes": sorted(boundary),
            "selections": {
                role: {
                    "cache_tag": selection.cache_tag,
                    "diffusion_dtype": selection.diffusion_dtype,
                    "target": selection.target,
                    "text_dtype": selection.text_dtype,
                    "vae_dtype": selection.vae_dtype,
                }
                for role, selection in selections.items()
            },
        }
    if receipt["cuda_initialized"]:
        raise ComparisonError(f"{args.engine} preflight initialized CUDA")
    print(json.dumps(receipt, sort_keys=True))
    return 0


def _run_gate(args: argparse.Namespace) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(GPU_INDEX):
        raise ComparisonError("run requires CUDA_VISIBLE_DEVICES=1")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ComparisonError("run output directory must be absent or empty")
    sources, artifacts = _common_receipt(args)
    preflight = json.loads(args.preflight_receipt.read_text(encoding="utf-8"))
    if preflight["sources"] != sources or preflight["artifacts"] != artifacts:
        raise ComparisonError("preflight source or artifact receipt differs from this run")
    hardware = _gpu_identity()
    claim = _gpu_claim()
    records = {
        engine: _run_process(args, engine, args.output_dir) for engine in ("comfyui", "dinkster")
    }
    workload = cast("dict[str, object]", json.loads(args.workload.read_text(encoding="utf-8")))
    verdict = _compare_records(records, sources, artifacts, workload)
    verdict["claim"] = claim
    verdict["hardware"] = hardware
    verdict["preflight"] = preflight
    _write_json(args.output_dir / "verdict.json", verdict)
    print(json.dumps(verdict, sort_keys=True))
    if not verdict["overall_pass"]:
        raise ComparisonError("matched MiniMax Music 3 gate failed; see verdict.json")
    return 0


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--comfyui-root", type=Path, required=True)
    parser.add_argument("--dinkster-root", type=Path, required=True)
    parser.add_argument("--templates-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--comfyui-python", type=Path, required=True)
    parser.add_argument("--dinkster-python", type=Path, required=True)
    parser.add_argument("--dependency-root", type=Path)
    parser.add_argument("--artifacts-receipt", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    _add_common_arguments(preflight)
    run = subparsers.add_parser("run")
    _add_common_arguments(run)
    run.add_argument("--preflight-receipt", type=Path, required=True)
    engine = subparsers.add_parser("engine")
    engine.add_argument("--engine", choices=("comfyui", "dinkster"), required=True)
    engine.add_argument("--root", type=Path, required=True)
    engine.add_argument("--artifacts-receipt", type=Path, required=True)
    engine.add_argument("--workload", type=Path, required=True)
    engine.add_argument("--output-dir", type=Path, required=True)
    engine.add_argument("--preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        return _run_preflight(args)
    if args.command == "run":
        return _run_gate(args)
    if args.preflight:
        return _engine_preflight(args)
    return _engine_comfyui(args) if args.engine == "comfyui" else _engine_dinkster(args)


if __name__ == "__main__":
    raise SystemExit(main())
