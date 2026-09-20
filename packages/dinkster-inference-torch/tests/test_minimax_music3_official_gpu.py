from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_assets import AssetRef, digest_file
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    OFFICIAL_ARTIFACTS,
    MiniMaxMusic3ComponentRole,
    build_music_prompt,
    load_safetensors_header,
    minimax_music3_component_runtime_identity,
    plan_minimax_music3_split_component,
)
from dinkster_inference.devices import DType
from dinkster_inference_torch import (
    MiniMaxMusic3Dav,
    MiniMaxMusic3DiT,
    MiniMaxMusic3TextModel,
    enroll_component,
    load_minimax_music3_component,
    soft_empty_cache,
)
from gpu_test_gate import require_gpu_tests_enabled

_PREFLIGHT = os.environ.get("DINKSTER_MINIMAX_MUSIC3_OFFICIAL_PREFLIGHT") == "1"
pytestmark = pytest.mark.skipif(
    not _PREFLIGHT and not torch.cuda.is_available(), reason="CUDA GPU required (see README)"
)

_MODELS = Path(
    os.environ.get(
        "DINKSTER_MINIMAX_MUSIC3_MODELS",
        "/home/kosin/model-artifacts/dinkster-1084-minimax-music3",
    )
)
_SOURCE_PARITY_RECEIPT = Path(
    os.environ.get(
        "DINKSTER_MINIMAX_MUSIC3_SOURCE_PARITY_RECEIPT",
        Path(
            os.environ.get(
                "DINKSTER_INFERENCE_PARITY_RECORDS",
                Path(__file__).resolve().parents[3].parent
                / "dinkster-evidence"
                / "inference-parity"
                / "records",
            )
        )
        / "minimax-music3-official/receipt.json",
    )
)
_ROLES: dict[str, MiniMaxMusic3ComponentRole] = {
    "diffusion_models/minimax_music3_dit_fp16.safetensors": "diffusion",
    "diffusion_models/minimax_music3_dit_fp32.safetensors": "diffusion",
    "diffusion_models/minimax_music3_dit_int8_convrot.safetensors": "diffusion",
    "text_encoders/minimax_music3_text_encoder_bf16.safetensors": "text",
    "text_encoders/minimax_music3_text_encoder_pruned_bf16.safetensors": "text",
    "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors": "text",
    "vae/minimax_music3_dav.safetensors": "vae",
}
_COMPUTE_DTYPES: dict[str, tuple[DType, torch.dtype]] = {
    "diffusion_models/minimax_music3_dit_fp16.safetensors": (FLOAT16, torch.float16),
    "diffusion_models/minimax_music3_dit_fp32.safetensors": (FLOAT32, torch.float32),
    "diffusion_models/minimax_music3_dit_int8_convrot.safetensors": (
        BFLOAT16,
        torch.bfloat16,
    ),
    "text_encoders/minimax_music3_text_encoder_bf16.safetensors": (
        BFLOAT16,
        torch.bfloat16,
    ),
    "text_encoders/minimax_music3_text_encoder_pruned_bf16.safetensors": (
        BFLOAT16,
        torch.bfloat16,
    ),
    "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors": (
        BFLOAT16,
        torch.bfloat16,
    ),
    "vae/minimax_music3_dav.safetensors": (FLOAT32, torch.float32),
}


class _Resolver:
    def __init__(self, digest: str, path: Path) -> None:
        self.digest = digest
        self.path = path

    def resolve(self, digest: str) -> Path | None:
        return self.path if digest == self.digest else None


def _load_official(path_key: str) -> tuple[Any, Any]:
    path = _MODELS / path_key
    source_url, expected_size, expected_sha256 = OFFICIAL_ARTIFACTS[path_key]
    assert source_url.startswith("https://huggingface.co/Comfy-Org/MiniMax-Music-3/resolve/")
    assert path.stat().st_size == expected_size
    with path.open("rb") as artifact:
        assert hashlib.file_digest(artifact, "sha256").hexdigest() == expected_sha256
    digest = digest_file(path)
    asset = AssetRef(digest, path.name, expected_size, resolver=_Resolver(digest, path))
    role = _ROLES[path_key]
    identity_dtype, compute_dtype = _COMPUTE_DTYPES[path_key]
    source = load_safetensors_header(
        path,
        asset_digest=asset.digest,
        asset_size=asset.size,
    )
    plan = plan_minimax_music3_split_component(source, role=role, path=path)
    identity = minimax_music3_component_runtime_identity(plan, role, identity_dtype)
    loaded = load_minimax_music3_component(
        path,
        asset=asset,
        expected_role=role,
        expected_identity=identity,
        compute_dtype=compute_dtype,
        attention_backend=cast("Any", {"diffusion": "flux", "text": "qwen"}.get(role)),
    )
    assert loaded.plan == plan
    assert loaded.runtime_identity == identity
    return loaded, compute_dtype


def _release(mechanism: Any) -> None:
    mechanism.unload()
    gc.collect()
    soft_empty_cache(torch.device("cuda:0"))


@pytest.mark.skipif(
    not all((_MODELS / path_key).is_file() for path_key in OFFICIAL_ARTIFACTS),
    reason="official MiniMax Music 3 artifacts not present (set DINKSTER_MINIMAX_MUSIC3_MODELS)",
)
def test_official_minimax_music3_artifacts_load_and_execute_on_cuda() -> None:
    require_gpu_tests_enabled()
    assert set(OFFICIAL_ARTIFACTS) == set(_ROLES) == set(_COMPUTE_DTYPES)
    assert torch.cuda.device_count() == 1

    if _PREFLIGHT:
        for path_key, (_url, size, sha256) in OFFICIAL_ARTIFACTS.items():
            path = _MODELS / path_key
            assert path.stat().st_size == size
            with path.open("rb") as artifact:
                assert hashlib.file_digest(artifact, "sha256").hexdigest() == sha256
        assert not torch.cuda.is_initialized()
        return

    prompt = build_music_prompt("Warm solo piano.", "[instrumental]")
    for path_key, role in _ROLES.items():
        loaded, compute_dtype = _load_official(path_key)
        mechanism = enroll_component(
            loaded.module,
            load_device="cuda:0",
            offload_device="cpu",
        )
        mechanism.partially_load(None)
        with torch.inference_mode():
            if role == "diffusion":
                module = cast("MiniMaxMusic3DiT", loaded.module)
                latent = torch.zeros((1, 128, 3), device="cuda:0", dtype=compute_dtype)
                timestep = torch.full((1,), 0.5, device="cuda:0")
                context = torch.zeros((1, 1, 8 * 4096), device="cuda:0", dtype=compute_dtype)
                output = module(
                    latent,
                    timestep,
                    context,
                    torch.ones((1, 1, 1), device="cuda:0", dtype=compute_dtype),
                )
                assert output.shape == latent.shape and bool(torch.isfinite(output).all())
            elif role == "text":
                module = cast("MiniMaxMusic3TextModel", loaded.module)
                assert loaded.tokenizer is not None
                tokens = loaded.tokenizer.encode(prompt, add_special_tokens=False).ids
                hidden = module.generate(
                    torch.tensor([tokens]),
                    197122968890040,
                    2,
                    compute_dtype=compute_dtype,
                    cfg_scale=1.7,
                    top_k=50,
                )
                assert 1 <= hidden.shape[0] <= 2
                assert hidden.shape[1] == 8 * 4096 and bool(torch.isfinite(hidden).all())
            else:
                module = cast("MiniMaxMusic3Dav", loaded.module)
                latent = torch.zeros((1, 128, 4), device="cuda:0")
                output = module.decode(latent)
                assert output.shape == (1, 2, 4 * 512)
                assert bool(torch.isfinite(output).all())
        _release(mechanism)
        del loaded, mechanism


@pytest.mark.skipif(
    not _SOURCE_PARITY_RECEIPT.is_file(),
    reason="official MiniMax Music 3 source-parity receipt is not present",
)
def test_official_minimax_music3_source_parity_on_cuda() -> None:
    require_gpu_tests_enabled()
    receipt = json.loads(_SOURCE_PARITY_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["schema"] == "dinkster.minimax_music3.matched-e2e.v1"
    assert receipt["overall_pass"] is True
    assert receipt["sources"]["comfyui"]["commit"] == ("345c9190497c82cff53e71fb4ae00d1e135a6542")
    assert receipt["sources"]["workflow_templates"]["commit"] == (
        "8417f4f2a8380556070721d0ff1da8285d6e5438"
    )
    assert receipt["correctness"]["pass"] is True
    assert receipt["performance"]["pass"] is True
    assert receipt["memory"]["pass"] is True
    assert receipt["cleanup"]["pass"] is True
    assert receipt["fallback_oom"]["pass"] is True
