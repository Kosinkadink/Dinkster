"""Scheduled runtime activation integration proof.

The regional condition-scale carrier below isolates runtime integration from
the native call-site adapter, which has separate end-to-end coverage.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from dinkster_inference import (
    FLUX_DEV,
    AreaDescriptor,
    AreaUnits,
    Conditioning,
    EncoderStream,
    InferenceTypeRegistry,
    PercentRange,
    SamplingGuidance,
    ScheduledEncodeRequest,
    ScheduledPrompt,
    ScheduledPromptRoute,
    StepEvent,
    cfg_combine,
    encode_conditioning_carrier,
)
from dinkster_inference_torch import (
    ScheduledSamplingError,
    materialize_regions,
    payload_binding_to_tensor,
)
from dinkster_inference_torch import scaled_patches as scaled_module
from golden_files import platform_golden_path
from test_denoise import tiny_cond, tiny_latent
from test_regional import _carrier as regional_carrier
from test_scheduled import flux_runtime
from test_scheduled_sampling import _metadata, _patch_resolver, _runtime

GOLDEN_ROOT = Path(__file__).parent / "goldens"
PINNED_COMFYUI = "b78cec879b9460d5cb25228a83a942fb78d2cd24"
GOLDEN_SHA256 = {
    "scheduled_sampling_goldens.json": (
        "463785eda348a3f2ca3bc44262200eb5704bb25ab14e54515b5bee643f88f45b"
    ),
    "grouped_regional_goldens.json": (
        "a12f7b0a1dc3558f042b773b70d52204c288328e5a330b9f07a2e040bc40bfdc"
    ),
    "regional_goldens.json": ("b878662b3a27a748c34640d92c2367acb606e66ddad7171fa0964343c7b02de9"),
    "regional_goldens.win32-py3.12.11-torch2.13.0+cpu.json": (
        "3f4687f6f3c8e51d987c6bbd7988d996e42b98fb55fe91382c6fe4f0f748c2c1"
    ),
    "regional_goldens.win32-py3.12.11-torch2.13.0+cu130.json": (
        "952203e2312bda63eaf4bdab1492a9e398205e26a7055eea0ef7b7ba6f7b8986"
    ),
    "regional_goldens.darwin-py3.12.11-torch2.13.0.json": (
        "b70e06f2084393f0655c160ba7af51b1b395f1aaa97699e8086bb0cc90abf198"
    ),
    "scaled_patch_goldens.json": (
        "5d022ad458e00eabf95bd50dbad0ec230a4cbe3da08f85c11f61fb8ea7cb6b28"
    ),
    "scaled_patch_goldens.darwin-py3.12.11-torch2.13.0.json": (
        "206e3c5f8c43fabcf0e15bbfc2ae19d8ac7316e901d2c331f20a37a78e7c054c"
    ),
    "scaled_patch_goldens.win32-py3.12.11-torch2.13.0+cpu.json": (
        "e63db53ec50755cc1866c9eb31882563f4806348db1401010f9291592ab734df"
    ),
    "scaled_patch_goldens.win32-py3.12.11-torch2.13.0+cu130.json": (
        "d0fa72145bb774ff88d8011efdec73b9ca6bbf68bd87ec6e5bde8e0b36b6f60c"
    ),
}
SCHEDULE = PercentRange(0.0, 0.8)


class _FixedEncoder:
    def __init__(self, value: Conditioning[torch.Tensor]) -> None:
        self._value = value

    def encode(self, _spans: object) -> Conditioning[torch.Tensor]:
        return self._value


def _tensor(record: dict[str, object]) -> torch.Tensor:
    return torch.tensor(record["data"], dtype=torch.float32).reshape(
        cast("list[int]", record["shape"])
    )


def _bytes(value: torch.Tensor) -> bytes:
    return bytes(value.detach().cpu().contiguous().view(torch.uint8).reshape(-1).tolist())


def _assert_clean(model: torch.nn.Module) -> None:
    assert not any(module._forward_hooks for module in model.modules())
    assert all(module not in scaled_module._ACTIVE_MODELS for module in model.modules())


def _goldens() -> dict[str, dict[str, object]]:
    loaded: dict[str, dict[str, object]] = {}
    for name in (
        "scheduled_sampling_goldens.json",
        "grouped_regional_goldens.json",
        "regional_goldens.json",
        "scaled_patch_goldens.json",
    ):
        path = GOLDEN_ROOT / name
        if name in ("regional_goldens.json", "scaled_patch_goldens.json"):
            path = platform_golden_path(path)
        expected_hash = GOLDEN_SHA256[path.name]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected_hash
        document = cast("dict[str, object]", json.loads(raw))
        reference = cast("dict[str, object]", document["reference"])
        assert reference["commit"] == PINNED_COMFYUI
        loaded[name] = document
    return loaded


def test_scheduled_encode_to_regional_scaled_sampling_is_transactional() -> None:
    """Exercise scheduled encoding, regional scaling, sampling, and cleanup."""
    goldens = _goldens()
    scheduled_golden = goldens["scheduled_sampling_goldens.json"]
    cond_fact = torch.tensor([[[[1.0, -2.0], [3.0, -4.0]]]], dtype=torch.float32)
    uncond_fact = torch.tensor([[[[-0.5, 0.25], [1.5, -2.0]]]], dtype=torch.float32)
    for case in cast("list[dict[str, Any]]", scheduled_golden["cases"]):
        combined = cfg_combine(cond_fact, uncond_fact, case["scale"])
        assert tuple(combined.shape) == tuple(case["shape"])
        assert combined.reshape(-1).tolist() == case["data"]

    grouped_golden = goldens["grouped_regional_goldens.json"]
    assert cast("dict[str, object]", grouped_golden["facts"])["compatible_physical_order"] == [
        "2",
        "1",
        "0",
    ]
    scaled_golden = goldens["scaled_patch_goldens.json"]
    assert cast(
        "dict[str, object]",
        cast("dict[str, object]", scaled_golden["inputs"])["scales"],
    )["data"] == [0.0, 1.0, -0.5]

    # Start from the real Flux ScheduledEncodeRequest/encode_text_scheduled path.
    encoded_runtime = flux_runtime()
    encoded_value = tiny_cond("scheduled-runtime-encoded")
    cast("Any", encoded_runtime)._t5_encoder = _FixedEncoder(Conditioning(encoded_value.embeddings))
    cast("Any", encoded_runtime)._clip_encoder = _FixedEncoder(encoded_value)
    request = ScheduledEncodeRequest(
        (
            ScheduledPrompt(
                SCHEDULE,
                (
                    ScheduledPromptRoute(EncoderStream.CLIP_L, "regional lighthouse"),
                    ScheduledPromptRoute(EncoderStream.T5, "regional lighthouse"),
                ),
            ),
        )
    )
    encoded = encoded_runtime.encode_text_scheduled(request, type_registry=InferenceTypeRegistry())
    repeated_encoded = encoded_runtime.encode_text_scheduled(
        request, type_registry=InferenceTypeRegistry()
    )
    encoded_bytes = encode_conditioning_carrier(encoded)
    assert encoded_bytes == encode_conditioning_carrier(repeated_encoded)
    assert encoded.conditioning.records[0].schedule == SCHEDULE
    text = payload_binding_to_tensor(
        next(binding for binding in encoded.bindings if binding.space == "conditioning-text")
    )
    pooled = payload_binding_to_tensor(
        next(binding for binding in encoded.bindings if binding.space == "conditioning-pooled")
    )
    assert torch.equal(text, encoded_value.embeddings)
    assert encoded_value.pooled is not None and torch.equal(pooled, encoded_value.pooled)

    # Build canonical regional, mask, scheduled diffusion, and scale facts to
    # isolate runtime integration from the native node adapter.
    metadata = _metadata(FLUX_DEV.id)
    mask = torch.tensor([[0.0, 1.0], [0.5, 0.25]], dtype=torch.float32)
    conditional = regional_carrier(
        (
            {
                "text": text,
                "pooled": pooled,
                "area": AreaDescriptor(0.5, 0.75, 0.25, 0.125, AreaUnits.PERCENT),
                "mask": mask,
                "mask_strength": 0.75,
                "scale": torch.tensor([1.5], dtype=torch.float32),
                "schedule": SCHEDULE,
                "extension_metadata": metadata,
            },
            {
                "text": text * 0.5,
                "pooled": pooled,
                "scale": torch.tensor([0.5], dtype=torch.float32),
                "schedule": PercentRange(0.0, 0.9),
                "extension_metadata": metadata,
            },
        ),
        family=FLUX_DEV.id,
        tokens=text.shape[1],
        features=text.shape[2],
    )
    unconditional = regional_carrier(
        (
            {
                "text": tiny_cond("scheduled-runtime-uncond").embeddings,
                "pooled": tiny_cond("scheduled-runtime-uncond").pooled,
                "mask": torch.flip(mask, (0,)),
                "scale": torch.tensor([0.75], dtype=torch.float32),
                "schedule": PercentRange(0.0, 0.6),
                "extension_metadata": metadata,
            },
        ),
        family=FLUX_DEV.id,
        tokens=text.shape[1],
        features=text.shape[2],
    )
    carrier_bytes = encode_conditioning_carrier(conditional)
    regions = materialize_regions(conditional, FLUX_DEV.id, 8, 8, "cpu")
    assert len(regions) == 2
    assert regions[0].area == (4, 6, 2, 1)
    assert regions[0].mask is not None and tuple(regions[0].mask.shape) == (1, 8, 8)
    assert regions[0].scale_vector is not None
    assert torch.equal(regions[0].scale_vector, torch.tensor([1.5]))
    assert regions[0].patch_digest == metadata[-1][1]
    assert regions[0].schedule == SCHEDULE

    regional_golden = goldens["regional_goldens.json"]
    golden_region = regional_carrier(
        ({"text": text, "pooled": pooled, "mask": mask},),
        family=FLUX_DEV.id,
        tokens=text.shape[1],
        features=text.shape[2],
    )
    golden_mask = materialize_regions(golden_region, FLUX_DEV.id, 3, 5, "cpu")[0].mask
    assert golden_mask is not None
    assert torch.equal(
        golden_mask,
        _tensor(cast("dict[str, object]", regional_golden["mask_resize"])),
    )

    runtime = _runtime("flux")
    model = runtime.assembled.diffusion
    identity = runtime.runtime_identity
    ordinary_cond = tiny_cond("ordinary-cond")
    ordinary_uncond = tiny_cond("ordinary-uncond")
    ordinary_args: dict[str, Any] = {
        "cond": ordinary_cond,
        "cfg": SamplingGuidance(ordinary_uncond, 2.0),
        "sampler_id": "dinkster.euler",
        "scheduler_id": "dinkster.normal",
        "steps": 2,
        "seed": 37,
        "compute_dtype": torch.float32,
    }
    ordinary_before = runtime.sample(tiny_latent(), **ordinary_args)
    resolver_calls: list[object] = []
    scheduled_args: dict[str, Any] = {
        "cond": conditional,
        "cfg": SamplingGuidance(unconditional, 2.0),
        "resolver": _patch_resolver(model, resolver_calls),
        "sampler_id": "dinkster.euler",
        "scheduler_id": "dinkster.normal",
        "steps": 2,
        "seed": 41,
        "compute_dtype": torch.float32,
    }
    first = runtime.sample_scheduled(tiny_latent(), **scheduled_args)
    second = runtime.sample_scheduled(tiny_latent(), **scheduled_args)
    assert torch.equal(first, second)
    assert _bytes(first) == _bytes(second)
    assert hashlib.sha256(_bytes(first)).hexdigest() == hashlib.sha256(_bytes(second)).hexdigest()
    assert runtime.runtime_identity == identity
    assert encode_conditioning_carrier(conditional) == carrier_bytes
    assert len(resolver_calls) == 2
    first_requests = cast("Any", resolver_calls[0])[0]
    second_requests = cast("Any", resolver_calls[1])[0]
    assert first_requests == second_requests
    assert first_requests[0].runtime_identity == identity
    assert first_requests[0].stack_digest == metadata[-1][1]
    _assert_clean(model)

    original_forward = model.forward

    def fail(*_args: object, **_kwargs: object) -> torch.Tensor:
        raise RuntimeError("scheduled-runtime-injected-model-failure")

    cast("Any", model).forward = fail
    try:
        with pytest.raises(RuntimeError, match="scheduled-runtime-injected-model-failure"):
            runtime.sample_scheduled(tiny_latent(), **scheduled_args)
    finally:
        cast("Any", model).forward = original_forward
    _assert_clean(model)

    cancellation: dict[str, Any] = {"cancelled": False, "progress": 0}

    def cancelled() -> bool:
        return cancellation["cancelled"]

    def progress(_event: StepEvent) -> None:
        cancellation["progress"] += 1
        cancellation["cancelled"] = True

    with pytest.raises(ScheduledSamplingError, match="cancelled"):
        runtime.sample_scheduled(
            tiny_latent(),
            **scheduled_args,
            cancelled=cancelled,
            on_step=progress,
        )
    assert cancellation["progress"] == 1
    _assert_clean(model)

    retried = runtime.sample_scheduled(tiny_latent(), **scheduled_args)
    ordinary_after = runtime.sample(tiny_latent(), **ordinary_args)
    assert torch.equal(retried, first)
    assert torch.equal(ordinary_after, ordinary_before)
    assert _bytes(ordinary_after) == _bytes(ordinary_before)
    assert runtime.runtime_identity == identity
    _assert_clean(model)
