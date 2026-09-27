from __future__ import annotations

import importlib
from collections.abc import Mapping
from types import SimpleNamespace
from typing import cast

import pytest
from dinkster_inference import ResidentConditioningCarrier
from dinkster_native import fork_nodes
from dinkster_native.native import register_native_types
from dinkster_values import TypeRegistry


def test_native_types_preserve_resident_conditioning_codec() -> None:
    class ResidentPayload:
        _dinkster_resident_fingerprint = "test-resident-conditioning"

    registry = TypeRegistry()
    register_native_types(registry)
    carrier = ResidentConditioningCarrier(ResidentPayload())

    spec = registry.spec("dinkster.conditioning")
    assert spec.decode(spec.encode(carrier)) is carrier


def test_generation_ksampler_normalizes_residency_before_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []
    model = object()
    latent = object()
    normalized = object()
    sampled = object()

    def unload_model(actual_model: object) -> None:
        events.append(("unload", actual_model))

    def fix_latent(
        actual_model: object,
        actual_latent: object,
        spatial: object,
        temporal: object,
    ) -> object:
        events.append(("fix", actual_model, actual_latent, spatial, temporal))
        return normalized

    def prepare_noise(actual_latent: object, seed: object, batch_index: object) -> object:
        events.append(("noise", actual_latent, seed, batch_index))
        return object()

    def sample_latent(*args: object, **kwargs: object) -> object:
        events.append(("sample", args, kwargs))
        return sampled

    sample = SimpleNamespace(
        fix_empty_latent_channels=fix_latent,
        prepare_noise=prepare_noise,
        sample=sample_latent,
    )
    model_management = SimpleNamespace(unload_model_and_clones=unload_model)
    modules = {
        "dinkster_comfy.sample": sample,
        "dinkster_comfy.model_management": model_management,
    }
    real_import = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: modules.get(name) or real_import(name),
    )

    result = fork_nodes.GenerationKSampler.execute(
        model=model,
        seed=459,
        steps=5,
        cfg=7.0,
        sampler_name="euler",
        scheduler="normal",
        positive=object(),
        negative=object(),
        latent_image={
            "samples": latent,
            "batch_index": (3,),
            "downscale_ratio_spacial": 8,
            "downscale_ratio_temporal": 4,
        },
        denoise=1.0,
    )

    assert [event[0] for event in events] == ["unload", "fix", "noise", "sample"]
    assert events[1] == ("fix", model, latent, 8, 4)
    assert events[2][1:] == (normalized, 459, (3,))
    assert cast("Mapping[str, object]", result["latent"])["samples"] is sampled
