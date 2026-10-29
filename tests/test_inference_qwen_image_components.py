from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from dinkster_inference import BFLOAT16, ComponentBinding, ComponentPlan
from dinkster_inference import qwen_image_assembly as assembly
from dinkster_inference.weights import WeightEntry, WeightSource


@dataclass(frozen=True)
class _Source:
    path: Path
    asset_size: int | None
    asset_digest: str | None

    def keys(self) -> tuple[str, ...]:
        return ()

    def entry(self, key: str) -> WeightEntry:
        raise KeyError(key)

    def metadata(self) -> dict[str, str]:
        return {}


def test_component_plan_binds_arbitrary_asset_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    size = 123
    digest = "blake3:" + "7" * 64
    source = _Source(tmp_path / "diffusion.safetensors", size, digest)
    base = ComponentPlan("diffusion", source.path, object(), {}, {}, {})
    monkeypatch.setattr(assembly, "plan_qwen_image_component", lambda *_args: base)

    plan = assembly.plan_qwen_image_official_component(
        cast("WeightSource", source), role="diffusion", path=source.path
    )
    identity = assembly.qwen_image_component_runtime_identity(plan, "diffusion", BFLOAT16)

    assert f"provider_revision={assembly.QWEN_IMAGE_PROVIDER_REVISION}" in plan.identity_facts
    assert f"asset_digest={digest}" in plan.identity_facts
    assert f"asset_size={size}" in plan.identity_facts
    ComponentBinding("diffusion", "dinkster.qwen_image", identity)
    assert identity == assembly.qwen_image_component_runtime_identity(plan, "diffusion", BFLOAT16)
    assert identity.startswith("native:dinkster.qwen_image:")


def test_component_plan_refuses_wrong_structure_and_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    size = 123
    digest = "blake3:" + "8" * 64
    source = _Source(tmp_path / "vae.safetensors", size, digest)
    monkeypatch.setattr(
        assembly,
        "plan_qwen_image_component",
        lambda *_args: (_ for _ in ()).throw(ValueError("geometry mismatch")),
    )
    with pytest.raises(ValueError, match="geometry mismatch"):
        assembly.plan_qwen_image_official_component(
            cast("WeightSource", source), role="diffusion", path=source.path
        )
    with pytest.raises(assembly.QwenImageComponentAssemblyError, match="path differs"):
        assembly.plan_qwen_image_official_component(
            cast("WeightSource", source), role="vae", path=tmp_path / "other.safetensors"
        )


@pytest.mark.parametrize(("size", "digest"), ((None, "blake3:" + "8" * 64), (123, None)))
def test_component_plan_refuses_partial_asset_identity(
    tmp_path: Path, size: int | None, digest: str | None
) -> None:
    source = _Source(tmp_path / "vae.safetensors", size, digest)

    with pytest.raises(assembly.QwenImageComponentAssemblyError, match="asset identity"):
        assembly.plan_qwen_image_official_component(
            cast("WeightSource", source), role="vae", path=source.path
        )
