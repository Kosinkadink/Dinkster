from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest
from dinkster_inference import ComponentPlan, Trellis2FlowRole
from dinkster_inference import trellis2_assembly as assembly
from dinkster_inference.weights import WeightEntry, WeightSource


@dataclass(frozen=True)
class _Source:
    path: Path
    asset_size: int
    asset_digest: str

    def keys(self) -> tuple[str, ...]:
        return ()

    def entry(self, key: str) -> WeightEntry:
        raise KeyError(key)

    def metadata(self) -> dict[str, str]:
        return {}


def test_split_flow_provider_revision_is_pinned() -> None:
    assert assembly.TRELLIS2_SPLIT_PROVIDER_REVISION == (
        "microsoft/TRELLIS.2-4B@af44b45f2e35a493886929c6d786e563ec68364d"
    )


def test_split_flow_plan_binds_provider_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    role = "shape-512"
    size = 456
    digest = "blake3:" + "7" * 64
    source = _Source(tmp_path / "shape-512.safetensors", size, digest)
    base = ComponentPlan(role, source.path, object(), {}, {}, {})
    monkeypatch.setattr(assembly, "plan_trellis2_flow_component", lambda *_args: base)

    plan = assembly.plan_trellis2_flow_artifact(
        cast("WeightSource", source), role=role, path=source.path
    )

    assert f"asset_digest={digest}" in plan.identity_facts
    assert f"asset_size={size}" in plan.identity_facts
    assert f"provider_revision={assembly.TRELLIS2_SPLIT_PROVIDER_REVISION}" in plan.identity_facts


@pytest.mark.parametrize("requested_role", ("shape", "texture"))
def test_split_flow_plan_refuses_wrong_structural_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested_role: str,
) -> None:
    source = _Source(
        tmp_path / "wrong-variant.safetensors",
        456,
        "blake3:" + "8" * 64,
    )
    monkeypatch.setattr(
        assembly,
        "plan_trellis2_flow_component",
        lambda *_args: (_ for _ in ()).throw(ValueError("geometry mismatch")),
    )

    with pytest.raises(assembly.Trellis2AssemblyError, match="geometry mismatch"):
        assembly.plan_trellis2_flow_artifact(
            cast("WeightSource", source),
            role=cast("Trellis2FlowRole", requested_role),
            path=source.path,
        )
