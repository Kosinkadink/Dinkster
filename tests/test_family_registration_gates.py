from __future__ import annotations

from pathlib import Path

import pytest
from dinkster_inference import EngineProperties, PreviewDecoderProperties, builtin_families
from dinkster_inference.component_catalog import default_component_registry
from family_gate_scanner import (
    EXTERNAL_PROOF_FAMILY_IDS,
    changed_paths_since_merge_base,
    family_literal_gates,
    unexpected_new_family_paths,
)

ROOT = Path(__file__).resolve().parents[1]
SHARED_ENGINE_FILES = (
    "packages/dinkster-inference/src/dinkster_inference/assembly.py",
    "packages/dinkster-inference/src/dinkster_inference/component_catalog.py",
    "packages/dinkster-inference/src/dinkster_inference/gguf.py",
    "packages/dinkster-inference/src/dinkster_inference/identity.py",
    "packages/dinkster-inference/src/dinkster_inference/ipadapter.py",
    "packages/dinkster-inference/src/dinkster_inference/runtime.py",
    "packages/dinkster-inference/src/dinkster_inference/taesd.py",
    "packages/dinkster-inference-torch/src/dinkster_inference_torch/assemble.py",
    "packages/dinkster-inference-torch/src/dinkster_inference_torch/component_runtime.py",
    "packages/dinkster-inference-torch/src/dinkster_inference_torch/denoise.py",
    "packages/dinkster-inference-torch/src/dinkster_inference_torch/memory.py",
    "packages/dinkster-inference-torch/src/dinkster_inference_torch/sampling_execution.py",
    "packages/dinkster-inference-torch/src/dinkster_inference_torch/sampling_runtime.py",
    "packages/dinkster-inference-torch/src/dinkster_inference_torch/schedules.py",
    "packages/dinkster-inference-torch/src/dinkster_inference_torch/wiring.py",
    "packages/dinkster-native/src/dinkster_native/preview_emit.py",
    "src/dinkster/native_policy.py",
)
# native_arm.py is asserted separately by the #170 scanner test.


def test_shared_engine_has_zero_literal_family_gates() -> None:
    findings = tuple(
        finding
        for relative_path in SHARED_ENGINE_FILES
        for finding in family_literal_gates(ROOT / relative_path, ROOT)
    )
    assert findings == (), "literal family gates remain:\n" + "\n".join(findings)


@pytest.mark.parametrize("family_id", sorted(EXTERNAL_PROOF_FAMILY_IDS))
def test_external_proof_family_is_in_fail_closed_scanner(tmp_path: Path, family_id: str) -> None:
    source = tmp_path / "shared_engine.py"
    source.write_text(f'enabled = family_id == "{family_id}"\n', encoding="utf-8")

    assert family_literal_gates(source, tmp_path) == (f"shared_engine.py:1: {family_id}",)


def test_new_family_proof_only_changes_registration_points() -> None:
    changed_paths = changed_paths_since_merge_base(ROOT)
    unexpected = unexpected_new_family_paths(changed_paths)
    assert unexpected == (), "new-family proof changed non-registration paths:\n" + "\n".join(
        unexpected
    )

    shared_edit = "packages/dinkster-inference/src/dinkster_inference/runtime.py"
    assert unexpected_new_family_paths(changed_paths | {shared_edit}) == (shared_edit,)


def test_registered_engine_properties_cover_shared_family_behavior() -> None:
    families = {family.id: family for family in builtin_families()}

    assert families["dinkster.flux_dev"].engine.sigma_space == "flux"
    assert families["dinkster.flux_schnell"].engine.sigma_space == "default"
    assert families["dinkster.sd15"].engine.controlnet_profile == "sd15"
    assert families["dinkster.sdxl"].engine.controlnet_profile == "sdxl"
    assert families["dinkster.sdxl_refiner"].engine.adm_profile == "sdxl_refiner"
    wan_preview = families["dinkster.wan21"].engine.preview_decoder
    assert wan_preview is not None and wan_preview.kind == "taehv"
    triposplat_preview = families["dinkster.triposplat"].engine.preview_decoder
    assert triposplat_preview is not None
    assert (triposplat_preview.kind, triposplat_preview.target) == (
        "asset",
        "triposplat_vae_decoder",
    )
    assert families["dinkster.wan21"].engine.supports_context_windows
    assert families["dinkster.wan22"].engine.supports_context_windows
    assert families["dinkster.ltxv"].engine.supports_context_windows
    assert not families["dinkster.sd15"].engine.supports_context_windows
    chroma = families["dinkster.chroma"]
    assert chroma.engine.quantized_component_load_device
    assert chroma.engine.attention_backends == (
        ("diffusion", "flux"),
        ("t5xxl", "t5"),
        ("vae", "vae"),
    )
    assert chroma.engine.attention_requires_route
    assert families["dinkster.chroma_radiance"].engine is chroma.engine

    components = {descriptor.id: descriptor for descriptor in default_component_registry()}
    assert components["dinkster.minimax_h3"].family.engine.attention_backend("diffusion") == "flux"
    assert components["dinkster.minimax_music3"].family.engine.attention_backends == (
        ("diffusion", "flux"),
        ("text", "qwen"),
    )
    assert components["dinkster.ltxav"].family.engine.attention_backend("gemma4_12b") == "qwen"


def test_preview_decoder_registration_rejects_invalid_kind_target_pairs() -> None:
    with pytest.raises(ValueError, match="does not accept a target"):
        PreviewDecoderProperties("taehv", "unexpected")
    with pytest.raises(ValueError, match="requires a target"):
        PreviewDecoderProperties("asset")


def test_engine_properties_reject_invalid_attention_registration() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        EngineProperties(attention_backends=(("diffusion", "flux"), ("diffusion", "qwen")))
    with pytest.raises(ValueError, match="need at least one attention backend"):
        EngineProperties(attention_requires_route=True)
