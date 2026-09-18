"""Strict Lumina Image 2.0 detection, component planning, and prompt policy."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT16,
    FLOAT32,
    GEMMA2_LUMINA_2B_CONFIG,
    LUMINA2,
    LUMINA2_CONFIG,
    LUMINA2_SYSTEM_PROMPTS,
    UINT8,
    AssemblyError,
    FamilyRegistry,
    Lumina2ComponentAssemblyError,
    TensorGeometry,
    WeightEntry,
    detect_gemma_text_config,
    detect_lumina2,
    detect_z_image,
    gemma_text_layout,
    load_safetensors_header,
    lumina2_layout,
    lumina2_system_prompt,
    plan_lumina2_artifact_components,
    plan_lumina2_checkpoint,
    plan_lumina2_checkpoint_components,
    plan_lumina2_component,
    plan_native,
    probe_native,
    tokenize_lumina2_prompt,
)
from dinkster_inference.component_checkpoint import ComponentCheckpointPlan

_KL_GOLDEN = (
    Path(__file__).parent.parent
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "kl_goldens.json"
)
_NETAYUME = Path("/mnt/data/comfy-models/checkpoints/NetaYumev35_pretrained_all_in_one.safetensors")
_NETAYUME_SIZE = 10_620_231_237
_NETAYUME_SHA256 = "4125cb490996ea85c8e3ba242866da02efd83a6a2c079dc8924a95eaa8327a44"
_NETAYUME_BLAKE3 = "3e84945252f6e757b51ada84726dffd5da38994a3c3fc612b85d3efb15e4cf35"


@dataclass(frozen=True)
class _Source:
    path: Path
    geometries: Mapping[str, TensorGeometry]
    asset_digest: str = "sha256:" + "1" * 64
    asset_size: int = 1234

    def keys(self) -> tuple[str, ...]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}

    def read_uint8_configuration(self, key: str, *, limit: int = 65_536) -> bytes:
        del key, limit
        return b"tokenizer"


def _geometries(
    layout: Mapping[str, tuple[int, ...]], dtype: object = BFLOAT16
) -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(shape, dtype) for key, shape in layout.items()}  # type: ignore[arg-type]


def _flux_vae_geometries() -> dict[str, TensorGeometry]:
    payload = json.loads(_KL_GOLDEN.read_text())
    geometries = {
        key: TensorGeometry(tuple(shape), BFLOAT16)
        for key, shape in payload["cases"]["standard"]["state_dict"]
    }
    for prefix in ("quant_conv.", "post_quant_conv."):
        for key in tuple(geometries):
            if key.startswith(prefix):
                del geometries[key]
    encoder_out = geometries["encoder.conv_out.weight"].shape
    decoder_in = geometries["decoder.conv_in.weight"].shape
    geometries["encoder.conv_out.weight"] = TensorGeometry((32, *encoder_out[1:]), BFLOAT16)
    geometries["encoder.conv_out.bias"] = TensorGeometry((32,), BFLOAT16)
    geometries["decoder.conv_in.weight"] = TensorGeometry(
        (decoder_in[0], 16, *decoder_in[2:]), BFLOAT16
    )
    return geometries


def _checkpoint_geometries() -> dict[str, TensorGeometry]:
    geometries = {
        "model.diffusion_model." + key: value
        for key, value in _geometries(lumina2_layout()).items()
    }
    geometries.update(
        {
            "text_encoders.gemma2_2b.transformer.model." + key: value
            for key, value in _geometries(gemma_text_layout(GEMMA2_LUMINA_2B_CONFIG)).items()
        }
    )
    geometries.update({"vae." + key: value for key, value in _flux_vae_geometries().items()})
    geometries["text_encoders.gemma2_2b.logit_scale"] = TensorGeometry((), FLOAT32)
    geometries["text_encoders.spiece_model"] = TensorGeometry((16,), UINT8)
    return geometries


def test_lumina2_exact_profile_layout_and_family_semantics() -> None:
    config = LUMINA2_CONFIG
    assert (
        config.hidden_width,
        config.caption_width,
        config.main_blocks,
        config.noise_refiner_blocks,
        config.context_refiner_blocks,
        config.attention_heads,
        config.kv_heads,
        config.attention_head_dim,
        config.ffn_width,
    ) == (2304, 2304, 26, 2, 2, 24, 8, 96, 9216)
    assert config.rope_axes == (32, 32, 32)
    assert config.rope_theta == 10_000.0
    assert config.timestep_embedding_width == 256
    assert config.modulation_width == 1024
    assert config.timestep_multiplier == 1.0
    assert config.block_modulation_silu is True
    assert config.learned_padding is False
    assert config.sampling_shift == 6.0

    layout = lumina2_layout()
    assert len(layout) == 400
    assert layout["cap_embedder.1.weight"] == (2304, 2304)
    assert layout["noise_refiner.0.attention.k_norm.weight"] == (96,)
    assert layout["layers.25.adaLN_modulation.1.weight"] == (9216, 1024)
    assert layout["final_layer.linear.weight"] == (64, 2304)
    assert layout["norm_final.weight"] == (2304,)

    family = LUMINA2
    assert family.id == "dinkster.lumina2"
    assert family.single_stream_latent().channels == 16
    assert family.single_stream_latent().scale_factor == 0.3611
    assert family.single_stream_latent().shift_factor == 0.1159
    assert family.sampling.shift == 6.0
    assert family.sampling.sigma_min == pytest.approx(6 / 1005)


@pytest.mark.parametrize("prefix", ("", "model.diffusion_model."))
def test_detector_accepts_only_complete_lumina2_and_never_z_image(prefix: str) -> None:
    geometries = {prefix + key: value for key, value in _geometries(lumina2_layout()).items()}
    source = _Source(Path("/fake/lumina2.safetensors"), geometries)
    evidence = detect_lumina2(source)
    assert evidence is not None
    assert evidence.key_prefix == prefix
    assert len(evidence.matched_keys) == 400
    assert evidence.fields["hidden_width"] == 2304
    assert LUMINA2.detector.detect(source) is not None
    assert detect_z_image(source) is None

    malformed = dict(geometries)
    malformed[prefix + "cap_embedder.1.weight"] = TensorGeometry((3840, 2304), BFLOAT16)
    assert detect_lumina2(_Source(source.path, malformed)) is None
    extra = dict(geometries)
    extra[prefix + "cap_pad_token"] = TensorGeometry((1, 2304), BFLOAT16)
    assert detect_lumina2(_Source(source.path, extra)) is None


def test_gemma2_profile_has_no_qk_norms_and_does_not_change_ltx_profile() -> None:
    layout = gemma_text_layout(GEMMA2_LUMINA_2B_CONFIG)
    assert len(layout) == 288
    assert layout["embed_tokens.weight"] == (256000, 2304)
    assert layout["layers.25.mlp.down_proj.weight"] == (2304, 9216)
    assert not any(key.endswith(("q_norm.weight", "k_norm.weight")) for key in layout)
    assert detect_gemma_text_config(_geometries(layout)) is GEMMA2_LUMINA_2B_CONFIG


def test_split_component_plans_are_strict_and_keep_tokenizer_out_of_model_state() -> None:
    diffusion_source = _Source(Path("/fake/lumina2.safetensors"), _geometries(lumina2_layout()))
    diffusion = plan_lumina2_component(diffusion_source, "diffusion")
    assert len(diffusion.keys) == 399
    assert diffusion.ignored == ("norm_final.weight",)

    text_geometries = {
        "model." + key: value
        for key, value in _geometries(gemma_text_layout(GEMMA2_LUMINA_2B_CONFIG)).items()
    }
    text_geometries["spiece_model"] = TensorGeometry((16,), UINT8)
    text = plan_lumina2_component(
        _Source(Path("/fake/gemma.safetensors"), text_geometries), "gemma2_2b"
    )
    assert len(text.keys) == 288
    assert text.ignored == ("spiece_model",)

    vae = plan_lumina2_component(
        _Source(Path("/fake/ae.safetensors"), _flux_vae_geometries()), "vae"
    )
    assert len(vae.keys) == 180

    broken = dict(text_geometries)
    broken["model.layers.0.self_attn.q_norm.weight"] = TensorGeometry((256,), BFLOAT16)
    with pytest.raises(AssemblyError, match="unexpected key"):
        plan_lumina2_component(_Source(Path("/fake/broken.safetensors"), broken), "gemma2_2b")

    missing_tokenizer = dict(text_geometries)
    del missing_tokenizer["spiece_model"]
    with pytest.raises(AssemblyError, match="exactly one uint8 spiece_model"):
        plan_lumina2_component(
            _Source(Path("/fake/missing-tokenizer.safetensors"), missing_tokenizer),
            "gemma2_2b",
        )

    malformed_tokenizer = dict(text_geometries)
    malformed_tokenizer["spiece_model"] = TensorGeometry((16,), BFLOAT16)
    with pytest.raises(AssemblyError, match="nonempty rank-1 uint8"):
        plan_lumina2_component(
            _Source(Path("/fake/malformed-tokenizer.safetensors"), malformed_tokenizer),
            "gemma2_2b",
        )


def test_all_in_one_checkpoint_has_native_component_plan(
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = _Source(Path("/fake/netayume.safetensors"), _checkpoint_geometries())
    capability = probe_native(source)
    assert capability.native, capability.reasons
    assert capability.family_id == LUMINA2.id
    plan = plan_native(source)
    assert isinstance(plan, ComponentCheckpointPlan)
    assert plan.family is LUMINA2
    assert plan.identity_components == plan_lumina2_checkpoint(source)
    assert plan.components["gemma2_2b"].config is GEMMA2_LUMINA_2B_CONFIG
    assert plan_native(source, registry=FamilyRegistry()) == plan
    assert "defaulting model, text, codec, sampling and dtype behavior" in caplog.text
    assert probe_native(source, vae=source).native
    assert plan_native(source, vae=source) == plan
    missing = dict(source.geometries)
    del missing["text_encoders.spiece_model"]
    refusal = probe_native(_Source(source.path, missing))
    assert not refusal.native
    assert "spiece_model" in "; ".join(refusal.reasons)


def test_exact_all_in_one_checkpoint_decomposes_into_three_component_plans() -> None:
    path = Path("/fake/netayume.safetensors")
    source = _Source(path, _checkpoint_geometries())
    diffusion, text, vae = plan_lumina2_checkpoint(source)
    assert (len(diffusion.keys), len(text.keys), len(vae.keys)) == (399, 288, 180)
    assert tuple(plan.component for plan in (diffusion, text, vae)) == (
        "diffusion",
        "gemma2_2b",
        "vae",
    )

    float16_geometries = {
        key: TensorGeometry(value.shape, FLOAT16) if value.dtype.kind == "float" else value
        for key, value in source.geometries.items()
    }
    float16_diffusion, _, float16_vae = plan_lumina2_checkpoint(_Source(path, float16_geometries))
    assert set(float16_diffusion.dtypes.values()) == {FLOAT16}
    assert set(float16_vae.dtypes.values()) == {FLOAT16}

    identified = plan_lumina2_checkpoint_components(source, path=path)
    assert all(f"asset_digest={source.asset_digest}" in plan.identity_facts for plan in identified)
    assert all(f"asset_size={source.asset_size}" in plan.identity_facts for plan in identified)
    artifact_components = plan_lumina2_artifact_components(source, path=path)
    assert tuple(role for role, _plan in artifact_components) == (
        "diffusion",
        "gemma2_2b",
        "vae",
    )
    assert tuple(plan for _role, plan in artifact_components) == identified

    extra = dict(source.geometries)
    extra["unknown.weight"] = TensorGeometry((1,), BFLOAT16)
    with pytest.raises(AssemblyError, match="unexpected unknown.weight"):
        plan_lumina2_checkpoint(_Source(path, extra))
    with pytest.raises(Lumina2ComponentAssemblyError, match="path differs"):
        plan_lumina2_checkpoint_components(source, path=Path("/fake/other.safetensors"))


def test_checkpoint_split_matches_comfyui_15eb748b() -> None:
    # ComfyUI 15eb748b: supported_models.Lumina2 and BASE.process_clip_state_dict,
    # sd.load_state_dict_guess_config, and model_base.BaseModel.load_model_weights.
    source = _Source(Path("/fake/netayume.safetensors"), _checkpoint_geometries())
    plan = plan_native(source)
    assert isinstance(plan, ComponentCheckpointPlan)
    diffusion, text, vae = (plan.components[role] for role in ("diffusion", "gemma2_2b", "vae"))
    unet_prefix = "model.diffusion_model."
    clip_prefix = "text_encoders."
    vae_prefix = "vae."
    reference_unet = {
        key.removeprefix(unet_prefix): key for key in source.keys() if key.startswith(unet_prefix)
    }
    reference_clip = {
        key.removeprefix(clip_prefix): key for key in source.keys() if key.startswith(clip_prefix)
    }
    reference_vae = {
        key.removeprefix(vae_prefix): key for key in source.keys() if key.startswith(vae_prefix)
    }
    assert dict(diffusion.keys) == {
        key: value for key, value in reference_unet.items() if key != "norm_final.weight"
    }
    assert dict(text.keys) == {
        key.removeprefix("gemma2_2b.transformer.model."): value
        for key, value in reference_clip.items()
        if key.startswith("gemma2_2b.transformer.model.")
    }
    assert dict(vae.keys) == reference_vae
    assert set(reference_clip.values()) - set(text.keys.values()) == {
        "text_encoders.spiece_model",
        "text_encoders.gemma2_2b.logit_scale",
    }


@pytest.mark.parametrize(
    ("role", "geometries"),
    (
        ("diffusion", _geometries(lumina2_layout())),
        (
            "gemma2_2b",
            {
                **{
                    "model." + key: value
                    for key, value in _geometries(
                        gemma_text_layout(GEMMA2_LUMINA_2B_CONFIG)
                    ).items()
                },
                "spiece_model": TensorGeometry((16,), UINT8),
            },
        ),
        ("vae", _flux_vae_geometries()),
    ),
)
def test_split_artifact_planner_returns_only_the_exact_component(
    role: str, geometries: Mapping[str, TensorGeometry]
) -> None:
    path = Path(f"/fake/{role}.safetensors")
    source = _Source(path, geometries)
    assert tuple(
        component_role
        for component_role, _plan in plan_lumina2_artifact_components(source, path=path)
    ) == (role,)


def test_system_prompts_and_weighted_tokenization_match_comfy_policy() -> None:
    assert lumina2_system_prompt("a red fox", "superior") == (
        LUMINA2_SYSTEM_PROMPTS["superior"] + " <Prompt Start> a red fox"
    )
    assert lumina2_system_prompt("a red fox", "alignment") == (
        LUMINA2_SYSTEM_PROMPTS["alignment"] + " <Prompt Start> a red fox"
    )
    with pytest.raises(ValueError, match="unknown Lumina2 system prompt"):
        lumina2_system_prompt("fox", "unknown")

    tokens = tokenize_lumina2_prompt(
        "plain (weighted:2.0)",
        encode=lambda word: tuple(ord(char) % 97 + 3 for char in word),
    )
    assert tokens.ids[0] == 2
    assert tokens.attention_mask == (1,) * len(tokens.ids)
    assert tokens.weights[0] == 1.0
    assert 2.0 in tokens.weights
    empty = tokenize_lumina2_prompt("", encode=lambda _word: ())
    assert empty.ids == (2,)
    assert empty.attention_mask == (1,)


def test_literal_end_of_turn_matches_comfy_gemma2_tokenization() -> None:
    encoded = {
        "plain ": (5, 6),
        " tail": (7, 8),
    }
    tokens = tokenize_lumina2_prompt(
        "plain <end_of_turn> tail",
        encode=encoded.__getitem__,
    )
    assert tokens.ids == (2, 6, 107, 7, 8)
    assert tokens.attention_mask == (1, 1, 1, 1, 1)
    assert tokens.weights == (1.0, 1.0, 1.0, 1.0, 1.0)
    leading = tokenize_lumina2_prompt(
        "<end_of_turn> tail",
        encode=encoded.__getitem__,
    )
    assert leading.ids == (2, 7, 8)


@pytest.mark.skipif(not _NETAYUME.exists(), reason="official NetaYume checkpoint absent")
def test_official_netayume_header_matches_pinned_complete_checkpoint() -> None:
    assert _NETAYUME.stat().st_size == _NETAYUME_SIZE
    source = load_safetensors_header(
        _NETAYUME,
        asset_digest="blake3:" + _NETAYUME_BLAKE3,
        asset_size=_NETAYUME_SIZE,
    )
    evidence = detect_lumina2(source)
    assert evidence is not None and len(evidence.matched_keys) == 400
    components = plan_lumina2_artifact_components(source, path=_NETAYUME)
    assert tuple(role for role, _plan in components) == ("diffusion", "gemma2_2b", "vae")
    diffusion, text, vae = (plan for _role, plan in components)
    assert (len(diffusion.keys), len(text.keys), len(vae.keys)) == (399, 288, 244)


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("all_descriptors", [False, True])
@pytest.mark.parametrize("bind_asset_identity", [False, True])
def test_component_detection_preserves_malformed_quantization(
    combined: bool, all_descriptors: bool, bind_asset_identity: bool
) -> None:
    from dinkster_inference.component_catalog import default_component_registry
    from dinkster_inference.component_registry import ComponentRegistry
    from dinkster_inference.quantization import QuantizationError

    class BadMetadata(_Source):
        def metadata(self) -> Mapping[str, str]:
            return {"_quantization_metadata": "not json"}

    source = BadMetadata(
        Path("lumina.safetensors"),
        _checkpoint_geometries() if combined else _geometries(lumina2_layout()),
    )
    registry = default_component_registry()
    if not all_descriptors:
        descriptor = registry.get("dinkster.lumina2")
        assert descriptor is not None
        registry = ComponentRegistry()
        registry.register(descriptor)
    with pytest.raises(QuantizationError, match="malformed _quantization_metadata"):
        registry.detect(source, source.path, bind_asset_identity=bind_asset_identity)
