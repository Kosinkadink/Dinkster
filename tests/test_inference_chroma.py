from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest
from dinkster_inference import (
    BFLOAT16,
    CHROMA,
    CHROMA_RADIANCE,
    FLOAT8_E4M3,
    FLOAT16,
    FLOAT32,
    FLUX_DIFFUSION_PREFIX,
    INT8,
    T5_XXL_CONFIG,
    UINT8,
    ChromaComponentRole,
    ChromaConfig,
    ChromaDetectError,
    ChromaRadianceConfig,
    DType,
    KLConfig,
    TensorGeometry,
    WeightEntry,
    chroma_component_family_id,
    chroma_component_runtime_identity,
    chroma_layout,
    chroma_radiance_layout,
    detect_chroma,
    detect_chroma_config,
    normalize_chroma_keys,
    plan_chroma_component,
    plan_chroma_split_component,
    t5_layout,
)
from dinkster_protocol import ATTENTION_ROLES, AttentionRoute, AttentionRouteToken

_KL_GOLDEN = (
    Path(__file__).parent.parent
    / "packages"
    / "dinkster-inference-torch"
    / "tests"
    / "goldens"
    / "kl_goldens.json"
)


@dataclass
class Source:
    geometries: dict[str, TensorGeometry]
    path: Path = Path("/fake/component.safetensors")
    payloads: dict[str, bytes] = field(default_factory=dict)
    asset_digest: str | None = None
    asset_size: int | None = None

    def keys(self) -> tuple[str, ...]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}

    def read_uint8_configuration(self, key: str) -> bytes:
        return self.payloads[key]


def geometrize(
    layout: Mapping[str, tuple[int, ...]], dtype: DType = BFLOAT16
) -> dict[str, TensorGeometry]:
    return {key: TensorGeometry(shape, dtype) for key, shape in layout.items()}


def prefixed(geometries: Mapping[str, TensorGeometry], prefix: str) -> dict[str, TensorGeometry]:
    return {prefix + key: geometry for key, geometry in geometries.items()}


def chroma_geometries(dtype: DType = BFLOAT16) -> dict[str, TensorGeometry]:
    return geometrize(chroma_layout(ChromaConfig(3072, 19, 38, 24)), dtype)


def radiance_geometries(
    *,
    final_head: str = "linear",
    x0: bool = False,
    sequential: bool = False,
) -> dict[str, TensorGeometry]:
    config = ChromaRadianceConfig(
        3072,
        19,
        38,
        24,
        16,
        nerf_final_head_type=final_head,
        use_x0=x0,
        use_sequential_txt_ids=sequential,
    )
    return geometrize(chroma_radiance_layout(config))


def t5_geometries() -> dict[str, TensorGeometry]:
    return geometrize(t5_layout(T5_XXL_CONFIG), FLOAT16)


def flux_vae_geometries() -> dict[str, TensorGeometry]:
    case = json.loads(_KL_GOLDEN.read_text())["cases"]["regularizer"]
    return {key: TensorGeometry(tuple(shape), FLOAT32) for key, shape in case["state_dict"]}


def fp8mixed_chroma_source(*, prefix: str = "") -> Source:
    geometries = chroma_geometries()
    payloads: dict[str, bytes] = {}
    quantized = 0
    for key in tuple(geometries):
        if not key.endswith(".weight"):
            continue
        layer = key[: -len(".weight")]
        if not (layer.startswith("double_blocks.") or layer.startswith("single_blocks.")):
            continue
        if layer.endswith(("query_norm", "key_norm")):
            continue
        geometries[key] = TensorGeometry(geometries[key].shape, FLOAT8_E4M3)
        scale = f"{layer}.weight_scale"
        config = f"{layer}.comfy_quant"
        payload = b'{"format":"float8_e4m3fn"}'
        geometries[scale] = TensorGeometry((), FLOAT32)
        geometries[config] = TensorGeometry((len(payload),), UINT8)
        payloads[prefix + config] = payload
        quantized += 1
    assert quantized == 228
    return Source(prefixed(geometries, prefix), payloads=payloads)


def test_exact_production_layouts_have_distinct_family_shapes() -> None:
    chroma = ChromaConfig(3072, 19, 38, 24)
    radiance = ChromaRadianceConfig(3072, 19, 38, 24, 16)
    assert len(chroma_layout(chroma)) == 643
    assert len(chroma_radiance_layout(radiance)) == 658
    assert chroma.latent_channels == 16
    assert radiance.latent_channels == 3
    assert chroma.patch_size == 2
    assert radiance.patch_size == 16
    assert chroma.modulation_count == radiance.modulation_count == 344


@pytest.mark.parametrize("prefix", ("", FLUX_DIFFUSION_PREFIX))
def test_detection_accepts_bare_and_loader_prefixes(prefix: str) -> None:
    source = Source(prefixed(chroma_geometries(), prefix))
    evidence = detect_chroma(source)
    assert evidence is not None
    assert evidence.family_id == CHROMA.id
    assert evidence.fields["key_prefix"] == prefix
    assert evidence.fields["radiance"] is False


@pytest.mark.parametrize(
    ("final_head", "x0", "sequential"),
    (("linear", False, False), ("conv", True, False), ("linear", False, True)),
)
def test_radiance_detection_preserves_architecture_markers(
    final_head: str, x0: bool, sequential: bool
) -> None:
    config = detect_chroma_config(
        radiance_geometries(final_head=final_head, x0=x0, sequential=sequential)
    )
    assert isinstance(config, ChromaRadianceConfig)
    assert config.nerf_final_head_type == final_head
    assert config.use_x0 is x0
    assert config.use_sequential_txt_ids is sequential
    evidence = detect_chroma(
        Source(radiance_geometries(final_head=final_head, x0=x0, sequential=sequential))
    )
    assert evidence is not None
    assert evidence.family_id == CHROMA_RADIANCE.id
    assert evidence.fields["radiance"] is True


def test_legacy_approximator_and_scale_spellings_normalize_once() -> None:
    legacy = {
        key.replace("distilled_guidance_layer.", "distilled_guidance_layer.0.", 1).replace(
            ".weight", ".scale"
        )
        if key.endswith(("query_norm.weight", "key_norm.weight"))
        or key.startswith("distilled_guidance_layer.norms.")
        else key: geometry
        for key, geometry in chroma_geometries().items()
    }
    assert detect_chroma_config(legacy) == ChromaConfig(3072, 19, 38, 24)
    collision = dict(chroma_geometries())
    collision["distilled_guidance_layer.norms.0.scale"] = collision[
        "distilled_guidance_layer.norms.0.weight"
    ]
    with pytest.raises(ValueError, match="both normalize"):
        normalize_chroma_keys(collision)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("missing", "layout mismatch"),
        ("extra", "layout mismatch"),
        ("shape", "shape mismatch"),
        ("dtype", "floating-point storage"),
        ("width", "3072-wide"),
    ),
)
def test_detection_refuses_non_exact_chroma_layouts(mutation: str, match: str) -> None:
    geometries = chroma_geometries()
    if mutation == "missing":
        geometries.pop("final_layer.linear.bias")
    elif mutation == "extra":
        geometries["foreign.weight"] = TensorGeometry((1,), FLOAT32)
    elif mutation == "shape":
        geometries["final_layer.linear.bias"] = TensorGeometry((63,), BFLOAT16)
    elif mutation == "dtype":
        geometries["final_layer.linear.bias"] = TensorGeometry((64,), INT8)
    else:
        geometries["txt_in.weight"] = TensorGeometry((2048, 4096), BFLOAT16)
    with pytest.raises(ChromaDetectError, match=match):
        detect_chroma_config(geometries)


@pytest.mark.parametrize("prefix", ("", FLUX_DIFFUSION_PREFIX))
def test_official_fp8mixed_shape_plans_all_quant_sidecars(prefix: str) -> None:
    source = fp8mixed_chroma_source(prefix=prefix)
    evidence = detect_chroma(source)
    assert evidence is not None
    assert evidence.family_id == CHROMA.id
    plan = plan_chroma_component(source, "diffusion")
    assert len(plan.keys) == 643
    assert len(plan.quant) == 228
    assert len(source.keys()) == 1099
    quant = plan.quant["double_blocks.0.img_attn.proj"]
    assert quant.format is None
    assert quant.weight == prefix + "double_blocks.0.img_attn.proj.weight"
    assert quant.weight_scale == prefix + "double_blocks.0.img_attn.proj.weight_scale"
    assert quant.config == prefix + "double_blocks.0.img_attn.proj.comfy_quant"
    assert plan.dtypes["double_blocks.0.img_attn.proj.weight"] == FLOAT8_E4M3


def test_chroma_components_plan_independently() -> None:
    text = Source(t5_geometries(), path=Path("/fake/t5.safetensors"))
    vae = Source(flux_vae_geometries(), path=Path("/fake/ae.safetensors"))
    chroma = plan_chroma_component(Source(chroma_geometries()), "diffusion")
    radiance = plan_chroma_component(Source(radiance_geometries(x0=True)), "diffusion")
    planned_text = plan_chroma_component(text, "t5xxl")
    planned_vae = plan_chroma_component(vae, "vae")

    assert chroma.component == radiance.component == "diffusion"
    assert chroma_component_family_id(chroma) == CHROMA.id
    assert chroma_component_family_id(radiance) == CHROMA_RADIANCE.id
    assert planned_text.component == "t5xxl"
    assert planned_text.config == T5_XXL_CONFIG
    assert planned_vae.component == "vae"
    vae_config = cast("KLConfig", planned_vae.config)
    assert vae_config.latent_channels == 16
    assert vae_config.batch_norm_latent is False


def test_chroma_planner_accepts_float32_storage() -> None:
    plan = plan_chroma_component(Source(chroma_geometries(FLOAT32)), "diffusion")
    assert set(plan.dtypes.values()) == {FLOAT32}


@pytest.mark.parametrize("role", ("diffusion", "t5xxl", "vae"))
def test_chroma_component_identity_binds_authenticated_attention_route(role: str) -> None:
    geometries = {
        "diffusion": chroma_geometries,
        "t5xxl": t5_geometries,
        "vae": flux_vae_geometries,
    }[role]()
    path = Path(f"/fake/{role}.safetensors")
    source = Source(
        geometries,
        path=path,
        asset_digest="blake3:" + "1" * 64,
        asset_size=123,
    )
    component_role = cast("ChromaComponentRole", role)
    planned = plan_chroma_split_component(source, role=component_role, path=path)
    token = AttentionRouteToken(
        version=1,
        routes=tuple(AttentionRoute(attention_role, "sdpa") for attention_role in ATTENTION_ROLES),
        provider_versions=(("torch", "2.13.0"),),
        adapter_contract_revision="dinkster.attention-kernel.v1",
        device_kind="cpu",
        device_sm=None,
        sdpa_torch_runtime="2.13.0",
        requested_policy="auto",
    )

    base = chroma_component_runtime_identity(planned, component_role, BFLOAT16)
    routed = chroma_component_runtime_identity(
        planned,
        component_role,
        BFLOAT16,
        attention_route_token=token,
    )

    assert routed != base


def test_component_registry_ignores_wrong_prefix_quantization_failures() -> None:
    from dinkster_inference.component_catalog import default_component_registry
    from dinkster_inference.quantization import QuantizationError

    geometries = chroma_geometries()
    geometries["img_in.weight"] = TensorGeometry(geometries["img_in.weight"].shape, FLOAT8_E4M3)
    geometries["img_in.scale_weight"] = TensorGeometry((), FLOAT32)
    geometries["scaled_fp8"] = TensorGeometry((), FLOAT32)
    source = Source(
        prefixed(geometries, FLUX_DIFFUSION_PREFIX),
        asset_digest="sha256:" + "1" * 64,
        asset_size=1234,
    )
    registry = default_component_registry()
    anima = registry.get("dinkster.anima")
    assert anima is not None
    with pytest.raises(QuantizationError, match="artifacts with no recognized spelling"):
        anima.detector(source, source.path)
    descriptor, role, plan = registry.select(source, source.path, "model")
    assert descriptor.id == "dinkster.chroma" and role == "diffusion"
    assert plan.keys["img_in.weight"] == FLUX_DIFFUSION_PREFIX + "img_in.weight"
    assert plan.quant["img_in"].weight_scale == FLUX_DIFFUSION_PREFIX + "img_in.scale_weight"


@pytest.mark.parametrize("radiance", [False, True])
@pytest.mark.parametrize("companions", [False, True])
def test_checkpoint_registry_preserves_chroma_component_plans(
    radiance: bool, companions: bool
) -> None:
    from dinkster_inference import FLUX_T5XXL_PREFIX, FLUX_VAE_PREFIX
    from dinkster_inference.component_checkpoint import plan_component_checkpoint

    geometries = prefixed(
        radiance_geometries(final_head="conv", x0=True, sequential=True)
        if radiance
        else chroma_geometries(),
        FLUX_DIFFUSION_PREFIX,
    )
    roles: list[ChromaComponentRole] = ["diffusion"]
    if companions:
        geometries.update(prefixed(t5_geometries(), FLUX_T5XXL_PREFIX))
        roles.append("t5xxl")
        if not radiance:
            geometries.update(prefixed(flux_vae_geometries(), FLUX_VAE_PREFIX))
            roles.append("vae")
    source = Source(geometries)
    checkpoint = plan_component_checkpoint(checkpoint=source)
    assert checkpoint.family == (CHROMA_RADIANCE if radiance else CHROMA)
    assert set(checkpoint.components) == set(roles)
    assert not checkpoint.unclaimed
    prefixes = {
        "diffusion": FLUX_DIFFUSION_PREFIX,
        "t5xxl": FLUX_T5XXL_PREFIX,
        "vae": FLUX_VAE_PREFIX,
    }
    for role in roles:
        scoped = Source(
            {
                key: geometry
                for key, geometry in geometries.items()
                if key.startswith(prefixes[role])
            }
        )
        baseline = plan_chroma_component(scoped, role)
        actual = checkpoint.components[role]
        assert actual.config == baseline.config
        assert actual.keys == baseline.keys
        assert actual.dtypes == baseline.dtypes
        assert actual.quant == baseline.quant
        assert actual.path == baseline.path
