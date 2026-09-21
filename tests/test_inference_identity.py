"""Torch-free native runtime identity and planning contract."""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import dinkster_inference.identity as identity_module
import dinkster_inference.registries as registry_module
import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT8_E4M3,
    FLOAT16,
    FLOAT32,
    INT8,
    NATIVE_WIRED_FAMILY_IDS,
    UINT8,
    ComponentPlan,
    DType,
    LayerQuant,
    NativeRefusalCategory,
    NativeRefusalError,
    TensorGeometry,
    build_runtime_identity,
    default_diffusion_dtype,
    default_text_dtype,
    default_vae_dtype,
    plan_native,
    probe_native,
    runtime_component_identity,
    split_quantization,
)
from dinkster_protocol import ActiveExtension, ExtensionSnapshot, extension_behavior_hash

from tests.test_inference_runtime import (
    combined_dev_checkpoint,
    source,
)


def identity(
    *,
    diffusion_dtype: DType = BFLOAT16,
    text_dtype: DType = FLOAT32,
    vae_dtype: DType = FLOAT32,
    fp8_matmul: bool = False,
    registry_token: str | None = None,
    extension_hash: str | None = None,
    patch_overlay_digests: tuple[str, ...] | None = None,
) -> str:
    plan = plan_native(combined_dev_checkpoint())
    return build_runtime_identity(
        plan.family.id,
        plan.identity_components,
        diffusion_dtype=diffusion_dtype,
        text_dtype=text_dtype,
        vae_dtype=vae_dtype,
        fp8_matmul=fp8_matmul,
        registry_token=registry_token,
        extension_behavior_hash=extension_hash,
        patch_overlay_digests=patch_overlay_digests,
    )


def test_identity_format_and_determinism() -> None:
    first = identity()
    assert first == identity()
    assert first == (
        "native:dinkster.flux_dev:1be4c1ca01b412a75ad055cc1989f0be56d095ecc9a7d720be9440ab43033200"
    )
    assert re.fullmatch(r"native:dinkster\.flux_dev:[0-9a-f]{64}", first)


def test_component_runtime_facts_rotate_runtime_but_not_structural_identity() -> None:
    component = ComponentPlan(
        component="diffusion",
        path=Path("model.gguf"),
        config="sdxl",
        keys={"weight": "weight"},
        dtypes={"weight": FLOAT32},
        quant={},
        source_format="gguf",
        runtime_facts=("gguf.artifact.file_sha256=" + "1" * 64,),
        payload_source=source({}, "model.gguf"),
    )
    changed = replace(
        component,
        runtime_facts=("gguf.artifact.file_sha256=" + "2" * 64,),
    )

    assert runtime_component_identity("dinkster.sdxl", (component,)) == (
        runtime_component_identity("dinkster.sdxl", (changed,))
    )
    knobs = {
        "diffusion_dtype": FLOAT32,
        "text_dtype": FLOAT32,
        "vae_dtype": FLOAT32,
        "fp8_matmul": False,
    }
    assert build_runtime_identity("dinkster.sdxl", (component,), **knobs) != (
        build_runtime_identity("dinkster.sdxl", (changed,), **knobs)
    )
    with pytest.raises(ValueError, match="requires bound runtime facts"):
        replace(component, runtime_facts=())


def test_nvfp4_identity_adds_only_format_specific_structural_facts() -> None:
    component = ComponentPlan(
        component="diffusion",
        path=Path("synthetic.safetensors"),
        config="classic-flux",
        keys={"img_in.weight": "img_in.weight"},
        dtypes={"img_in.weight": UINT8},
        quant={
            "img_in": LayerQuant(
                layer="img_in",
                format="nvfp4",
                weight="img_in.weight",
                weight_scale="img_in.weight_scale",
                input_scale="img_in.input_scale",
                weight_scale_2="img_in.weight_scale_2",
                pre_quant_scale="img_in.pre_quant_scale",
            )
        },
    )
    lines = runtime_component_identity("dinkster.flux_dev", (component,))
    assert (
        "quant=img_in format=nvfp4 input_scale=True"
        " full_precision_matmul=False weight_scale_2=True pre_quant_scale=True"
    ) in lines


def test_payload_nvfp4_identity_is_stable() -> None:
    component = ComponentPlan(
        component="diffusion",
        path=Path("payload.safetensors"),
        config="flux",
        keys={"a.weight": "a.weight", "b.weight": "b.weight"},
        dtypes={"a.weight": UINT8, "b.weight": UINT8},
        quant={
            layer: LayerQuant(
                layer=layer,
                format="nvfp4",
                weight=f"{layer}.weight",
                weight_scale=f"{layer}.weight_scale",
                weight_scale_2=f"{layer}.weight_scale_2",
                config=f"{layer}.comfy_quant",
            )
            for layer in ("a", "b")
        },
    )
    first = runtime_component_identity("dinkster.flux_dev", (component,))
    assert first == runtime_component_identity("dinkster.flux_dev", (component,))
    assert sum("format=nvfp4" in line for line in first) == 2


def test_payload_fp8_facts_remain_asset_borne_in_identity() -> None:
    payload = b'{"format":"float8_e4m3fn","full_precision_matrix_mult":true}'
    split = split_quantization(
        {
            "block.weight": TensorGeometry((16, 16), FLOAT8_E4M3),
            "block.weight_scale": TensorGeometry((), FLOAT32),
            "block.comfy_quant": TensorGeometry((len(payload),), UINT8),
        },
        payload_reader=lambda _key: payload,
    )
    component = ComponentPlan(
        component="diffusion",
        path=Path("payload-fp8.safetensors"),
        config="flux",
        keys={"block.weight": "block.weight"},
        dtypes={"block.weight": FLOAT8_E4M3},
        quant=split.layers,
    )
    assert (
        "quant=block format=payload input_scale=False full_precision_matmul=False"
    ) in runtime_component_identity("dinkster.flux_dev", (component,))


def test_unsupported_quant_contract_parameters_payloads_and_geometry_rotate_identity() -> None:
    def lines(*, group_size: int, correction: bool) -> tuple[str, ...]:
        payloads = {
            "weight_s_rel": "block.weight_s_rel",
            "weight_s_channel": "block.weight_s_channel",
        }
        if correction:
            payloads["weight_correction"] = "block.weight_correction"
        component = ComponentPlan(
            component="diffusion",
            path=Path("typed-only.safetensors"),
            config="typed-only",
            keys={"block.weight": "block.weight"},
            dtypes={"block.weight": UINT8},
            quant={
                "block": LayerQuant(
                    layer="block",
                    format="asym_w4a8_int8",
                    weight="block.weight",
                    weight_scale="",
                    payloads=payloads,
                    parameters={
                        "group_size": group_size,
                        "convrot_groupsize": 256,
                    },
                    logical_shape=(128, 256),
                )
            },
        )
        return runtime_component_identity("typed-only", (component,))

    base = lines(group_size=16, correction=False)
    assert base != lines(group_size=32, correction=False)
    assert base != lines(group_size=16, correction=True)
    assert any(
        "format=asym_w4a8_int8" in line
        and "logical_shape=(128, 256)" in line
        and "executable=False" in line
        for line in base
    )


def test_int8_convrot_parameters_rotate_executable_identity() -> None:
    def lines(*, convrot: bool, group: int) -> tuple[str, ...]:
        component = ComponentPlan(
            component="diffusion",
            path=Path("int8.safetensors"),
            config="int8",
            keys={"block.weight": "block.weight"},
            dtypes={"block.weight": INT8},
            quant={
                "block": LayerQuant(
                    layer="block",
                    format="int8_tensorwise",
                    weight="block.weight",
                    weight_scale="block.weight_scale",
                    parameters={"convrot": convrot, "convrot_groupsize": group},
                )
            },
        )
        return runtime_component_identity("int8", (component,))

    base = lines(convrot=True, group=256)
    assert base != lines(convrot=False, group=256)
    assert base != lines(convrot=True, group=64)
    assert any(
        "format=int8_tensorwise" in line and "parameters=convrot=True,convrot_groupsize=256" in line
        for line in base
    )


def test_mxfp8_optional_input_scale_presence_rotates_structural_identity() -> None:
    def lines(input_scale: str | None) -> tuple[str, ...]:
        payloads = {"weight_scale": "block.weight_scale"}
        if input_scale is not None:
            payloads["input_scale"] = input_scale
        component = ComponentPlan(
            component="diffusion",
            path=Path("typed-mxfp8.safetensors"),
            config="typed-only",
            keys={"block.weight": "block.weight"},
            dtypes={"block.weight": FLOAT8_E4M3},
            quant={
                "block": LayerQuant(
                    layer="block",
                    format="mxfp8",
                    weight="block.weight",
                    weight_scale="block.weight_scale",
                    input_scale=input_scale,
                    payloads=payloads,
                    logical_shape=(128, 256),
                )
            },
        )
        return runtime_component_identity("typed-only", (component,))

    absent = lines(None)
    present = lines("block.input_scale")
    assert absent != present
    present_quant = next(line for line in present if line.startswith("quant=block"))
    absent_quant = next(line for line in absent if line.startswith("quant=block"))
    assert present_quant.count(" input_scale=") == 1
    assert absent_quant.count(" input_scale=") == 1
    assert "payloads=input_scale,weight_scale" in present_quant
    assert "input_scale=True" in present_quant
    assert "input_scale=False" in absent_quant


def test_each_knob_and_registry_token_rotate_identity() -> None:
    base = identity()
    assert identity(diffusion_dtype=FLOAT16) != base
    assert identity(text_dtype=FLOAT16) != base
    assert identity(vae_dtype=FLOAT16) != base
    assert identity(fp8_matmul=True) != base
    assert identity(registry_token="registry-v2") != base


def test_extension_snapshot_hash_rotates_identity_deterministically() -> None:
    first_snapshot = ExtensionSnapshot(
        (
            ActiveExtension(
                id="example.sampler",
                version="1.0.0",
                package_digest="sha256:" + "1" * 64,
                contribution_ids=("samplers/example",),
                capabilities=("model-family-registration",),
                behavior_configuration=(("precision", "strict"),),
            ),
        )
    )
    second_snapshot = ExtensionSnapshot(
        (
            ActiveExtension(
                id="example.sampler",
                version="1.0.0",
                package_digest="sha256:" + "1" * 64,
                contribution_ids=("samplers/example",),
                capabilities=("model-family-registration",),
                behavior_configuration=(("precision", "fast"),),
            ),
        )
    )
    first_hash = extension_behavior_hash(first_snapshot)
    second_hash = extension_behavior_hash(second_snapshot)

    assert identity(extension_hash=first_hash) == identity(extension_hash=first_hash)
    assert identity(extension_hash=first_hash) != identity()
    assert identity(extension_hash=first_hash) != identity(extension_hash=second_hash)


def test_extension_behavior_hash_must_be_canonical_sha256() -> None:
    with pytest.raises(ValueError, match="lowercase sha256"):
        identity(extension_hash="not-a-hash")


def test_patch_overlay_identity_is_optional_and_ordered() -> None:
    baseline = identity()
    first = "1" * 64
    second = "2" * 64
    assert identity(patch_overlay_digests=None) == baseline
    assert identity(patch_overlay_digests=()) == baseline
    assert identity(patch_overlay_digests=(first,)) != baseline
    assert identity(patch_overlay_digests=(first, second)) != identity(
        patch_overlay_digests=(second, first)
    )
    assert baseline == (
        "native:dinkster.flux_dev:1be4c1ca01b412a75ad055cc1989f0be56d095ecc9a7d720be9440ab43033200"
    )


def test_patch_overlay_identity_requires_canonical_sha256() -> None:
    with pytest.raises(ValueError, match="lowercase sha256"):
        identity(patch_overlay_digests=("not-a-hash",))


def test_component_order_comes_from_the_plan_property() -> None:
    plan = plan_native(combined_dev_checkpoint())
    from_property = build_runtime_identity(
        plan.family.id,
        plan.identity_components,
        diffusion_dtype=BFLOAT16,
        text_dtype=FLOAT32,
        vae_dtype=FLOAT32,
        fp8_matmul=False,
    )
    assert from_property == identity()
    assert (
        build_runtime_identity(
            plan.family.id,
            tuple(reversed(plan.identity_components)),
            diffusion_dtype=BFLOAT16,
            text_dtype=FLOAT32,
            vae_dtype=FLOAT32,
            fp8_matmul=False,
        )
        != from_property
    )


@pytest.mark.parametrize(
    ("family_id", "expected"),
    [
        ("dinkster.flux2_dev", BFLOAT16),
        ("dinkster.flux2_klein_4b", BFLOAT16),
        ("dinkster.flux2_klein_9b", BFLOAT16),
        ("dinkster.flux_dev", BFLOAT16),
        ("dinkster.flux_schnell", BFLOAT16),
        ("dinkster.ideogram4", BFLOAT16),
        ("dinkster.krea2", BFLOAT16),
        ("dinkster.sd15", FLOAT16),
        ("dinkster.sdxl", FLOAT16),
        ("dinkster.sdxl_refiner", FLOAT16),
    ],
)
def test_default_diffusion_dtype(family_id: str, expected: object) -> None:
    assert default_diffusion_dtype(family_id) is expected


def test_default_registry_cache_rebuilds_for_provider_changes_and_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity_module._cached_default_inference_registries.cache_clear()
    original = registry_module.component_catalog.default_component_registry
    calls = 0

    def provider_a():
        nonlocal calls
        calls += 1
        return original()

    def provider_b():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(
        registry_module.component_catalog,
        "default_component_registry",
        provider_a,
    )
    try:
        original_registries = identity_module._default_inference_registries()
        assert default_diffusion_dtype("dinkster.sd15") is FLOAT16
        assert default_text_dtype("dinkster.flux_dev") is BFLOAT16
        assert default_vae_dtype("dinkster.sdxl") is BFLOAT16
        assert identity_module._default_inference_registries() is original_registries
        assert calls == 1

        monkeypatch.setattr(
            registry_module.component_catalog,
            "default_component_registry",
            provider_b,
        )
        changed_registries = identity_module._default_inference_registries()
        assert changed_registries is not original_registries
        assert calls == 2

        monkeypatch.setattr(
            registry_module.component_catalog,
            "default_component_registry",
            provider_a,
        )
        assert identity_module._default_inference_registries() is original_registries
        assert calls == 2

        for _ in range(16):
            monkeypatch.setattr(
                registry_module.component_catalog,
                "default_component_registry",
                lambda: original(),
            )
            identity_module._default_inference_registries()
        cache_info = identity_module._cached_default_inference_registries.cache_info()
        assert cache_info.maxsize == 8
        assert cache_info.currsize == 8
    finally:
        identity_module._cached_default_inference_registries.cache_clear()


def test_default_diffusion_dtype_reports_unknown_label_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert default_diffusion_dtype("dinkster.unknown") is BFLOAT16
    assert "no diffusion dtype specialization; defaulting to bfloat16" in caplog.text


def test_every_wired_family_has_a_default_diffusion_dtype() -> None:
    """default_diffusion_dtype's family table is hardcoded; this pins
    it against NATIVE_WIRED_FAMILY_IDS drift - a newly wired family
    without a default would otherwise only fail at torch load time."""
    for family_id in NATIVE_WIRED_FAMILY_IDS:
        default_diffusion_dtype(family_id)


@pytest.mark.parametrize(
    "family_id",
    [
        "dinkster.flux_dev",
        "dinkster.flux_schnell",
    ],
)
def test_default_text_dtype(family_id: str) -> None:
    assert default_text_dtype(family_id) is BFLOAT16


@pytest.mark.parametrize(
    "family_id",
    [
        "dinkster.anima",
        "dinkster.chroma",
        "dinkster.chroma_radiance",
        "dinkster.ideogram4",
        "dinkster.krea2",
        "dinkster.lumina2",
        "dinkster.sd15",
        "dinkster.sdxl",
        "dinkster.sdxl_refiner",
        "dinkster.wan21",
        "dinkster.wan22",
        "dinkster.z_image",
        "dinkster.z_image_pixel_space",
    ],
)
def test_reference_float32_text_families_default_to_float32(family_id: str) -> None:
    assert default_text_dtype(family_id) is FLOAT32


def test_default_text_dtype_reports_unknown_label_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert default_text_dtype("dinkster.unknown") is BFLOAT16
    assert "no text dtype specialization; defaulting to bfloat16" in caplog.text


def test_every_wired_family_has_a_default_text_dtype() -> None:
    """default_text_dtype validates the family id against the same table
    as default_diffusion_dtype; this pins it against
    NATIVE_WIRED_FAMILY_IDS drift."""
    float32_families = {
        "dinkster.anima",
        "dinkster.chroma",
        "dinkster.chroma_radiance",
        "dinkster.ideogram4",
        "dinkster.krea2",
        "dinkster.lumina2",
        "dinkster.sd15",
        "dinkster.sdxl",
        "dinkster.sdxl_refiner",
        "dinkster.wan21",
        "dinkster.wan22",
        "dinkster.z_image",
        "dinkster.z_image_pixel_space",
    }
    for family_id in NATIVE_WIRED_FAMILY_IDS:
        expected = FLOAT32 if family_id in float32_families else BFLOAT16
        assert default_text_dtype(family_id) is expected


@pytest.mark.parametrize(
    ("family_id", "supported", "expected"),
    (
        ("dinkster.sdxl", (FLOAT16, BFLOAT16, FLOAT32), BFLOAT16),
        ("dinkster.flux_dev", (FLOAT16, FLOAT32), FLOAT32),
        ("dinkster.wan21", (FLOAT16, BFLOAT16, FLOAT32), BFLOAT16),
        ("dinkster.wan21", (FLOAT16, FLOAT32), FLOAT16),
        ("dinkster.minimax_h3", (BFLOAT16, FLOAT32), FLOAT32),
        ("dinkster.minimax_h3", (FLOAT16, BFLOAT16, FLOAT32), FLOAT16),
        ("dinkster.triposplat", (FLOAT16, BFLOAT16, FLOAT32), FLOAT16),
    ),
)
def test_default_vae_dtype_matches_comfyui_working_dtype_order(
    family_id: str, supported: tuple[DType, ...], expected: DType
) -> None:
    assert default_vae_dtype(family_id, supported) is expected


def test_sd_vae_policy_excludes_overflow_prone_float16() -> None:
    assert default_vae_dtype("dinkster.sd15", (FLOAT16, FLOAT32)) is FLOAT32
    assert default_vae_dtype("dinkster.sdxl", (FLOAT16, FLOAT32)) is FLOAT32
    assert default_vae_dtype("dinkster.sdxl_refiner", (FLOAT16, FLOAT32)) is FLOAT32


def test_default_vae_dtype_reports_unknown_label_default(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert default_vae_dtype("dinkster.unknown") is BFLOAT16
    assert "no VAE dtype specialization; defaulting to bfloat16" in caplog.text
    assert default_vae_dtype("dinkster.unknown", (FLOAT32,)) is FLOAT32
    with pytest.raises(ValueError, match="no VAE dtype supported by device set"):
        default_vae_dtype("dinkster.unknown", ())


def test_every_wired_family_has_a_default_vae_dtype() -> None:
    for family_id in NATIVE_WIRED_FAMILY_IDS:
        default_vae_dtype(family_id)


def test_plan_native_and_probe_native_agree_on_acceptance() -> None:
    checkpoint = combined_dev_checkpoint()
    capability = probe_native(checkpoint)
    plan = plan_native(checkpoint)
    assert capability.native
    assert plan.family.id == capability.family_id


def test_plan_native_and_probe_native_agree_on_refusal_reasons() -> None:
    unknown = source({}, "unknown.safetensors")
    capability = probe_native(unknown)
    with pytest.raises(NativeRefusalError) as caught:
        plan_native(unknown)
    assert caught.value.reasons == capability.reasons
    assert (
        caught.value.category
        is capability.refusal_category
        is NativeRefusalCategory.INVALID_INPUT_OR_IDENTITY
    )


def test_plan_native_and_probe_native_share_no_source_error() -> None:
    with pytest.raises(ValueError) as probe_error:
        probe_native()
    with pytest.raises(ValueError) as plan_error:
        plan_native()
    assert str(plan_error.value) == str(probe_error.value)
