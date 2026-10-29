"""Exact torch-free Z-Image architecture, text identity, and layout proofs."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from dinkster_inference import (
    BFLOAT16,
    FLOAT8_E4M3,
    FLOAT32,
    UINT8,
    Z_IMAGE,
    Z_IMAGE_CONFIG,
    Z_IMAGE_PIXEL_CONFIG,
    Z_IMAGE_PIXEL_SPACE,
    Z_IMAGE_QWEN3_4B_CONFIG,
    Z_IMAGE_SIGMAS,
    TensorGeometry,
    WeightEntry,
    detect_qwen_text_config,
    detect_z_image,
    load_safetensors_header,
    plan_native,
    plan_z_image_assembly,
    plan_z_image_control,
    plan_z_image_token_layout,
    probe_native,
    qwen_text_layout,
    tokenize_z_image_prompt,
    z_image_layout,
    z_image_pixel_layout,
)
from dinkster_inference.z_image import z_image_control_layout


class HeaderSource:
    def __init__(self, shapes: Mapping[str, tuple[int, ...]]) -> None:
        self.shapes = dict(shapes)

    def keys(self) -> Sequence[str]:
        return tuple(self.shapes)

    def entry(self, key: str) -> WeightEntry:
        geometry = TensorGeometry(self.shapes[key], BFLOAT16)
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        raise AssertionError("Z-Image detection must not read metadata")


class PathHeaderSource(HeaderSource):
    def __init__(self, shapes: Mapping[str, tuple[int, ...]], path: str) -> None:
        super().__init__(shapes)
        self.path = Path(path)

    def metadata(self) -> Mapping[str, str]:
        return {}


class QuantizedPathHeaderSource:
    def __init__(
        self,
        geometries: Mapping[str, TensorGeometry],
        payloads: Mapping[str, bytes],
        path: str,
    ) -> None:
        self.geometries = dict(geometries)
        self.payloads = dict(payloads)
        self.path = Path(path)

    def keys(self) -> Sequence[str]:
        return tuple(self.geometries)

    def entry(self, key: str) -> WeightEntry:
        geometry = self.geometries[key]
        return WeightEntry(key, geometry, 0, geometry.nbytes)

    def metadata(self) -> Mapping[str, str]:
        return {}

    def read_uint8_configuration(self, key: str) -> bytes:
        return self.payloads[key]


def test_z_image_exact_profile_and_453_entry_layout() -> None:
    config = Z_IMAGE_CONFIG
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
    ) == (3840, 2560, 30, 2, 2, 30, 30, 128, 10240)
    assert config.patch == (2, 2)
    assert config.frame_patch == 1
    assert config.rope_axes == (32, 48, 48)
    assert config.rope_theta == 256.0
    assert config.qk_norm_eps == 1e-5
    assert config.modulation_width == 256
    assert config.pad_tokens_multiple == 32
    assert config.sampling_shift == 3.0
    assert Z_IMAGE_SIGMAS.shift == 3.0

    layout = z_image_layout()
    assert len(layout) == 453
    assert layout["cap_embedder.1.weight"] == (3840, 2560)
    assert layout["layers.29.adaLN_modulation.0.weight"] == (15360, 256)
    assert layout["noise_refiner.1.attention.qkv.weight"] == (11520, 3840)
    assert layout["context_refiner.1.feed_forward.w2.weight"] == (3840, 10240)
    assert layout["final_layer.linear.weight"] == (64, 3840)
    with pytest.raises(TypeError):
        layout["extra"] = (1,)  # type: ignore[index]
    with pytest.raises(ValueError, match="exact latent Z-Image"):
        replace(config, hidden_width=2304)

    family_evidence = Z_IMAGE.detector.detect(HeaderSource(z_image_layout()))
    assert family_evidence is not None and family_evidence.family_id == config.family_id
    assert Z_IMAGE.single_stream_latent().channels == 16
    assert Z_IMAGE.sampling.shift == 3.0
    assert Z_IMAGE.sampling.sigma_min == pytest.approx(3 / 1002)
    assert Z_IMAGE.display_name == "Z-Image"


@pytest.mark.parametrize("prefix", ("", "model.diffusion_model."))
def test_z_image_detector_accepts_only_exact_latent_non_omni_shape(prefix: str) -> None:
    shapes = {prefix + key: shape for key, shape in z_image_layout().items()}
    evidence = detect_z_image(HeaderSource(shapes))
    assert evidence is not None
    assert evidence.config is Z_IMAGE_CONFIG
    assert evidence.key_prefix == prefix
    assert evidence.fields["sampling_shift"] == 3.0


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("cap_embedder.1.weight", (2304, 2560)),
        ("dec_net.cond_embed.weight", (3840, 3840)),
        ("siglip_embedder.0.weight", (1152,)),
        ("layers.30.attention.q_norm.weight", (128,)),
        ("layers.17.feed_forward.w2.weight", (3840, 10239)),
    ),
)
def test_z_image_detector_refuses_lumina_pixel_omni_and_extra_depth(
    mutation: str, value: tuple[int, ...]
) -> None:
    shapes = dict(z_image_layout())
    shapes[mutation] = value
    assert detect_z_image(HeaderSource(shapes)) is None


def test_z_image_detector_refuses_incomplete_layout() -> None:
    shapes = dict(z_image_layout())
    del shapes["noise_refiner.1.ffn_norm2.weight"]
    assert detect_z_image(HeaderSource(shapes)) is None


def test_z_image_pixel_space_profile_is_exact_and_separate() -> None:
    layout = z_image_pixel_layout()
    assert len(layout) == 557
    assert layout["x_embedder.weight"] == (3840, 3072)
    assert layout["layers.29.attention.to_q.weight"] == (3840, 3840)
    assert layout["dec_net.input_embedder.embedder.0.weight"] == (3840, 3136)
    assert layout["dec_net.res_blocks.3.adaLN_modulation.1.weight"] == (11520, 3840)
    assert layout["dec_net.final_layer.linear.weight"] == (3072, 3840)
    evidence = detect_z_image(HeaderSource(layout))
    assert evidence is not None and evidence.config is Z_IMAGE_PIXEL_CONFIG
    assert evidence.fields["patch"] == "32x32"
    family_evidence = Z_IMAGE_PIXEL_SPACE.detector.detect(HeaderSource(layout))
    assert family_evidence is not None
    assert family_evidence.family_id == "dinkster.z_image_pixel_space"
    assert Z_IMAGE.detector.detect(HeaderSource(layout)) is None
    assert Z_IMAGE_PIXEL_SPACE.single_stream_latent().channels == 3
    assert Z_IMAGE_PIXEL_SPACE.single_stream_latent().spatial_downscale == 1


def test_z_image_pixel_space_native_plan_uses_no_vae() -> None:
    diffusion = PathHeaderSource(z_image_pixel_layout(), "/fake/zeta-chroma.safetensors")
    qwen = PathHeaderSource(
        {"model." + key: shape for key, shape in qwen_text_layout(Z_IMAGE_QWEN3_4B_CONFIG).items()},
        "/fake/qwen_3_4b.safetensors",
    )
    plan = plan_native(diffusion=diffusion, qwen3_4b=qwen)
    from dinkster_inference.assembly import ZImageAssemblyPlan

    assert isinstance(plan, ZImageAssemblyPlan)
    assert plan.family is Z_IMAGE_PIXEL_SPACE
    assert plan.vae is None
    assert len(plan.diffusion.keys) == 555
    assert plan.diffusion.ignored == ("__sequential__", "__x0__")
    assert plan.identity_components == (plan.diffusion, plan.qwen3_4b)
    capability = probe_native(diffusion=diffusion, qwen3_4b=qwen)
    assert capability.native and capability.family_id == Z_IMAGE_PIXEL_SPACE.id
    with pytest.raises(ValueError, match="accepts no VAE"):
        plan_z_image_assembly(diffusion=diffusion, qwen3_4b=qwen, vae=qwen)


def test_z_image_union_control_plan_is_exact_and_fail_closed(tmp_path: Path) -> None:
    class PlanSource(HeaderSource):
        def __init__(self, shapes: Mapping[str, tuple[int, ...]], path: Path) -> None:
            super().__init__(shapes)
            self.path = path

        def metadata(self) -> Mapping[str, str]:
            return {}

    layout = z_image_control_layout()
    assert len(layout) == 136
    assert layout["control_all_x_embedder.2-1.weight"] == (3840, 64)
    assert layout["control_layers.5.attention.to_v.weight"] == (3840, 3840)
    source = PlanSource(layout, tmp_path / "control.safetensors")
    plan = plan_z_image_control(source, asset_digest="blake3:" + "0" * 64)
    assert plan.source_role == "z_image_control"
    assert tuple(plan.control.keys) == tuple(layout)
    missing = dict(layout)
    missing.pop("control_layers.5.after_proj.bias")
    incomplete = PlanSource(missing, source.path)
    with pytest.raises(ValueError, match="exact 136-key BF16"):
        plan_z_image_control(incomplete, asset_digest="blake3:" + "0" * 64)


def test_z_image_detector_accepts_unrelated_combined_checkpoint_components() -> None:
    prefix = "model.diffusion_model."
    shapes = {prefix + key: shape for key, shape in z_image_layout().items()}
    shapes["vae.decoder.conv_in.weight"] = (128, 16, 3, 3)
    shapes["text_encoders.qwen3_4b.transformer.model.embed_tokens.weight"] = (151936, 2560)
    evidence = detect_z_image(HeaderSource(shapes))
    assert evidence is not None and evidence.key_prefix == prefix


def test_z_image_qwen3_4b_identity_and_layout() -> None:
    config = Z_IMAGE_QWEN3_4B_CONFIG
    assert (
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    ) == (2560, 9728, 36, 32, 8, 128)
    assert config.pad_token_id == 151643
    assert config.qk_norm
    assert config.output_hidden_layer == -2
    assert not config.layer_norm_hidden_state
    assert not config.zero_masked
    layout = qwen_text_layout(config)
    assert len(layout) == 398
    assert layout["layers.0.self_attn.q_proj.weight"] == (4096, 2560)
    assert layout["layers.0.self_attn.k_proj.weight"] == (1024, 2560)
    assert layout["layers.35.self_attn.o_proj.weight"] == (2560, 4096)
    geometries = {key: TensorGeometry(shape, BFLOAT16) for key, shape in layout.items()}
    assert detect_qwen_text_config(geometries) is config


def test_z_image_qwen3_4b_mixed_fp8_nvfp4_plan_is_exact() -> None:
    prefix = "model."
    geometries = {
        prefix + key: TensorGeometry(shape, BFLOAT16)
        for key, shape in qwen_text_layout(Z_IMAGE_QWEN3_4B_CONFIG).items()
    }
    fp8_layer = prefix + "layers.0.self_attn.k_proj"
    fp8_config = b'{"format":"float8_e4m3fn"}'
    geometries[fp8_layer + ".weight"] = TensorGeometry((1024, 2560), FLOAT8_E4M3)
    geometries[fp8_layer + ".weight_scale"] = TensorGeometry((), FLOAT32)
    geometries[fp8_layer + ".comfy_quant"] = TensorGeometry((len(fp8_config),), UINT8)

    nvfp4_layer = prefix + "layers.0.self_attn.q_proj"
    nvfp4_config = b'{"format":"nvfp4"}'
    geometries[nvfp4_layer + ".weight"] = TensorGeometry((4096, 1280), UINT8)
    geometries[nvfp4_layer + ".weight_scale"] = TensorGeometry((4096, 160), FLOAT8_E4M3)
    geometries[nvfp4_layer + ".weight_scale_2"] = TensorGeometry((), FLOAT32)
    geometries[nvfp4_layer + ".comfy_quant"] = TensorGeometry((len(nvfp4_config),), UINT8)

    qwen = QuantizedPathHeaderSource(
        geometries,
        {
            fp8_layer + ".comfy_quant": fp8_config,
            nvfp4_layer + ".comfy_quant": nvfp4_config,
        },
        "/fake/qwen_3_4b_mixed.safetensors",
    )
    goldens = json.loads(
        (
            Path(__file__).parents[1]
            / "packages/dinkster-inference-torch/tests/goldens/kl_goldens.json"
        ).read_text()
    )
    vae_layout = {key: tuple(shape) for key, shape in goldens["cases"]["regularizer"]["state_dict"]}
    plan = plan_z_image_assembly(
        diffusion=PathHeaderSource(z_image_layout(), "/fake/z_image.safetensors"),
        qwen3_4b=qwen,
        vae=PathHeaderSource(vae_layout, "/fake/ae.safetensors"),
    )

    assert len(plan.qwen3_4b.keys) == 398
    assert plan.qwen3_4b.quant["layers.0.self_attn.k_proj"].format is None
    assert plan.qwen3_4b.quant["layers.0.self_attn.k_proj"].config == (
        "model.layers.0.self_attn.k_proj.comfy_quant"
    )
    assert plan.qwen3_4b.quant["layers.0.self_attn.q_proj"].format == "nvfp4"
    assert plan.qwen3_4b.keys["layers.0.self_attn.q_proj.weight"] == (
        "model.layers.0.self_attn.q_proj.weight"
    )


def test_z_image_prompt_uses_fixed_chat_template_without_padding() -> None:
    class Tokenizer:
        text = ""

        def encode(self, text: str) -> list[int]:
            self.text = text
            return [151644, 872, 198, 42, 151645, 198, 151644, 77091, 198]

    tokenizer = Tokenizer()
    tokens = tokenize_z_image_prompt("cat", tokenizer=tokenizer)  # type: ignore[arg-type]
    assert tokenizer.text == "<|im_start|>user\ncat<|im_end|>\n<|im_start|>assistant\n"
    assert tokens.ids == (151644, 872, 198, 42, 151645, 198, 151644, 77091, 198)
    assert tokens.attention_mask == (1,) * len(tokens.ids)


def test_z_image_layout_declares_learned_padding_and_corrected_rope_positions() -> None:
    plan = plan_z_image_token_layout(5, (3, 3))
    assert tuple(segment.identity for segment in plan.layout.segments) == (
        "caption",
        "caption_pad",
        "image",
        "image_pad",
    )
    assert tuple(segment.role for segment in plan.layout.segments) == (
        "conditioning",
        "learned_padding",
        "target",
        "learned_padding",
    )
    assert (plan.caption_padding_rows, plan.image_padding_rows) == (27, 23)
    assert plan.layout.padded_rows == 0
    assert plan.layout.total_rows == 64
    positions = plan.positions.values
    assert positions[4] == (5.0, 0.0, 0.0)
    assert positions[31] == (32.0, 0.0, 0.0)
    assert positions[32] == (33.0, 0.0, 0.0)
    assert positions[40] == (33.0, 2.0, 2.0)
    assert positions[41:] == ((0.0, 0.0, 0.0),) * 23


def test_z_image_layout_omits_empty_learned_padding_segments() -> None:
    plan = plan_z_image_token_layout(32, (4, 8))
    assert tuple(segment.identity for segment in plan.layout.segments) == ("caption", "image")
    assert plan.caption_padding_rows == plan.image_padding_rows == 0
    assert plan.positions.values[32] == (33.0, 0.0, 0.0)


def test_z_image_split_assembly_maps_exact_components() -> None:
    goldens = json.loads(
        (
            Path(__file__).parents[1]
            / "packages/dinkster-inference-torch/tests/goldens/kl_goldens.json"
        ).read_text()
    )
    vae_layout = {key: tuple(shape) for key, shape in goldens["cases"]["regularizer"]["state_dict"]}
    plan = plan_z_image_assembly(
        diffusion=PathHeaderSource(z_image_layout(), "/fake/z_image.safetensors"),
        qwen3_4b=PathHeaderSource(
            {
                "model." + key: shape
                for key, shape in qwen_text_layout(Z_IMAGE_QWEN3_4B_CONFIG).items()
            },
            "/fake/qwen_3_4b.safetensors",
        ),
        vae=PathHeaderSource(vae_layout, "/fake/ae.safetensors"),
    )
    assert plan.family is Z_IMAGE
    assert len(plan.diffusion.keys) == 453
    assert len(plan.qwen3_4b.keys) == 398
    assert plan.qwen3_4b.keys["layers.35.mlp.down_proj.weight"] == (
        "model.layers.35.mlp.down_proj.weight"
    )
    assert plan.vae is not None
    assert plan.vae.config.embed_dim == 16
    assert plan.identity_components == (plan.diffusion, plan.qwen3_4b, plan.vae)


def test_z_image_is_admitted_by_native_probe_with_qwen3_4b_slot() -> None:
    goldens = json.loads(
        (
            Path(__file__).parents[1]
            / "packages/dinkster-inference-torch/tests/goldens/kl_goldens.json"
        ).read_text()
    )
    sources = {
        "diffusion": PathHeaderSource(z_image_layout(), "/fake/z_image.safetensors"),
        "qwen3_4b": PathHeaderSource(
            {
                "model." + key: shape
                for key, shape in qwen_text_layout(Z_IMAGE_QWEN3_4B_CONFIG).items()
            },
            "/fake/qwen_3_4b.safetensors",
        ),
        "vae": PathHeaderSource(
            {key: tuple(shape) for key, shape in goldens["cases"]["regularizer"]["state_dict"]},
            "/fake/ae.safetensors",
        ),
    }
    capability = probe_native(
        diffusion=sources["diffusion"],
        qwen3_4b=sources["qwen3_4b"],
        vae=sources["vae"],
    )
    assert capability.native and capability.family_id == Z_IMAGE.id
    plan = plan_native(
        diffusion=sources["diffusion"],
        qwen3_4b=sources["qwen3_4b"],
        vae=sources["vae"],
    )
    assert plan.family is Z_IMAGE


@pytest.fixture
def z_image_split_paths(model_root: Path, qwen_name: str) -> tuple[Path, Path, Path]:
    paths = (
        model_root / "diffusion_models/z_image_turbo_bf16.safetensors",
        model_root / "text_encoders" / qwen_name,
        model_root / "vae/ae.safetensors",
    )
    if not all(path.exists() for path in paths):
        pytest.skip("Z-Image split assets absent")
    return paths


@pytest.mark.parametrize(
    ("qwen_name", "quant_formats"),
    (
        pytest.param("qwen_3_4b.safetensors", {}, id="bf16"),
        pytest.param(
            "qwen_3_4b_fp8_mixed.safetensors",
            {None: 177, "nvfp4": 12},
            id="fp8-mixed",
        ),
        pytest.param(
            "qwen_3_4b_fp4_mixed.safetensors",
            {None: 58, "nvfp4": 189},
            id="fp4-mixed",
        ),
    ),
)
def test_real_z_image_split_assembly(
    z_image_split_paths: tuple[Path, Path, Path], quant_formats: Mapping[str | None, int]
) -> None:
    paths = z_image_split_paths
    plan = plan_z_image_assembly(
        diffusion=load_safetensors_header(paths[0]),
        qwen3_4b=load_safetensors_header(paths[1]),
        vae=load_safetensors_header(paths[2]),
    )
    assert plan.vae is not None
    assert (len(plan.diffusion.keys), len(plan.qwen3_4b.keys), len(plan.vae.keys)) == (
        453,
        398,
        244,
    )
    assert tuple(component.component for component in plan.identity_components) == (
        "diffusion",
        "qwen3_4b",
        "vae",
    )
    assert tuple(component.path for component in plan.identity_components) == paths
    assert Counter(item.format for item in plan.qwen3_4b.quant.values()) == quant_formats


@pytest.fixture
def real_z_image_base(model_root: Path) -> Path:
    path = model_root / "diffusion_models/z_image_bf16.safetensors"
    if not path.exists():
        pytest.skip("Z-Image Base weights absent")
    return path


@pytest.fixture
def real_z_image_pixel(model_root: Path) -> Path:
    path = model_root / "diffusion_models/zeta-chroma-base-x0-pixel-no-dino-1024.safetensors"
    if not path.exists():
        pytest.skip("Zeta-Chroma weights absent")
    return path


def test_real_z_image_base_uses_the_shared_latent_family_contract(real_z_image_base: Path) -> None:
    source = load_safetensors_header(real_z_image_base)
    evidence = detect_z_image(source)
    assert evidence is not None
    assert evidence.config is Z_IMAGE_CONFIG
    assert evidence.key_prefix == ""
    assert len(evidence.matched_keys) == 453


def test_real_z_image_pixel_header_matches_exact_native_profile(real_z_image_pixel: Path) -> None:
    source = load_safetensors_header(real_z_image_pixel)
    evidence = detect_z_image(source)
    assert evidence is not None and evidence.config is Z_IMAGE_PIXEL_CONFIG
    assert evidence.key_prefix == ""
    assert len(evidence.matched_keys) == 557
