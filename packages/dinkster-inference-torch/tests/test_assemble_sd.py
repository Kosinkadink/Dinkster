"""Executing SD-era assembly plans: planned slices -> modules.

assemble_sd shares assemble_flux's per-component executor - dtype
preservation, quant handling, and the documented refusals are proven
in test_assemble.py and not repeated. These tests cover what the SD
families ADD: the UNetModel builder, optional text-encoder slots
staying None (SD 1.5 wires no CLIP-G, the refiner no CLIP-L), and the
planned OpenCLIP tensor transforms (the fused in_proj RowChunk split
and the text_projection Transpose2D - comfy/utils.py
transformers_convert / clip_text_transformers_convert @ 947c2749),
which no Flux plan ever carries. Detection + full-size assembly
against installed checkpoints lives in the capability-gated GPU
suite.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from dinkster_inference import (
    FLOAT32,
    Q8_0,
    SD15,
    SDXL,
    SDXL_REFINER,
    ClipTextConfig,
    ComponentPlan,
    GGUFComponentMap,
    GGUFComponentTensor,
    GGUFResidencyMode,
    GGUFSource,
    KLConfig,
    LinearToConv2D,
    RowChunk,
    SDAssemblyPlan,
    TAESDCodecPlan,
    TAESDConfig,
    TensorGeometry,
    Transpose2D,
    UNetConfig,
    load_gguf_weight_source,
    load_safetensors_header,
    plan_sd_assembly,
)
from dinkster_inference.autoencoder_kl import diffusers_kl_key
from dinkster_inference_torch import (
    TAESD,
    AssembledSD,
    AssembleError,
    AutoencoderKL,
    CastOperations,
    ClipTextModel,
    GgufEncodedLinear,
    TAESDDecoder,
    TAESDEncoder,
    UNetModel,
    assemble_sd,
    enroll_assembled,
)
from dinkster_inference_torch.controlnet import (
    ControlResourceBindingError,
    _validate_sdxl_base_resource,  # pyright: ignore[reportPrivateUsage]
)
from test_assemble import component_plan, component_state, write_checkpoint, write_tiny_gguf
from test_autoencoder_kl import build_model
from unet_fill import fill_state_dict, hashed_input

from tests.test_inference_assembly import (  # pyright: ignore[reportMissingImports]
    clip_l_sd_geometries,
    sdxl_combined_geometries,
    source,
    unet_geometries,
)
from tests.test_inference_gguf import (  # pyright: ignore[reportMissingImports]
    _decode_vectors,
    _encoded_storage,
)
from tests.test_inference_kl import (  # pyright: ignore[reportMissingImports]
    diffusers_geometries,
)

UNET_GOLDENS = json.loads((Path(__file__).parent / "goldens/unet_goldens.json").read_text())

TINY_CLIP_L = ClipTextConfig(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=128,
    hidden_act="quick_gelu",
    vocab_size=96,
    eos_token_id=95,
)
TINY_CLIP_G = ClipTextConfig(
    hidden_size=48,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=96,
    hidden_act="gelu",
    vocab_size=96,
    eos_token_id=95,
)
TINY_SD1_UNET = UNetConfig(
    in_channels=4,
    out_channels=4,
    model_channels=32,
    num_res_blocks=(1, 1),
    channel_mult=(1, 2),
    transformer_depth=(1, 1),
    transformer_depth_output=(1, 1, 1, 1),
    transformer_depth_middle=1,
    context_dim=TINY_CLIP_L.hidden_size,
    use_linear_in_transformer=False,
    num_heads=8,
)
TINY_SDXL_UNET = UNetConfig(
    in_channels=4,
    out_channels=4,
    model_channels=32,
    num_res_blocks=(1, 1),
    channel_mult=(1, 2),
    transformer_depth=(0, 2),
    transformer_depth_output=(0, 0, 2, 2),
    transformer_depth_middle=2,
    context_dim=TINY_CLIP_L.hidden_size + TINY_CLIP_G.hidden_size,
    use_linear_in_transformer=True,
    adm_in_channels=12,
    num_head_channels=16,
)
TINY_SDXL_INPAINT_UNET = replace(TINY_SDXL_UNET, in_channels=9, context_dim=24)
TINY_REFINER_UNET = UNetConfig(
    in_channels=4,
    out_channels=4,
    model_channels=32,
    num_res_blocks=(1, 1),
    channel_mult=(1, 2),
    transformer_depth=(0, 2),
    transformer_depth_output=(0, 0, 2, 2),
    transformer_depth_middle=2,
    context_dim=TINY_CLIP_G.hidden_size,
    use_linear_in_transformer=True,
    adm_in_channels=16,
    num_head_channels=16,
)
TINY_KL = KLConfig(
    in_channels=3,
    out_channels=3,
    ch=32,
    decoder_ch=32,
    ch_mult=(1, 2),
    num_res_blocks=1,
    z_channels=4,
    embed_dim=4,
)


def unet_state(config: UNetConfig) -> dict[str, torch.Tensor]:
    """The UNet's deterministic state (the golden fill: rank-1 weights
    are norm scales and center on 1.0, which clip_fill's name rule
    would miss)."""
    entries = [(key, list(value.shape)) for key, value in UNetModel(config).state_dict().items()]
    return dict(fill_state_dict(entries))


@pytest.fixture(scope="module")
def standard(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One ordinary fp32 split-checkpoint set per component, written
    once; (plan, state) pairs like test_assemble's fixture."""
    tmp = tmp_path_factory.mktemp("tiny-sd")
    built: dict[str, Any] = {}
    states: dict[str, dict[str, torch.Tensor]] = {
        "diffusion_sd1": unet_state(TINY_SD1_UNET),
        "diffusion_sdxl": unet_state(TINY_SDXL_UNET),
        "diffusion_refiner": unet_state(TINY_REFINER_UNET),
        "clip_l": component_state(ClipTextModel(TINY_CLIP_L)),
        "clip_g": component_state(ClipTextModel(TINY_CLIP_G)),
        "vae": component_state(AutoencoderKL(TINY_KL)),
    }
    configs: dict[str, object] = {
        "diffusion_sd1": TINY_SD1_UNET,
        "diffusion_sdxl": TINY_SDXL_UNET,
        "diffusion_refiner": TINY_REFINER_UNET,
        "clip_l": TINY_CLIP_L,
        "clip_g": TINY_CLIP_G,
        "vae": TINY_KL,
    }
    for name, state in states.items():
        path = write_checkpoint(tmp / f"{name}.safetensors", state)
        component = "diffusion" if name.startswith("diffusion") else name
        built[name] = (
            component_plan(component, path, configs[name], state),
            state,
        )
    return built


def sd1_plan(standard: dict[str, Any], **overrides: Any) -> SDAssemblyPlan:
    plans: dict[str, ComponentPlan[Any]] = {
        "diffusion": standard["diffusion_sd1"][0],
        "clip_l": standard["clip_l"][0],
        "vae": standard["vae"][0],
    }
    plans.update(overrides)
    return SDAssemblyPlan(
        family=SD15,
        diffusion=plans["diffusion"],
        clip_l=plans["clip_l"],
        clip_g=None,
        vae=overrides.get("vae", plans["vae"]),
    )


def sdxl_plan(standard: dict[str, Any], **overrides: ComponentPlan[Any]) -> SDAssemblyPlan:
    plans: dict[str, ComponentPlan[Any]] = {
        "diffusion": standard["diffusion_sdxl"][0],
        "clip_l": standard["clip_l"][0],
        "clip_g": standard["clip_g"][0],
        "vae": standard["vae"][0],
    }
    plans.update(overrides)
    return SDAssemblyPlan(
        family=SDXL,
        diffusion=plans["diffusion"],
        clip_l=plans["clip_l"],
        clip_g=plans["clip_g"],
        vae=plans["vae"],
    )


def forward_diffusion(assembled: AssembledSD, config: UNetConfig) -> None:
    """A bounded UNet forward (shapes only; numerics are golden-pinned
    in test_unet.py)."""
    y = None if config.adm_in_channels is None else torch.randn(1, config.adm_in_channels)
    out = assembled.diffusion(
        torch.randn(1, config.in_channels, 8, 8),
        torch.tensor([500.0]),
        context=torch.randn(1, 3, config.context_dim or 1),
        y=y,
    )
    assert out.shape == (1, config.out_channels, 8, 8)


# ------------------------------------------------------- happy paths


def test_sd1_assembly_wires_no_clip_g(standard: dict[str, Any]) -> None:
    assembled = assemble_sd(sd1_plan(standard), diffusion_dtype=torch.float32)
    assert assembled.family is SD15
    assert assembled.clip_g is None
    assert assembled.clip_l is not None
    forward_diffusion(assembled, TINY_SD1_UNET)
    embeds = assembled.clip_l.embed_tokens(torch.tensor([[1, 2, 95]]))
    assert assembled.clip_l(embeds, torch.tensor([2])).pooled.shape == (1, 64)
    assert isinstance(assembled.vae, AutoencoderKL)
    latent = assembled.vae.encode(torch.randn(1, 3, 16, 16))
    assert assembled.vae.decode(latent).shape == (1, 3, 16, 16)


def test_sd_assembly_uses_gguf_payload_loader(
    standard: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    diffusion, state = standard["diffusion_sd1"]
    authority, _model_key, _payload = _encoded_storage(tmp_path, Q8_0, _decode_vectors()[1][1])
    diffusion = replace(
        diffusion,
        path=authority.path,
        source_format="gguf",
        runtime_facts=authority.runtime_facts,
        payload_source=authority,
    )
    calls: list[tuple[Path, set[str]]] = []

    def load(
        payload_source: object,
        keys: set[str],
        *,
        expected_runtime_facts: tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        assert expected_runtime_facts == diffusion.runtime_facts
        assert payload_source is authority
        calls.append((authority.path, keys))
        return {key: state[key] for key in keys}

    monkeypatch.setattr("dinkster_inference_torch.assemble.load_gguf_tensors", load)
    assembled = assemble_sd(sd1_plan(standard, diffusion=diffusion), diffusion_dtype=torch.float32)

    assert calls == [(diffusion.path, set(diffusion.keys.values()))]
    retained_plan = assembled.diffusion.__dict__["_dinkster_component_plan"]
    assert retained_plan.source_format == "gguf"
    assert retained_plan.runtime_facts == diffusion.runtime_facts
    assert retained_plan.payload_source is None
    assert retained_plan.payload_consumed is True
    forward_diffusion(assembled, TINY_SD1_UNET)


def test_sd_assembly_gguf_encoded_residency_matches_speed_bit_exactly(
    standard: dict[str, Any], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Encoded residency modes must change weight storage only:
    swapped Linears decode the same float32 values the eager loader
    materializes, so every mode produces bit-identical outputs. The
    balanced mode also fills its decoded cache on the first forward."""
    diffusion, state = standard["diffusion_sd1"]
    path = write_tiny_gguf(tmp_path / "tiny-sd1.gguf", state, architecture="sd1")
    shapes = {key: tuple(value.shape) for key, value in state.items()}

    def tiny_map(gguf_source: GGUFSource) -> GGUFComponentMap:
        """The production mapper admits only full-size family
        geometry; this stand-in maps the tiny UNet identically."""
        tensors = {
            name: GGUFComponentTensor(
                model_key=name,
                source_name=name,
                logical_shape=shapes[name],
                ggml_type=tensor.ggml_type,
                offset=tensor.offset,
                nbytes=tensor.nbytes,
            )
            for name, tensor in gguf_source.tensors.items()
        }
        return GGUFComponentMap(
            mapper_id="dinkster.gguf.diffusion.v1",
            architecture="sd1",
            family_id="dinkster.sd15",
            component="diffusion",
            tensor_prefix="",
            tensors=tensors,
        )

    monkeypatch.setattr("dinkster_inference.gguf.map_gguf_component", tiny_map)

    assembled: dict[GGUFResidencyMode, UNetModel] = {}
    modes: tuple[GGUFResidencyMode, ...] = ("speed", "memory", "balanced")
    for mode in modes:
        # An explicit balanced budget keeps the cache assertions
        # independent of this host's free memory.
        budget = (1 << 20) if mode == "balanced" else None
        authority = load_gguf_weight_source(path, residency_mode=mode, decoded_cache_budget=budget)
        plan = replace(
            diffusion,
            path=path,
            source_format="gguf",
            runtime_facts=authority.runtime_facts,
            payload_source=authority,
        )
        assembled[mode] = assemble_sd(
            sd1_plan(standard, diffusion=plan), diffusion_dtype=torch.float32
        ).diffusion.eval()

    swapped = {
        name
        for name, module in assembled["memory"].named_modules()
        if isinstance(module, GgufEncodedLinear)
    }
    eligible = {
        name
        for name, module in assembled["speed"].named_modules()
        if isinstance(module, torch.nn.Linear) and module.weight.numel() % 32 == 0
    }
    assert swapped == eligible
    assert swapped
    assert not any(isinstance(module, GgufEncodedLinear) for module in assembled["speed"].modules())
    for name in swapped:
        blocks = assembled["memory"].get_submodule(name).get_buffer("weight_blocks")
        assert blocks.dtype == torch.uint8

    balanced_swapped = {
        name
        for name, module in assembled["balanced"].named_modules()
        if isinstance(module, GgufEncodedLinear)
    }
    assert balanced_swapped == swapped
    caches = {
        module.decoded_cache
        for module in assembled["balanced"].modules()
        if isinstance(module, GgufEncodedLinear)
    }
    assert len(caches) == 1
    (cache,) = caches
    assert cache is not None
    assert cache.used_bytes == 0

    x = hashed_input("gguf-residency:x", (1, 4, 8, 8))
    timesteps = torch.tensor([7.0], dtype=torch.float32)
    context = hashed_input("gguf-residency:context", (1, 5, 64))
    expected = assembled["speed"](x, timesteps, context=context)
    assert torch.equal(assembled["memory"](x, timesteps, context=context), expected)
    assert torch.equal(assembled["balanced"](x, timesteps, context=context), expected)
    assert 0 < cache.used_bytes <= 1 << 20


def test_sd15_inpaint_assembly_strict_loads_and_forwards(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    config = replace(TINY_SD1_UNET, in_channels=9, context_dim=16)
    state = unet_state(config)
    path = write_checkpoint(tmp_path / "sd15-inpaint.safetensors", state)
    diffusion = component_plan("diffusion", path, config, state)
    assembled = assemble_sd(sd1_plan(standard, diffusion=diffusion), diffusion_dtype=torch.float32)
    assert tuple(assembled.diffusion.named_buffers()) == ()
    case = UNET_GOLDENS["cases"]["sd15_inpaint"]
    x = hashed_input("sd15_inpaint:x", (1, 9, 8, 8))
    timesteps = torch.tensor(case["timesteps"], dtype=torch.float32)
    context = hashed_input("sd15_inpaint:context", (1, 5, 16))
    expected = torch.tensor(case["output"]["data"], dtype=torch.float32).reshape(
        case["output"]["shape"]
    )
    torch.testing.assert_close(
        assembled.diffusion(x, timesteps, context=context),
        expected,
        rtol=1e-4,
        atol=1e-5,
    )


def test_sd15_inpaint_wider_storage_rounds_to_compute_dtype(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    """An f32 checkpoint under f16 compute loads as f16 parameters equal to
    the f16 roundtrip, with no cast-at-use layers: storage wider than the
    compute dtype rounds down at load, exactly as the reference loads it
    into model-dtype parameters."""
    config = replace(TINY_SD1_UNET, in_channels=9, context_dim=16)
    state = unet_state(config)
    path = write_checkpoint(tmp_path / "sd15-inpaint-fp16.safetensors", state)
    diffusion = component_plan("diffusion", path, config, state)
    model = assemble_sd(
        sd1_plan(standard, diffusion=diffusion), diffusion_dtype=torch.float16
    ).diffusion.eval()
    cast_conv_type = type(CastOperations(torch.float16).conv2d(1, 1, 1))
    assert not any(isinstance(module, cast_conv_type) for module in model.modules())
    assert not any(hasattr(module, "_compute_dtype") for module in model.modules())
    assert tuple(model.named_buffers()) == ()

    loaded = model.state_dict()
    for key, tensor in state.items():
        assert loaded[key].dtype is torch.float16, key
        expected = tensor.to(torch.float16)
        assert torch.equal(loaded[key].view(torch.uint8), expected.view(torch.uint8)), key

    case = UNET_GOLDENS["cases"]["sd15_inpaint"]
    x = hashed_input("sd15_inpaint:x", (1, 9, 8, 8)).to(torch.float16)
    timesteps = torch.tensor(case["timesteps"], dtype=torch.float32)
    context = hashed_input("sd15_inpaint:context", (1, 5, 16)).to(torch.float16)
    with torch.inference_mode():
        output = model(x, timesteps, context=context)
    assert output.dtype is torch.float16


def test_module_dtype_conversion_preserves_executed_integer_buffer() -> None:
    class BufferedConv(torch.nn.Module):
        sentinel: torch.Tensor

        def __init__(self) -> None:
            super().__init__()
            self.conv = torch.nn.Conv2d(2, 3, 3, padding=1)
            self.register_buffer("sentinel", torch.tensor([7], dtype=torch.int64))

        def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return self.conv(value), self.sentinel + 1

    stored_fp16 = BufferedConv().to(dtype=torch.float16)
    value = torch.randn(1, 2, 6, 6, dtype=torch.float16)
    _, executed_sentinel = stored_fp16(value)

    assert stored_fp16.conv.weight.dtype is torch.float16
    assert stored_fp16.sentinel.dtype is torch.int64
    assert executed_sentinel.item() == 8


@pytest.mark.parametrize("target", [torch.float16, torch.bfloat16])
def test_sd15_inpaint_storage_dtype_policy_is_value_neutral(
    standard: dict[str, Any], tmp_path: Path, target: torch.dtype
) -> None:
    config = replace(TINY_SD1_UNET, in_channels=9, context_dim=16)
    state = unet_state(config)
    path = write_checkpoint(tmp_path / f"sd15-inpaint-{target}.safetensors", state)
    diffusion = component_plan("diffusion", path, config, state)
    plan = sd1_plan(standard, diffusion=diffusion)
    default = assemble_sd(
        plan,
        diffusion_dtype=target,
        text_dtype=target,
        vae_dtype=target,
    )
    policy = replace(
        assemble_sd(
            plan,
            diffusion_dtype=target,
            text_dtype=target,
            vae_dtype=target,
        ),
        _storage_dtype_follows_compute=True,
    )
    components = {
        "diffusion": policy.diffusion,
        "clip_l": policy.clip_l,
        "vae": policy.vae,
    }
    expected = {
        f"{component}.{name}": parameter.detach().to(target).clone()
        for component, module in components.items()
        if module is not None
        for name, parameter in module.named_parameters()
    }
    assert expected

    default_residency = enroll_assembled(
        default,
        load_device="cpu",
        offload_device="cpu",
    )
    policy_residency = enroll_assembled(
        policy,
        load_device="cpu",
        offload_device="cpu",
    )
    # Wider-than-compute storage already rounds to the compute dtype at
    # load, so the follow-compute policy finds nothing left to convert.
    assert dict(policy_residency.storage_dtype_report.outcomes) == {
        "diffusion": "already_at_target",
        "clip_l": "already_at_target",
        "vae": "already_at_target",
    }
    for mechanism in default_residency.values():
        mechanism.partially_load(None)
    for mechanism in policy_residency.values():
        mechanism.partially_load(None)

    actual = {
        f"{component}.{name}": parameter
        for component, module in components.items()
        if module is not None
        for name, parameter in module.named_parameters()
    }
    assert actual.keys() == expected.keys()
    for name, parameter in actual.items():
        assert parameter.dtype is target
        assert torch.equal(parameter.view(torch.uint8), expected[name].view(torch.uint8)), name

    case = UNET_GOLDENS["cases"]["sd15_inpaint"]
    x = hashed_input("sd15_inpaint:x", (1, 9, 8, 8)).to(target)
    timesteps = torch.tensor(case["timesteps"], dtype=torch.float32)
    context = hashed_input("sd15_inpaint:context", (1, 5, 16)).to(target)
    with torch.inference_mode():
        default_output = default.diffusion(x, timesteps, context=context)
        policy_output = policy.diffusion(x, timesteps, context=context)
    assert torch.equal(policy_output, default_output)


def test_sdxl_inpaint_assembly_strict_loads_and_forwards_with_adm(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    state = unet_state(TINY_SDXL_INPAINT_UNET)
    path = write_checkpoint(tmp_path / "sdxl-inpaint.safetensors", state)
    diffusion = component_plan("diffusion", path, TINY_SDXL_INPAINT_UNET, state)
    assembled = assemble_sd(sdxl_plan(standard, diffusion=diffusion), diffusion_dtype=torch.float32)
    case = UNET_GOLDENS["cases"]["xl_wide_inpaint"]
    x = hashed_input("xl_wide_inpaint:x", (1, 9, 8, 8))
    timesteps = torch.tensor(case["timesteps"], dtype=torch.float32)
    context = hashed_input(
        "xl_wide_inpaint:context", (1, 5, TINY_SDXL_INPAINT_UNET.context_dim or 1)
    )
    adm_channels = TINY_SDXL_INPAINT_UNET.adm_in_channels
    assert adm_channels is not None
    y = hashed_input("xl_wide_inpaint:y", (1, adm_channels))
    expected = torch.tensor(case["output"]["data"], dtype=torch.float32).reshape(
        case["output"]["shape"]
    )
    torch.testing.assert_close(
        assembled.diffusion(x, timesteps, context=context, y=y),
        expected,
        rtol=1e-4,
        atol=1e-5,
    )


def test_sdxl_assembly_wires_both_towers(standard: dict[str, Any]) -> None:
    assembled = assemble_sd(sdxl_plan(standard), diffusion_dtype=torch.float32)
    assert assembled.family is SDXL
    assert assembled.clip_l is not None and assembled.clip_g is not None
    forward_diffusion(assembled, TINY_SDXL_UNET)
    embeds = assembled.clip_g.embed_tokens(torch.tensor([[1, 2, 95]]))
    assert assembled.clip_g(embeds, torch.tensor([2])).pooled.shape == (1, 48)


@pytest.mark.parametrize("family_id", ("dinkster.sdxl", "extension.custom_sdxl"))
def test_sdxl_assembly_binds_planned_base_provenance(
    monkeypatch: pytest.MonkeyPatch, family_id: str
) -> None:
    import dinkster_inference_torch.assemble as assembly_module

    digest = "blake3:" + "a" * 64
    plan = plan_sd_assembly(checkpoint=source(sdxl_combined_geometries(), "sdxl.safetensors"))
    plan = replace(plan, family=replace(plan.family, id=family_id), diffusion_asset_digest=digest)

    def load_component(component: Any, factory: Any, **_kwargs: Any) -> Any:
        with torch.device("meta"):
            return factory(component.config)

    monkeypatch.setattr(assembly_module, "_load_component", load_component)
    assembled = assemble_sd(plan)
    _validate_sdxl_base_resource(assembled.diffusion, digest)
    with pytest.raises(ControlResourceBindingError):
        _validate_sdxl_base_resource(assembled.diffusion, "blake3:" + "b" * 64)
    with pytest.raises(AssembleError, match="does not match the planned source"):
        assemble_sd(plan, diffusion_asset_digest="blake3:" + "b" * 64)


def test_diffusers_vae_assembly_strict_loads_canonical_module(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    """The planned rank-2 attention payloads reshape at load and the
    resulting module state is byte-identical to canonical storage."""
    canonical_model = build_model("standard")
    canonical_state = canonical_model.state_dict()
    canonical_geometries = {
        key: TensorGeometry(tuple(tensor.shape), FLOAT32) for key, tensor in canonical_state.items()
    }
    diffusers_geometry = diffusers_geometries(canonical_geometries)
    diffusers_state = {
        source_key: canonical_state[diffusers_kl_key(source_key)].reshape(geometry.shape)
        for source_key, geometry in diffusers_geometry.items()
    }
    path = write_checkpoint(tmp_path / "diffusers-vae.safetensors", diffusers_state)
    planned = plan_sd_assembly(
        diffusion=source(unet_geometries(), "unet.safetensors"),
        clip_l=source(clip_l_sd_geometries(), "clip-l.safetensors"),
        vae=load_safetensors_header(path),
    )
    vae_plan = planned.vae
    assert isinstance(vae_plan, ComponentPlan)
    assert vae_plan.config == canonical_model.config
    assert vae_plan.keys == {
        diffusers_kl_key(source_key): source_key for source_key in diffusers_state
    }
    assert vae_plan.transforms == {
        model_key: LinearToConv2D()
        for model_key, source_key in vae_plan.keys.items()
        if diffusers_state[source_key].ndim == 2
        and model_key.endswith((".q.weight", ".k.weight", ".v.weight", ".proj_out.weight"))
    }
    assembled = assemble_sd(sd1_plan(standard, vae=vae_plan), diffusion_dtype=torch.float32)
    actual = assembled.vae.state_dict()
    assert set(actual) == set(canonical_state)
    for key, expected in canonical_state.items():
        assert torch.equal(actual[key], expected), key
    content = torch.linspace(-1.0, 1.0, 3 * 16 * 16).reshape(1, 3, 16, 16)
    expected_latent = canonical_model.encode(content)
    actual_latent = assembled.vae.encode(content)
    assert torch.equal(actual_latent, expected_latent)
    assert torch.equal(
        assembled.vae.decode(actual_latent),
        canonical_model.decode(expected_latent),
    )


def test_taesd_assembly_strictly_loads_both_planned_halves(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    encoder_state = component_state(TAESDEncoder())
    decoder_state = component_state(TAESDDecoder())
    encoder_payload = dict(encoder_state)
    encoder_payload.update(
        {
            "first_stage_model.vae_scale": torch.tensor(0.18215),
            "first_stage_model.vae_shift": torch.tensor(0.0),
        }
    )
    encoder_path = write_checkpoint(tmp_path / "encoder.safetensors", encoder_payload)
    decoder_path = write_checkpoint(tmp_path / "decoder.safetensors", decoder_state)
    config = TAESDConfig("sd15", "encoder")
    codec = TAESDCodecPlan(
        config,
        component_plan("taesd_encoder", encoder_path, config, encoder_state),
        component_plan(
            "taesd_decoder",
            decoder_path,
            TAESDConfig("sd15", "decoder"),
            decoder_state,
        ),
        scale_keys=(
            "first_stage_model.vae_scale",
            "first_stage_model.vae_shift",
        ),
    )
    assembled = assemble_sd(sd1_plan(standard, vae=codec), diffusion_dtype=torch.float32)
    assert isinstance(assembled.vae, TAESD)
    assert assembled.vae.config.family == "sd15"
    assert assembled.vae.encode(torch.zeros(1, 3, 8, 8)).shape == (1, 4, 1, 1)


def test_sd_plan_refuses_taesd_from_the_wrong_family(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    encoder_state = component_state(TAESDEncoder())
    decoder_state = component_state(TAESDDecoder())
    encoder_path = write_checkpoint(tmp_path / "wrong-encoder.safetensors", encoder_state)
    decoder_path = write_checkpoint(tmp_path / "wrong-decoder.safetensors", decoder_state)
    config = TAESDConfig("sdxl", "encoder")
    codec = TAESDCodecPlan(
        config,
        component_plan("taesd_encoder", encoder_path, config, encoder_state),
        component_plan(
            "taesd_decoder",
            decoder_path,
            TAESDConfig("sdxl", "decoder"),
            decoder_state,
        ),
    )
    with pytest.raises(ValueError, match="requires sd15 TAESD"):
        sd1_plan(standard, vae=codec)


def test_refiner_assembly_wires_no_clip_l(standard: dict[str, Any]) -> None:
    assembled = assemble_sd(
        SDAssemblyPlan(
            family=SDXL_REFINER,
            diffusion=standard["diffusion_refiner"][0],
            clip_l=None,
            clip_g=standard["clip_g"][0],
            vae=standard["vae"][0],
        ),
        diffusion_dtype=torch.float32,
    )
    assert assembled.family is SDXL_REFINER
    assert assembled.clip_l is None
    assert assembled.clip_g is not None
    forward_diffusion(assembled, TINY_REFINER_UNET)


# ------------------------------------------- the OpenCLIP transforms


ATTN = "text_model.encoder.layers.0.self_attn"
PROJECTION = "text_projection.weight"


def openclip_clip_g_plan(tmp_path: Path, state: dict[str, torch.Tensor]) -> ComponentPlan[Any]:
    """A CLIP-G plan whose layer-0 q/k/v derive from one fused
    ``in_proj_weight`` and whose text_projection is stored in the
    OpenCLIP ``x @ W`` layout - the shapes transformers_convert
    normalizes @ 947c2749."""
    source = dict(state)
    fused = torch.cat([source.pop(f"{ATTN}.{name}_proj.weight") for name in "qkv"], dim=0)
    source["in_proj_weight"] = fused
    source[PROJECTION] = source[PROJECTION].transpose(0, 1).contiguous()
    path = write_checkpoint(tmp_path / "openclip_g.safetensors", source)
    keys = {key: key for key in state}
    for name in "qkv":
        keys[f"{ATTN}.{name}_proj.weight"] = "in_proj_weight"
    return ComponentPlan(
        component="clip_g",
        path=path,
        config=TINY_CLIP_G,
        keys=keys,
        dtypes=dict.fromkeys(keys, FLOAT32),
        quant={},
        transforms={
            f"{ATTN}.q_proj.weight": RowChunk(part=0, parts=3),
            f"{ATTN}.k_proj.weight": RowChunk(part=1, parts=3),
            f"{ATTN}.v_proj.weight": RowChunk(part=2, parts=3),
            PROJECTION: Transpose2D(),
        },
    )


def test_openclip_transforms_derive_the_planned_tensors(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    """The fused-source assembly must produce bit-identical modules to
    the already-split source: RowChunk picks each q/k/v slice,
    Transpose2D restores the Linear layout."""
    state = standard["clip_g"][1]
    plan = sdxl_plan(standard, clip_g=openclip_clip_g_plan(tmp_path, state))
    assembled = assemble_sd(plan, diffusion_dtype=torch.float32)
    plain = assemble_sd(sdxl_plan(standard), diffusion_dtype=torch.float32)
    assert assembled.clip_g is not None and plain.clip_g is not None
    actual = assembled.clip_g.state_dict()
    expected = plain.clip_g.state_dict()
    assert set(actual) == set(expected)
    for key in expected:
        assert torch.equal(actual[key], expected[key]), key


def test_row_chunk_refuses_an_uneven_split(standard: dict[str, Any], tmp_path: Path) -> None:
    state = dict(standard["clip_g"][1])
    key = f"{ATTN}.q_proj.weight"
    source = dict(state)
    # 3 does not divide 48+1 rows
    source["in_proj_weight"] = torch.randn(49, TINY_CLIP_G.hidden_size)
    del source[key]
    path = write_checkpoint(tmp_path / "uneven.safetensors", source)
    keys = {k: k for k in state}
    keys[key] = "in_proj_weight"
    plan = sdxl_plan(
        standard,
        clip_g=ComponentPlan(
            component="clip_g",
            path=path,
            config=TINY_CLIP_G,
            keys=keys,
            dtypes=dict.fromkeys(keys, FLOAT32),
            quant={},
            transforms={key: RowChunk(part=0, parts=3)},
        ),
    )
    with pytest.raises(AssembleError, match="does not split"):
        assemble_sd(plan, diffusion_dtype=torch.float32)


def test_transpose_refuses_a_non_matrix(standard: dict[str, Any], tmp_path: Path) -> None:
    state = dict(standard["clip_g"][1])
    source = dict(state)
    source[PROJECTION] = torch.randn(TINY_CLIP_G.hidden_size)
    path = write_checkpoint(tmp_path / "flat.safetensors", source)
    keys = {k: k for k in state}
    plan = sdxl_plan(
        standard,
        clip_g=ComponentPlan(
            component="clip_g",
            path=path,
            config=TINY_CLIP_G,
            keys=keys,
            dtypes=dict.fromkeys(keys, FLOAT32),
            quant={},
            transforms={PROJECTION: Transpose2D()},
        ),
    )
    with pytest.raises(AssembleError, match="transpose source has rank"):
        assemble_sd(plan, diffusion_dtype=torch.float32)
