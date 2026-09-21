"""Executing classic-Flux assembly plans: planned slices -> modules.

The torch-free planner is proven in tests/test_inference_assembly.py;
these tests cover the executing half only. Tiny hand-built plans over
real safetensors payloads (written by the same byte layout the header
parser reads) prove: per-parameter storage dtype preservation, the
INITLESS/CastOperations decision, plain-fp8 cast-at-use, scaled-fp8
module swapping in both spellings, default fills, and every documented
refusal. Detection + full-size assembly against installed checkpoints
lives in the capability-gated GPU suite.

Run with the torch venv: .venv-torch/bin/python -m pytest -q
packages/dinkster-inference-torch/tests
"""

from __future__ import annotations

import json
import struct
from collections.abc import Callable, Iterable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from clip_fill import fill_value
from dinkster_inference import (
    BFLOAT16,
    FLOAT8_E4M3,
    FLOAT8_E5M2,
    FLOAT16,
    FLOAT32,
    FLUX2_DEV,
    FLUX_DEV,
    FLUX_SCHNELL,
    INT8,
    Q4_0,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_0,
    QWEN_IMAGE,
    QWEN_IMAGE_CONFIG,
    QWEN_IMAGE_EDIT_2511_CONFIG,
    QWEN_IMAGE_LAYERED_CONFIG,
    UINT8,
    UMT5_XXL_CONFIG,
    WAN21,
    WAN21_CAUSAL_AR_1_3B,
    WAN21_CLIP_VISION,
    WAN21_FLOW_RVS_1_3B,
    WAN21_FLOW_RVS_VAE_CONFIG,
    WAN21_I2V_14B,
    WAN21_SCAIL2_14B,
    WAN21_SCAIL_14B,
    WAN21_T2V_1_3B,
    WAN21_VAE_CONFIG,
    Z_IMAGE,
    AttentionRoute,
    ClipTextConfig,
    ComponentPlan,
    DType,
    FluxAssemblyPlan,
    FluxConfig,
    GGMLType,
    GGUFComponentMap,
    GGUFComponentTensor,
    GGUFResidencyMode,
    GGUFSource,
    GGUFValueType,
    KLConfig,
    LayerQuant,
    QwenTextConfig,
    RowChunk,
    T5Config,
    Transpose2D,
    Wan21AssemblyPlan,
    load_gguf_weight_source,
)
from dinkster_inference_torch import (
    ATTENTION_ADAPTER_CONTRACT,
    INITLESS,
    AssembledFlux,
    AssembledFlux2,
    AssembledQwenImage,
    AssembledWan21,
    AssembleError,
    AttentionPolicy,
    AttentionSelectionError,
    AutoencoderKL,
    ClipTextModel,
    Flux,
    FluxRuntime,
    Fp8Linear,
    GgufEncodedLinear,
    Int8Embedding,
    Int8Linear,
    QwenTextModel,
    T5TextModel,
    assemble_flux,
    assemble_flux2,
    assemble_qwen_image,
    assemble_wan21,
    assemble_z_image,
    discover_attention_route_token,
)
from dinkster_inference_torch import assemble as assemble_mod
from dinkster_inference_torch import attention as attention_mod
from dinkster_inference_torch import quant_linear as quant_linear_mod
from dinkster_inference_torch._nvfp4_diagnostics import (
    Nvfp4DiagnosticsRecorder,
    nvfp4_runtime_status,
)
from dinkster_inference_torch.operations import (
    CastOperations,
    _CastMixin,  # pyright: ignore[reportPrivateUsage]
)
from dinkster_inference_torch.quant_linear import Nvfp4Linear
from dinkster_inference_torch.sources import tensor_file_slice
from dinkster_inference_torch.wan21_causal import Wan21CausalModel
from dinkster_inference_torch.wan21_scail import WanScailModel
from dinkster_inference_torch.wan21_vae import WanVAE
from test_gguf_linear import encode_q8_0, reference_quant_blocks

from tests.test_inference_gguf import (  # pyright: ignore[reportMissingImports]
    _llama_t5_name,
    _metadata,
    _string,
)

E4M3 = torch.float8_e4m3fn

TINY_FLUX = FluxConfig(
    in_channels=16,
    out_channels=16,
    vec_in_dim=12,
    context_in_dim=24,
    hidden_size=32,
    depth=1,
    depth_single_blocks=1,
    num_heads=2,
    axes_dim=(4, 6, 6),
    guidance_embed=True,
)
TINY_CLIP = ClipTextConfig(
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=128,
    hidden_act="quick_gelu",
    vocab_size=96,
    eos_token_id=95,
)
TINY_T5 = T5Config(
    d_model=48,
    d_ff=96,
    d_kv=12,
    num_heads=4,
    num_layers=2,
    vocab_size=128,
    dense_act_fn="gelu_pytorch_tanh",
    is_gated_act=True,
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
TINY_QWEN = QwenTextConfig(
    architecture="ovis_qwen3_2b",
    vocab_size=64,
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=32,
    rms_norm_eps=1e-6,
    rope_theta=1_000_000.0,
    qkv_bias=False,
    qk_norm=True,
    prompt_template="{}",
    min_tokens=1,
    pad_token_id=0,
    slice_marker_id=1,
    slice_marker_suffix_id=2,
    zero_masked=True,
)
TINY_VECTOR_FREE_FLUX = replace(
    TINY_FLUX,
    vec_in_dim=None,
    context_in_dim=TINY_QWEN.hidden_size,
    guidance_embed=False,
    txt_norm=True,
    yak_mlp=True,
    txt_ids_dims=(1, 2),
)

SAFETENSORS_NAMES = {
    torch.float32: "F32",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float8_e4m3fn: "F8_E4M3",
    torch.float8_e5m2: "F8_E5M2",
    torch.int8: "I8",
    torch.uint8: "U8",
}
DINKSTER_DTYPES: dict[torch.dtype, DType] = {
    torch.float32: FLOAT32,
    torch.float16: FLOAT16,
    torch.bfloat16: BFLOAT16,
    torch.float8_e4m3fn: FLOAT8_E4M3,
    torch.float8_e5m2: FLOAT8_E5M2,
    torch.int8: INT8,
    torch.uint8: UINT8,
}


def write_checkpoint(path: Path, tensors: dict[str, torch.Tensor]) -> Path:
    header: dict[str, Any] = {}
    payload = bytearray()
    for key, tensor in tensors.items():
        flat = tensor.detach().contiguous().reshape(-1)
        raw = b"" if flat.numel() == 0 else bytes(flat.view(torch.uint8).tolist())
        header[key] = {
            "dtype": SAFETENSORS_NAMES[tensor.dtype],
            "shape": list(tensor.shape),
            "data_offsets": [len(payload), len(payload) + len(raw)],
        }
        payload.extend(raw)
    raw_header = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + bytes(payload))
    return path


def component_state(
    module: torch.nn.Module, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    """A deterministic full state dict for one tiny module (the shared
    key-name hash, so values are stable across runs)."""
    return {
        key: fill_value(key, tuple(tensor.shape)).to(dtype)
        for key, tensor in module.state_dict().items()
    }


def component_plan(
    component: str,
    path: Path,
    config: object,
    state: dict[str, torch.Tensor],
    *,
    prefix: str = "",
    quant: dict[str, LayerQuant] | None = None,
    absent: tuple[str, ...] = (),
) -> ComponentPlan[Any]:
    keys = {key: prefix + key for key in state}
    dtypes = {key: DINKSTER_DTYPES[state[key].dtype] for key in state}
    return ComponentPlan(
        component=component,
        path=path,
        config=config,
        keys=keys,
        dtypes=dtypes,
        quant=quant or {},
        absent=absent,
    )


def quantize_into(
    state: dict[str, torch.Tensor],
    layer: str,
    *,
    fp8_dtype: torch.dtype = E4M3,
) -> torch.Tensor:
    """Scale-quantize one layer's weight in place; returns the 0-dim
    float32 scale the checkpoint should carry."""
    weight = state[f"{layer}.weight"]
    scale = (weight.abs().amax() / torch.finfo(fp8_dtype).max).to(torch.float32)
    state[f"{layer}.weight"] = (weight / scale).to(fp8_dtype)
    return scale.reshape(())


def quant_json(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(list(json.dumps(payload).encode()), dtype=torch.uint8)


@pytest.fixture(scope="module")
def standard(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One ordinary fp32 split-checkpoint set, written once: each
    entry carries (plan, state) for reuse and for swapping single
    components out in error tests."""
    tmp = tmp_path_factory.mktemp("tiny-flux")
    built: dict[str, Any] = {}
    components = (
        ("diffusion", TINY_FLUX, Flux),
        ("clip_l", TINY_CLIP, ClipTextModel),
        ("t5xxl", TINY_T5, T5TextModel),
        ("vae", TINY_KL, AutoencoderKL),
    )
    for name, config, factory in components:
        # cast: pyright narrows the heterogeneous tuple literal (and
        # any re-annotation), unioning the four __init__ signatures.
        build = cast("Callable[..., torch.nn.Module]", factory)
        state = component_state(build(config))
        path = write_checkpoint(tmp / f"{name}.safetensors", state)
        built[name] = (
            component_plan(name, path, config, state),
            state,
        )
    return built


def make_plan(
    standard: dict[str, Any],
    **overrides: ComponentPlan[Any],
) -> FluxAssemblyPlan:
    plans = {name: plan for name, (plan, _) in standard.items()}
    plans.update(overrides)
    return FluxAssemblyPlan(
        family=FLUX_DEV,
        diffusion=plans["diffusion"],
        clip_l=plans["clip_l"],
        t5xxl=plans["t5xxl"],
        vae=plans["vae"],
    )


def forward_all(assembled: AssembledFlux) -> None:
    """A bounded forward through every component (shapes only; the
    per-model numerics are golden-pinned in their own suites)."""
    compute = assembled.diffusion.img_in.weight.dtype
    if isinstance(assembled.diffusion.img_in, Fp8Linear) or isinstance(
        assembled.diffusion.img_in, _CastMixin
    ):
        compute = torch.float32
    out = assembled.diffusion(
        torch.randn(1, 16, 4, 4).to(compute),
        torch.tensor([0.5]).to(compute),
        torch.randn(1, 6, 24).to(compute),
        torch.randn(1, 12).to(compute),
        torch.tensor([3.5]).to(compute),
    )
    assert out.shape == (1, 16, 4, 4)
    assert assembled.clip_l is not None
    assert assembled.t5xxl is not None
    clip_embeds = assembled.clip_l.embed_tokens(torch.tensor([[1, 2, 95]]))
    clip_out = assembled.clip_l(clip_embeds, torch.tensor([2]))
    assert clip_out.pooled.shape == (1, 64)
    t5_out = assembled.t5xxl(assembled.t5xxl.embed_tokens(torch.tensor([[3, 1]])))
    assert t5_out.shape == (1, 2, 48)
    latent = assembled.vae.encode(torch.randn(1, 3, 16, 16))
    assert assembled.vae.decode(latent).shape == (1, 3, 16, 16)


# ------------------------------------------------------- happy paths


def test_split_assembly_all_fp32(standard: dict[str, Any]) -> None:
    assembled = assemble_flux(make_plan(standard), diffusion_dtype=torch.float32)
    assert assembled.clip_l is not None and assembled.t5xxl is not None
    assert assembled.family is FLUX_DEV
    # Storage matched compute everywhere: initless modules, no casts.
    for module in (
        assembled.diffusion,
        assembled.clip_l,
        assembled.t5xxl,
        assembled.vae,
    ):
        assert not any(isinstance(child, _CastMixin) for child in module.modules())
    # Exact storage: what shipped is what loaded.
    _, state = standard["diffusion"]
    assert torch.equal(
        assembled.diffusion.state_dict()["img_in.weight"],
        state["img_in.weight"],
    )
    forward_all(assembled)


def test_authenticated_six_role_status_is_immutable_and_reused_by_runtime(
    standard: dict[str, Any],
) -> None:
    token = discover_attention_route_token("sdpa")
    assembled = assemble_flux(
        make_plan(standard),
        diffusion_dtype=torch.float32,
        attention_policy="sdpa",
        attention_route_token=token,
    )
    assert tuple(assembled.attention_status) == (
        "unet",
        "flux",
        "vae",
        "clip",
        "t5",
        "qwen",
    )
    assert all(
        status.role == role
        and status.requested_policy == "sdpa"
        and status.primary == "sdpa"
        and status.fallback is None
        and status.authenticated is True
        and status.provider_versions == token.provider_versions
        and status.adapter_contract == token.adapter_contract_revision
        and status.device_kind == token.device_kind
        and status.device_sm == token.device_sm
        and status.sdpa_torch_runtime == token.sdpa_torch_runtime
        for role, status in assembled.attention_status.items()
    )
    with pytest.raises(TypeError):
        assembled.attention_status["flux"] = assembled.attention_status["unet"]  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        assembled.attention_status["flux"].device_kind = "forged"  # type: ignore[misc]
    runtime = FluxRuntime(assembled, runtime_identity="native:test:attention")
    assert runtime.attention_status is assembled.attention_status


def test_role_policy_overrides_split_kitchen_and_sdpa_across_roles(
    standard: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(attention_mod, "_KITCHEN_AVAILABLE", lambda: True)
    overrides: tuple[tuple[str, AttentionPolicy], ...] = (
        ("vae", "sdpa"),
        ("clip", "sdpa"),
        ("t5", "sdpa"),
    )
    token = discover_attention_route_token(
        "dinkster_kitchen_int8", requested_role_policies=overrides
    )
    assert token.version == 2
    assembled = assemble_flux(
        make_plan(standard),
        attention_policy="dinkster_kitchen_int8",
        attention_route_token=token,
    )
    statuses = assembled.attention_status
    for role in ("unet", "flux", "qwen"):
        assert statuses[role].requested_policy == "dinkster_kitchen_int8"
        assert statuses[role].primary == "dinkster_kitchen_int8"
        assert statuses[role].fallback == "sdpa"
    for role in ("vae", "clip", "t5"):
        assert statuses[role].requested_policy == "sdpa"
        assert statuses[role].primary == "sdpa"
        assert statuses[role].fallback is None
    assert all(status.authenticated for status in statuses.values())


def test_forged_attention_evidence_refuses_before_payload_realization(
    standard: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    token = replace(
        discover_attention_route_token("sdpa"),
        provider_versions=(("sentinel", "provider-v9"), ("torch", "sentinel-build")),
        device_kind="sentinel-device",
        device_sm=123,
        sdpa_torch_runtime="sentinel-runtime",
    )

    def unexpected_payload(*args: object, **kwargs: object) -> object:
        raise AssertionError("payload realization must not start")

    monkeypatch.setattr(assemble_mod, "load_tensors", unexpected_payload)
    with pytest.raises(AssembleError, match="rediscovered runtime evidence"):
        assemble_flux(
            make_plan(standard),
            attention_policy="sdpa",
            attention_route_token=token,
        )


def test_attention_contract_mismatch_refuses_before_payload_realization(
    standard: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    token = replace(
        discover_attention_route_token("auto", device_kind="cpu"),
        adapter_contract_revision="forged.contract.v999",
        provider_versions=(("forged-provider", "sentinel-version"),),
        sdpa_torch_runtime="forged-runtime",
    )

    def unexpected_payload(*args: object, **kwargs: object) -> object:
        raise AssertionError("payload realization must not start")

    monkeypatch.setattr(assemble_mod, "load_tensors", unexpected_payload)
    with pytest.raises(AssembleError, match="adapter contract"):
        assemble_flux(make_plan(standard), attention_route_token=token)


def test_attention_route_mismatch_refuses_at_token_construction() -> None:
    token = discover_attention_route_token("auto", device_kind="cpu")
    with pytest.raises(ValueError, match="inconsistent with its effective policy"):
        replace(
            token,
            routes=tuple(
                AttentionRoute(route.role, "flash" if route.role == "flux" else route.primary)
                for route in token.routes
            ),
        )


def test_named_policy_without_token_refuses_before_payload_realization(
    standard: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_payload(*args: object, **kwargs: object) -> object:
        raise AssertionError("payload realization must not start")

    monkeypatch.setattr(assemble_mod, "load_tensors", unexpected_payload)
    with pytest.raises(ValueError, match="requires an authenticated route token"):
        assemble_flux(make_plan(standard), attention_policy="sdpa")


def test_discovery_names_mps_when_cuda_is_absent() -> None:
    fake_torch = SimpleNamespace(
        __version__="2.13.0",
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    token = discover_attention_route_token("auto", torch_module=fake_torch)
    assert token.device_kind == "mps"
    assert token.device_sm is None


def test_discovery_prefers_cuda_over_mps() -> None:
    fake_torch = SimpleNamespace(
        __version__="2.13.0",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda: (12, 0),
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    token = discover_attention_route_token("auto", torch_module=fake_torch)
    assert token.device_kind == "cuda"
    assert token.device_sm == 120


def test_discovery_falls_back_to_cpu_without_accelerators() -> None:
    fake_torch = SimpleNamespace(
        __version__="2.13.0",
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    token = discover_attention_route_token("auto", torch_module=fake_torch)
    assert token.device_kind == "cpu"
    assert token.device_sm is None


def test_discovery_splits_hip_builds_from_nvidia() -> None:
    fake_torch = SimpleNamespace(
        __version__="2.12.0+rocm7.1.4",
        version=SimpleNamespace(hip="7.1.44064"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda: (11, 0),
        ),
    )
    token = discover_attention_route_token("auto", torch_module=fake_torch)
    assert token.device_kind == "rocm"
    assert token.device_sm == 110
    assert token.provider_versions == (("hip", "7.1.44064"), ("torch", "2.12.0+rocm7.1.4"))
    assert token.sdpa_torch_runtime == "2.12.0"


def test_discovery_nvidia_token_shape_is_unchanged() -> None:
    fake_torch = SimpleNamespace(
        __version__="2.13.0+cu130",
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda: (12, 0),
        ),
    )
    token = discover_attention_route_token("auto", torch_module=fake_torch)
    assert token.device_kind == "cuda"
    assert token.device_sm == 120
    assert token.provider_versions == (("torch", "2.13.0+cu130"),)


def test_discovery_names_intel_accelerator_before_mps() -> None:
    fake_torch = SimpleNamespace(
        __version__="2.13.0+xpu",
        version=SimpleNamespace(hip=None, xpu="20260500"),
        cuda=SimpleNamespace(is_available=lambda: False),
        xpu=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    token = discover_attention_route_token("auto", torch_module=fake_torch)
    assert token.device_kind == "xpu"
    assert token.device_sm is None
    assert token.provider_versions == (("torch", "2.13.0+xpu"), ("xpu", "20260500"))


def test_discovery_intel_accelerator_without_runtime_version_evidence() -> None:
    fake_torch = SimpleNamespace(
        __version__="2.13.0+xpu",
        version=SimpleNamespace(hip=None, xpu=None),
        cuda=SimpleNamespace(is_available=lambda: False),
        xpu=SimpleNamespace(is_available=lambda: True),
    )
    token = discover_attention_route_token("auto", torch_module=fake_torch)
    assert token.device_kind == "xpu"
    assert token.provider_versions == (("torch", "2.13.0+xpu"),)


def test_explicit_hip_family_requires_hip_runtime() -> None:
    fake_torch = SimpleNamespace(
        __version__="2.13.0",
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    with pytest.raises(AttentionSelectionError, match="torch.version.hip"):
        discover_attention_route_token("auto", torch_module=fake_torch, device_kind="rocm")


def test_explicit_hip_family_records_hip_evidence() -> None:
    fake_torch = SimpleNamespace(
        __version__="2.12.0+rocm7.1.4",
        version=SimpleNamespace(hip="7.1.44064"),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    token = discover_attention_route_token(
        "auto", torch_module=fake_torch, device_kind="rocm", device_sm=110
    )
    assert token.device_kind == "rocm"
    assert token.device_sm == 110
    assert token.provider_versions == (("hip", "7.1.44064"), ("torch", "2.12.0+rocm7.1.4"))


def test_explicit_nvidia_family_refuses_hip_builds() -> None:
    fake_torch = SimpleNamespace(
        __version__="2.12.0+rocm7.1.4",
        version=SimpleNamespace(hip="7.1.44064"),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    with pytest.raises(AttentionSelectionError, match="ROCm torch build"):
        discover_attention_route_token("auto", torch_module=fake_torch, device_kind="cuda")


def test_explicit_intel_family_refuses_sm_capability() -> None:
    fake_torch = SimpleNamespace(
        __version__="2.13.0+xpu",
        version=SimpleNamespace(hip=None, xpu="20260500"),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    with pytest.raises(AttentionSelectionError, match="no SM capability"):
        discover_attention_route_token(
            "auto", torch_module=fake_torch, device_kind="xpu", device_sm=120
        )


def test_legacy_missing_token_auto_retains_canonical_sdpa_status(
    standard: dict[str, Any],
) -> None:
    assembled = assemble_flux(make_plan(standard), diffusion_dtype=torch.float32)
    assert tuple(assembled.attention_status) == (
        "unet",
        "flux",
        "vae",
        "clip",
        "t5",
        "qwen",
    )
    assert all(
        status.requested_policy == "auto"
        and status.primary == "sdpa"
        and status.fallback is None
        and status.authenticated is False
        and status.provider_versions == ()
        and status.adapter_contract == ATTENTION_ADAPTER_CONTRACT
        and status.device_kind == "unknown"
        and status.device_sm is None
        and status.sdpa_torch_runtime == "unknown"
        for status in assembled.attention_status.values()
    )


def test_qwen_text_component_assembles_strictly(standard: dict[str, Any], tmp_path: Path) -> None:
    diffusion_config = replace(TINY_FLUX, context_in_dim=TINY_QWEN.hidden_size)
    diffusion_state = component_state(Flux(diffusion_config))
    diffusion_path = write_checkpoint(tmp_path / "diffusion-qwen.safetensors", diffusion_state)
    qwen_state = component_state(QwenTextModel(TINY_QWEN))
    qwen_path = write_checkpoint(tmp_path / "ovis-text.safetensors", qwen_state)
    vae_plan, _ = standard["vae"]
    plan = FluxAssemblyPlan(
        family=FLUX_DEV,
        diffusion=component_plan(
            "diffusion",
            diffusion_path,
            diffusion_config,
            diffusion_state,
        ),
        clip_l=None,
        t5xxl=None,
        vae=vae_plan,
        qwen3_2b=component_plan("qwen3_2b", qwen_path, TINY_QWEN, qwen_state),
    )
    assembled = assemble_flux(plan, diffusion_dtype=torch.float32)
    assert assembled.clip_l is None and assembled.t5xxl is None
    assert assembled.qwen3_2b is not None
    assert assembled.qwen3_2b.config == TINY_QWEN
    assert set(assembled.qwen3_2b.state_dict()) == set(qwen_state)
    assert torch.equal(
        assembled.qwen3_2b.state_dict()["layers.1.mlp.down_proj.weight"],
        qwen_state["layers.1.mlp.down_proj.weight"],
    )


def test_z_image_assembly_routes_three_exact_component_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Component:
        def __init__(self, name: str) -> None:
            self.name = name

        def without_payload_source(self) -> Component:
            return self

    class Loaded:
        pass

    components = {name: Component(name) for name in ("diffusion", "qwen3_4b", "vae")}
    plan = type("Plan", (), {"family": Z_IMAGE, **components})()
    calls: list[tuple[str, str, torch.dtype]] = []

    def load(
        component: Component, build: Any, *, compute_dtype: torch.dtype, **_kwargs: Any
    ) -> Loaded:
        calls.append((component.name, build.func.__name__, compute_dtype))
        return Loaded()

    monkeypatch.setattr(assemble_mod, "_load_component", load)
    assembled = assemble_z_image(cast("Any", plan))
    assert assembled.family is Z_IMAGE
    assert calls == [
        ("diffusion", "ZImage", torch.bfloat16),
        ("qwen3_4b", "QwenTextModel", torch.float32),
        ("vae", "AutoencoderKL", torch.float32),
    ]
    assert assembled.compute_dtype("diffusion") == torch.bfloat16


def test_flux2_assembly_routes_three_exact_component_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Component:
        def __init__(self, name: str) -> None:
            self.name = name

        def without_payload_source(self) -> Component:
            return self

    class Loaded:
        def __init__(self, name: str) -> None:
            self.name = name

    components = {name: Component(name) for name in ("diffusion", "text_encoder", "vae")}
    plan = type("Plan", (), {"family": FLUX2_DEV, **components})()
    calls: list[tuple[str, str, torch.dtype]] = []

    def load(
        component: Component, build: Any, *, compute_dtype: torch.dtype, **_kwargs: Any
    ) -> Loaded:
        calls.append((component.name, build.func.__name__, compute_dtype))
        return Loaded(component.name)

    monkeypatch.setattr(assemble_mod, "_load_component", load)
    assembled = assemble_flux2(cast("Any", plan))
    assert isinstance(assembled, AssembledFlux2)
    assert assembled.family is FLUX2_DEV
    assert assembled.diffusion.name == "diffusion"  # type: ignore[attr-defined]
    assert assembled.text_encoder.name == "text_encoder"  # type: ignore[attr-defined]
    assert assembled.vae.name == "vae"  # type: ignore[attr-defined]
    assert calls == [
        ("diffusion", "Flux", torch.bfloat16),
        ("text_encoder", "QwenTextModel", torch.float32),
        ("vae", "AutoencoderKL", torch.float32),
    ]
    assert assembled.compute_dtype("diffusion") == torch.bfloat16
    assert assembled.compute_dtype("text_encoder") == torch.float32
    assert assembled.compute_dtype("vae") == torch.float32


@pytest.mark.parametrize(
    "variant",
    (QWEN_IMAGE_CONFIG, QWEN_IMAGE_EDIT_2511_CONFIG, QWEN_IMAGE_LAYERED_CONFIG),
)
def test_qwen_image_assembly_routes_each_variant_and_shared_components(
    monkeypatch: pytest.MonkeyPatch,
    variant: object,
) -> None:
    class Component:
        def __init__(self, name: str, config: object) -> None:
            self.component = name
            self.config = config

        def without_payload_source(self) -> Component:
            return self

    class Loaded:
        def __init__(self, name: str) -> None:
            self.name = name

    components = {
        "diffusion": Component("diffusion", variant),
        "qwen2_5_vl_7b": Component("qwen2_5_vl_7b", object()),
        "vae": Component("vae", object()),
    }
    plan = type("Plan", (), {"family": QWEN_IMAGE, **components})()
    calls: list[tuple[str, torch.dtype]] = []

    def load(
        component: Component, _build: object, *, compute_dtype: torch.dtype, **_kwargs: object
    ) -> Loaded:
        calls.append((component.component, compute_dtype))
        return Loaded(component.component)

    monkeypatch.setattr(assemble_mod, "_load_component", load)
    assembled = assemble_qwen_image(
        cast("Any", plan),
        diffusion_dtype=torch.float32,
        text_dtype=torch.bfloat16,
        vae_dtype=torch.float32,
    )

    assert isinstance(assembled, AssembledQwenImage)
    assert assembled.family is QWEN_IMAGE
    assert assembled.diffusion.name == "diffusion"  # type: ignore[attr-defined]
    assert assembled.text.name == "qwen2_5_vl_7b"  # type: ignore[attr-defined]
    assert assembled.vae.name == "vae"  # type: ignore[attr-defined]
    assert calls == [
        ("diffusion", torch.float32),
        ("qwen2_5_vl_7b", torch.bfloat16),
        ("vae", torch.float32),
    ]
    assert set(assembled._component_compute_dtypes) == {  # pyright: ignore[reportPrivateUsage]
        "diffusion",
        "text",
        "vae",
    }
    assert assembled.compute_dtype("diffusion") is torch.float32
    assert assembled.compute_dtype("text") is torch.bfloat16
    assert assembled.compute_dtype("vae") is torch.float32


def test_vector_free_ovis_diffusion_and_qwen_assemble_strictly(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    diffusion_state = component_state(Flux(TINY_VECTOR_FREE_FLUX), torch.bfloat16)
    diffusion_path = write_checkpoint(tmp_path / "vector-free-ovis.safetensors", diffusion_state)
    qwen_state = component_state(QwenTextModel(TINY_QWEN))
    qwen_path = write_checkpoint(tmp_path / "ovis-text.safetensors", qwen_state)
    vae_plan, _ = standard["vae"]
    plan = FluxAssemblyPlan(
        family=FLUX_SCHNELL,
        diffusion=component_plan(
            "diffusion",
            diffusion_path,
            TINY_VECTOR_FREE_FLUX,
            diffusion_state,
        ),
        clip_l=None,
        t5xxl=None,
        vae=vae_plan,
        qwen3_2b=component_plan("qwen3_2b", qwen_path, TINY_QWEN, qwen_state),
    )
    assembled = assemble_flux(plan, diffusion_dtype=torch.bfloat16)
    assert assembled.family is FLUX_SCHNELL
    assert assembled.diffusion.config == TINY_VECTOR_FREE_FLUX
    assert assembled.diffusion.vector_in is None
    assert assembled.diffusion.guidance_in is None
    assert set(assembled.diffusion.state_dict()) == set(diffusion_state)
    assert not any(
        key.startswith(("vector_in.", "guidance_in.")) for key in assembled.diffusion.state_dict()
    )
    for key, expected in diffusion_state.items():
        assert assembled.diffusion.state_dict()[key].dtype == expected.dtype
        assert torch.equal(assembled.diffusion.state_dict()[key], expected)
    with pytest.raises(ValueError, match="vector-free Ovis.*qwen3_2b"):
        replace(assembled, qwen3_2b=None)


def test_storage_below_compute_lands_on_cast_operations(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    """bf16 storage under the default bf16 diffusion compute is
    initless; fp32 text encoders at fp32 stay initless; but bf16
    storage at fp32 compute must cast at use - and per-parameter
    dtypes stay exactly as shipped either way."""
    state = component_state(Flux(TINY_FLUX), torch.bfloat16)
    path = write_checkpoint(tmp_path / "dit-bf16.safetensors", state)
    plan = make_plan(
        standard,
        diffusion=component_plan("diffusion", path, TINY_FLUX, state),
    )
    assembled = assemble_flux(plan)  # default diffusion_dtype=bf16
    assert not any(isinstance(child, _CastMixin) for child in assembled.diffusion.modules())
    assembled32 = assemble_flux(plan, diffusion_dtype=torch.float32)
    assert any(isinstance(child, _CastMixin) for child in assembled32.diffusion.modules())
    for loaded in (assembled, assembled32):
        for key, tensor in loaded.diffusion.state_dict().items():
            assert tensor.dtype == torch.bfloat16, key


@pytest.mark.parametrize(
    ("storage_dtype", "compute_dtype"),
    ((torch.float16, torch.bfloat16), (torch.bfloat16, torch.float16)),
)
def test_equal_width_cast_storage_keeps_file_backing(
    tmp_path: Path, storage_dtype: torch.dtype, compute_dtype: torch.dtype
) -> None:
    class Component(torch.nn.Module):
        def __init__(self, operations: Any) -> None:
            super().__init__()
            self.linear = operations.linear(4, 3)

    def build(_config: object, *, operations: Any) -> Component:
        return Component(operations)

    generator = torch.Generator().manual_seed(667)
    state = {
        "linear.weight": torch.randn((3, 4), generator=generator).to(storage_dtype),
        "linear.bias": torch.randn((3,), generator=generator).to(storage_dtype),
    }
    path = write_checkpoint(tmp_path / "equal-width.safetensors", state)
    module = assemble_mod._load_component(  # pyright: ignore[reportPrivateUsage]
        component_plan("diffusion", path, SimpleNamespace(), state),
        build,
        compute_dtype=compute_dtype,
        fp8_matmul=False,
        preserve_equal_width_cast_storage=True,
    )
    assert isinstance(module.linear, _CastMixin)
    for name, tensor in module.state_dict().items():
        assert tensor.dtype is storage_dtype
        assert tensor_file_slice(tensor) is not None
        assert torch.equal(tensor, state[name])
    inputs = torch.randn((2, 4), generator=generator).to(compute_dtype)
    expected = torch.nn.functional.linear(
        inputs, state["linear.weight"].to(compute_dtype), state["linear.bias"].to(compute_dtype)
    )
    assert torch.equal(module.linear(inputs), expected)


def test_default_equal_width_loading_preserves_patch_rounding(tmp_path: Path) -> None:
    from dataclasses import dataclass, field

    from dinkster_inference.patches import DiffPatch, PatchEntry, PatchSet
    from dinkster_inference_torch.module_residency import enroll_assembled

    class Component(torch.nn.Module):
        def __init__(self, operations: Any) -> None:
            super().__init__()
            self.linear = operations.linear(1, 1, bias=False)

    def build(_config: object, *, operations: Any) -> Component:
        return Component(operations)

    @dataclass
    class Assembly:
        diffusion: torch.nn.Module
        _storage_dtype_follows_compute: bool = True
        _component_compute_dtypes: dict[str, torch.dtype] = field(
            default_factory=lambda: {"diffusion": torch.bfloat16}
        )

    state = {"linear.weight": torch.tensor([[1.00390625]], dtype=torch.float16)}
    path = write_checkpoint(tmp_path / "rounding.safetensors", state)
    module = assemble_mod._load_component(  # pyright: ignore[reportPrivateUsage]
        component_plan("diffusion", path, SimpleNamespace(), state),
        build,
        compute_dtype=torch.bfloat16,
        fp8_matmul=False,
    )
    enrolled = enroll_assembled(
        cast("Any", Assembly(module)),
        load_device="cpu",
        offload_device="cpu",
        patch_sets={
            "diffusion": PatchSet(
                {"linear.weight": (PatchEntry(DiffPatch(torch.tensor([[0.001953125]]))),)}
            )
        },
    )
    assert enrolled.storage_dtype_report.outcomes == {"diffusion": "already_at_target"}
    assert module.linear(torch.ones((1, 1), dtype=torch.bfloat16)).item() == 1.0


def test_component_can_materialize_storage_at_compute_dtype(tmp_path: Path) -> None:
    class Component(torch.nn.Module):
        def __init__(self, operations: Any) -> None:
            super().__init__()
            self.conv = operations.conv1d(2, 3, 1)

    def build(_config: object, *, operations: Any) -> Component:
        return Component(operations)

    generator = torch.Generator().manual_seed(20260831)
    state = {
        "conv.weight": torch.randn((3, 2, 1), generator=generator, dtype=torch.bfloat16),
        "conv.bias": torch.randn((3,), generator=generator, dtype=torch.bfloat16),
    }
    path = write_checkpoint(tmp_path / "component-bf16.safetensors", state)
    module = assemble_mod._load_component(  # pyright: ignore[reportPrivateUsage]
        component_plan("audio_vae", path, SimpleNamespace(), state),
        build,
        compute_dtype=torch.float32,
        fp8_matmul=False,
        storage_dtype_follows_compute=True,
    )

    assert not any(isinstance(child, _CastMixin) for child in module.modules())
    assert module.conv.weight.dtype is torch.float32
    assert module.conv.bias is not None and module.conv.bias.dtype is torch.float32
    assert torch.equal(module.conv.weight, state["conv.weight"].float())
    assert torch.equal(module.conv.bias, state["conv.bias"].float())


def test_mixed_per_parameter_dtypes_are_preserved(standard: dict[str, Any], tmp_path: Path) -> None:
    """A checkpoint whose parameters disagree about dtype loads with
    every parameter at ITS OWN shipped dtype - no model-wide cast."""
    state = component_state(Flux(TINY_FLUX))
    mixed = {}
    for index, (key, tensor) in enumerate(state.items()):
        mixed[key] = tensor.to((torch.float32, torch.bfloat16, torch.float16)[index % 3])
    path = write_checkpoint(tmp_path / "dit-mixed.safetensors", mixed)
    plan = make_plan(
        standard,
        diffusion=component_plan("diffusion", path, TINY_FLUX, mixed),
    )
    assembled = assemble_flux(plan, diffusion_dtype=torch.float32)
    loaded = assembled.diffusion.state_dict()
    for key, tensor in mixed.items():
        assert loaded[key].dtype == tensor.dtype, key
    forward_all(assembled)


def test_wider_storage_rounds_down_to_compute_dtype(
    standard: dict[str, Any], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Checkpoint tensors stored WIDER than the compute dtype round down
    to the compute dtype, exactly as the reference loads them into
    model-dtype parameters (narrower storage keeps casting at use, see
    test_plain_fp8_storage_casts_at_use)."""
    caplog.set_level("INFO", logger="dinkster.inference_torch.assemble")
    state = component_state(Flux(TINY_FLUX), torch.float32)
    path = write_checkpoint(tmp_path / "dit-wide.safetensors", state)
    plan = make_plan(
        standard,
        diffusion=component_plan("diffusion", path, TINY_FLUX, state),
    )
    assembled = assemble_flux(plan)  # default diffusion_dtype=bf16
    loaded = assembled.diffusion.state_dict()
    for key, tensor in state.items():
        assert loaded[key].dtype == torch.bfloat16, key
        assert torch.equal(loaded[key], tensor.to(torch.bfloat16)), key
    assert (
        "diffusion: casting checkpoint storage dtype torch.float32 to compute dtype torch.bfloat16"
    ) in caplog.messages
    forward_all(assembled)


def test_plain_fp8_storage_casts_at_use(standard: dict[str, Any], tmp_path: Path) -> None:
    """flux1-dev-fp8-style files: fp8 dtypes with NO scales are not
    quantization - just storage below compute, dequantized at use."""
    state = component_state(Flux(TINY_FLUX))
    plain = {key: tensor.to(E4M3) for key, tensor in state.items()}
    path = write_checkpoint(tmp_path / "dit-fp8.safetensors", plain)
    plan = make_plan(
        standard,
        diffusion=component_plan("diffusion", path, TINY_FLUX, plain),
    )
    assembled = assemble_flux(plan, diffusion_dtype=torch.float32)
    assert not any(isinstance(child, Fp8Linear) for child in assembled.diffusion.modules())
    for key, tensor in assembled.diffusion.state_dict().items():
        assert tensor.dtype == E4M3, key
    forward_all(assembled)


def test_plain_fp8_matmul_route_binds_cast_linears(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    state = component_state(Flux(TINY_FLUX))
    plain = {key: tensor.to(E4M3) for key, tensor in state.items()}
    path = write_checkpoint(tmp_path / "dit-fp8-route.safetensors", plain)
    plan = make_plan(
        standard,
        diffusion=component_plan("diffusion", path, TINY_FLUX, plain),
    )
    assembled = assemble_flux(
        plan,
        diffusion_dtype=torch.float32,
        fp8_matmul=True,
    )
    linears = [
        child for child in assembled.diffusion.modules() if isinstance(child, torch.nn.Linear)
    ]
    assert linears
    assert all(getattr(child, "fp8_matmul", False) for child in linears)


def test_prefixed_source_keys_follow_the_planned_rename(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    state = component_state(Flux(TINY_FLUX))
    prefix = "model.diffusion_model."
    path = write_checkpoint(
        tmp_path / "combined.safetensors",
        {prefix + key: tensor for key, tensor in state.items()},
    )
    plan = make_plan(
        standard,
        diffusion=component_plan("diffusion", path, TINY_FLUX, state, prefix=prefix),
    )
    assembled = assemble_flux(plan, diffusion_dtype=torch.float32)
    assert torch.equal(
        assembled.diffusion.state_dict()["img_in.weight"],
        state["img_in.weight"],
    )


# -------------------------------------------------------- scaled fp8


def scaled_diffusion(
    tmp_path: Path,
    *,
    input_scale: torch.Tensor | None = None,
    quant_override: LayerQuant | None = None,
    extra: dict[str, torch.Tensor] | None = None,
) -> tuple[ComponentPlan[Any], dict[str, torch.Tensor]]:
    state = component_state(Flux(TINY_FLUX))
    scale = quantize_into(state, "img_in")
    tensors = dict(state)
    tensors["img_in.scale_weight"] = scale
    quant = quant_override or LayerQuant(
        layer="img_in",
        format="float8_e4m3fn",
        weight="img_in.weight",
        weight_scale="img_in.scale_weight",
    )
    if input_scale is not None:
        tensors["img_in.scale_input"] = input_scale
    if extra:
        tensors.update(extra)
    path = write_checkpoint(tmp_path / "dit-scaled.safetensors", tensors)
    plan = component_plan("diffusion", path, TINY_FLUX, state, quant={"img_in": quant})
    return plan, tensors


def nvfp4_diffusion(
    tmp_path: Path,
    *,
    pre_quant_scale: bool = False,
    input_scale: bool = True,
    mutate: Callable[[dict[str, torch.Tensor]], None] | None = None,
) -> tuple[ComponentPlan[Any], dict[str, torch.Tensor]]:
    state = component_state(Flux(TINY_FLUX))
    kitchen = quant_linear_mod._require_nvfp4_kitchen()  # pyright: ignore[reportPrivateUsage]
    weight = state["img_in.weight"]
    tensor_scale = torch.tensor(weight.abs().amax().item() / (448.0 * 6.0), dtype=torch.float32)
    qweight, block_scale = kitchen.quantize(weight, tensor_scale)
    state["img_in.weight"] = qweight
    tensors = dict(state)
    tensors.update(
        {
            "img_in.weight_scale": block_scale,
            "img_in.weight_scale_2": tensor_scale,
        }
    )
    if input_scale:
        tensors["img_in.input_scale"] = torch.tensor(0.125, dtype=torch.float32)
    if pre_quant_scale:
        tensors["img_in.pre_quant_scale"] = torch.linspace(0.5, 1.5, 64)
    if mutate is not None:
        mutate(tensors)
    path = write_checkpoint(tmp_path / "dit-nvfp4.safetensors", tensors)
    quant = LayerQuant(
        layer="img_in",
        format="nvfp4",
        weight="img_in.weight",
        weight_scale="img_in.weight_scale",
        input_scale="img_in.input_scale" if input_scale else None,
        weight_scale_2="img_in.weight_scale_2",
        pre_quant_scale=("img_in.pre_quant_scale" if pre_quant_scale else None),
    )
    return (
        component_plan("diffusion", path, TINY_FLUX, state, quant={"img_in": quant}),
        tensors,
    )


def int8_convrot_diffusion(
    tmp_path: Path,
    *,
    config: dict[str, Any] | None = None,
    bias_dtype: torch.dtype = torch.float32,
) -> tuple[ComponentPlan[Any], dict[str, torch.Tensor]]:
    state = component_state(Flux(TINY_FLUX))
    state["img_in.bias"] = state["img_in.bias"].to(bias_dtype)
    rows, columns = state["img_in.weight"].shape
    generator = torch.Generator().manual_seed(84)
    state["img_in.weight"] = torch.randint(
        -100, 101, (rows, columns), generator=generator, dtype=torch.int8
    )
    state["img_in.weight_scale"] = (
        torch.rand((rows, 1), generator=generator, dtype=torch.float32) / 100
    )
    if config is not None:
        state["img_in.comfy_quant"] = quant_json(config)
    path = write_checkpoint(tmp_path / "dit-int8-convrot.safetensors", state)
    quant = LayerQuant(
        layer="img_in",
        format="int8_tensorwise",
        weight="img_in.weight",
        weight_scale="img_in.weight_scale",
        config="img_in.comfy_quant" if config is not None else None,
        parameters={"convrot": True, "convrot_groupsize": columns},
    )
    return (
        component_plan("diffusion", path, TINY_FLUX, state, quant={"img_in": quant}),
        state,
    )


def test_int8_convrot_assembles_and_executes_selected_provider(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    diffusion, tensors = int8_convrot_diffusion(tmp_path)
    assembled = assemble_flux(
        make_plan(standard, diffusion=diffusion), diffusion_dtype=torch.float32
    )
    layer = assembled.diffusion.img_in
    assert isinstance(layer, Int8Linear)
    assert layer.convrot is True
    assert layer.convrot_groupsize == layer.in_features
    input = torch.randn(3, layer.in_features)
    expected = quant_linear_mod._int8_linear(  # pyright: ignore[reportPrivateUsage]
        input,
        tensors["img_in.weight"],
        tensors["img_in.weight_scale"],
        tensors["img_in.bias"],
        out_dtype=torch.float32,
        convrot=True,
        convrot_groupsize=layer.in_features,
    )
    torch.testing.assert_close(layer(input), expected, rtol=0, atol=0)


def test_int8_quantized_bias_is_loaded_at_compute_dtype(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    diffusion, tensors = int8_convrot_diffusion(tmp_path, bias_dtype=torch.bfloat16)
    assembled = assemble_flux(
        make_plan(standard, diffusion=diffusion), diffusion_dtype=torch.float16
    )
    layer = assembled.diffusion.img_in
    assert isinstance(layer, Int8Linear)
    assert layer.bias is not None
    assert layer.bias.dtype is torch.float16
    assert torch.equal(layer.bias, tensors["img_in.bias"].to(torch.float16))


def test_quantized_component_loads_cast_state_at_bound_compute_dtype(tmp_path: Path) -> None:
    class MixedPrecisionModule(torch.nn.Module):
        def __init__(self, operations: Any) -> None:
            super().__init__()
            self.quantized = INITLESS.linear(4, 3)
            self.precision_quantized = CastOperations(torch.float32).linear(4, 3)
            self.plain = operations.linear(4, 4)
            self.patch = CastOperations(torch.float32).conv3d(1, 2, 1)

    def build(_config: object, *, operations: Any) -> MixedPrecisionModule:
        return MixedPrecisionModule(operations)

    generator = torch.Generator().manual_seed(20260822)
    state = {
        "quantized.weight": torch.randint(-100, 101, (3, 4), generator=generator, dtype=torch.int8),
        "quantized.weight_scale": torch.rand((), generator=generator),
        "quantized.bias": torch.randn((3,), generator=generator, dtype=torch.bfloat16),
        "precision_quantized.weight": torch.randint(
            -100, 101, (3, 4), generator=generator, dtype=torch.int8
        ),
        "precision_quantized.weight_scale": torch.rand((), generator=generator),
        "precision_quantized.bias": torch.randn((3,), generator=generator, dtype=torch.bfloat16),
        "plain.weight": torch.randn((4, 4), generator=generator, dtype=torch.bfloat16),
        "plain.bias": torch.randn((4,), generator=generator, dtype=torch.bfloat16),
        "patch.weight": torch.randn((2, 1, 1, 1, 1), generator=generator, dtype=torch.bfloat16),
        "patch.bias": torch.randn((2,), generator=generator, dtype=torch.bfloat16),
    }
    path = write_checkpoint(tmp_path / "mixed-precision.safetensors", state)
    quant = LayerQuant(
        layer="quantized",
        format="int8_tensorwise",
        weight="quantized.weight",
        weight_scale="quantized.weight_scale",
    )
    precision_quant = LayerQuant(
        layer="precision_quantized",
        format="int8_tensorwise",
        weight="precision_quantized.weight",
        weight_scale="precision_quantized.weight_scale",
    )
    plan = component_plan(
        "diffusion",
        path,
        SimpleNamespace(),
        state,
        quant={"quantized": quant, "precision_quantized": precision_quant},
    )

    module = assemble_mod._load_component(  # pyright: ignore[reportPrivateUsage]
        plan,
        build,
        compute_dtype=torch.float16,
        fp8_matmul=False,
    )

    assert isinstance(module.quantized, Int8Linear)
    assert module.quantized.bias is not None
    assert module.quantized.bias.dtype is torch.float16
    assert torch.equal(module.quantized.bias, state["quantized.bias"].half())
    assert isinstance(module.precision_quantized, Int8Linear)
    assert module.precision_quantized.compute_dtype is torch.float32
    assert module.precision_quantized.bias is not None
    assert module.precision_quantized.bias.dtype is torch.float32
    assert torch.equal(
        module.precision_quantized.bias,
        state["precision_quantized.bias"].float(),
    )
    assert module.plain.weight.dtype is torch.float16
    assert module.plain.bias is not None and module.plain.bias.dtype is torch.float16
    assert torch.equal(module.plain.weight, state["plain.weight"].half())
    assert torch.equal(module.plain.bias, state["plain.bias"].half())
    assert module.patch.weight.dtype is torch.float32
    assert module.patch.bias is not None and module.patch.bias.dtype is torch.float32
    assert torch.equal(module.patch.weight, state["patch.weight"].float())
    assert torch.equal(module.patch.bias, state["patch.bias"].float())


def test_quantized_component_preserves_narrower_plain_storage(tmp_path: Path) -> None:
    class MixedPrecisionModule(torch.nn.Module):
        def __init__(self, operations: Any) -> None:
            super().__init__()
            self.quantized = INITLESS.linear(4, 3, bias=False)
            self.embedding = operations.embedding(8, 4)

    def build(_config: object, *, operations: Any) -> MixedPrecisionModule:
        return MixedPrecisionModule(operations)

    generator = torch.Generator().manual_seed(20260831)
    state = {
        "quantized.weight": torch.randint(-100, 101, (3, 4), generator=generator, dtype=torch.int8),
        "quantized.weight_scale": torch.rand((), generator=generator),
        "embedding.weight": torch.randn((8, 4), generator=generator, dtype=torch.float16),
    }
    path = write_checkpoint(tmp_path / "narrow-plain-storage.safetensors", state)
    quant = LayerQuant(
        layer="quantized",
        format="int8_tensorwise",
        weight="quantized.weight",
        weight_scale="quantized.weight_scale",
    )
    plan = component_plan(
        "text",
        path,
        SimpleNamespace(),
        state,
        quant={"quantized": quant},
    )

    module = assemble_mod._load_component(  # pyright: ignore[reportPrivateUsage]
        plan,
        build,
        compute_dtype=torch.float32,
        fp8_matmul=False,
    )

    assert module.embedding.weight.dtype is torch.float16
    output = module.embedding(torch.tensor([[1, 2]]))
    assert output.dtype is torch.float32
    assert torch.equal(output[0], state["embedding.weight"][[1, 2]].float())


def test_int8_embedding_assembles_and_executes_selected_rows(tmp_path: Path) -> None:
    class EmbeddingModule(torch.nn.Module):
        def __init__(self, operations: Any) -> None:
            super().__init__()
            self.embed_tokens = operations.embedding(11, 256)

    def build(_config: object, *, operations: Any) -> EmbeddingModule:
        return EmbeddingModule(operations)

    generator = torch.Generator().manual_seed(20260830)
    state = {
        "embed_tokens.weight": torch.randint(
            -100, 101, (11, 256), generator=generator, dtype=torch.int8
        ),
        "embed_tokens.weight_scale": (
            torch.rand((11, 1), generator=generator, dtype=torch.float32) / 100
        ),
    }
    path = write_checkpoint(tmp_path / "embedding-int8-convrot.safetensors", state)
    quant = LayerQuant(
        layer="embed_tokens",
        format="int8_tensorwise",
        weight="embed_tokens.weight",
        weight_scale="embed_tokens.weight_scale",
        parameters={"convrot": True, "convrot_groupsize": 256},
    )
    plan = component_plan(
        "gemma4_12b",
        path,
        SimpleNamespace(),
        state,
        quant={"embed_tokens": quant},
    )

    module = assemble_mod._load_component(  # pyright: ignore[reportPrivateUsage]
        plan,
        build,
        compute_dtype=torch.float32,
        fp8_matmul=False,
    )

    assert isinstance(module.embed_tokens, Int8Embedding)
    indices = torch.tensor([[1, 7, 4], [10, 0, 3]])
    expected = torch.ops.dinkster_kitchen.dequantize_int8_embedding(
        state["embed_tokens.weight"],
        state["embed_tokens.weight_scale"],
        indices,
        256,
        0,
    )
    torch.testing.assert_close(module.embed_tokens(indices), expected, rtol=0, atol=0)


def test_int8_embedding_refuses_max_norm() -> None:
    module = torch.nn.Module()
    module.embed_tokens = torch.nn.Embedding(11, 256, max_norm=1.0)
    resolved = assemble_mod._ResolvedQuant(  # pyright: ignore[reportPrivateUsage]
        format="int8_tensorwise",
        fp8_dtype=None,
        full_precision_matmul=False,
        convrot=True,
    )

    with pytest.raises(AssembleError, match="max_norm"):
        assemble_mod._swap_in_int8_layer(  # pyright: ignore[reportPrivateUsage]
            "gemma4_12b",
            module,
            "embed_tokens",
            resolved,
            compute_dtype=torch.float32,
        )


def test_non_quantized_component_normalizes_equal_and_wider_state_and_pins(tmp_path: Path) -> None:
    class MixedPrecisionModule(torch.nn.Module):
        def __init__(self, operations: Any) -> None:
            super().__init__()
            self.body = operations.linear(4, 4)
            self.wide = operations.linear(4, 4)
            self.patch = CastOperations(torch.float32).conv3d(1, 2, 1)

    def build(_config: object, *, operations: Any) -> MixedPrecisionModule:
        return MixedPrecisionModule(operations)

    generator = torch.Generator().manual_seed(20260823)
    state = {
        "body.weight": torch.randn((4, 4), generator=generator, dtype=torch.bfloat16),
        "body.bias": torch.randn((4,), generator=generator, dtype=torch.bfloat16),
        "wide.weight": torch.randn((4, 4), generator=generator, dtype=torch.float32),
        "wide.bias": torch.randn((4,), generator=generator, dtype=torch.float32),
        "patch.weight": torch.randn((2, 1, 1, 1, 1), generator=generator, dtype=torch.bfloat16),
        "patch.bias": torch.randn((2,), generator=generator, dtype=torch.bfloat16),
    }
    path = write_checkpoint(tmp_path / "mixed-precision.safetensors", state)
    plan = component_plan("diffusion", path, SimpleNamespace(), state)

    module = assemble_mod._load_component(  # pyright: ignore[reportPrivateUsage]
        plan,
        build,
        compute_dtype=torch.float16,
        fp8_matmul=False,
    )

    assert module.body.weight.dtype is torch.float16
    assert module.body.bias is not None and module.body.bias.dtype is torch.float16
    assert torch.equal(module.body.weight, state["body.weight"].half())
    assert torch.equal(module.body.bias, state["body.bias"].half())
    assert module.wide.weight.dtype is torch.float16
    assert module.wide.bias is not None and module.wide.bias.dtype is torch.float16
    assert torch.equal(module.wide.weight, state["wide.weight"].half())
    assert torch.equal(module.wide.bias, state["wide.bias"].half())
    assert module.patch.weight.dtype is torch.float32
    assert module.patch.bias is not None and module.patch.bias.dtype is torch.float32
    assert torch.equal(module.patch.weight, state["patch.weight"].float())
    assert torch.equal(module.patch.bias, state["patch.bias"].float())


def test_int8_payload_full_precision_pin_must_be_boolean(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    diffusion, _ = int8_convrot_diffusion(
        tmp_path,
        config={
            "format": "int8_tensorwise",
            "full_precision_matrix_mult": "false",
            "convrot": True,
            "convrot_groupsize": 64,
        },
    )
    with pytest.raises(AssembleError, match="full_precision_matrix_mult must be a bool"):
        assemble_flux(make_plan(standard, diffusion=diffusion), diffusion_dtype=torch.float32)


@pytest.mark.parametrize(
    ("group", "match"),
    [
        (8, "power of 4"),
        (256, "width 64 must be divisible"),
    ],
)
def test_int8_hand_built_plan_refuses_invalid_convrot_group(
    standard: dict[str, Any], tmp_path: Path, group: int, match: str
) -> None:
    diffusion, _ = int8_convrot_diffusion(tmp_path)
    quant = diffusion.quant["img_in"]
    diffusion = replace(
        diffusion,
        quant={
            "img_in": replace(
                quant,
                parameters={"convrot": True, "convrot_groupsize": group},
            )
        },
    )
    with pytest.raises(AssembleError, match=match):
        assemble_flux(make_plan(standard, diffusion=diffusion), diffusion_dtype=torch.float32)


def test_nvfp4_assembles_strict_ordinary_state(standard: dict[str, Any], tmp_path: Path) -> None:
    diffusion, tensors = nvfp4_diffusion(tmp_path, pre_quant_scale=True)
    assembled = assemble_flux(
        make_plan(standard, diffusion=diffusion), diffusion_dtype=torch.float32
    )
    layer = assembled.diffusion.img_in
    assert isinstance(layer, Nvfp4Linear)
    assert set(layer.state_dict()) == {
        "weight",
        "weight_scale",
        "weight_scale_2",
        "input_scale",
        "pre_quant_scale",
        "bias",
    }
    for name in layer.state_dict():
        assert torch.equal(
            layer.state_dict()[name].reshape(-1).view(torch.uint8),
            tensors[f"img_in.{name}"].reshape(-1).view(torch.uint8),
        )
    assert not isinstance(assembled.diffusion.txt_in, Nvfp4Linear)
    status = nvfp4_runtime_status(assembled)
    recorder = layer._diagnostics  # pyright: ignore[reportPrivateUsage]
    assert recorder is not None
    recorder.record("route_non_cuda")
    reconstructed = replace(assembled)
    assert reconstructed.diffusion is assembled.diffusion
    assert layer._diagnostics is recorder  # pyright: ignore[reportPrivateUsage]
    assert nvfp4_runtime_status(reconstructed) == recorder.snapshot()
    assert nvfp4_runtime_status(reconstructed).lifetime["route_non_cuda"] == (
        status.lifetime.get("route_non_cuda", 0) + 1
    )


def test_nvfp4_reconstruction_rejects_mixed_recorders(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    diffusion, _ = nvfp4_diffusion(tmp_path)
    assembled = assemble_flux(
        make_plan(standard, diffusion=diffusion), diffusion_dtype=torch.float32
    )
    second = Nvfp4Linear(64, 64, bias=False, compute_dtype=torch.float32)
    second._bind_diagnostics(  # pyright: ignore[reportPrivateUsage]
        Nvfp4DiagnosticsRecorder()
    )
    cast(Any, assembled.diffusion).txt_in = second
    with pytest.raises(RuntimeError, match="different diagnostics recorders"):
        replace(assembled)


def test_nvfp4_assembles_without_optional_input_scale(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    diffusion, _ = nvfp4_diffusion(tmp_path, input_scale=False)
    assembled = assemble_flux(
        make_plan(standard, diffusion=diffusion), diffusion_dtype=torch.float32
    )
    layer = assembled.diffusion.img_in
    assert isinstance(layer, Nvfp4Linear)
    assert layer.input_scale is None
    assert "input_scale" not in layer.state_dict()


@pytest.mark.parametrize(
    "name",
    [
        "img_in.weight_scale",
        "img_in.weight_scale_2",
        "img_in.input_scale",
        "img_in.pre_quant_scale",
    ],
)
def test_nvfp4_nonfinite_scale_refuses_before_assignment(
    standard: dict[str, Any], tmp_path: Path, name: str
) -> None:
    def poison(tensors: dict[str, torch.Tensor]) -> None:
        if name == "img_in.weight_scale":
            tensors[name] = torch.full(tensors[name].shape, float("nan"), dtype=torch.float32).to(
                torch.float8_e4m3fn
            )
        elif name == "img_in.pre_quant_scale":
            tensors[name] = torch.full(tensors[name].shape, float("nan"))
        else:
            tensors[name] = torch.tensor(float("nan"), dtype=torch.float32)

    diffusion, _ = nvfp4_diffusion(
        tmp_path,
        pre_quant_scale=name == "img_in.pre_quant_scale",
        mutate=poison,
    )
    with pytest.raises(AssembleError, match="non-finite|not finite"):
        assemble_flux(
            make_plan(standard, diffusion=diffusion),
            diffusion_dtype=torch.float32,
        )


def test_nvfp4_nonscalar_tensor_scale_refuses_before_assignment(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    def reshape_scale(tensors: dict[str, torch.Tensor]) -> None:
        tensors["img_in.weight_scale_2"] = tensors["img_in.weight_scale_2"].reshape(1)

    diffusion, _ = nvfp4_diffusion(tmp_path, mutate=reshape_scale)
    with pytest.raises(AssembleError, match="scalar float32"):
        assemble_flux(
            make_plan(standard, diffusion=diffusion),
            diffusion_dtype=torch.float32,
        )


def test_scaled_fp8_swaps_in_fp8_linear(standard: dict[str, Any], tmp_path: Path) -> None:
    diffusion, tensors = scaled_diffusion(tmp_path)
    assembled = assemble_flux(
        make_plan(standard, diffusion=diffusion),
        diffusion_dtype=torch.float32,
    )
    layer = assembled.diffusion.img_in
    assert isinstance(layer, Fp8Linear)
    assert layer.weight.dtype == E4M3
    assert torch.equal(
        layer.weight.view(torch.uint8),
        tensors["img_in.weight"].view(torch.uint8),
    )
    assert torch.equal(layer.weight_scale, tensors["img_in.scale_weight"])
    assert layer.input_scale.item() == 1.0  # neutral default
    assert not layer.fp8_matmul
    # Only the quantized layer swapped; siblings stay ordinary.
    assert not isinstance(assembled.diffusion.txt_in, Fp8Linear)
    forward_all(assembled)


def test_scaled_fp8_loads_checkpoint_input_scale(standard: dict[str, Any], tmp_path: Path) -> None:
    diffusion, _ = scaled_diffusion(
        tmp_path,
        input_scale=torch.tensor(0.25, dtype=torch.float32),
        quant_override=LayerQuant(
            layer="img_in",
            format="float8_e4m3fn",
            weight="img_in.weight",
            weight_scale="img_in.scale_weight",
            input_scale="img_in.scale_input",
        ),
    )
    assembled = assemble_flux(
        make_plan(standard, diffusion=diffusion),
        diffusion_dtype=torch.float32,
    )
    layer = assembled.diffusion.img_in
    assert isinstance(layer, Fp8Linear)
    assert layer.input_scale.item() == 0.25


def test_payload_config_spelling_resolves_format_and_pin(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    """The per-layer .comfy_quant JSON spelling: format and the
    full-precision-matmul pin live in payload bytes, resolved at load;
    fp8_matmul=True must respect the pin and enable the rest."""
    state = component_state(Flux(TINY_FLUX))
    img_scale = quantize_into(state, "img_in")
    txt_scale = quantize_into(state, "txt_in")
    tensors = dict(state)
    tensors["img_in.scale_weight"] = img_scale
    tensors["img_in.comfy_quant"] = quant_json(
        {
            "format": "float8_e4m3fn",
            "full_precision_matrix_mult": "enabled",
            "future_parameter": 1,
        }
    )
    tensors["txt_in.scale_weight"] = txt_scale
    path = write_checkpoint(tmp_path / "dit-config.safetensors", tensors)
    plan = component_plan(
        "diffusion",
        path,
        TINY_FLUX,
        state,
        quant={
            "img_in": LayerQuant(
                layer="img_in",
                format=None,
                weight="img_in.weight",
                weight_scale="img_in.scale_weight",
                config="img_in.comfy_quant",
            ),
            "txt_in": LayerQuant(
                layer="txt_in",
                format="float8_e4m3fn",
                weight="txt_in.weight",
                weight_scale="txt_in.scale_weight",
            ),
        },
    )
    assembled = assemble_flux(
        make_plan(standard, diffusion=plan),
        diffusion_dtype=torch.float32,
        fp8_matmul=True,
    )
    img_in = assembled.diffusion.img_in
    txt_in = assembled.diffusion.txt_in
    assert isinstance(img_in, Fp8Linear) and isinstance(txt_in, Fp8Linear)
    assert img_in.full_precision_matmul and not img_in.fp8_matmul
    assert not txt_in.full_precision_matmul and txt_in.fp8_matmul


# ------------------------------------------------------ default fills


def test_absent_text_projection_fills_identity(standard: dict[str, Any], tmp_path: Path) -> None:
    state = component_state(ClipTextModel(TINY_CLIP))
    del state["text_projection.weight"]
    path = write_checkpoint(tmp_path / "clip-sd1.safetensors", state)
    plan = make_plan(
        standard,
        clip_l=component_plan(
            "clip_l",
            path,
            TINY_CLIP,
            state,
            absent=("text_projection.weight",),
        ),
    )
    assembled = assemble_flux(plan, diffusion_dtype=torch.float32)
    assert assembled.clip_l is not None
    projection = assembled.clip_l.state_dict()["text_projection.weight"]
    assert torch.equal(projection, torch.eye(64))


def test_absent_key_without_default_refuses(standard: dict[str, Any], tmp_path: Path) -> None:
    state = component_state(Flux(TINY_FLUX))
    del state["img_in.bias"]
    path = write_checkpoint(tmp_path / "dit-truncated.safetensors", state)
    plan = make_plan(
        standard,
        diffusion=component_plan("diffusion", path, TINY_FLUX, state, absent=("img_in.bias",)),
    )
    with pytest.raises(AssembleError, match="no default fill"):
        assemble_flux(plan, diffusion_dtype=torch.float32)


# ---------------------------------------------------------- refusals


def broken_config_case(
    standard: dict[str, Any],
    tmp_path: Path,
    config_tensor: torch.Tensor,
) -> FluxAssemblyPlan:
    state = component_state(Flux(TINY_FLUX))
    scale = quantize_into(state, "img_in")
    tensors = dict(state)
    tensors["img_in.scale_weight"] = scale
    tensors["img_in.comfy_quant"] = config_tensor
    path = write_checkpoint(tmp_path / "dit-broken.safetensors", tensors)
    plan = component_plan(
        "diffusion",
        path,
        TINY_FLUX,
        state,
        quant={
            "img_in": LayerQuant(
                layer="img_in",
                format=None,
                weight="img_in.weight",
                weight_scale="img_in.scale_weight",
                config="img_in.comfy_quant",
            )
        },
    )
    return make_plan(standard, diffusion=plan)


def test_config_tensor_must_be_uint8(standard: dict[str, Any], tmp_path: Path) -> None:
    plan = broken_config_case(standard, tmp_path, torch.zeros(4, dtype=torch.float32))
    with pytest.raises(AssembleError, match="expected uint8"):
        assemble_flux(plan, diffusion_dtype=torch.float32)


def test_config_tensor_must_be_json(standard: dict[str, Any], tmp_path: Path) -> None:
    plan = broken_config_case(
        standard, tmp_path, torch.tensor(list(b"not json"), dtype=torch.uint8)
    )
    with pytest.raises(AssembleError, match="not JSON"):
        assemble_flux(plan, diffusion_dtype=torch.float32)


def test_config_json_must_be_an_object(standard: dict[str, Any], tmp_path: Path) -> None:
    plan = broken_config_case(standard, tmp_path, quant_json([1, 2]))  # type: ignore[arg-type]
    with pytest.raises(AssembleError, match="not an object"):
        assemble_flux(plan, diffusion_dtype=torch.float32)


def test_config_without_format_refuses(standard: dict[str, Any], tmp_path: Path) -> None:
    plan = broken_config_case(standard, tmp_path, quant_json({"group_size": 32}))
    with pytest.raises(AssembleError, match="no.*format"):
        assemble_flux(plan, diffusion_dtype=torch.float32)


def test_payload_nvfp4_with_nonpacked_weight_refuses(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    plan = broken_config_case(standard, tmp_path, quant_json({"format": "nvfp4"}))
    with pytest.raises(AssembleError, match="NVFP4 weight must be rank-2 uint8"):
        assemble_flux(plan, diffusion_dtype=torch.float32)


def test_payload_full_precision_fact_must_match_planned_identity(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    plan = broken_config_case(
        standard,
        tmp_path,
        quant_json({"format": "nvfp4", "full_precision_matrix_mult": True}),
    )
    planned = plan.diffusion.quant["img_in"]
    diffusion = replace(
        plan.diffusion,
        quant={
            "img_in": replace(
                planned,
                format="nvfp4",
                full_precision_matmul=False,
            )
        },
    )
    with pytest.raises(AssembleError, match="contradicts the planned value"):
        assemble_flux(replace(plan, diffusion=diffusion), diffusion_dtype=torch.float32)


def test_weight_payload_contradicting_declared_dtype_refuses(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    state = component_state(Flux(TINY_FLUX))
    tensors = dict(state)  # weight stays fp32: the header lied
    tensors["img_in.scale_weight"] = torch.ones((), dtype=torch.float32)
    path = write_checkpoint(tmp_path / "dit-lied.safetensors", tensors)
    plan = component_plan(
        "diffusion",
        path,
        TINY_FLUX,
        state,
        quant={
            "img_in": LayerQuant(
                layer="img_in",
                format="float8_e4m3fn",
                weight="img_in.weight",
                weight_scale="img_in.scale_weight",
            )
        },
    )
    with pytest.raises(AssembleError, match="declares.*float8_e4m3fn"):
        assemble_flux(
            make_plan(standard, diffusion=plan),
            diffusion_dtype=torch.float32,
        )


@pytest.mark.parametrize(
    "bad_scale",
    [
        torch.ones((), dtype=torch.float16),
        torch.ones(2, dtype=torch.float32),
    ],
    ids=["wrong-dtype", "wrong-numel"],
)
def test_malformed_weight_scale_refuses(
    standard: dict[str, Any], tmp_path: Path, bad_scale: torch.Tensor
) -> None:
    state = component_state(Flux(TINY_FLUX))
    quantize_into(state, "img_in")
    tensors = dict(state)
    tensors["img_in.scale_weight"] = bad_scale
    path = write_checkpoint(tmp_path / "dit-badscale.safetensors", tensors)
    plan = component_plan(
        "diffusion",
        path,
        TINY_FLUX,
        state,
        quant={
            "img_in": LayerQuant(
                layer="img_in",
                format="float8_e4m3fn",
                weight="img_in.weight",
                weight_scale="img_in.scale_weight",
            )
        },
    )
    with pytest.raises(AssembleError, match="weight_scale must be one"):
        assemble_flux(
            make_plan(standard, diffusion=plan),
            diffusion_dtype=torch.float32,
        )


def test_quantized_non_linear_layer_refuses(standard: dict[str, Any], tmp_path: Path) -> None:
    """T5's shared embedding has a weight but no fp8-Linear port."""
    state = component_state(T5TextModel(TINY_T5))
    quantize_into(state, "shared")
    tensors = dict(state)
    tensors["shared.scale_weight"] = torch.ones((), dtype=torch.float32)
    path = write_checkpoint(tmp_path / "t5-embed-quant.safetensors", tensors)
    plan = make_plan(
        standard,
        t5xxl=component_plan(
            "t5xxl",
            path,
            TINY_T5,
            state,
            quant={
                "shared": LayerQuant(
                    layer="shared",
                    format="float8_e4m3fn",
                    weight="shared.weight",
                    weight_scale="shared.scale_weight",
                )
            },
        ),
    )
    with pytest.raises(AssembleError, match="only Linear"):
        assemble_flux(plan, diffusion_dtype=torch.float32)


def test_quantized_layer_missing_from_module_refuses(
    standard: dict[str, Any], tmp_path: Path
) -> None:
    state = component_state(Flux(TINY_FLUX))
    ghost = torch.randn(8, 8)
    state["phantom.weight"] = ghost  # planner-claimed, module-unknown
    quantize_into(state, "phantom")
    tensors = dict(state)
    tensors["phantom.scale_weight"] = torch.ones((), dtype=torch.float32)
    path = write_checkpoint(tmp_path / "dit-phantom.safetensors", tensors)
    plan = component_plan(
        "diffusion",
        path,
        TINY_FLUX,
        state,
        quant={
            "phantom": LayerQuant(
                layer="phantom",
                format="float8_e4m3fn",
                weight="phantom.weight",
                weight_scale="phantom.scale_weight",
            )
        },
    )
    with pytest.raises(AssembleError, match="does not exist"):
        assemble_flux(
            make_plan(standard, diffusion=plan),
            diffusion_dtype=torch.float32,
        )


# ------------------------------------------------- planned transforms


def transformed_plan(path: Path) -> ComponentPlan[Any]:
    """A hand-built plan whose model tensors are DERIVED: fused q/k/v
    row chunks plus a transposed projection (the OpenCLIP text-encoder
    conversion shape, comfy/utils.py transformers_convert @ 947c2749)."""
    return ComponentPlan(
        component="clip_g",
        path=path,
        config=None,
        keys={
            "q.weight": "attn.in_proj_weight",
            "k.weight": "attn.in_proj_weight",
            "v.weight": "attn.in_proj_weight",
            "proj.weight": "text_projection",
        },
        dtypes={
            "q.weight": FLOAT32,
            "k.weight": FLOAT32,
            "v.weight": FLOAT32,
            "proj.weight": FLOAT32,
        },
        quant={},
        transforms={
            "q.weight": RowChunk(part=0, parts=3),
            "k.weight": RowChunk(part=1, parts=3),
            "v.weight": RowChunk(part=2, parts=3),
            "proj.weight": Transpose2D(),
        },
    )


def test_transforms_derive_model_tensors(tmp_path: Path) -> None:
    from dinkster_inference_torch.assemble import (
        _component_state,  # pyright: ignore[reportPrivateUsage]
    )

    fused = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    projection = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    state = _component_state(
        transformed_plan(tmp_path / "unused.safetensors"),
        {"attn.in_proj_weight": fused, "text_projection": projection},
        {},
    )
    q, k, v = fused.chunk(3, dim=0)
    assert torch.equal(state["q.weight"], q)
    assert torch.equal(state["k.weight"], k)
    assert torch.equal(state["v.weight"], v)
    assert torch.equal(state["proj.weight"], projection.transpose(0, 1))
    # Chunks are zero-copy views of the exactly-sized fused buffer;
    # the transpose is materialized into the Linear memory layout.
    assert all(tensor.is_contiguous() for tensor in state.values())
    assert state["v.weight"].untyped_storage().data_ptr() == (fused.untyped_storage().data_ptr())
    assert state["proj.weight"].untyped_storage().data_ptr() != (
        projection.untyped_storage().data_ptr()
    )


def test_row_chunk_refuses_indivisible_source(tmp_path: Path) -> None:
    from dinkster_inference_torch.assemble import (
        _component_state,  # pyright: ignore[reportPrivateUsage]
    )

    fused = torch.zeros(7, 2)
    projection = torch.zeros(2, 3)
    with pytest.raises(AssembleError, match="equal chunks"):
        _component_state(
            transformed_plan(tmp_path / "unused.safetensors"),
            {"attn.in_proj_weight": fused, "text_projection": projection},
            {},
        )


def test_transpose_refuses_wrong_rank(tmp_path: Path) -> None:
    from dinkster_inference_torch.assemble import (
        _component_state,  # pyright: ignore[reportPrivateUsage]
    )

    fused = torch.zeros(6, 2)
    projection = torch.zeros(2, 3, 1)
    with pytest.raises(AssembleError, match="rank"):
        _component_state(
            transformed_plan(tmp_path / "unused.safetensors"),
            {"attn.in_proj_weight": fused, "text_projection": projection},
            {},
        )


def test_wan21_assembly_reconstructs_embedded_sentencepiece_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "wan.safetensors"
    plan = Wan21AssemblyPlan(
        WAN21,
        component_plan("diffusion", path, WAN21_T2V_1_3B, {}),
        component_plan("umt5xxl", path, UMT5_XXL_CONFIG, {}),
        component_plan("vae", path, WAN21_VAE_CONFIG, {}),
        "text_encoders.umt5xxl.spiece_model",
    )
    components = {name: INITLESS.linear(2, 2) for name in ("diffusion", "umt5xxl", "vae")}
    cast("Any", components["diffusion"]).config = SimpleNamespace(model_type="t2v")
    payload = b"embedded sentencepiece bytes"

    def fake_load_component(
        component: ComponentPlan[Any], *_args: object, **_kwargs: object
    ) -> torch.nn.Module:
        return components[component.component]

    def fake_load_tensors(
        _path: Path, keys: Iterable[str] | None = None
    ) -> dict[str, torch.Tensor]:
        assert keys is not None
        return {key: torch.tensor(list(payload), dtype=torch.uint8) for key in keys}

    monkeypatch.setattr(assemble_mod, "_load_component", fake_load_component)
    monkeypatch.setattr(assemble_mod, "load_tensors", fake_load_tensors)

    assembled = assemble_wan21(
        plan,
        diffusion_dtype=torch.float32,
        text_dtype=torch.float32,
        vae_dtype=torch.float32,
    )

    assert isinstance(assembled, AssembledWan21)
    assert assembled.family is WAN21
    assert assembled.diffusion is components["diffusion"]
    assert assembled.umt5xxl is components["umt5xxl"]
    assert assembled.vae is components["vae"]
    assert assembled.tokenizer_model == payload


def test_wan21_flow_rvs_assembly_builds_the_one_channel_vae(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "wan-flow-rvs.safetensors"
    plan = Wan21AssemblyPlan(
        WAN21,
        component_plan("diffusion", path, WAN21_FLOW_RVS_1_3B, {}),
        component_plan("umt5xxl", path, UMT5_XXL_CONFIG, {}),
        component_plan("vae", path, WAN21_FLOW_RVS_VAE_CONFIG, {}),
        "spiece_model",
    )
    diffusion = INITLESS.linear(2, 2)
    cast("Any", diffusion).config = WAN21_FLOW_RVS_1_3B
    text = INITLESS.linear(2, 2)

    def fake_load_component(
        component: ComponentPlan[Any], factory: object, **_kwargs: object
    ) -> torch.nn.Module:
        if component.component == "diffusion":
            return diffusion
        if component.component == "umt5xxl":
            return text
        return cast("Any", factory)(component.config, operations=INITLESS)

    def fake_load_tensors(
        _path: Path, _keys: Iterable[str] | None = None
    ) -> dict[str, torch.Tensor]:
        return {"spiece_model": torch.tensor([1], dtype=torch.uint8)}

    monkeypatch.setattr(assemble_mod, "_load_component", fake_load_component)
    monkeypatch.setattr(assemble_mod, "load_tensors", fake_load_tensors)

    assembled = assemble_wan21(plan)

    assert isinstance(assembled.vae, WanVAE)
    assert assembled.vae.config.image_channels == 3
    assert assembled.vae.config.conv_out_channels == 1


def test_wan21_assembly_selects_causal_ar_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "wan-causal-ar.safetensors"
    plan = Wan21AssemblyPlan(
        WAN21,
        component_plan("diffusion", path, WAN21_CAUSAL_AR_1_3B, {}),
        component_plan("umt5xxl", path, UMT5_XXL_CONFIG, {}),
        component_plan("vae", path, WAN21_VAE_CONFIG, {}),
        "text_encoders.umt5xxl.spiece_model",
    )
    components = {name: INITLESS.linear(2, 2) for name in ("diffusion", "umt5xxl", "vae")}
    cast("Any", components["diffusion"]).config = WAN21_CAUSAL_AR_1_3B
    factories: dict[str, object] = {}

    def fake_load_component(
        component: ComponentPlan[Any], factory: object, **_kwargs: object
    ) -> torch.nn.Module:
        factories[component.component] = factory
        return components[component.component]

    def fake_load_tensors(
        _path: Path, keys: Iterable[str] | None = None
    ) -> dict[str, torch.Tensor]:
        assert keys is not None
        return {key: torch.tensor(list(b"sentencepiece"), dtype=torch.uint8) for key in keys}

    monkeypatch.setattr(assemble_mod, "_load_component", fake_load_component)
    monkeypatch.setattr(assemble_mod, "load_tensors", fake_load_tensors)

    assembled = assemble_wan21(plan)

    diffusion_factory = cast("Any", factories["diffusion"])
    assert diffusion_factory.func is Wan21CausalModel
    assert assembled.diffusion is components["diffusion"]


@pytest.mark.parametrize("config", (WAN21_SCAIL_14B, WAN21_SCAIL2_14B))
def test_wan21_assembly_selects_scail_model(
    config: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "wan-scail.safetensors"
    plan = Wan21AssemblyPlan(
        WAN21,
        component_plan("diffusion", path, cast("Any", config), {}),
        component_plan("umt5xxl", path, UMT5_XXL_CONFIG, {}),
        component_plan("vae", path, WAN21_VAE_CONFIG, {}),
        "text_encoders.umt5xxl.spiece_model",
        clip_vision=component_plan("clip_vision", path, WAN21_CLIP_VISION, {}),
    )
    components = {
        name: INITLESS.linear(2, 2) for name in ("diffusion", "umt5xxl", "vae", "clip_vision")
    }
    cast("Any", components["diffusion"]).config = config
    factories: dict[str, object] = {}

    def fake_load_component(
        component: ComponentPlan[Any], factory: object, **_kwargs: object
    ) -> torch.nn.Module:
        factories[component.component] = factory
        return components[component.component]

    def fake_load_tensors(
        _path: Path, keys: Iterable[str] | None = None
    ) -> dict[str, torch.Tensor]:
        assert keys is not None
        return {key: torch.tensor(list(b"sentencepiece"), dtype=torch.uint8) for key in keys}

    monkeypatch.setattr(assemble_mod, "_load_component", fake_load_component)
    monkeypatch.setattr(assemble_mod, "load_tensors", fake_load_tensors)

    assembled = assemble_wan21(plan)

    diffusion_factory = cast("Any", factories["diffusion"])
    assert diffusion_factory.func is WanScailModel
    assert assembled.diffusion is components["diffusion"]


def test_wan21_clip_vision_keeps_reference_float32_compute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "wan-i2v.safetensors"
    plan = Wan21AssemblyPlan(
        WAN21,
        component_plan("diffusion", path, WAN21_I2V_14B, {}),
        component_plan("umt5xxl", path, UMT5_XXL_CONFIG, {}),
        component_plan("vae", path, WAN21_VAE_CONFIG, {}),
        "text_encoders.umt5xxl.spiece_model",
        clip_vision=component_plan("clip_vision", path, WAN21_CLIP_VISION, {}),
    )
    components = {
        name: INITLESS.linear(2, 2) for name in ("diffusion", "umt5xxl", "vae", "clip_vision")
    }
    cast("Any", components["diffusion"]).config = SimpleNamespace(model_type="i2v")
    compute_dtypes: dict[str, torch.dtype] = {}

    def fake_load_component(
        component: ComponentPlan[Any],
        *_args: object,
        compute_dtype: torch.dtype,
        **_kwargs: object,
    ) -> torch.nn.Module:
        compute_dtypes[component.component] = compute_dtype
        return components[component.component]

    def fake_load_tensors(
        _path: Path, keys: Iterable[str] | None = None
    ) -> dict[str, torch.Tensor]:
        assert keys is not None
        return {key: torch.tensor(list(b"sentencepiece"), dtype=torch.uint8) for key in keys}

    monkeypatch.setattr(assemble_mod, "_load_component", fake_load_component)
    monkeypatch.setattr(assemble_mod, "load_tensors", fake_load_tensors)

    assembled = assemble_wan21(
        plan,
        diffusion_dtype=torch.bfloat16,
        text_dtype=torch.float16,
        vae_dtype=torch.bfloat16,
    )

    assert compute_dtypes == {
        "diffusion": torch.bfloat16,
        "umt5xxl": torch.float16,
        "vae": torch.bfloat16,
        "clip_vision": torch.float32,
    }
    assert assembled.compute_dtype("clip_vision") is torch.float32


def write_tiny_gguf(
    path: Path,
    state: dict[str, torch.Tensor],
    *,
    architecture: str,
    quant_layout: GGMLType | Callable[[str], GGMLType] = Q8_0,
) -> Path:
    """A real GGUF holding one tiny component: rank-2 weights whose
    element counts split into whole blocks of the requested layout
    are quantized (Q8_0 payloads encode the state values; other
    layouts carry deterministic random blocks, which residency-mode
    equivalence tests decode identically on every path), everything
    else is F32, and every tensor carries its logical shape as
    comfy.gguf.orig_shape metadata over a flat wire shape."""
    alignment = 32
    fields = [
        _metadata("general.architecture", GGUFValueType.STRING, architecture),
        _metadata("general.quantization_version", GGUFValueType.UINT32, 2),
    ]
    index = bytearray()
    payload = bytearray()
    for seed, (key, value) in enumerate(state.items()):
        layout = quant_layout(key) if callable(quant_layout) else quant_layout
        if (
            key.endswith(".weight")
            and value.ndim == 2
            and value.numel() % layout.block_elements == 0
        ):
            if layout.name == "Q8_0":
                blocks = encode_q8_0(value).reshape(-1)
            else:
                blocks = reference_quant_blocks(
                    layout, value.numel() // layout.block_elements, seed=seed
                ).reshape(-1)
            data = bytes(blocks.contiguous().untyped_storage())[: blocks.numel()]
            type_code = layout.code
        else:
            data = bytes(value.contiguous().untyped_storage())[: value.numel() * 4]
            type_code = 0  # F32
        fields.append(
            _metadata(
                "comfy.gguf.orig_shape." + key,
                GGUFValueType.ARRAY,
                (GGUFValueType.INT32, tuple(value.shape)),
            )
        )
        index.extend(_string(key))
        index.extend(struct.pack("<Iq", 1, value.numel()))
        index.extend(struct.pack("<IQ", type_code, len(payload)))
        payload.extend(data)
        payload.extend(bytes(-len(payload) % alignment))
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(state), len(fields)))
    header.extend(b"".join(fields))
    header.extend(index)
    header.extend(bytes(-len(header) % alignment))
    path.write_bytes(bytes(header) + bytes(payload))
    return path


_ENCODED_TEST_LAYOUTS: dict[str, GGMLType] = {
    "Q4_0": Q4_0,
    "Q4_K": Q4_K,
    "Q5_K": Q5_K,
    "Q6_K": Q6_K,
    "Q8_0": Q8_0,
}


def _write_and_map_tiny_t5_gguf(
    t5_state: dict[str, torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    quant_layout: GGMLType | Callable[[str], GGMLType],
) -> Path:
    """Write the tiny T5 as a GGUF checkpoint and point the component
    mapper at its geometry (the production text mapper admits only
    full-size XXL geometry; this stand-in maps the tiny T5 through
    the same llama.cpp source-name translation)."""
    names = {_llama_t5_name(key): key for key in t5_state}
    path = write_tiny_gguf(
        tmp_path / "tiny-t5.gguf",
        {name: t5_state[key] for name, key in names.items()},
        architecture="t5encoder",
        quant_layout=quant_layout,
    )
    shapes = {key: tuple(value.shape) for key, value in t5_state.items()}

    def tiny_map(gguf_source: GGUFSource) -> GGUFComponentMap:
        tensors = {
            names[name]: GGUFComponentTensor(
                model_key=names[name],
                source_name=name,
                logical_shape=shapes[names[name]],
                ggml_type=tensor.ggml_type,
                offset=tensor.offset,
                nbytes=tensor.nbytes,
            )
            for name, tensor in gguf_source.tensors.items()
        }
        return GGUFComponentMap(
            mapper_id="dinkster.gguf.text.v1",
            architecture="t5encoder",
            family_id="dinkster.text.t5xxl",
            component="t5xxl",
            tensor_prefix="",
            tensors=tensors,
        )

    monkeypatch.setattr("dinkster_inference.gguf.map_gguf_component", tiny_map)
    return path


@pytest.mark.parametrize("layout_name", (*_ENCODED_TEST_LAYOUTS, "mixed"))
def test_flux_text_gguf_encoded_residency_matches_speed_bit_exactly(
    standard: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    layout_name: str,
) -> None:
    """Text components ride the same encoded-residency swap as
    diffusion: swapped T5 Linears decode the same float32 values the
    eager loader materializes, so every residency mode assembles a
    bit-identical encoder at the bfloat16 text compute default -
    for every encoded-resident layout, and for a checkpoint mixing
    them. The balanced mode additionally fills its budgeted decoded
    cache on the first forward, and a zero budget degrades to
    memory-mode behavior."""
    t5_plan, t5_state = standard["t5xxl"]
    quant_layout: GGMLType | Callable[[str], GGMLType]
    if layout_name == "mixed":
        names = {_llama_t5_name(key): key for key in t5_state}
        cycle = tuple(_ENCODED_TEST_LAYOUTS.values())
        assignment = {name: cycle[i % len(cycle)] for i, name in enumerate(sorted(names))}
        quant_layout = assignment.__getitem__
    else:
        quant_layout = _ENCODED_TEST_LAYOUTS[layout_name]
    path = _write_and_map_tiny_t5_gguf(t5_state, monkeypatch, tmp_path, quant_layout)

    assembled: dict[GGUFResidencyMode, T5TextModel] = {}
    modes: tuple[GGUFResidencyMode, ...] = ("speed", "memory", "balanced")
    for mode in modes:
        # An explicit balanced budget keeps the cache assertions
        # independent of this host's free memory.
        budget = (1 << 20) if mode == "balanced" else None
        authority = load_gguf_weight_source(path, residency_mode=mode, decoded_cache_budget=budget)
        plan = replace(
            t5_plan,
            path=path,
            source_format="gguf",
            runtime_facts=authority.runtime_facts,
            payload_source=authority,
        )
        flux = assemble_flux(
            make_plan(standard, t5xxl=plan),
            diffusion_dtype=torch.float32,
            text_dtype=torch.bfloat16,
        )
        assert flux.t5xxl is not None
        assembled[mode] = flux.t5xxl.eval()

    swapped = {
        name
        for name, module in assembled["memory"].named_modules()
        if isinstance(module, GgufEncodedLinear)
    }
    projections = {
        name
        for name, module in assembled["speed"].named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    assert swapped == projections
    assert len(swapped) == 14
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
    # Every swapped linear shares one per-component cache; memory-mode
    # swaps carry none.
    caches = {
        module.decoded_cache
        for module in assembled["balanced"].modules()
        if isinstance(module, GgufEncodedLinear)
    }
    assert len(caches) == 1
    (cache,) = caches
    assert cache is not None
    assert cache.used_bytes == 0
    assert all(
        module.decoded_cache is None
        for module in assembled["memory"].modules()
        if isinstance(module, GgufEncodedLinear)
    )

    ids = torch.tensor([[3, 1, 7, 42, 0, 0]])
    embeds = {mode: assembled[mode].embed_tokens(ids) for mode in modes}
    assert torch.equal(embeds["memory"], embeds["speed"])
    assert torch.equal(embeds["balanced"], embeds["speed"])
    with torch.no_grad():
        outputs = {mode: assembled[mode](embeds[mode]) for mode in modes}
    assert torch.equal(outputs["memory"], outputs["speed"])
    assert torch.equal(outputs["balanced"], outputs["speed"])
    # The first forward filled the decoded cache with bfloat16 weights.
    assert cache.budget_bytes == 1 << 20
    assert 0 < cache.used_bytes <= 1 << 20

    # A zero budget keeps encoded residency but caches nothing.
    pinched_authority = load_gguf_weight_source(
        path, residency_mode="balanced", decoded_cache_budget=0
    )
    pinched_plan = replace(
        t5_plan,
        path=path,
        source_format="gguf",
        runtime_facts=pinched_authority.runtime_facts,
        payload_source=pinched_authority,
    )
    pinched_flux = assemble_flux(
        make_plan(standard, t5xxl=pinched_plan),
        diffusion_dtype=torch.float32,
        text_dtype=torch.bfloat16,
    )
    assert pinched_flux.t5xxl is not None
    pinched = pinched_flux.t5xxl.eval()
    pinched_caches = {
        module.decoded_cache
        for module in pinched.modules()
        if isinstance(module, GgufEncodedLinear)
    }
    (pinched_cache,) = pinched_caches
    assert pinched_cache is not None
    with torch.no_grad():
        assert torch.equal(pinched(pinched.embed_tokens(ids)), outputs["speed"])
    assert pinched_cache.used_bytes == 0


def test_gguf_residency_modes_use_decode_routes(
    standard: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Both GGUF residency modes assemble encoded decode-route linears."""

    t5_plan, t5_state = standard["t5xxl"]
    path = _write_and_map_tiny_t5_gguf(t5_state, monkeypatch, tmp_path, Q8_0)

    encoded: dict[GGUFResidencyMode, dict[str, GgufEncodedLinear]] = {}
    modes: tuple[GGUFResidencyMode, ...] = ("memory", "balanced")
    for mode in modes:
        authority = load_gguf_weight_source(path, residency_mode=mode)
        plan = replace(
            t5_plan,
            path=path,
            source_format="gguf",
            runtime_facts=authority.runtime_facts,
            payload_source=authority,
        )
        flux = assemble_flux(
            make_plan(standard, t5xxl=plan),
            diffusion_dtype=torch.float32,
            text_dtype=torch.bfloat16,
        )
        assert flux.t5xxl is not None
        encoded[mode] = {
            name: module
            for name, module in flux.t5xxl.named_modules()
            if isinstance(module, GgufEncodedLinear)
        }
        assert len(encoded[mode]) == 14
