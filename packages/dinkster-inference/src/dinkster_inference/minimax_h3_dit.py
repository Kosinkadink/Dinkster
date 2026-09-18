"""Exact torch-free MiniMax H3 DiT layout and inert assembly planning.

The state layout follows ComfyUI's MiniMax H3 module at commit
``2a68ce33b4c9ea6ee4283e618a74560cefb32694`` with the exact staged H3
profile detected by :mod:`dinkster_inference.minimax_h3`. Planning accepts only
one complete bare or ``model.diffusion_model.``-prefixed diffusion source.
It does not register a family or provide an executable runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

from dinkster_protocol import AttentionPolicy, AttentionRouteToken, validate_attention_policy

from .devices import FLOAT32, DType
from .minimax_h3 import MINIMAX_H3_CONFIG, MiniMaxH3Config
from .refusal import NativeRefusalCategory, NativeRefusalError
from .weights import TensorGeometry, WeightSource

_PREFIXES = ("", "model.diffusion_model.")
_TIME_EMBED_DIM = 2688
_TOKEN_REFINER_DEPTH = 2
_ROPE_AXIS_DIM = 16

MiniMaxH3DiTRole = Literal["fl2va-dit", "ref2va-dit"]

MINIMAX_H3_DIT_PROVIDER_REVISION = "3f57e8291d2ef846f9a074b1b76d2767db434abe"


def _validate_minimax_h3_dit_role(role: MiniMaxH3DiTRole) -> None:
    if role not in ("fl2va-dit", "ref2va-dit"):
        raise ValueError(f"unsupported MiniMax H3 DiT role {role!r}")


class MiniMaxH3DiTAssemblyError(ValueError):
    """The source is not the exact unpacked MiniMax H3 DiT layout."""


class MiniMaxH3DiTExecutionRefusal(NativeRefusalError):
    """A typed refusal for an H3 format with no runtime provider."""

    def __init__(self, execution_format: str, runtime_provider: str | None = None) -> None:
        format_obj = cast("object", execution_format)
        if not isinstance(format_obj, str) or not execution_format:
            raise TypeError("execution_format must be a non-empty string")
        if runtime_provider is not None:
            raise ValueError("MiniMax H3 has no supported runtime provider")
        self.execution_format = execution_format
        self.runtime_provider = runtime_provider
        super().__init__(
            (
                f"MiniMax H3 execution format {execution_format!r} has no"
                " supported runtime provider",
            ),
            NativeRefusalCategory.NATIVE_INELIGIBLE,
        )


def minimax_h3_dit_provider_facts(
    role: MiniMaxH3DiTRole,
    *,
    quantized: bool,
    torch_version: str,
    comfy_kitchen_version: str | None = None,
    attention_policy: AttentionPolicy = "auto",
) -> tuple[str, ...]:
    """Build behavior-bearing provider facts shared by dispatch and assembly."""

    _validate_minimax_h3_dit_role(role)
    if type(quantized) is not bool:
        raise TypeError("MiniMax H3 quantized provider selection must be boolean")
    if type(torch_version) is not str or not torch_version:
        raise ValueError("MiniMax H3 provider facts require a torch version")
    validate_attention_policy(attention_policy)
    kitchen_attention_provider = {
        "comfy_kitchen_int8": "comfy-kitchen.int8_attention",
        "sol": "comfy-kitchen.sol_attn",
    }.get(attention_policy)
    if (quantized or kitchen_attention_provider is not None) and (
        type(comfy_kitchen_version) is not str or not comfy_kitchen_version
    ):
        raise ValueError(
            "MiniMax H3 provider facts require a comfy-kitchen version for comfy-kitchen providers"
        )
    return (
        f"provider_revision={MINIMAX_H3_DIT_PROVIDER_REVISION}",
        *(
            (
                "int8_provider=comfy-kitchen.int8_linear",
                f"comfy_kitchen_version={comfy_kitchen_version}",
            )
            if quantized
            else ()
        ),
        *(
            (
                f"attention_provider={kitchen_attention_provider}",
                *(() if quantized else (f"comfy_kitchen_version={comfy_kitchen_version}",)),
            )
            if kitchen_attention_provider is not None
            else ("attention_provider=torch-sdpa",)
            if attention_policy in ("auto", "sdpa")
            else (f"attention_provider={attention_policy}",)
        ),
        f"torch_version={torch_version}",
        f"artifact_role={role}",
    )


def minimax_h3_dit_runtime_identity(
    *,
    asset_digest: str,
    asset_size: int,
    role: MiniMaxH3DiTRole,
    diffusion_dtype: str,
    attention_policy: AttentionPolicy = "auto",
    attention_route_token: AttentionRouteToken | None = None,
    runtime_facts: Sequence[str] = (),
) -> str:
    """Build the native identity for one ordinary H3 MODEL asset."""

    if type(asset_digest) is not str or not asset_digest:
        raise ValueError("MiniMax H3 model identity requires an asset digest")
    if type(asset_size) is not int or asset_size < 0:
        raise ValueError("MiniMax H3 model identity requires a nonnegative asset size")
    _validate_minimax_h3_dit_role(role)
    if type(diffusion_dtype) is not str or not diffusion_dtype:
        raise ValueError("MiniMax H3 model identity requires a diffusion dtype")
    facts = tuple(runtime_facts)
    if any(type(fact) is not str or not fact for fact in facts):
        raise ValueError("MiniMax H3 model runtime facts must be non-empty strings")
    from .identity import build_runtime_identity_from_facts

    return build_runtime_identity_from_facts(
        MINIMAX_H3_CONFIG.family_id,
        minimax_h3_dit_component_identity(asset_digest, asset_size, role),
        diffusion_dtype=diffusion_dtype,
        text_dtype="unloaded",
        vae_dtype="unloaded",
        fp8_matmul=False,
        attention_policy=attention_policy,
        attention_route_token=attention_route_token,
        runtime_facts=facts,
    )


def minimax_h3_dit_component_identity(
    asset_digest: str,
    asset_size: int,
    role: MiniMaxH3DiTRole,
) -> tuple[str, ...]:
    """Return the path-free component facts retained by a model recipe."""

    if type(asset_digest) is not str or not asset_digest:
        raise ValueError("MiniMax H3 model identity requires an asset digest")
    if type(asset_size) is not int or asset_size < 0:
        raise ValueError("MiniMax H3 model identity requires a nonnegative asset size")
    _validate_minimax_h3_dit_role(role)
    return (
        f"family={MINIMAX_H3_CONFIG.family_id}",
        "component=diffusion",
        f"artifact_role={role}",
        f"asset_digest={asset_digest}",
        f"asset_size={asset_size}",
    )


def _linear(
    keys: dict[str, tuple[int, ...]],
    name: str,
    out_features: int,
    in_features: int,
    *,
    bias: bool,
) -> None:
    keys[f"{name}.weight"] = (out_features, in_features)
    if bias:
        keys[f"{name}.bias"] = (out_features,)


def _attention(keys: dict[str, tuple[int, ...]], name: str, config: MiniMaxH3Config) -> None:
    inner = config.attention_heads * config.attention_head_dim
    _linear(keys, f"{name}.qkv_proj", inner * 3, config.hidden_width, bias=False)
    keys[f"{name}.q_norm.weight"] = (config.attention_head_dim,)
    keys[f"{name}.k_norm.weight"] = (config.attention_head_dim,)
    _linear(keys, f"{name}.out_proj", config.hidden_width, inner, bias=False)


def _mlp(keys: dict[str, tuple[int, ...]], name: str, config: MiniMaxH3Config) -> None:
    _linear(keys, f"{name}.fc1", config.ffn_width * 2, config.hidden_width, bias=False)
    _linear(keys, f"{name}.fc2", config.hidden_width, config.ffn_width, bias=False)


MiniMaxH3TimeEmbeddingKind = Literal["curve", "mlp"]


def _minimax_h3_dit_keys(
    config: MiniMaxH3Config, time_embedding_kind: MiniMaxH3TimeEmbeddingKind
) -> dict[str, tuple[int, ...]]:
    keys: dict[str, tuple[int, ...]] = {}
    video_patch_width = (
        config.video_latent_channels * config.patch[0] * config.patch[1] * config.patch[2]
    )
    _linear(
        keys,
        "video_patch_proj",
        config.hidden_width,
        video_patch_width,
        bias=True,
    )
    _linear(
        keys,
        "audio_patch_proj",
        config.hidden_width,
        config.audio_latent_channels,
        bias=True,
    )
    _linear(
        keys,
        "condition_proj",
        config.hidden_width,
        config.text_width,
        bias=True,
    )
    if time_embedding_kind == "curve":
        keys["adaln_t_table"] = (1000, _TIME_EMBED_DIM)
    else:
        _linear(keys, "time_embedder.proj_in", config.hidden_width, 256, bias=True)
        _linear(
            keys,
            "time_embedder.proj_out",
            _TIME_EMBED_DIM,
            config.hidden_width,
            bias=True,
        )
    keys["rope.inv_freq"] = (_ROPE_AXIS_DIM,)

    for index in range(_TOKEN_REFINER_DEPTH):
        root = f"token_refiner.blocks.{index}"
        keys[f"{root}.norm1.weight"] = (config.hidden_width,)
        keys[f"{root}.norm2.weight"] = (config.hidden_width,)
        _attention(keys, f"{root}.attn", config)
        _mlp(keys, f"{root}.mlp", config)
    keys["token_refiner.final_norm.weight"] = (config.hidden_width,)

    adaln_width = 6 * 3 * config.hidden_width
    for index in range(config.depth):
        root = f"blocks.{index}"
        keys[f"{root}.norm1.weight"] = (config.hidden_width,)
        keys[f"{root}.norm2.weight"] = (config.hidden_width,)
        _attention(keys, f"{root}.attn", config)
        _mlp(keys, f"{root}.mlp", config)
        _linear(
            keys,
            f"{root}.adaln_proj.linear",
            adaln_width,
            _TIME_EMBED_DIM,
            bias=True,
        )

    keys["final_layer.norm.weight"] = (config.hidden_width,)
    _linear(
        keys,
        "final_layer.adaln_proj.linear",
        2 * config.hidden_width,
        _TIME_EMBED_DIM,
        bias=True,
    )
    _linear(
        keys,
        "final_layer.video_out",
        video_patch_width,
        config.hidden_width,
        bias=True,
    )
    _linear(
        keys,
        "final_layer.audio_out",
        config.audio_latent_channels,
        config.hidden_width,
        bias=True,
    )
    return keys


def _minimax_h3_fp32_storage_keys(
    config: MiniMaxH3Config, time_embedding_kind: MiniMaxH3TimeEmbeddingKind
) -> frozenset[str]:
    keys = {
        "video_patch_proj.weight",
        "video_patch_proj.bias",
        "audio_patch_proj.weight",
        "audio_patch_proj.bias",
        "rope.inv_freq",
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
    }
    if time_embedding_kind == "curve":
        keys.add("adaln_t_table")
        keys.add("final_layer.adaln_proj.linear.weight")
        keys.add("final_layer.adaln_proj.linear.bias")
        for index in range(config.depth):
            root = f"blocks.{index}.adaln_proj.linear"
            keys.add(f"{root}.weight")
            keys.add(f"{root}.bias")
    else:
        keys.update(
            {
                "time_embedder.proj_in.weight",
                "time_embedder.proj_in.bias",
                "time_embedder.proj_out.weight",
                "time_embedder.proj_out.bias",
            }
        )
    return frozenset(keys)


@dataclass(frozen=True)
class MiniMaxH3DiTLayout:
    """Complete state and packed-attention facts for the exact H3 DiT."""

    config: MiniMaxH3Config
    keys: Mapping[str, tuple[int, ...]] = field(repr=False)
    fp32_storage_keys: frozenset[str] = field(repr=False)
    time_embedding_kind: MiniMaxH3TimeEmbeddingKind = "curve"
    depth: int = field(default=50, init=False)
    hidden_width: int = field(default=5376, init=False)
    attention_heads: int = field(default=56, init=False)
    attention_head_dim: int = field(default=128, init=False)
    ffn_width: int = field(default=14336, init=False)
    video_patch: tuple[int, int, int] = field(default=(1, 2, 2), init=False)
    video_patch_width: int = field(default=96, init=False)
    audio_patch_width: int = field(default=32, init=False)
    attention_kind: Literal["full"] = field(default="full", init=False)
    attention_mask: None = field(default=None, init=False)
    rope_axes: tuple[str, str, str] = field(default=("time", "height", "width"), init=False)
    rope_axis_dim: int = field(default=16, init=False)
    rope_rotary_dim: int = field(default=96, init=False)
    rope_style: Literal["split_half"] = field(default="split_half", init=False)
    sampler_stream_order: tuple[str, str] = field(default=("video", "audio"), init=False)
    packed_target_order: tuple[str, str] = field(default=("audio", "video"), init=False)

    def __post_init__(self) -> None:
        if not isinstance(cast("object", self.config), MiniMaxH3Config):
            raise TypeError("layout config must be MiniMaxH3Config")
        if self.config != MINIMAX_H3_CONFIG:
            raise ValueError("layout config must be the exact H3 profile")
        keys_obj = cast("object", self.keys)
        if not isinstance(keys_obj, Mapping):
            raise TypeError("layout keys must be a mapping")
        frozen = dict(cast("Mapping[str, tuple[int, ...]]", keys_obj))
        if self.time_embedding_kind not in ("curve", "mlp"):
            raise ValueError("time_embedding_kind must be curve or mlp")
        expected = _minimax_h3_dit_keys(MINIMAX_H3_CONFIG, self.time_embedding_kind)
        if frozen != expected:
            raise ValueError("keys must equal the exact H3 DiT layout")
        fp32_keys_obj = cast("object", self.fp32_storage_keys)
        if not isinstance(fp32_keys_obj, frozenset):
            raise TypeError("fp32_storage_keys must be a frozenset")
        expected_fp32_keys = _minimax_h3_fp32_storage_keys(
            MINIMAX_H3_CONFIG, self.time_embedding_kind
        )
        if fp32_keys_obj != expected_fp32_keys:
            raise ValueError("fp32_storage_keys must equal the exact H3 FP32 islands")
        object.__setattr__(self, "config", MINIMAX_H3_CONFIG)
        object.__setattr__(self, "keys", MappingProxyType(frozen))
        object.__setattr__(self, "fp32_storage_keys", expected_fp32_keys)


def minimax_h3_dit_layout(
    config: MiniMaxH3Config = MINIMAX_H3_CONFIG,
    *,
    time_embedding_kind: MiniMaxH3TimeEmbeddingKind = "curve",
) -> MiniMaxH3DiTLayout:
    """Return one exact H3 DiT state layout."""
    if not isinstance(cast("object", config), MiniMaxH3Config):
        raise TypeError("config must be MiniMaxH3Config")
    if config != MINIMAX_H3_CONFIG:
        raise ValueError("config must be the exact H3 profile")
    if time_embedding_kind not in ("curve", "mlp"):
        raise ValueError("time_embedding_kind must be curve or mlp")
    return MiniMaxH3DiTLayout(
        config,
        _minimax_h3_dit_keys(config, time_embedding_kind),
        _minimax_h3_fp32_storage_keys(config, time_embedding_kind),
        time_embedding_kind,
    )


@dataclass(frozen=True)
class MiniMaxH3DiTAssemblyPlan:
    """Deterministic complete-key plan for one inert H3 diffusion source."""

    layout: MiniMaxH3DiTLayout
    source_prefix: str
    keys: Mapping[str, str]
    dtypes: Mapping[str, DType]
    claims: tuple[str, ...]
    execution_format: Literal["bfloat16", "float32"]
    family_id: str = field(default="dinkster.minimax_h3", init=False)
    source_role: str = field(default="diffusion", init=False)
    runtime_provider: None = field(default=None, init=False)
    runnable: Literal[False] = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.layout != minimax_h3_dit_layout(
            time_embedding_kind=self.layout.time_embedding_kind
        ):
            raise ValueError("assembly plan must use an exact H3 DiT layout")
        if self.source_prefix not in _PREFIXES:
            raise ValueError("H3 source prefix must be bare or model.diffusion_model.")
        keys = dict(self.keys)
        dtypes = dict(self.dtypes)
        expected = {key: self.source_prefix + key for key in self.layout.keys}
        if keys != expected:
            raise ValueError("H3 plan must map every layout key exactly once")
        if set(dtypes) != set(expected):
            raise ValueError("H3 plan dtypes must cover every layout key")
        if any(dtype.kind != "float" for dtype in dtypes.values()):
            raise MiniMaxH3DiTAssemblyError("H3 plan weights require floating storage")
        if not isinstance(cast("object", self.claims), tuple):
            raise TypeError("H3 claims must be a tuple")
        if self.claims != tuple(sorted(expected.values())):
            raise ValueError("H3 claims must exactly cover every source key")
        if self.execution_format not in ("bfloat16", "float32"):
            raise MiniMaxH3DiTAssemblyError("H3 execution format must be bfloat16 or float32")
        expected_format = "float32" if set(dtypes.values()) == {FLOAT32} else "bfloat16"
        if self.execution_format != expected_format:
            raise MiniMaxH3DiTAssemblyError(
                f"H3 execution format {self.execution_format} does not match floating storage"
            )
        object.__setattr__(self, "keys", MappingProxyType(keys))
        object.__setattr__(self, "dtypes", MappingProxyType(dtypes))

    def execution_refusal(self) -> MiniMaxH3DiTExecutionRefusal:
        """Return the typed reason this inert plan cannot execute."""
        return MiniMaxH3DiTExecutionRefusal(self.execution_format)


def _source_prefix(source_keys: tuple[str, ...], layout: MiniMaxH3DiTLayout) -> str:
    if not source_keys:
        raise MiniMaxH3DiTAssemblyError("empty MiniMax H3 DiT source")
    actual = set(source_keys)
    bare = set(layout.keys)
    prefixed = {"model.diffusion_model." + key for key in layout.keys}
    if actual == bare:
        return ""
    if actual == prefixed:
        return "model.diffusion_model."
    has_bare = bool(actual & bare)
    has_prefixed = bool(actual & prefixed)
    if has_bare and has_prefixed:
        raise MiniMaxH3DiTAssemblyError("mixed bare and model.diffusion_model.-prefixed H3 keys")
    expected = prefixed if has_prefixed else bare
    missing = tuple(sorted(expected - actual))
    foreign = tuple(sorted(actual - expected))
    if missing:
        raise MiniMaxH3DiTAssemblyError("missing required H3 DiT keys: " + ", ".join(missing[:3]))
    raise MiniMaxH3DiTAssemblyError("foreign or leftover H3 DiT keys: " + ", ".join(foreign[:3]))


def _entry_geometry(source: WeightSource, key: str) -> TensorGeometry:
    try:
        return source.entry(key).geometry
    except KeyError as error:
        raise MiniMaxH3DiTAssemblyError(
            f"inconsistent H3 source omitted advertised key {key}"
        ) from error


def plan_minimax_h3_dit_assembly(
    source: WeightSource,
) -> MiniMaxH3DiTAssemblyPlan:
    """Plan one exact bare or prefixed H3 DiT without reading payloads."""
    source_keys = tuple(source.keys())
    if len(source_keys) != len(set(source_keys)):
        raise MiniMaxH3DiTAssemblyError("duplicate H3 DiT source keys")
    layouts = tuple(minimax_h3_dit_layout(time_embedding_kind=kind) for kind in ("curve", "mlp"))
    for layout in layouts:
        actual = set(source_keys)
        if actual in (
            set(layout.keys),
            {"model.diffusion_model." + key for key in layout.keys},
        ):
            break
    else:
        layout = layouts[0]
    prefix = _source_prefix(source_keys, layout)
    model_to_source = {key: prefix + key for key in layout.keys}
    dtypes: dict[str, DType] = {}
    for model_key, source_key in model_to_source.items():
        geometry = _entry_geometry(source, source_key)
        expected = layout.keys[model_key]
        if geometry.shape != expected:
            raise MiniMaxH3DiTAssemblyError(
                f"geometry mismatch for {source_key}: got {geometry.shape}, expected {expected}"
            )
        if geometry.dtype.kind != "float":
            raise MiniMaxH3DiTAssemblyError(
                f"H3 weight {source_key} requires floating storage, got {geometry.dtype.name}"
            )
        dtypes[model_key] = geometry.dtype
    execution_format: Literal["bfloat16", "float32"] = (
        "float32" if set(dtypes.values()) == {FLOAT32} else "bfloat16"
    )
    return MiniMaxH3DiTAssemblyPlan(
        layout=layout,
        source_prefix=prefix,
        keys=model_to_source,
        dtypes=dtypes,
        claims=tuple(sorted(source_keys)),
        execution_format=execution_format,
    )


__all__ = [
    "MINIMAX_H3_DIT_PROVIDER_REVISION",
    "MiniMaxH3DiTRole",
    "MiniMaxH3DiTAssemblyError",
    "MiniMaxH3DiTAssemblyPlan",
    "MiniMaxH3DiTExecutionRefusal",
    "MiniMaxH3DiTLayout",
    "MiniMaxH3TimeEmbeddingKind",
    "minimax_h3_dit_component_identity",
    "minimax_h3_dit_provider_facts",
    "minimax_h3_dit_runtime_identity",
    "minimax_h3_dit_layout",
    "plan_minimax_h3_dit_assembly",
]
