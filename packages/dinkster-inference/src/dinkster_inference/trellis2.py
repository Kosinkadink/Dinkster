"""Torch-free TRELLIS.2 component contracts.

The contracts match Microsoft TRELLIS.2 revision
75fbf0183001ed9876c8dbb35de6b68552ee08bd and ComfyUI revision
8a33128f2f8c5585c57486c07de481241e70a39c. TRELLIS.2 and Pixal3D
share flow dimensions, while Pixal3D adds projected image features to
every cross-attention block. Split Microsoft flow checkpoints and the
fused ComfyUI checkpoints use the same model keys once their component
prefix is removed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .weights import TensorGeometry

TRELLIS2_FAMILY_ID = "dinkster.trellis2"


class Trellis2DetectError(ValueError):
    """A checkpoint does not match an exact TRELLIS.2 component."""


Trellis2FlowStage = Literal["structure", "shape", "texture"]
Trellis2ImageAttention = Literal["global", "projected"]
Trellis2DecoderKind = Literal["shape", "texture", "structure"]


@dataclass(frozen=True, slots=True)
class Trellis2FlowConfig:
    stage: Trellis2FlowStage
    image_attention: Trellis2ImageAttention
    in_channels: int
    out_channels: int
    projected_channels: int | None
    model_channels: int = 1536
    condition_channels: int = 1024
    num_blocks: int = 30
    num_heads: int = 12
    head_channels: int = 128
    mlp_channels: int = 8192
    timestep_channels: int = 256


@dataclass(frozen=True, slots=True)
class Trellis2DecoderConfig:
    kind: Trellis2DecoderKind
    latent_channels: int
    out_channels: int


def _flow_config(
    stage: Trellis2FlowStage, image_attention: Trellis2ImageAttention
) -> Trellis2FlowConfig:
    channels = {
        "structure": (8, 8),
        "shape": (32, 32),
        "texture": (64, 32),
    }
    in_channels, out_channels = channels[stage]
    projected = None
    if image_attention == "projected":
        projected = 1024 if stage == "structure" else 2048
    return Trellis2FlowConfig(stage, image_attention, in_channels, out_channels, projected)


def trellis2_flow_layout(config: Trellis2FlowConfig) -> dict[str, tuple[int, ...]]:
    """Return the exact flow-transformer key and shape listing."""
    hidden = config.model_channels
    layout: dict[str, tuple[int, ...]] = {
        "t_embedder.mlp.0.weight": (hidden, config.timestep_channels),
        "t_embedder.mlp.0.bias": (hidden,),
        "t_embedder.mlp.2.weight": (hidden, hidden),
        "t_embedder.mlp.2.bias": (hidden,),
        "adaLN_modulation.1.weight": (hidden * 6, hidden),
        "adaLN_modulation.1.bias": (hidden * 6,),
        "input_layer.weight": (hidden, config.in_channels),
        "input_layer.bias": (hidden,),
        "out_layer.weight": (config.out_channels, hidden),
        "out_layer.bias": (config.out_channels,),
    }
    for index in range(config.num_blocks):
        prefix = f"blocks.{index}."
        layout[f"{prefix}modulation"] = (hidden * 6,)
        layout[f"{prefix}norm2.weight"] = (hidden,)
        layout[f"{prefix}norm2.bias"] = (hidden,)
        layout[f"{prefix}self_attn.to_qkv.weight"] = (hidden * 3, hidden)
        layout[f"{prefix}self_attn.to_qkv.bias"] = (hidden * 3,)
        layout[f"{prefix}self_attn.q_rms_norm.gamma"] = (
            config.num_heads,
            config.head_channels,
        )
        layout[f"{prefix}self_attn.k_rms_norm.gamma"] = (
            config.num_heads,
            config.head_channels,
        )
        layout[f"{prefix}self_attn.to_out.weight"] = (hidden, hidden)
        layout[f"{prefix}self_attn.to_out.bias"] = (hidden,)
        cross = f"{prefix}cross_attn."
        if config.image_attention == "projected":
            assert config.projected_channels is not None
            layout[f"{cross}proj_linear.weight"] = (hidden, config.projected_channels)
            layout[f"{cross}proj_linear.bias"] = (hidden,)
            cross += "cross_attn_block."
        layout[f"{cross}to_q.weight"] = (hidden, hidden)
        layout[f"{cross}to_q.bias"] = (hidden,)
        layout[f"{cross}to_kv.weight"] = (hidden * 2, config.condition_channels)
        layout[f"{cross}to_kv.bias"] = (hidden * 2,)
        layout[f"{cross}q_rms_norm.gamma"] = (config.num_heads, config.head_channels)
        layout[f"{cross}k_rms_norm.gamma"] = (config.num_heads, config.head_channels)
        layout[f"{cross}to_out.weight"] = (hidden, hidden)
        layout[f"{cross}to_out.bias"] = (hidden,)
        layout[f"{prefix}mlp.mlp.0.weight"] = (config.mlp_channels, hidden)
        layout[f"{prefix}mlp.mlp.0.bias"] = (config.mlp_channels,)
        layout[f"{prefix}mlp.mlp.2.weight"] = (hidden, config.mlp_channels)
        layout[f"{prefix}mlp.mlp.2.bias"] = (hidden,)
    return layout


def _sparse_decoder_layout(*, predict_subdivision: bool) -> dict[str, tuple[int, ...]]:
    channels = (1024, 512, 256, 128, 64)
    blocks = (4, 16, 8, 4, 0)
    layout: dict[str, tuple[int, ...]] = {
        "from_latent.weight": (channels[0], 32),
        "from_latent.bias": (channels[0],),
        "output_layer.weight": ((7 if predict_subdivision else 6), channels[-1]),
        "output_layer.bias": ((7 if predict_subdivision else 6),),
    }
    for stage, (channel, count) in enumerate(zip(channels, blocks, strict=True)):
        for index in range(count):
            prefix = f"blocks.{stage}.{index}."
            layout[f"{prefix}conv.weight"] = (channel, 3, 3, 3, channel)
            layout[f"{prefix}conv.bias"] = (channel,)
            layout[f"{prefix}norm.weight"] = (channel,)
            layout[f"{prefix}norm.bias"] = (channel,)
            layout[f"{prefix}mlp.0.weight"] = (channel * 4, channel)
            layout[f"{prefix}mlp.0.bias"] = (channel * 4,)
            layout[f"{prefix}mlp.2.weight"] = (channel, channel * 4)
            layout[f"{prefix}mlp.2.bias"] = (channel,)
        if stage == len(channels) - 1:
            continue
        prefix = f"blocks.{stage}.{count}."
        out_channel = channels[stage + 1]
        layout[f"{prefix}norm1.weight"] = (channel,)
        layout[f"{prefix}norm1.bias"] = (channel,)
        layout[f"{prefix}conv1.weight"] = (out_channel * 8, 3, 3, 3, channel)
        layout[f"{prefix}conv1.bias"] = (out_channel * 8,)
        layout[f"{prefix}conv2.weight"] = (out_channel, 3, 3, 3, out_channel)
        layout[f"{prefix}conv2.bias"] = (out_channel,)
        if predict_subdivision:
            layout[f"{prefix}to_subdiv.weight"] = (8, channel)
            layout[f"{prefix}to_subdiv.bias"] = (8,)
    return layout


def trellis2_structure_decoder_layout() -> dict[str, tuple[int, ...]]:
    """Return the exact dense occupancy decoder layout."""
    layout: dict[str, tuple[int, ...]] = {
        "input_layer.weight": (512, 8, 3, 3, 3),
        "input_layer.bias": (512,),
    }

    def add_resblock(prefix: str, channels: int) -> None:
        for norm in ("norm1", "norm2"):
            layout[f"{prefix}{norm}.weight"] = (channels,)
            layout[f"{prefix}{norm}.bias"] = (channels,)
        for conv in ("conv1", "conv2"):
            layout[f"{prefix}{conv}.weight"] = (channels, channels, 3, 3, 3)
            layout[f"{prefix}{conv}.bias"] = (channels,)

    add_resblock("middle_block.0.", 512)
    add_resblock("middle_block.1.", 512)
    block_index = 0
    for stage, channels in enumerate((512, 128, 32)):
        for _ in range(2):
            add_resblock(f"blocks.{block_index}.", channels)
            block_index += 1
        if stage < 2:
            out_channels = (128, 32)[stage]
            layout[f"blocks.{block_index}.conv.weight"] = (
                out_channels * 8,
                channels,
                3,
                3,
                3,
            )
            layout[f"blocks.{block_index}.conv.bias"] = (out_channels * 8,)
            block_index += 1
    layout["out_layer.0.weight"] = (32,)
    layout["out_layer.0.bias"] = (32,)
    layout["out_layer.2.weight"] = (1, 32, 3, 3, 3)
    layout["out_layer.2.bias"] = (1,)
    return layout


def trellis2_decoder_layout(kind: Trellis2DecoderKind) -> dict[str, tuple[int, ...]]:
    if kind == "shape":
        sparse = _sparse_decoder_layout(predict_subdivision=True)
        layout = {f"shape_dec.{key}": shape for key, shape in sparse.items()}
        layout.update(
            {
                f"struct_dec.{key}": shape
                for key, shape in trellis2_structure_decoder_layout().items()
            }
        )
        return layout
    if kind == "texture":
        return {
            f"txt_dec.{key}": shape
            for key, shape in _sparse_decoder_layout(predict_subdivision=False).items()
        }
    return trellis2_structure_decoder_layout()


def _require_layout(
    geometries: Mapping[str, TensorGeometry],
    expected: Mapping[str, tuple[int, ...]],
    *,
    component: str,
) -> None:
    problems: list[str] = []
    for key, shape in expected.items():
        found = geometries.get(key)
        if found is None:
            problems.append(f"missing {key}")
        elif found.shape != shape:
            problems.append(f"{key}: expected {shape}, found {found.shape}")
    problems.extend(f"unexpected key {key}" for key in sorted(set(geometries) - set(expected)))
    if problems:
        shown = "; ".join(problems[:6])
        if len(problems) > 6:
            shown += f"; and {len(problems) - 6} more"
        raise Trellis2DetectError(f"not an exact TRELLIS.2 {component}: {shown}")


def detect_trellis2_flow(
    geometries: Mapping[str, TensorGeometry],
) -> Trellis2FlowConfig:
    """Detect stage and global/projected conditioning from exact geometry."""
    input_weight = geometries.get("input_layer.weight")
    output_weight = geometries.get("out_layer.weight")
    if input_weight is None or output_weight is None:
        raise Trellis2DetectError("not a TRELLIS.2 flow (missing input/output layers)")
    channel_pair = (input_weight.shape, output_weight.shape)
    stage: Trellis2FlowStage
    if channel_pair == ((1536, 8), (8, 1536)):
        stage = "structure"
    elif channel_pair == ((1536, 32), (32, 1536)):
        stage = "shape"
    elif channel_pair == ((1536, 64), (32, 1536)):
        stage = "texture"
    else:
        raise Trellis2DetectError(f"not a TRELLIS.2 flow (channel geometry {channel_pair})")
    attention: Trellis2ImageAttention = (
        "projected" if "blocks.0.cross_attn.proj_linear.weight" in geometries else "global"
    )
    config = _flow_config(stage, attention)
    _require_layout(geometries, trellis2_flow_layout(config), component=f"{stage} flow")
    return config


def detect_trellis2_decoder(
    geometries: Mapping[str, TensorGeometry],
) -> Trellis2DecoderConfig:
    """Detect an exact fused or split TRELLIS.2 decoder."""
    if "shape_dec.from_latent.weight" in geometries:
        kind: Trellis2DecoderKind = "shape"
        config = Trellis2DecoderConfig(kind, 32, 7)
        expected = trellis2_decoder_layout(kind)
    elif "txt_dec.from_latent.weight" in geometries:
        kind = "texture"
        config = Trellis2DecoderConfig(kind, 32, 6)
        expected = trellis2_decoder_layout(kind)
    elif geometries.get("output_layer.weight") is not None:
        output_shape = geometries["output_layer.weight"].shape
        if output_shape == (7, 64):
            kind = "shape"
            config = Trellis2DecoderConfig(kind, 32, 7)
            expected = _sparse_decoder_layout(predict_subdivision=True)
        elif output_shape == (6, 64):
            kind = "texture"
            config = Trellis2DecoderConfig(kind, 32, 6)
            expected = _sparse_decoder_layout(predict_subdivision=False)
        else:
            raise Trellis2DetectError(
                f"not a TRELLIS.2 sparse decoder (output geometry {output_shape})"
            )
    elif "input_layer.weight" in geometries:
        kind = "structure"
        config = Trellis2DecoderConfig(kind, 8, 1)
        expected = trellis2_decoder_layout(kind)
    else:
        raise Trellis2DetectError("not a TRELLIS.2 decoder")
    _require_layout(geometries, expected, component=f"{kind} decoder")
    return config


__all__ = [
    "TRELLIS2_FAMILY_ID",
    "Trellis2DecoderConfig",
    "Trellis2DecoderKind",
    "Trellis2DetectError",
    "Trellis2FlowConfig",
    "Trellis2FlowStage",
    "Trellis2ImageAttention",
    "detect_trellis2_decoder",
    "detect_trellis2_flow",
    "trellis2_decoder_layout",
    "trellis2_flow_layout",
    "trellis2_structure_decoder_layout",
]
