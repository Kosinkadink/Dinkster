"""SD1/SDXL diffusion UNet: torch-free config, layout, detection.

The reference infers this architecture from state-dict geometry
(comfy/model_detection.py detect_unet_config @ b78cec87, the
``input_blocks.0.0.weight`` leg) and constructs it in
comfy/ldm/modules/diffusionmodules/openaimodel.py UNetModel.
:class:`UNetConfig` carries exactly the fields that construction
consumes for the supported 2D image UNets; the torch modules
(dinkster_inference_torch.unet) construct from that config through the
typed Operations seam with state-dict keys IDENTICAL to the
reference.

Attention-head counts are NOT derivable from tensor shapes (q/k/v
projections are full width). The reference resolves them by matching
the detected config against comfy/supported_models.py profiles
(unet_extra_config: SD1.x num_heads=8, SDXL-era num_head_channels=64);
detection here does the same through :data:`UNET_HEAD_PROFILES`,
keyed on the (context_dim, use_linear_in_transformer) pair that
separates the supported families.

Detection REJECTS rather than guesses: every family the reference's
detection would route elsewhere (MMDiT, Stable Cascade, audio DiT,
Flux - all missing ``input_blocks.0.0.weight``), temporal/video UNets
(SVD ``time_stack`` keys), SD2.x (context 1024 profile, whose
fp32-attention pin is not ported), and the pruned-SDXL distillates
(SSD1B/Segmind Vega/KOALA, ``transformer_depth_middle`` -1/-2)
raise :class:`UNetDetectError` naming what was found. Deferred
variants are ledgered in ROADMAP.md ("Native inference"), never
silently dropped. A config candidate that survives the scan is then
required to reproduce the ENTIRE key/shape listing
(:func:`unet_layout`); any drift refuses with the differences.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .weights import TensorGeometry


class UNetDetectError(ValueError):
    """The geometry mapping is not a supported SD1/SDXL diffusion
    UNet checkpoint; the message names what was found instead."""


#: (context_dim, use_linear_in_transformer) -> (num_heads,
#: num_head_channels), from comfy/supported_models.py
#: unet_extra_config @ b78cec87 (SD15 num_heads=8; SDXL and
#: SDXLRefiner num_head_channels=64). SD2.x (1024, True) is
#: deliberately absent: its fp32-attention pin is not ported
#: (ROADMAP: Native inference).
UNET_HEAD_PROFILES: Mapping[tuple[int, bool], tuple[int, int]] = MappingProxyType(
    {
        (768, False): (8, -1),
        (2048, True): (-1, 64),
        (1280, True): (-1, 64),
    }
)


@dataclass(frozen=True)
class UNetConfig:
    """The construction-relevant subset of the reference UNetModel
    arguments for supported 2D image UNets.

    ``transformer_depth`` carries one entry per input-side res block
    (the constructor consumes it left to right);
    ``transformer_depth_output`` one entry per output-side res block
    in INPUT-scan order (the constructor pops from the end, so the
    deepest level reads the tail). Exactly one of ``num_heads`` /
    ``num_head_channels`` is set, the other -1, mirroring the
    reference contract. ``adm_in_channels`` set means the
    ``"sequential"`` class-embedding MLP (the only num_classes mode a
    supported family uses)."""

    in_channels: int
    out_channels: int
    model_channels: int
    num_res_blocks: tuple[int, ...]
    channel_mult: tuple[int, ...]
    transformer_depth: tuple[int, ...]
    transformer_depth_output: tuple[int, ...]
    transformer_depth_middle: int
    context_dim: int
    use_linear_in_transformer: bool
    adm_in_channels: int | None = None
    num_heads: int = -1
    num_head_channels: int = -1
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("in_channels", "out_channels", "model_channels"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.model_channels % 2:
            raise ValueError(
                "model_channels must be even (timestep embedding"
                f" halves it), got {self.model_channels}"
            )
        if not self.num_res_blocks or any(n < 1 for n in self.num_res_blocks):
            raise ValueError(f"num_res_blocks entries must be >= 1, got {self.num_res_blocks}")
        if len(self.channel_mult) != len(self.num_res_blocks):
            raise ValueError(
                f"channel_mult has {len(self.channel_mult)} levels,"
                f" num_res_blocks has {len(self.num_res_blocks)}"
            )
        if any(m < 1 for m in self.channel_mult):
            raise ValueError(f"channel_mult entries must be >= 1, got {self.channel_mult}")
        if self.channel_mult[0] != 1:
            raise ValueError(
                "channel_mult must start at 1 (the reference output head"
                f" assumes it), got {self.channel_mult}"
            )
        if len(self.transformer_depth) != sum(self.num_res_blocks):
            raise ValueError(
                "transformer_depth needs one entry per input res block"
                f" ({sum(self.num_res_blocks)}), got"
                f" {len(self.transformer_depth)}"
            )
        expected_output = sum(n + 1 for n in self.num_res_blocks)
        if len(self.transformer_depth_output) != expected_output:
            raise ValueError(
                "transformer_depth_output needs one entry per output res"
                f" block ({expected_output}), got"
                f" {len(self.transformer_depth_output)}"
            )
        if any(d < 0 for d in self.transformer_depth) or any(
            d < 0 for d in self.transformer_depth_output
        ):
            raise ValueError("transformer depths must be >= 0")
        if self.transformer_depth_middle < 1:
            raise ValueError(
                "transformer_depth_middle must be >= 1; the reference's"
                " -1/-2 middle-block variants (SSD1B/Segmind Vega/KOALA)"
                " are not ported (ROADMAP: Native inference), got"
                f" {self.transformer_depth_middle}"
            )
        if self.context_dim < 1:
            raise ValueError(f"context_dim must be >= 1, got {self.context_dim}")
        if (self.num_heads == -1) == (self.num_head_channels == -1):
            raise ValueError(
                "exactly one of num_heads / num_head_channels must be set"
                f" (the other -1), got {self.num_heads} /"
                f" {self.num_head_channels}"
            )
        if self.num_heads != -1 and self.num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {self.num_heads}")
        if self.num_head_channels != -1 and self.num_head_channels < 1:
            raise ValueError(f"num_head_channels must be >= 1, got {self.num_head_channels}")
        if self.adm_in_channels is not None and self.adm_in_channels < 1:
            raise ValueError(f"adm_in_channels must be >= 1, got {self.adm_in_channels}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        for ch in self._attention_widths():
            self.heads_for(ch)

    def _attention_widths(self) -> set[int]:
        """Every channel width that carries a spatial transformer."""
        widths: set[int] = set()
        depths = list(self.transformer_depth)
        out_depths = list(self.transformer_depth_output)
        for level, mult in enumerate(self.channel_mult):
            ch = mult * self.model_channels
            for _ in range(self.num_res_blocks[level]):
                if depths.pop(0) > 0:
                    widths.add(ch)
            for _ in range(self.num_res_blocks[level] + 1):
                if out_depths.pop(0) > 0:
                    widths.add(ch)
        widths.add(self.channel_mult[-1] * self.model_channels)  # middle
        return widths

    @property
    def time_embed_dim(self) -> int:
        """The reference's fixed 4x widening of model_channels."""
        return self.model_channels * 4

    def heads_for(self, channels: int) -> tuple[int, int]:
        """(num_heads, dim_head) at a given channel width - the
        reference's per-site resolution with legacy=False."""
        if self.num_head_channels == -1:
            heads, rem = divmod(channels, self.num_heads)
            if rem or heads < 1:
                raise ValueError(f"{channels} channels do not divide into {self.num_heads} heads")
            return self.num_heads, channels // self.num_heads
        heads, rem = divmod(channels, self.num_head_channels)
        if rem or heads < 1:
            raise ValueError(
                f"{channels} channels do not divide into {self.num_head_channels}-wide heads"
            )
        return heads, self.num_head_channels


#: The standard txt2img configs, verbatim from the reference's own
#: pinned listings (comfy/model_detection.py
#: unet_config_from_diffusers_unet SD15/SDXL/SDXL_refiner @ b78cec87).
#: Inpainting/ip2p variants differ only in in_channels and detect
#: fine; these constants pin the canonical geometry.
SD15_UNET_CONFIG = UNetConfig(
    in_channels=4,
    out_channels=4,
    model_channels=320,
    num_res_blocks=(2, 2, 2, 2),
    channel_mult=(1, 2, 4, 4),
    transformer_depth=(1, 1, 1, 1, 1, 1, 0, 0),
    transformer_depth_output=(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
    transformer_depth_middle=1,
    context_dim=768,
    use_linear_in_transformer=False,
    num_heads=8,
)

SD15_INPAINT_UNET_CONFIG = UNetConfig(
    in_channels=9,
    out_channels=4,
    model_channels=320,
    num_res_blocks=(2, 2, 2, 2),
    channel_mult=(1, 2, 4, 4),
    transformer_depth=(1, 1, 1, 1, 1, 1, 0, 0),
    transformer_depth_output=(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0),
    transformer_depth_middle=1,
    context_dim=768,
    use_linear_in_transformer=False,
    num_heads=8,
)

SDXL_UNET_CONFIG = UNetConfig(
    in_channels=4,
    out_channels=4,
    model_channels=320,
    num_res_blocks=(2, 2, 2),
    channel_mult=(1, 2, 4),
    transformer_depth=(0, 0, 2, 2, 10, 10),
    transformer_depth_output=(0, 0, 0, 2, 2, 2, 10, 10, 10),
    transformer_depth_middle=10,
    context_dim=2048,
    use_linear_in_transformer=True,
    adm_in_channels=2816,
    num_head_channels=64,
)

SDXL_INPAINT_UNET_CONFIG = UNetConfig(
    in_channels=9,
    out_channels=4,
    model_channels=320,
    num_res_blocks=(2, 2, 2),
    channel_mult=(1, 2, 4),
    transformer_depth=(0, 0, 2, 2, 10, 10),
    transformer_depth_output=(0, 0, 0, 2, 2, 2, 10, 10, 10),
    transformer_depth_middle=10,
    context_dim=2048,
    use_linear_in_transformer=True,
    adm_in_channels=2816,
    num_head_channels=64,
)

SDXL_REFINER_UNET_CONFIG = UNetConfig(
    in_channels=4,
    out_channels=4,
    model_channels=384,
    num_res_blocks=(2, 2, 2, 2),
    channel_mult=(1, 2, 4, 4),
    transformer_depth=(0, 0, 4, 4, 4, 4, 0, 0),
    transformer_depth_output=(0, 0, 0, 4, 4, 4, 4, 4, 4, 0, 0, 0),
    transformer_depth_middle=4,
    context_dim=1280,
    use_linear_in_transformer=True,
    adm_in_channels=2560,
    num_head_channels=64,
)

KNOWN_UNET_CONFIGS = (
    SD15_UNET_CONFIG,
    SD15_INPAINT_UNET_CONFIG,
    SDXL_UNET_CONFIG,
    SDXL_INPAINT_UNET_CONFIG,
    SDXL_REFINER_UNET_CONFIG,
)


def _resblock(prefix: str, ch: int, out_ch: int, emb: int) -> Iterator[tuple[str, tuple[int, ...]]]:
    yield f"{prefix}in_layers.0.weight", (ch,)
    yield f"{prefix}in_layers.0.bias", (ch,)
    yield f"{prefix}in_layers.2.weight", (out_ch, ch, 3, 3)
    yield f"{prefix}in_layers.2.bias", (out_ch,)
    yield f"{prefix}emb_layers.1.weight", (out_ch, emb)
    yield f"{prefix}emb_layers.1.bias", (out_ch,)
    yield f"{prefix}out_layers.0.weight", (out_ch,)
    yield f"{prefix}out_layers.0.bias", (out_ch,)
    yield f"{prefix}out_layers.3.weight", (out_ch, out_ch, 3, 3)
    yield f"{prefix}out_layers.3.bias", (out_ch,)
    if ch != out_ch:
        yield f"{prefix}skip_connection.weight", (out_ch, ch, 1, 1)
        yield f"{prefix}skip_connection.bias", (out_ch,)


def _transformer(
    prefix: str, ch: int, depth: int, config: UNetConfig
) -> Iterator[tuple[str, tuple[int, ...]]]:
    heads, dim_head = config.heads_for(ch)
    inner = heads * dim_head
    context = config.context_dim
    yield f"{prefix}norm.weight", (ch,)
    yield f"{prefix}norm.bias", (ch,)
    if config.use_linear_in_transformer:
        yield f"{prefix}proj_in.weight", (inner, ch)
    else:
        yield f"{prefix}proj_in.weight", (inner, ch, 1, 1)
    yield f"{prefix}proj_in.bias", (inner,)
    for d in range(depth):
        block = f"{prefix}transformer_blocks.{d}."
        yield f"{block}attn1.to_q.weight", (inner, inner)
        yield f"{block}attn1.to_k.weight", (inner, inner)
        yield f"{block}attn1.to_v.weight", (inner, inner)
        yield f"{block}attn1.to_out.0.weight", (inner, inner)
        yield f"{block}attn1.to_out.0.bias", (inner,)
        yield f"{block}attn2.to_q.weight", (inner, inner)
        yield f"{block}attn2.to_k.weight", (inner, context)
        yield f"{block}attn2.to_v.weight", (inner, context)
        yield f"{block}attn2.to_out.0.weight", (inner, inner)
        yield f"{block}attn2.to_out.0.bias", (inner,)
        yield f"{block}ff.net.0.proj.weight", (inner * 8, inner)
        yield f"{block}ff.net.0.proj.bias", (inner * 8,)
        yield f"{block}ff.net.2.weight", (inner, inner * 4)
        yield f"{block}ff.net.2.bias", (inner,)
        yield f"{block}norm1.weight", (inner,)
        yield f"{block}norm1.bias", (inner,)
        yield f"{block}norm2.weight", (inner,)
        yield f"{block}norm2.bias", (inner,)
        yield f"{block}norm3.weight", (inner,)
        yield f"{block}norm3.bias", (inner,)
    if config.use_linear_in_transformer:
        # The reference constructs proj_out as Linear(in_channels,
        # inner_dim) - transposed relative to proj_in, coincidentally
        # square because inner == ch for every supported profile.
        yield f"{prefix}proj_out.weight", (inner, ch)
    else:
        yield f"{prefix}proj_out.weight", (ch, inner, 1, 1)
    yield f"{prefix}proj_out.bias", (ch,)


def unet_layout(config: UNetConfig) -> dict[str, tuple[int, ...]]:
    """Every state-dict key and shape of the constructed UNet - the
    reference constructor's loops replayed over geometry."""
    m = config.model_channels
    emb = config.time_embed_dim
    out: dict[str, tuple[int, ...]] = {}

    def put(entries: Iterator[tuple[str, tuple[int, ...]]]) -> None:
        for key, shape in entries:
            out[key] = shape

    out["time_embed.0.weight"] = (emb, m)
    out["time_embed.0.bias"] = (emb,)
    out["time_embed.2.weight"] = (emb, emb)
    out["time_embed.2.bias"] = (emb,)
    if config.adm_in_channels is not None:
        out["label_emb.0.0.weight"] = (emb, config.adm_in_channels)
        out["label_emb.0.0.bias"] = (emb,)
        out["label_emb.0.2.weight"] = (emb, emb)
        out["label_emb.0.2.bias"] = (emb,)

    out["input_blocks.0.0.weight"] = (m, config.in_channels, 3, 3)
    out["input_blocks.0.0.bias"] = (m,)
    depths = list(config.transformer_depth)
    input_chans = [m]
    ch = m
    index = 1
    for level, mult in enumerate(config.channel_mult):
        for _ in range(config.num_res_blocks[level]):
            prefix = f"input_blocks.{index}."
            out_ch = mult * m
            put(_resblock(f"{prefix}0.", ch, out_ch, emb))
            ch = out_ch
            depth = depths.pop(0)
            if depth > 0:
                put(_transformer(f"{prefix}1.", ch, depth, config))
            input_chans.append(ch)
            index += 1
        if level != len(config.channel_mult) - 1:
            out[f"input_blocks.{index}.0.op.weight"] = (ch, ch, 3, 3)
            out[f"input_blocks.{index}.0.op.bias"] = (ch,)
            input_chans.append(ch)
            index += 1

    put(_resblock("middle_block.0.", ch, ch, emb))
    put(_transformer("middle_block.1.", ch, config.transformer_depth_middle, config))
    put(_resblock("middle_block.2.", ch, ch, emb))

    out_depths = list(config.transformer_depth_output)
    index = 0
    for level, mult in reversed(list(enumerate(config.channel_mult))):
        for block in range(config.num_res_blocks[level] + 1):
            prefix = f"output_blocks.{index}."
            ich = input_chans.pop()
            out_ch = mult * m
            put(_resblock(f"{prefix}0.", ch + ich, out_ch, emb))
            ch = out_ch
            depth = out_depths.pop()
            layer = 1
            if depth > 0:
                put(_transformer(f"{prefix}{layer}.", ch, depth, config))
                layer += 1
            if level and block == config.num_res_blocks[level]:
                out[f"{prefix}{layer}.conv.weight"] = (ch, ch, 3, 3)
                out[f"{prefix}{layer}.conv.bias"] = (ch,)
            index += 1

    out["out.0.weight"] = (m,)
    out["out.0.bias"] = (m,)
    out["out.2.weight"] = (config.out_channels, m, 3, 3)
    out["out.2.bias"] = (config.out_channels,)
    return out


def _count_blocks(keys: frozenset[str], template: str) -> int:
    """Contiguous block count, comfy/model_detection.py count_blocks
    @ b78cec87 over key names."""
    count = 0
    while any(key.startswith(template.format(count)) for key in keys):
        count += 1
    return count


def _prefixed(keys: frozenset[str], prefix: str) -> bool:
    return any(key.startswith(prefix) for key in keys)


def detect_unet_config(
    geometries: Mapping[str, TensorGeometry],
) -> UNetConfig:
    """Classify a diffusion-model-scoped header (``input_blocks.*``
    keys, any checkpoint prefix already stripped) as a supported
    SD1/SDXL UNet, or refuse loudly. Dtypes are ignored - checkpoints
    legitimately ship fp16. The scan is the reference's standard-UNet
    leg (comfy/model_detection.py detect_unet_config @ b78cec87) made
    strict, then the candidate must reproduce the full layout."""
    if not geometries:
        raise UNetDetectError("empty state dict header")
    first_conv = geometries.get("input_blocks.0.0.weight")
    if first_conv is None:
        raise UNetDetectError(
            "not an SD-era UNet (no input_blocks.0.0.weight); MMDiT,"
            " Stable Cascade, audio, and Flux checkpoints route to other"
            " families"
        )
    if len(first_conv.shape) != 4:
        raise UNetDetectError(
            f"input_blocks.0.0.weight has rank {len(first_conv.shape)}, expected a 2D conv (rank 4)"
        )
    keys = frozenset(geometries)
    if any(".time_stack." in key or ".time_mix_blocks." in key for key in keys):
        raise UNetDetectError(
            "temporal/video UNet (time_stack keys); SVD-era video models"
            " are not ported (ROADMAP: Native inference)"
        )
    model_channels = first_conv.shape[0]
    in_channels = first_conv.shape[1]
    out_conv = geometries.get("out.2.weight")
    if out_conv is None:
        raise UNetDetectError("no output projection (out.2.weight)")
    if len(out_conv.shape) != 4:
        raise UNetDetectError(
            f"out.2.weight has rank {len(out_conv.shape)}, expected a 2D conv (rank 4)"
        )
    out_channels = out_conv.shape[0]

    adm_in_channels: int | None = None
    label = geometries.get("label_emb.0.0.weight")
    if label is not None:
        if len(label.shape) != 2:
            raise UNetDetectError(
                f"label_emb.0.0.weight has rank {len(label.shape)}, expected a linear (rank 2)"
            )
        adm_in_channels = label.shape[1]

    num_res_blocks: list[int] = []
    channel_mult: list[int] = []
    transformer_depth: list[int] = []
    transformer_depth_output: list[int] = []
    context_dim: int | None = None
    use_linear = False

    def block_depth(prefix: str) -> int | None:
        """Transformer depth of one block, or None when the block has
        no spatial transformer (calculate_transformer_depth)."""
        nonlocal context_dim, use_linear
        template = f"{prefix}1.transformer_blocks." + "{}."
        if not _prefixed(keys, f"{prefix}1.transformer_blocks."):
            return None
        depth = _count_blocks(keys, template)
        to_k = geometries.get(f"{prefix}1.transformer_blocks.0.attn2.to_k.weight")
        proj_in = geometries.get(f"{prefix}1.proj_in.weight")
        if to_k is None or proj_in is None:
            raise UNetDetectError(
                f"transformer block under {prefix}1. is missing its attn2.to_k / proj_in geometry"
            )
        if len(to_k.shape) != 2:
            raise UNetDetectError(
                f"{prefix}1.transformer_blocks.0.attn2.to_k.weight has"
                f" rank {len(to_k.shape)}, expected a linear (rank 2)"
            )
        if len(proj_in.shape) not in (2, 4):
            raise UNetDetectError(
                f"{prefix}1.proj_in.weight has rank"
                f" {len(proj_in.shape)}, expected a linear (rank 2) or"
                " 1x1 conv (rank 4)"
            )
        if context_dim is None:
            context_dim = to_k.shape[1]
            use_linear = len(proj_in.shape) == 2
        return depth

    input_block_count = _count_blocks(keys, "input_blocks.{}.")
    last_res_blocks = 0
    last_channel_mult = 0
    for count in range(input_block_count):
        prefix = f"input_blocks.{count}."
        prefix_output = f"output_blocks.{input_block_count - count - 1}."
        if f"{prefix}0.op.weight" in keys:
            num_res_blocks.append(last_res_blocks)
            channel_mult.append(last_channel_mult)
            last_res_blocks = 0
            last_channel_mult = 0
            depth = block_depth(prefix_output)
            transformer_depth_output.append(depth if depth is not None else 0)
        else:
            # The reference scans the output side on every
            # non-downsample step, including the initial conv block.
            if f"{prefix}0.in_layers.0.weight" in keys:
                last_res_blocks += 1
                widening = geometries.get(f"{prefix}0.out_layers.3.weight")
                if widening is None:
                    raise UNetDetectError(f"res block {prefix}0. is missing out_layers.3.weight")
                if len(widening.shape) != 4:
                    raise UNetDetectError(
                        f"{prefix}0.out_layers.3.weight has rank"
                        f" {len(widening.shape)}, expected a 2D conv"
                        " (rank 4)"
                    )
                last_channel_mult = widening.shape[0] // model_channels
                depth = block_depth(prefix)
                transformer_depth.append(depth if depth is not None else 0)
            if f"{prefix_output}0.in_layers.0.weight" in keys:
                depth = block_depth(prefix_output)
                transformer_depth_output.append(depth if depth is not None else 0)
    num_res_blocks.append(last_res_blocks)
    channel_mult.append(last_channel_mult)

    if _prefixed(keys, "middle_block.1.proj_in."):
        transformer_depth_middle = _count_blocks(keys, "middle_block.1.transformer_blocks.{}")
    elif "middle_block.0.in_layers.0.weight" in keys:
        raise UNetDetectError(
            "middle block without a transformer (transformer_depth_middle"
            " -1, SSD1B/Segmind Vega lineage); not ported (ROADMAP: Native"
            " inference)"
        )
    else:
        raise UNetDetectError(
            "no middle block (transformer_depth_middle -2, KOALA lineage);"
            " not ported (ROADMAP: Native inference)"
        )

    if context_dim is None:
        raise UNetDetectError(
            "no spatial transformer anywhere in the input blocks; not a supported SD1/SDXL UNet"
        )
    profile = UNET_HEAD_PROFILES.get((context_dim, use_linear))
    if profile is None:
        raise UNetDetectError(
            f"unknown head profile: context_dim {context_dim},"
            f" {'linear' if use_linear else 'conv'} transformer"
            " projections; attention-head counts are not derivable from"
            " shapes, so only SD1.x (768/conv) and SDXL base/refiner"
            " (2048/1280 linear) are accepted (SD2.x is ledgered in"
            " ROADMAP: Native inference)"
        )
    num_heads, num_head_channels = profile

    try:
        config = UNetConfig(
            in_channels=in_channels,
            out_channels=out_channels,
            model_channels=model_channels,
            num_res_blocks=tuple(num_res_blocks),
            channel_mult=tuple(channel_mult),
            transformer_depth=tuple(transformer_depth),
            transformer_depth_output=tuple(transformer_depth_output),
            transformer_depth_middle=transformer_depth_middle,
            context_dim=context_dim,
            use_linear_in_transformer=use_linear,
            adm_in_channels=adm_in_channels,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
        )
    except ValueError as error:
        raise UNetDetectError(
            f"scanned geometry does not form a valid UNet config: {error}"
        ) from error

    layout = unet_layout(config)
    problems: list[str] = []
    for key, shape in layout.items():
        found = geometries.get(key)
        if found is None:
            problems.append(f"missing {key}")
        elif found.shape != shape:
            problems.append(f"{key}: expected shape {shape}, found {found.shape}")
    problems.extend(f"unexpected key {key}" for key in sorted(set(geometries) - set(layout)))
    if problems:
        shown = "; ".join(problems[:6])
        more = len(problems) - 6
        if more > 0:
            shown += f"; and {more} more"
        raise UNetDetectError(f"geometry does not match the detected UNet layout: {shown}")
    return config


__all__ = [
    "KNOWN_UNET_CONFIGS",
    "SD15_INPAINT_UNET_CONFIG",
    "SD15_UNET_CONFIG",
    "SDXL_INPAINT_UNET_CONFIG",
    "SDXL_REFINER_UNET_CONFIG",
    "SDXL_UNET_CONFIG",
    "UNET_HEAD_PROFILES",
    "UNetConfig",
    "UNetDetectError",
    "detect_unet_config",
    "unet_layout",
]
