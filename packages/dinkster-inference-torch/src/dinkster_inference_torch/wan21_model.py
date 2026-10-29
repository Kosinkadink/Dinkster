"""Native core Wan diffusion transformer.

Faithful standalone port of ``comfy/ldm/wan/model.py`` and
``comfy/ldm/wan/model_animate.py`` at ComfyUI commit 76135e55, with the Uni3C
consumer from commit 947c2749 and the Wan FFN fusion from commit 15989f87.
State-dict names match the reference.
Full-reference, camera, Animate pose/face, and typed Uni3C conditioning are
supported; generic patch replacement is absent.

Inputs and outputs use ``[B, C, T, H, W]``. Patch tokens are row-major over
time, height, then width with patch size ``(1, 2, 2)``. ``context`` contains
UMT5 rows. Wan 2.1 I2V additionally projects 1280-wide CLIP vision rows and
uses the reference split image/text cross-attention.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from dinkster_inference import (
    WAN21_CAMERA_1_3B,
    WAN21_CAMERA_14B,
    WAN21_FLF_I2V_14B,
    WAN21_FUN_CONTROL_1_3B,
    WAN21_FUN_INPAINT_1_3B,
    WAN21_I2V_14B,
    WAN21_T2V_1_3B,
    WAN21_T2V_14B,
    WAN21_VACE_1_3B,
    WAN21_VACE_14B,
    WAN22_ANIMATE_14B,
    WAN22_BERNINI_14B,
    WAN22_CAMERA_14B,
    WAN22_FUN_CONTROL_5B,
    WAN22_FUN_CONTROL_14B,
    WAN22_FUN_INPAINT_5B,
    WAN22_I2V_14B,
    WAN22_TI2V_5B,
    Wan21Config,
)

from .attention import AttentionKernel, attention_kernel_context, select_attention
from .flux import EmbedND, apply_rope, rope
from .operations import INITLESS, CastOperations, Operations, ResidencyRouted
from .ops import cast_weight
from .quant_linear import linear_input_act
from .wan21_animate import AnimateMotionEncoder, FaceAdapter, FaceEncoder

if TYPE_CHECKING:
    from .wan21_multitalk import Wan21MultiTalkExecution
    from .wan21_uni3c import Wan21Uni3CExecution

_DEFAULT_ATTENTION = select_attention("flux").kernel


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    half = dim // 2
    position = position.to(torch.float32)
    sinusoid = torch.outer(
        position,
        torch.pow(
            10000,
            -torch.arange(half, device=position.device, dtype=torch.float32).div(half),
        ),
    )
    return torch.cat((torch.cos(sinusoid), torch.sin(sinusoid)), dim=1)


def _repeat_time_rows(value: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    repeats = 1
    if value.shape[1] > 1:
        repeats = tokens.shape[1] // value.shape[1]
    if repeats == 1:
        return value
    if repeats * value.shape[1] == tokens.shape[1]:
        return torch.repeat_interleave(value, repeats, dim=1)
    return torch.repeat_interleave(value, repeats + 1, dim=1)[:, : tokens.shape[1]]


class WanSelfAttention(torch.nn.Module):
    _attention_kernel: AttentionKernel

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qk_norm: bool,
        eps: float,
        operations: Operations,
        attention_kernel: AttentionKernel,
        kv_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        kv_dim = dim if kv_dim is None else kv_dim
        self.q = operations.linear(dim, dim)
        self.k = operations.linear(kv_dim, dim)
        self.v = operations.linear(kv_dim, dim)
        self.o = operations.linear(dim, dim)
        self.norm_q = operations.rms_norm(dim, eps=eps) if qk_norm else torch.nn.Identity()
        self.norm_k = operations.rms_norm(dim, eps=eps) if qk_norm else torch.nn.Identity()
        object.__setattr__(self, "_attention_kernel", attention_kernel)

    def forward_with_qk(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, heads, width = (
            x.shape[0],
            x.shape[1],
            self.num_heads,
            self.head_dim,
        )

        def query() -> torch.Tensor:
            return self.norm_q(self.q(x)).view(batch, sequence, heads, width)

        def key() -> torch.Tensor:
            return self.norm_k(self.k(x)).view(batch, sequence, heads, width)

        q, k = apply_rope(query(), key(), freqs)
        v = self.v(x).view(batch, sequence, heads, width)
        attended = self._attention_kernel(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        output = self.o(attended.transpose(1, 2).reshape(batch, sequence, heads * width))
        return output, q, k

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        return self.forward_with_qk(x, freqs)[0]


class WanCrossAttention(WanSelfAttention):
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_rows = x.shape[:2]
        key_rows = context.shape[1]
        q = self.norm_q(self.q(x)).view(batch, query_rows, self.num_heads, self.head_dim)
        k = self.norm_k(self.k(context)).view(batch, key_rows, self.num_heads, self.head_dim)
        v = self.v(context).view(batch, key_rows, self.num_heads, self.head_dim)
        attended = self._attention_kernel(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )
        return self.o(attended.transpose(1, 2).reshape(batch, query_rows, -1))


class WanI2VCrossAttention(WanCrossAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qk_norm: bool,
        eps: float,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            qk_norm=qk_norm,
            eps=eps,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.k_img = operations.linear(dim, dim)
        self.v_img = operations.linear(dim, dim)
        self.norm_k_img = operations.rms_norm(dim, eps=eps) if qk_norm else torch.nn.Identity()

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        image_rows: int | None,
    ) -> torch.Tensor:
        batch, query_rows = x.shape[:2]
        q = (
            self.norm_q(self.q(x))
            .view(batch, query_rows, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        text = context[:, image_rows:]
        image = context[:, :image_rows]
        k = (
            self.norm_k(self.k(text))
            .view(batch, text.shape[1], self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = self.v(text).view(batch, text.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        attended = self._attention_kernel(q, k, v)
        k_img = (
            self.norm_k_img(self.k_img(image))
            .view(batch, image.shape[1], self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v_img = (
            self.v_img(image)
            .view(batch, image.shape[1], self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        attended = attended + self._attention_kernel(q, k_img, v_img)
        return self.o(attended.transpose(1, 2).reshape(batch, query_rows, -1))


class WanFeedForward(torch.nn.Sequential):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return linear_input_act(self[2], self[0](input), "gelu_tanh")


class WanAttentionBlock(ResidencyRouted, torch.nn.Module):
    def __init__(
        self,
        config: Wan21Config,
        *,
        operations: Operations,
        attention_kernel: AttentionKernel,
        _self_attention_type: type[WanSelfAttention] = WanSelfAttention,
    ) -> None:
        super().__init__()
        dim = config.hidden_size
        self.norm1 = operations.layer_norm(dim, eps=config.eps, elementwise_affine=False)
        self.self_attn = _self_attention_type(
            dim,
            config.num_heads,
            qk_norm=config.qk_norm,
            eps=config.eps,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.norm3 = (
            operations.layer_norm(dim, eps=config.eps)
            if config.cross_attn_norm
            else torch.nn.Identity()
        )
        self.image_to_video = config.model_type == "i2v"
        cross_type = WanI2VCrossAttention if self.image_to_video else WanCrossAttention
        self.cross_attn = cross_type(
            dim,
            config.num_heads,
            qk_norm=config.qk_norm,
            eps=config.eps,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.norm2 = operations.layer_norm(dim, eps=config.eps, elementwise_affine=False)
        self.ffn = WanFeedForward(
            operations.linear(dim, config.ffn_hidden_size),
            torch.nn.GELU(approximate="tanh"),
            operations.linear(config.ffn_hidden_size, dim),
        )
        self.modulation = torch.nn.Parameter(torch.empty(1, 6, dim))

    def _forward_owned(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        image_rows: int | None,
        modulation_state: torch.Tensor,
        multitalk: Wan21MultiTalkExecution | None,
        multitalk_block_index: int,
        grid_shape: tuple[int, int, int] | None,
    ) -> torch.Tensor:
        modulation = (modulation_state.unsqueeze(0) + time).unbind(2)
        x = x.contiguous()
        normalized = torch.addcmul(
            _repeat_time_rows(modulation[0], x),
            self.norm1(x),
            1 + _repeat_time_rows(modulation[1], x),
        )
        if multitalk is not None and multitalk.target_masks is not None:
            update, query, key = self.self_attn.forward_with_qk(normalized, freqs)
        else:
            update = self.self_attn(normalized, freqs)
            query = key = None
        x = torch.addcmul(x, update, _repeat_time_rows(modulation[2], x))
        if self.image_to_video:
            cross_attn = cast("WanI2VCrossAttention", self.cross_attn)
            x = x + cross_attn(self.norm3(x), context, image_rows)
        else:
            x = x + self.cross_attn(self.norm3(x), context)
        if multitalk is not None:
            assert grid_shape is not None
            if multitalk.target_masks is None:
                x = multitalk.model.forward_block(
                    multitalk_block_index,
                    x,
                    multitalk.audio_context,
                    grid_shape,
                    strength=multitalk.strength,
                )
            else:
                assert query is not None and key is not None
                lanes = []
                for lane in range(x.shape[0]):
                    attention_map = multitalk.model.attention_map(
                        query[lane : lane + 1],
                        key[lane : lane + 1],
                        grid_shape,
                        multitalk.target_masks,
                    )
                    lanes.append(
                        multitalk.model.forward_block(
                            multitalk_block_index,
                            x[lane : lane + 1],
                            multitalk.audio_context,
                            grid_shape,
                            attention_map=attention_map,
                            strength=multitalk.strength,
                        )
                    )
                x = torch.cat(lanes)
        normalized = torch.addcmul(
            _repeat_time_rows(modulation[3], x),
            self.norm2(x),
            1 + _repeat_time_rows(modulation[4], x),
        )
        return torch.addcmul(
            x,
            self.ffn(normalized),
            _repeat_time_rows(modulation[5], x),
        )

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
        image_rows: int | None,
        *,
        multitalk: Wan21MultiTalkExecution | None = None,
        multitalk_block_index: int = 0,
        grid_shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            modulation = cast_weight(self.modulation, device=x.device, dtype=x.dtype)
            return self._forward_owned(
                x,
                time,
                freqs,
                context,
                image_rows,
                modulation,
                multitalk,
                multitalk_block_index,
                grid_shape,
            )
        with binding.lease() as lease:
            return self._forward_owned(
                x,
                time,
                freqs,
                context,
                image_rows,
                lease.get("modulation", dtype=x.dtype),
                multitalk,
                multitalk_block_index,
                grid_shape,
            )


class WanVaceAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        config: Wan21Config,
        *,
        block_id: int,
        operations: Operations,
        attention_kernel: AttentionKernel,
    ) -> None:
        super().__init__(
            config,
            operations=operations,
            attention_kernel=attention_kernel,
        )
        self.before_proj = (
            operations.linear(config.hidden_size, config.hidden_size) if block_id == 0 else None
        )
        self.after_proj = operations.linear(config.hidden_size, config.hidden_size)

    def forward_vace(
        self,
        control: torch.Tensor,
        original: torch.Tensor,
        time: torch.Tensor,
        freqs: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.before_proj is not None:
            control = self.before_proj(control) + original
        control = super().forward(control, time, freqs, context, 0)
        return self.after_proj(control), control


class WanCameraResidualBlock(torch.nn.Module):
    def __init__(self, dim: int, *, operations: Operations) -> None:
        super().__init__()
        self.conv1 = operations.conv2d(dim, dim, 3, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = operations.conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.relu(self.conv1(x))) + x


class WanCameraAdapter(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, operations: Operations) -> None:
        super().__init__()
        self.conv = operations.conv2d(in_channels * 64, out_channels, 2, stride=2)
        self.residual_blocks = torch.nn.Sequential(
            WanCameraResidualBlock(out_channels, operations=operations)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        x = self.residual_blocks(self.conv(F.pixel_unshuffle(x, 8)))
        return x.view(batch, frames, *x.shape[1:]).permute(0, 2, 1, 3, 4)


class WanHead(ResidencyRouted, torch.nn.Module):
    def __init__(
        self,
        config: Wan21Config,
        *,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.norm = operations.layer_norm(
            config.hidden_size, eps=config.eps, elementwise_affine=False
        )
        self.head = operations.linear(
            config.hidden_size,
            math.prod(config.patch_size) * config.out_channels,
        )
        self.modulation = torch.nn.Parameter(torch.empty(1, 2, config.hidden_size))

    def _forward_owned(
        self, x: torch.Tensor, time: torch.Tensor, modulation_state: torch.Tensor
    ) -> torch.Tensor:
        modulation = (modulation_state + time.unsqueeze(2)).unbind(2)
        return self.head(
            torch.addcmul(
                _repeat_time_rows(modulation[0], x),
                self.norm(x),
                1 + _repeat_time_rows(modulation[1], x),
            )
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        binding = self._offloaded_residency()
        if binding is None:
            modulation = cast_weight(self.modulation, device=x.device, dtype=x.dtype)
            return self._forward_owned(x, time, modulation)
        with binding.lease() as lease:
            return self._forward_owned(x, time, lease.get("modulation", dtype=x.dtype))


class WanImageEmbedding(ResidencyRouted, torch.nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        flf_pos_embed_token_number: int | None,
        operations: Operations,
    ) -> None:
        super().__init__()
        self.proj = torch.nn.Sequential(
            operations.layer_norm(1280),
            operations.linear(1280, 1280),
            torch.nn.GELU(),
            operations.linear(1280, dim),
            operations.layer_norm(dim),
        )
        self.emb_pos = (
            None
            if flf_pos_embed_token_number is None
            else torch.nn.Parameter(torch.empty(1, flf_pos_embed_token_number, 1280))
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        if self.emb_pos is not None:
            binding = self._offloaded_residency()
            if binding is None:
                position = cast_weight(self.emb_pos, device=rows.device, dtype=rows.dtype)
                rows = rows[:, : self.emb_pos.shape[1]] + position[:, : rows.shape[1]]
            else:
                with binding.lease() as lease:
                    position = lease.get("emb_pos", dtype=rows.dtype)
                    rows = rows[:, : self.emb_pos.shape[1]] + position[:, : rows.shape[1]]
        return self.proj(rows)


class Wan21Model(torch.nn.Module):
    """Native Wan 2.1/2.2 core, Fun, camera, or VACE diffusion transformer."""

    _attention_kernel: AttentionKernel

    def __init__(
        self,
        config: Wan21Config = WAN21_T2V_1_3B,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_ATTENTION,
        _block_type: type[WanAttentionBlock] = WanAttentionBlock,
    ) -> None:
        super().__init__()
        self.config = config
        object.__setattr__(self, "_attention_kernel", attention_kernel)
        self.patch_embedding = CastOperations(torch.float32).conv3d(
            config.in_channels,
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
        )
        self.text_embedding = torch.nn.Sequential(
            operations.linear(config.text_dim, config.hidden_size),
            torch.nn.GELU(approximate="tanh"),
            operations.linear(config.hidden_size, config.hidden_size),
        )
        self.time_embedding = torch.nn.Sequential(
            operations.linear(config.time_freq_dim, config.hidden_size),
            torch.nn.SiLU(),
            operations.linear(config.hidden_size, config.hidden_size),
        )
        self.time_projection = torch.nn.Sequential(
            torch.nn.SiLU(),
            operations.linear(config.hidden_size, config.hidden_size * 6),
        )
        self.blocks = torch.nn.ModuleList(
            _block_type(
                config,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            for _ in range(config.num_layers)
        )
        self.vace_blocks = (
            None
            if config.vace_layers is None
            else torch.nn.ModuleList(
                WanVaceAttentionBlock(
                    config,
                    block_id=index,
                    operations=operations,
                    attention_kernel=attention_kernel,
                )
                for index in range(config.vace_layers)
            )
        )
        self.vace_patch_embedding = (
            None
            if config.vace_layers is None
            else CastOperations(torch.float32).conv3d(
                96,
                config.hidden_size,
                config.patch_size,
                stride=config.patch_size,
            )
        )
        self.head = WanHead(config, operations=operations)
        head_dim = config.hidden_size // config.num_heads
        axis_width = head_dim // 6
        self.rope_embedder = EmbedND(
            dim=head_dim,
            theta=10000,
            axes_dim=(head_dim - 4 * axis_width, 2 * axis_width, 2 * axis_width),
        )
        self.img_emb = (
            WanImageEmbedding(
                config.hidden_size,
                flf_pos_embed_token_number=config.flf_pos_embed_token_number,
                operations=operations,
            )
            if config.model_type == "i2v"
            else None
        )
        self.ref_conv = (
            None
            if config.reference_channels is None
            else operations.conv2d(
                config.reference_channels,
                config.hidden_size,
                2,
                stride=2,
            )
        )
        self.control_adapter = (
            None
            if config.camera_channels is None
            else WanCameraAdapter(
                config.camera_channels,
                config.hidden_size,
                operations=operations,
            )
        )
        animate = config.model_variant == "animate"
        self.pose_patch_embedding = (
            operations.conv3d(
                config.out_channels,
                config.hidden_size,
                config.patch_size,
                stride=config.patch_size,
            )
            if animate
            else None
        )
        self.motion_encoder = AnimateMotionEncoder() if animate else None
        self.face_adapter = (
            FaceAdapter(
                config.hidden_size,
                config.num_heads,
                config.num_layers // 5,
                operations=operations,
                attention_kernel=attention_kernel,
            )
            if animate
            else None
        )
        self.face_encoder = (
            FaceEncoder(config.hidden_size, operations=operations) if animate else None
        )

    def _validate(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        vision: torch.Tensor | None,
        temporal_reference: torch.Tensor | None,
        reference_latent: torch.Tensor | None,
        vace_context: torch.Tensor | None,
        vace_strength: tuple[float, ...] | None,
        camera_conditions: torch.Tensor | None,
        pose_latents: torch.Tensor | None,
        face_pixel_values: torch.Tensor | None,
        multitalk: Wan21MultiTalkExecution | None = None,
        uni3c: Wan21Uni3CExecution | None = None,
        uni3c_input: torch.Tensor | None = None,
        context_latents: tuple[torch.Tensor, ...] = (),
    ) -> None:
        config = self.config
        animate = config.model_variant == "animate"
        if not animate and (pose_latents is not None or face_pixel_values is not None):
            raise ValueError("pose and face conditioning are supported only by Wan Animate")
        for name, value in (("x", x), ("timesteps", timesteps), ("context", context)):
            if (
                type(value) is not torch.Tensor
                or not value.is_floating_point()
                or value.layout != torch.strided
            ):
                raise TypeError(f"{name} must be an exact strided floating torch.Tensor")
        if x.ndim != 5 or x.shape[1] != config.in_channels or any(size == 0 for size in x.shape):
            raise ValueError(
                "x must be a non-empty [batch, in_channels, time, height, width]"
                f" tensor with {config.in_channels} channels, got {tuple(x.shape)}"
            )
        batch = x.shape[0]
        timestep_shapes = ((batch,), (batch, x.shape[2]))
        if tuple(timesteps.shape) not in timestep_shapes:
            raise ValueError(
                f"timesteps must have shape ({batch},) or ({batch}, {x.shape[2]}), "
                f"got {tuple(timesteps.shape)}"
            )
        if context.ndim != 3 or context.shape[0] != batch or context.shape[2] != config.text_dim:
            raise ValueError(
                f"context must have shape ({batch}, text_rows, {config.text_dim}),"
                f" got {tuple(context.shape)}"
            )
        if context.shape[1] == 0:
            raise ValueError("context must contain at least one text row")
        if type(context_latents) is not tuple:
            raise TypeError("context_latents must be an exact tuple")
        if context_latents:
            if config is not WAN22_BERNINI_14B:
                raise ValueError("context_latents require exact Wan 2.2 Bernini 14B")
            if any(
                value is not None
                for value in (
                    temporal_reference,
                    reference_latent,
                    vace_context,
                    camera_conditions,
                    pose_latents,
                    face_pixel_values,
                    uni3c,
                )
            ):
                raise ValueError(
                    "Bernini context latents cannot be combined with another intervention"
                )
            for index, latent in enumerate(context_latents):
                if (
                    type(latent) is not torch.Tensor
                    or not latent.is_floating_point()
                    or latent.layout != torch.strided
                ):
                    raise TypeError(
                        f"context_latents[{index}] must be an exact strided floating torch.Tensor"
                    )
                if (
                    latent.ndim != 5
                    or latent.shape[0] != batch
                    or latent.shape[1] != config.out_channels
                    or any(size <= 0 for size in latent.shape[2:])
                ):
                    raise ValueError(
                        f"context_latents[{index}] must have shape "
                        f"({batch}, {config.out_channels}, time, height, width)"
                    )
        if multitalk is not None:
            from .wan21_multitalk import Wan21MultiTalkExecution

            if type(multitalk) is not Wan21MultiTalkExecution:
                raise TypeError("multitalk must be exact Wan21MultiTalkExecution")
            if config is not WAN21_I2V_14B:
                raise ValueError("InfiniteTalk requires exact base Wan 2.1 I2V 14B")
            if (
                any(
                    value is not None
                    for value in (
                        temporal_reference,
                        reference_latent,
                        vace_context,
                        camera_conditions,
                        pose_latents,
                        face_pixel_values,
                        uni3c,
                    )
                )
                or context_latents
            ):
                raise ValueError("InfiniteTalk cannot be combined with another Wan intervention")
            patch_t, patch_h, patch_w = config.patch_size
            expected_time = math.ceil(x.shape[2] / patch_t)
            if multitalk.audio_context.shape[1] != expected_time:
                raise ValueError("InfiniteTalk audio context must match the target latent time")
            if multitalk.target_masks is not None:
                expected_spatial = math.ceil(x.shape[3] / patch_h) * math.ceil(x.shape[4] / patch_w)
                if multitalk.target_masks.shape[1] != expected_spatial:
                    raise ValueError(
                        "InfiniteTalk target masks must match the target patch-token grid"
                    )
        if (uni3c is None) != (uni3c_input is None):
            raise ValueError("Wan 2.1 Uni3C execution and input must be provided together")
        if uni3c is not None:
            from .wan21_uni3c import (
                Wan21Uni3CExecution,
                validate_wan21_uni3c_resource,
            )

            if type(uni3c) is not Wan21Uni3CExecution:
                raise TypeError("uni3c must be exact Wan21Uni3CExecution")
            if config is not WAN21_T2V_14B and config is not WAN21_I2V_14B:
                raise ValueError("Uni3C supports only exact base Wan 2.1 T2V or I2V 14B")
            if any(
                value is not None
                for value in (
                    temporal_reference,
                    reference_latent,
                    vace_context,
                    camera_conditions,
                    pose_latents,
                    face_pixel_values,
                )
            ):
                raise ValueError("Uni3C cannot be combined with another Wan model intervention")
            validate_wan21_uni3c_resource(uni3c.model, uni3c.model_digest)
            assert uni3c_input is not None
            if (
                type(uni3c_input) is not torch.Tensor
                or not uni3c_input.is_floating_point()
                or uni3c_input.layout is not torch.strided
            ):
                raise TypeError("Uni3C input must be an exact strided floating torch.Tensor")
            if (
                uni3c_input.ndim != 5
                or uni3c_input.shape[0] < 1
                or uni3c_input.shape[1] != uni3c.model.config.input_channels
                or uni3c_input.shape[2:] != x.shape[2:]
                or batch % uni3c_input.shape[0]
            ):
                raise ValueError(
                    "Uni3C input must be [batch,36,T,H,W], divide the model batch, and match x"
                )
        if temporal_reference is not None:
            if (
                type(temporal_reference) is not torch.Tensor
                or not temporal_reference.is_floating_point()
                or temporal_reference.layout != torch.strided
            ):
                raise TypeError("temporal_reference must be an exact strided floating torch.Tensor")
            if (
                temporal_reference.ndim != 5
                or temporal_reference.shape[0] != batch
                or temporal_reference.shape[1] != config.in_channels
                or temporal_reference.shape[2] <= 0
                or temporal_reference.shape[3:] != x.shape[3:]
            ):
                raise ValueError(
                    "temporal_reference must match x batch, channels, and spatial shape"
                )
        if reference_latent is not None:
            if self.ref_conv is None or config.reference_channels is None:
                raise ValueError("reference latents are supported only by full-reference geometry")
            if (
                type(reference_latent) is not torch.Tensor
                or not reference_latent.is_floating_point()
                or reference_latent.layout != torch.strided
            ):
                raise TypeError("reference_latent must be an exact strided floating torch.Tensor")
            expected = (batch, config.reference_channels, x.shape[3], x.shape[4])
            if tuple(reference_latent.shape) != expected:
                raise ValueError(f"reference_latent must have shape {expected}")
        if self.vace_blocks is None:
            if vace_context is not None or vace_strength is not None:
                raise ValueError("VACE conditioning is supported only by VACE geometry")
        else:
            if vace_context is None or vace_strength is None:
                raise ValueError("Wan VACE requires context tensors and strengths")
            if (
                type(vace_context) is not torch.Tensor
                or not vace_context.is_floating_point()
                or vace_context.layout != torch.strided
            ):
                raise TypeError("VACE context must be an exact strided floating torch.Tensor")
            expected_tail = (96, *x.shape[2:])
            if (
                vace_context.ndim != 6
                or vace_context.shape[0] != batch
                or vace_context.shape[1] <= 0
                or tuple(vace_context.shape[2:]) != expected_tail
            ):
                raise ValueError(
                    f"VACE context must have shape ({batch}, controls, 96, time, height, width)"
                )
            if (
                type(vace_strength) is not tuple
                or len(vace_strength) != vace_context.shape[1]
                or any(
                    type(value) is not float or not math.isfinite(value) or value < 0.0
                    for value in vace_strength
                )
            ):
                raise ValueError("VACE strengths must be one finite non-negative float per control")
        if self.control_adapter is None:
            if camera_conditions is not None:
                raise ValueError("camera conditioning is supported only by camera geometry")
        elif camera_conditions is not None:
            if (
                type(camera_conditions) is not torch.Tensor
                or not camera_conditions.is_floating_point()
                or camera_conditions.layout != torch.strided
            ):
                raise TypeError("camera_conditions must be an exact strided floating torch.Tensor")
            expected = (
                batch,
                self.config.camera_channels,
                x.shape[2],
                x.shape[3] * 8,
                x.shape[4] * 8,
            )
            if tuple(camera_conditions.shape) != expected:
                raise ValueError(f"camera_conditions must have shape {expected}")
        if pose_latents is not None:
            if (
                type(pose_latents) is not torch.Tensor
                or not pose_latents.is_floating_point()
                or pose_latents.layout != torch.strided
            ):
                raise TypeError("pose_latents must be an exact strided floating torch.Tensor")
            if (
                pose_latents.ndim != 5
                or pose_latents.shape[0] != batch
                or pose_latents.shape[1] != config.out_channels
                or pose_latents.shape[2] <= 0
                or pose_latents.shape[3:] != x.shape[3:]
            ):
                raise ValueError("pose_latents must match x batch and spatial latent geometry")
        if face_pixel_values is not None:
            if (
                type(face_pixel_values) is not torch.Tensor
                or not face_pixel_values.is_floating_point()
                or face_pixel_values.layout != torch.strided
            ):
                raise TypeError("face_pixel_values must be an exact strided floating torch.Tensor")
            if (
                face_pixel_values.ndim != 5
                or face_pixel_values.shape[0] != batch
                or face_pixel_values.shape[1] != 3
                or face_pixel_values.shape[2] <= 0
                or face_pixel_values.shape[3:] != (512, 512)
            ):
                raise ValueError("face_pixel_values must have shape [batch,3,frames,512,512]")
        if vision is None:
            return
        if self.img_emb is None:
            raise ValueError("vision rows are supported only by I2V geometry")
        if (
            type(vision) is not torch.Tensor
            or not vision.is_floating_point()
            or vision.layout != torch.strided
        ):
            raise TypeError("vision must be an exact strided floating torch.Tensor")
        if vision.ndim != 3 or vision.shape[0] != batch or vision.shape[2] != 1280:
            raise ValueError(
                f"vision must have shape ({batch}, vision_rows, 1280), got {tuple(vision.shape)}"
            )
        if vision.shape[1] == 0:
            raise ValueError("vision must contain at least one row")

    def _rope(
        self,
        shape: tuple[int, int, int],
        x: torch.Tensor,
        *,
        time_start: int = 0,
        source_id: int = 0,
    ) -> torch.Tensor:
        time, height, width = shape
        ids = torch.zeros((time, height, width, 3), device=x.device, dtype=x.dtype)
        ids[..., 0] += torch.linspace(
            time_start,
            time_start + time - 1,
            time,
            device=x.device,
            dtype=x.dtype,
        )[:, None, None]
        ids[..., 1] += torch.linspace(0, height - 1, height, device=x.device, dtype=x.dtype)[
            None, :, None
        ]
        ids[..., 2] += torch.linspace(0, width - 1, width, device=x.device, dtype=x.dtype)[
            None, None, :
        ]
        freqs = self.rope_embedder(ids.reshape(1, -1, 3)).movedim(1, 2)
        if source_id:
            head_dim = self.config.hidden_size // self.config.num_heads
            position = torch.tensor(
                [[float(source_id)]],
                device=freqs.device,
                dtype=torch.float32,
            )
            source_rotation = rope(position, head_dim, self.rope_embedder.theta).reshape(
                1, 1, 1, head_dim // 2, 2, 2
            )
            freqs = torch.einsum(
                "...ij,...jk->...ik",
                freqs,
                source_rotation.to(freqs.dtype),
            )
        return freqs

    def _animate_features(
        self,
        x: torch.Tensor,
        pose_latents: torch.Tensor | None,
        face_pixel_values: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if pose_latents is not None:
            assert self.pose_patch_embedding is not None
            pose = self.pose_patch_embedding(pose_latents)
            pose_frames = min(pose.shape[2], x.shape[2] - 1)
            if pose_frames:
                x[:, :, 1 : pose_frames + 1] += pose[:, :, :pose_frames]
        if face_pixel_values is None:
            return x, None
        assert self.motion_encoder is not None
        assert self.face_encoder is not None
        batch, _, frames, height, width = face_pixel_values.shape
        flattened = face_pixel_values.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, 3, height, width
        )
        motion = torch.cat(
            tuple(self.motion_encoder(part) for part in flattened.split(8)),
            dim=0,
        ).view(batch, frames, 512)
        motion = self.face_encoder(motion)
        motion = torch.cat((motion.new_zeros(batch, 1, *motion.shape[2:]), motion), dim=1)
        if motion.shape[1] < x.shape[2]:
            motion = torch.cat(
                (motion, motion.new_zeros(batch, x.shape[2] - motion.shape[1], *motion.shape[2:])),
                dim=1,
            )
        return x, motion[:, : x.shape[2]]

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        vision: torch.Tensor | None = None,
        *,
        temporal_reference: torch.Tensor | None = None,
        reference_latent: torch.Tensor | None = None,
        vace_context: torch.Tensor | None = None,
        vace_strength: tuple[float, ...] | None = None,
        camera_conditions: torch.Tensor | None = None,
        pose_latents: torch.Tensor | None = None,
        face_pixel_values: torch.Tensor | None = None,
        multitalk: Wan21MultiTalkExecution | None = None,
        uni3c: Wan21Uni3CExecution | None = None,
        uni3c_input: torch.Tensor | None = None,
        context_latents: tuple[torch.Tensor, ...] = (),
    ) -> torch.Tensor:
        self._validate(
            x,
            timesteps,
            context,
            vision,
            temporal_reference,
            reference_latent,
            vace_context,
            vace_strength,
            camera_conditions,
            pose_latents,
            face_pixel_values,
            multitalk,
            uni3c,
            uni3c_input,
            context_latents,
        )
        original_shape = x.shape[2:]
        pad_t = (-x.shape[2]) % self.config.patch_size[0]
        pad_h = (-x.shape[3]) % self.config.patch_size[1]
        pad_w = (-x.shape[4]) % self.config.patch_size[2]
        if pad_t or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode="circular")
            if vace_context is not None:
                batch, control_count = vace_context.shape[:2]
                vace_context = F.pad(
                    vace_context.flatten(0, 1),
                    (0, pad_w, 0, pad_h, 0, pad_t),
                    mode="circular",
                ).unflatten(0, (batch, control_count))
            if camera_conditions is not None:
                camera_conditions = F.pad(
                    camera_conditions,
                    (0, pad_w * 8, 0, pad_h * 8, 0, pad_t),
                    mode="circular",
                )
            if pose_latents is not None:
                pose_latents = F.pad(
                    pose_latents,
                    (0, pad_w, 0, pad_h, 0, pad_t),
                    mode="circular",
                )
            if uni3c_input is not None:
                uni3c_input = F.pad(
                    uni3c_input,
                    (0, pad_w, 0, pad_h, 0, pad_t),
                    mode="circular",
                )
        padded_context_latents: list[torch.Tensor] = []
        for latent in context_latents:
            context_pad_t = (-latent.shape[2]) % self.config.patch_size[0]
            context_pad_h = (-latent.shape[3]) % self.config.patch_size[1]
            context_pad_w = (-latent.shape[4]) % self.config.patch_size[2]
            if context_pad_t or context_pad_h or context_pad_w:
                latent = F.pad(
                    latent,
                    (0, context_pad_w, 0, context_pad_h, 0, context_pad_t),
                    mode="circular",
                )
            padded_context_latents.append(latent)
        if temporal_reference is not None:
            reference_pad_t = (-temporal_reference.shape[2]) % self.config.patch_size[0]
            if reference_pad_t or pad_h or pad_w:
                temporal_reference = F.pad(
                    temporal_reference,
                    (0, pad_w, 0, pad_h, 0, reference_pad_t),
                    mode="circular",
                )
            x = torch.cat((x, temporal_reference), dim=2)
        x = self.patch_embedding(x.float()).to(x.dtype)
        motion: torch.Tensor | None = None
        if self.config.model_variant == "animate":
            x, motion = self._animate_features(x, pose_latents, face_pixel_values)
        if camera_conditions is not None:
            assert self.control_adapter is not None
            x = x + self.control_adapter(camera_conditions).to(x.dtype)
        projected_context_latents = tuple(
            self.patch_embedding(latent.float()).to(x.dtype) for latent in padded_context_latents
        )
        grid = (x.shape[2], x.shape[3], x.shape[4])
        rope_grid = (grid[0] + (reference_latent is not None), grid[1], grid[2])
        freqs = self._rope(rope_grid, x)
        for source_id, projected_context in enumerate(projected_context_latents, start=1):
            freqs = torch.cat(
                (
                    freqs,
                    self._rope(
                        (
                            projected_context.shape[2],
                            projected_context.shape[3],
                            projected_context.shape[4],
                        ),
                        x,
                        source_id=source_id,
                    ),
                ),
                dim=1,
            )
        x = x.flatten(2).transpose(1, 2)

        time = self.time_embedding(
            sinusoidal_embedding_1d(self.config.time_freq_dim, timesteps.flatten()).to(x.dtype)
        ).reshape(timesteps.shape[0], -1, self.config.hidden_size)
        projected_time = self.time_projection(time).unflatten(2, (6, self.config.hidden_size))
        reference_rows = 0
        if reference_latent is not None:
            assert self.ref_conv is not None
            projected_reference = self.ref_conv(reference_latent).flatten(2).transpose(1, 2)
            reference_rows = projected_reference.shape[1]
            x = torch.cat((projected_reference, x), dim=1)
        target_rows = x.shape[1]
        for projected_context in projected_context_latents:
            x = torch.cat((x, projected_context.flatten(2).transpose(1, 2)), dim=1)
        context = self.text_embedding(context)
        image_rows: int | None = None
        if vision is not None:
            assert self.img_emb is not None
            projected_vision = self.img_emb(vision)
            image_rows = projected_vision.shape[1]
            context = torch.cat((projected_vision, context), dim=1)

        controls: list[torch.Tensor] = []
        if vace_context is not None:
            assert self.vace_patch_embedding is not None
            batch, control_count = vace_context.shape[:2]
            projected = self.vace_patch_embedding(
                vace_context.movedim(0, 1).flatten(0, 1).float()
            ).to(vace_context.dtype)
            controls = list(projected.flatten(2).transpose(1, 2).split(batch, dim=0))
            if len(controls) != control_count:
                raise RuntimeError("VACE control projection produced an invalid batch layout")

        original = x
        mapping_step = (
            None
            if self.config.vace_layers is None
            else self.config.num_layers // self.config.vace_layers
        )
        uni3c_hidden: torch.Tensor | None = None
        uni3c_freqs: torch.Tensor | None = None
        uni3c_temb: torch.Tensor | None = None
        uni3c_execution = uni3c
        if uni3c_execution is not None:
            assert uni3c_input is not None
            uni3c_hidden, uni3c_freqs = uni3c_execution.model.process_input(uni3c_input)
            uni3c_temb = time[: uni3c_input.shape[0], 0].to(x.dtype)
            expected_rows = grid[0] * grid[1] * grid[2]
            if uni3c_hidden.shape[1] != expected_rows:
                raise ValueError("Uni3C residual rows must match the target video token grid")
        with attention_kernel_context(self._attention_kernel, x.numel(), device=x.device):
            for index, block in enumerate(self.blocks):
                x = block(
                    x,
                    projected_time,
                    freqs,
                    context,
                    image_rows,
                    multitalk=multitalk,
                    multitalk_block_index=index,
                    grid_shape=grid,
                )
                if (
                    uni3c_execution is not None
                    and uni3c_hidden is not None
                    and index < uni3c_execution.model.config.layers
                ):
                    assert uni3c_freqs is not None
                    assert uni3c_temb is not None
                    uni3c_hidden, residual = uni3c_execution.model.forward_block(
                        index,
                        uni3c_hidden,
                        uni3c_temb,
                        uni3c_freqs,
                    )
                    residual = residual.to(x.dtype) * uni3c_execution.strength
                    if residual.shape[0] != x.shape[0]:
                        residual = residual.repeat(x.shape[0] // residual.shape[0], 1, 1)
                    residual_end = reference_rows + residual.shape[1]
                    x = torch.cat(
                        (
                            x[:, :reference_rows],
                            x[:, reference_rows:residual_end] + residual,
                            x[:, residual_end:],
                        ),
                        dim=1,
                    )
                if motion is not None and index % 5 == 0:
                    assert self.face_adapter is not None
                    x = x + self.face_adapter.fuser_blocks[index // 5](x, motion)
                if mapping_step is not None and index % mapping_step == 0:
                    assert self.vace_blocks is not None
                    assert vace_strength is not None
                    vace_block = cast(
                        "WanVaceAttentionBlock", self.vace_blocks[index // mapping_step]
                    )
                    for control_index, control in enumerate(controls):
                        residual, controls[control_index] = vace_block.forward_vace(
                            control,
                            original,
                            projected_time,
                            freqs,
                            context,
                        )
                        x = x + residual * vace_strength[control_index]
        x = self.head(x, time)
        if padded_context_latents:
            x = x[:, :target_rows]
        if reference_rows:
            x = x[:, reference_rows:]
        batch = x.shape[0]
        patch_t, patch_h, patch_w = self.config.patch_size
        x = x.view(batch, *grid, patch_t, patch_h, patch_w, self.config.out_channels)
        x = torch.einsum("bthwpqrc->bctphqwr", x)
        x = x.reshape(
            batch,
            self.config.out_channels,
            grid[0] * patch_t,
            grid[1] * patch_h,
            grid[2] * patch_w,
        )
        return x[:, :, : original_shape[0], : original_shape[1], : original_shape[2]]


__all__ = [
    "WAN21_CAMERA_1_3B",
    "WAN21_CAMERA_14B",
    "WAN21_FLF_I2V_14B",
    "WAN21_FUN_CONTROL_1_3B",
    "WAN21_FUN_INPAINT_1_3B",
    "WAN21_I2V_14B",
    "WAN21_T2V_1_3B",
    "WAN21_T2V_14B",
    "WAN21_VACE_1_3B",
    "WAN21_VACE_14B",
    "WAN22_ANIMATE_14B",
    "WAN22_BERNINI_14B",
    "WAN22_CAMERA_14B",
    "WAN22_FUN_CONTROL_5B",
    "WAN22_FUN_CONTROL_14B",
    "WAN22_FUN_INPAINT_5B",
    "WAN22_I2V_14B",
    "WAN22_TI2V_5B",
    "Wan21Config",
    "Wan21Model",
    "WanCameraAdapter",
    "WanCameraResidualBlock",
    "WanVaceAttentionBlock",
]
