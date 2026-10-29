"""Native Chroma and Chroma Radiance diffusion transformers."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from dinkster_inference.chroma import ChromaConfig, ChromaRadianceConfig

from .attention import AttentionKernel, attention_kernel_context, select_attention
from .flux import (
    DoubleStreamBlock,
    EmbedND,
    MLPEmbedder,
    ModulationOut,
    SingleStreamBlock,
    apply_rope_comfy,
    flux_timestep_embedding,
)
from .model_prefetch import PrefetchPlan, close_prefetch_queue, prefetch_queue_pop
from .operations import INITLESS, CastOperations, Operations, ResidencyRouted

_DEFAULT_CHROMA_ATTENTION = select_attention("flux").kernel


@dataclass(frozen=True)
class ChromaRadianceOptions:
    nerf_tile_size: int | None = None
    force_sequential_txt_ids: bool = False

    def __post_init__(self) -> None:
        if self.nerf_tile_size is not None and self.nerf_tile_size < 0:
            raise ValueError("nerf_tile_size must be non-negative or None")


class Approximator(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        layers: int,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.in_proj = operations.linear(in_dim, hidden_dim)
        self.layers = torch.nn.ModuleList(
            MLPEmbedder(hidden_dim, hidden_dim, operations=operations) for _ in range(layers)
        )
        self.norms = torch.nn.ModuleList(operations.rms_norm(hidden_dim) for _ in range(layers))
        self.out_proj = operations.linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for layer, norm in zip(self.layers, self.norms, strict=True):
            x = x + layer(norm(x))
        return self.out_proj(x)


class ChromaFinalLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.norm_final = operations.layer_norm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = operations.linear(hidden_size, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        modulation: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        shift, scale = modulation
        return self.linear(torch.addcmul(shift, self.norm_final(x), 1 + scale))


class NerfEmbedder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        max_freqs: int,
    ) -> None:
        super().__init__()
        self.max_freqs = max_freqs
        self.embedder = torch.nn.Sequential(
            CastOperations(torch.float32).linear(in_channels + max_freqs**2, hidden_size)
        )
        self._position_cache: OrderedDict[tuple[int, torch.device], torch.Tensor] = OrderedDict()

    def _positions(
        self,
        patch_size: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        key = (patch_size, device)
        cached = self._position_cache.get(key)
        if cached is not None:
            self._position_cache.move_to_end(key)
            return cached
        pos = torch.linspace(0, 1, patch_size, device=device, dtype=torch.float32)
        pos_y, pos_x = torch.meshgrid(pos, pos, indexing="ij")
        pos_x = pos_x.reshape(-1, 1, 1)
        pos_y = pos_y.reshape(-1, 1, 1)
        frequencies = torch.linspace(
            0, self.max_freqs - 1, self.max_freqs, device=device, dtype=torch.float32
        )
        frequency_x = frequencies[None, :, None]
        frequency_y = frequencies[None, None, :]
        coefficients = (1 + frequency_x * frequency_y).reciprocal()
        basis_x = torch.cos(pos_x * frequency_x * torch.pi)
        basis_y = torch.cos(pos_y * frequency_y * torch.pi)
        result = (basis_x * basis_y * coefficients).view(1, -1, self.max_freqs**2)
        self._position_cache[key] = result
        if len(self._position_cache) > 4:
            self._position_cache.popitem(last=False)
        return result

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, pixels, _ = inputs.shape
        patch_size = math.isqrt(pixels)
        if patch_size**2 != pixels:
            raise ValueError("NeRF input pixel count must be a square")
        input_dtype = inputs.dtype
        values = inputs.to(dtype=torch.float32)
        positions = self._positions(patch_size, device=inputs.device).expand(batch, -1, -1)
        return self.embedder(torch.cat((values, positions), dim=-1)).to(dtype=input_dtype)


class NerfGLUBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_size_s: int,
        hidden_size_x: int,
        mlp_ratio: int,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.hidden_size_x = hidden_size_x
        self.mlp_width = hidden_size_x * mlp_ratio
        self.param_generator = operations.linear(hidden_size_s, 3 * hidden_size_x * self.mlp_width)
        self.norm = operations.rms_norm(hidden_size_x)

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        gate_params, value_params, output_params = self.param_generator(state).chunk(3, dim=-1)
        gate = gate_params.view(batch, self.hidden_size_x, self.mlp_width)
        value = value_params.view(batch, self.hidden_size_x, self.mlp_width)
        output = output_params.view(batch, self.mlp_width, self.hidden_size_x)
        gate = F.normalize(gate, dim=-2)
        value = F.normalize(value, dim=-2)
        output = F.normalize(output, dim=-2)
        normalized = self.norm(x)
        hidden = F.silu(torch.bmm(normalized, gate)) * torch.bmm(normalized, value)
        return x + torch.bmm(hidden, output)


class NerfFinalLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.norm = operations.rms_norm(hidden_size)
        self.linear = operations.linear(hidden_size, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x.movedim(1, -1))).movedim(-1, 1)


class NerfFinalLayerConv(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        out_channels: int,
        *,
        operations: Operations = INITLESS,
    ) -> None:
        super().__init__()
        self.norm = operations.rms_norm(hidden_size)
        self.conv = operations.conv2d(hidden_size, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x.movedim(1, -1)).movedim(-1, 1))


class Chroma(torch.nn.Module):
    def __init__(
        self,
        config: ChromaConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_CHROMA_ATTENTION,
    ) -> None:
        super().__init__()
        self.config = config
        self._attention_kernel: AttentionKernel = attention_kernel
        self.pe_embedder = EmbedND(config.head_dim, config.theta, config.axes_dim)
        self.img_in = operations.linear(
            config.latent_channels * config.patch_size**2, config.hidden_size
        )
        self.txt_in = operations.linear(config.context_in_dim, config.hidden_size)
        self.distilled_guidance_layer = Approximator(
            config.approximator_input_dim,
            config.approximator_hidden_dim,
            config.hidden_size,
            config.approximator_layers,
            operations=operations,
        )
        self.double_blocks = torch.nn.ModuleList(
            DoubleStreamBlock(
                config.hidden_size,
                config.num_heads,
                config.mlp_hidden_dim,
                qkv_bias=config.qkv_bias,
                modulation=False,
                operations=operations,
                attention_kernel=attention_kernel,
                rope_kernel=apply_rope_comfy,
            )
            for _ in range(config.depth)
        )
        self.single_blocks = torch.nn.ModuleList(
            SingleStreamBlock(
                config.hidden_size,
                config.num_heads,
                config.mlp_hidden_dim,
                modulation=False,
                operations=operations,
                attention_kernel=attention_kernel,
                rope_kernel=apply_rope_comfy,
            )
            for _ in range(config.depth_single_blocks)
        )
        self.final_layer = ChromaFinalLayer(
            config.hidden_size, config.out_channels, operations=operations
        )
        self._double_prefetch = PrefetchPlan(self.double_blocks)
        self._single_prefetch = PrefetchPlan(self.single_blocks)

    @staticmethod
    def _modulation(
        vectors: torch.Tensor,
        offset: int,
    ) -> ModulationOut:
        return ModulationOut(
            vectors[:, offset : offset + 1],
            vectors[:, offset + 1 : offset + 2],
            vectors[:, offset + 2 : offset + 3],
        )

    def _single_modulation(self, vectors: torch.Tensor, index: int) -> ModulationOut:
        return self._modulation(vectors, 3 * index)

    def _double_modulation(
        self,
        vectors: torch.Tensor,
        index: int,
    ) -> tuple[tuple[ModulationOut, ModulationOut], tuple[ModulationOut, ModulationOut]]:
        image_offset = 3 * self.config.depth_single_blocks + 6 * index
        text_offset = image_offset + 6 * self.config.depth
        return (
            (
                self._modulation(vectors, image_offset),
                self._modulation(vectors, image_offset + 3),
            ),
            (
                self._modulation(vectors, text_offset),
                self._modulation(vectors, text_offset + 3),
            ),
        )

    def _distilled_modulation(
        self,
        timesteps: torch.Tensor,
        guidance: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        timestep = flux_timestep_embedding(timesteps, 16).to(reference)
        strength = flux_timestep_embedding(guidance, 16).to(reference)
        indices = flux_timestep_embedding(
            torch.arange(
                self.config.modulation_count,
                device=reference.device,
                dtype=timesteps.dtype,
            ),
            32,
        ).to(reference)
        indices = indices.unsqueeze(0).expand(reference.shape[0], -1, -1)
        condition = torch.cat((timestep, strength), dim=-1)
        condition = condition.unsqueeze(1).expand(-1, self.config.modulation_count, -1)
        return self.distilled_guidance_layer(torch.cat((condition, indices), dim=-1))

    def _transform(
        self,
        image: torch.Tensor,
        image_ids: torch.Tensor,
        context: torch.Tensor,
        text_ids: torch.Tensor,
        timesteps: torch.Tensor,
        guidance: torch.Tensor,
        *,
        image_already_embedded: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not image_already_embedded:
            image = self.img_in(image)
        modulations = self._distilled_modulation(timesteps, guidance, image)
        text = self.txt_in(context)
        positional = self.pe_embedder(torch.cat((text_ids, image_ids), dim=1))
        with attention_kernel_context(self._attention_kernel, image.numel(), device=image.device):
            double_prefetch = self._double_prefetch.make_queue()
            try:
                for index, block in enumerate(self.double_blocks):
                    prefetch_queue_pop(double_prefetch, block)
                    image, text = block(
                        image,
                        text,
                        self._double_modulation(modulations, index),
                        positional,
                    )
                prefetch_queue_pop(double_prefetch, None)
            finally:
                close_prefetch_queue(double_prefetch)
            tokens = torch.cat((text, image), dim=1)
            single_prefetch = self._single_prefetch.make_queue()
            try:
                for index, block in enumerate(self.single_blocks):
                    prefetch_queue_pop(single_prefetch, block)
                    tokens = block(
                        tokens,
                        self._single_modulation(modulations, index),
                        positional,
                    )
                prefetch_queue_pop(single_prefetch, None)
            finally:
                close_prefetch_queue(single_prefetch)
        return tokens[:, text.shape[1] :], modulations

    def _position_ids(
        self,
        batch: int,
        height: int,
        width: int,
        tokens: int,
        reference: torch.Tensor,
        *,
        sequential_text: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_ids = torch.zeros((height, width, 3), device=reference.device, dtype=reference.dtype)
        image_ids[:, :, 1] = torch.arange(height, device=reference.device, dtype=reference.dtype)[
            :, None
        ]
        image_ids[:, :, 2] = torch.arange(width, device=reference.device, dtype=reference.dtype)[
            None, :
        ]
        image_ids = image_ids.reshape(1, height * width, 3).expand(batch, -1, -1)
        text_ids = torch.zeros((batch, tokens, 3), device=reference.device, dtype=reference.dtype)
        if sequential_text:
            text_ids[:, :, 0] = torch.arange(
                tokens, device=reference.device, dtype=reference.dtype
            )[None]
        return image_ids, text_ids

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        config = self.config
        batch, channels, height, width = x.shape
        if channels != config.latent_channels:
            raise ValueError(f"Chroma expects {config.latent_channels} latent channels")
        if timesteps.shape != (batch,) or guidance.shape != (batch,):
            raise ValueError("timesteps and guidance must each have shape [batch]")
        if context.shape[:1] != (batch,) or context.shape[-1] != config.context_in_dim:
            raise ValueError("context has the wrong batch or feature width")
        patch = config.patch_size
        pad_h = (patch - height % patch) % patch
        pad_w = (patch - width % patch) % patch
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="circular")
        h_len = x.shape[-2] // patch
        w_len = x.shape[-1] // patch
        image = (
            x.view(batch, channels, h_len, patch, w_len, patch)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch, h_len * w_len, channels * patch**2)
        )
        image_ids, text_ids = self._position_ids(batch, h_len, w_len, context.shape[1], x)
        output, modulations = self._transform(
            image, image_ids, context, text_ids, timesteps, guidance
        )
        final = self.config.modulation_count - 2
        output = self.final_layer(
            output,
            (
                modulations[:, final : final + 1],
                modulations[:, final + 1 : final + 2],
            ),
        )
        return (
            output.view(batch, h_len, w_len, channels, patch, patch)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, h_len * patch, w_len * patch)[:, :, :height, :width]
        )


class ChromaRadiance(ResidencyRouted, Chroma):
    config: ChromaRadianceConfig

    def __init__(
        self,
        config: ChromaRadianceConfig,
        *,
        operations: Operations = INITLESS,
        attention_kernel: AttentionKernel = _DEFAULT_CHROMA_ATTENTION,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.config = config
        self._attention_kernel: AttentionKernel = attention_kernel
        self.pe_embedder = EmbedND(config.head_dim, config.theta, config.axes_dim)
        self.img_in_patch = operations.conv2d(
            config.latent_channels,
            config.hidden_size,
            config.patch_size,
            stride=config.patch_size,
        )
        self.txt_in = operations.linear(config.context_in_dim, config.hidden_size)
        self.distilled_guidance_layer = Approximator(
            config.approximator_input_dim,
            config.approximator_hidden_dim,
            config.hidden_size,
            config.approximator_layers,
            operations=operations,
        )
        self.double_blocks = torch.nn.ModuleList(
            DoubleStreamBlock(
                config.hidden_size,
                config.num_heads,
                config.mlp_hidden_dim,
                qkv_bias=config.qkv_bias,
                modulation=False,
                operations=operations,
                attention_kernel=attention_kernel,
                rope_kernel=apply_rope_comfy,
            )
            for _ in range(config.depth)
        )
        self.single_blocks = torch.nn.ModuleList(
            SingleStreamBlock(
                config.hidden_size,
                config.num_heads,
                config.mlp_hidden_dim,
                modulation=False,
                operations=operations,
                attention_kernel=attention_kernel,
                rope_kernel=apply_rope_comfy,
            )
            for _ in range(config.depth_single_blocks)
        )
        self._double_prefetch = PrefetchPlan(self.double_blocks)
        self._single_prefetch = PrefetchPlan(self.single_blocks)
        self.nerf_image_embedder = NerfEmbedder(
            config.latent_channels,
            config.nerf_hidden_size,
            config.nerf_max_freqs,
        )
        self.nerf_blocks = torch.nn.ModuleList(
            NerfGLUBlock(
                config.hidden_size,
                config.nerf_hidden_size,
                config.nerf_mlp_ratio,
                operations=operations,
            )
            for _ in range(config.nerf_depth)
        )
        if config.nerf_final_head_type == "conv":
            self.nerf_final_layer_conv = NerfFinalLayerConv(
                config.nerf_hidden_size, config.out_channels, operations=operations
            )
        else:
            self.nerf_final_layer = NerfFinalLayer(
                config.nerf_hidden_size, config.out_channels, operations=operations
            )
        if config.use_x0:
            self.register_buffer("__x0__", torch.empty(0))
        if config.use_sequential_txt_ids:
            self.register_buffer("__sequential__", torch.empty(0))

    def _nerf(
        self,
        image: torch.Tensor,
        hidden: torch.Tensor,
        tile_size: int,
    ) -> torch.Tensor:
        config = self.config
        batch, channels, height, width = image.shape
        patch = config.patch_size
        patches = hidden.shape[1]
        pixels = F.unfold(image, kernel_size=patch, stride=patch).transpose(1, 2)
        states = hidden.reshape(batch * patches, config.hidden_size)
        pixels = pixels.reshape(batch * patches, channels, patch**2).transpose(1, 2)

        def run(pixel_tile: torch.Tensor, state_tile: torch.Tensor) -> torch.Tensor:
            result = self.nerf_image_embedder(pixel_tile)
            for block in self.nerf_blocks:
                result = block(result, state_tile)
            return result

        if tile_size > 0 and patches > tile_size:
            output = torch.cat(
                [
                    run(pixels[start:end], states[start:end])
                    for start in range(0, batch * patches, batch * tile_size)
                    for end in (min(start + batch * tile_size, batch * patches),)
                ],
                dim=0,
            )
        else:
            output = run(pixels, states)
        output = output.transpose(1, 2).reshape(batch, patches, -1).transpose(1, 2)
        output = F.fold(
            output,
            output_size=(height, width),
            kernel_size=patch,
            stride=patch,
        )
        if config.nerf_final_head_type == "conv":
            return self.nerf_final_layer_conv(output)
        return self.nerf_final_layer(output)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: torch.Tensor,
        guidance: torch.Tensor,
        *,
        options: ChromaRadianceOptions | None = None,
    ) -> torch.Tensor:
        config = self.config
        batch, channels, height, width = x.shape
        if channels != config.latent_channels:
            raise ValueError(f"Chroma Radiance expects {config.latent_channels} pixel channels")
        if timesteps.shape != (batch,) or guidance.shape != (batch,):
            raise ValueError("timesteps and guidance must each have shape [batch]")
        if context.shape[:1] != (batch,) or context.shape[-1] != config.context_in_dim:
            raise ValueError("context has the wrong batch or feature width")
        patch = config.patch_size
        pad_h = (patch - height % patch) % patch
        pad_w = (patch - width % patch) % patch
        image = F.pad(x, (0, pad_w, 0, pad_h), mode="circular") if pad_h or pad_w else x
        embedded = self.img_in_patch(image).flatten(2).transpose(1, 2)
        h_len = image.shape[-2] // patch
        w_len = image.shape[-1] // patch
        sequential = config.use_sequential_txt_ids or (
            options is not None and options.force_sequential_txt_ids
        )
        image_ids, text_ids = self._position_ids(
            batch,
            h_len,
            w_len,
            context.shape[1],
            image,
            sequential_text=sequential,
        )
        output, _ = self._transform(
            embedded,
            image_ids,
            context,
            text_ids,
            timesteps,
            guidance,
            image_already_embedded=True,
        )
        tile_size = (
            config.nerf_tile_size
            if options is None or options.nerf_tile_size is None
            else options.nerf_tile_size
        )
        predicted = self._nerf(image, output, tile_size)[:, :, :height, :width]
        if config.use_x0:
            return (x - predicted) / timesteps.view(-1, 1, 1, 1)
        return predicted


__all__ = [
    "Approximator",
    "Chroma",
    "ChromaFinalLayer",
    "ChromaRadiance",
    "ChromaRadianceOptions",
    "NerfEmbedder",
    "NerfFinalLayer",
    "NerfFinalLayerConv",
    "NerfGLUBlock",
]
