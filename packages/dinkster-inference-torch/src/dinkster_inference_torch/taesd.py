"""Native TAESD architecture matching comfy/taesd/taesd.py @ f4b99bc."""

from __future__ import annotations

import torch
from dinkster_inference.taesd import TAESDConfig, TAESDMemoryEstimator, taesd_descriptor

from .autoencoder_kl import crop_to_multiple, process_input, process_output
from .codecs import CodecPlugin
from .operations import INITLESS, Operations


class Clamp(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3.0) * 3.0


class Block(torch.nn.Module):
    def __init__(self, inc: int, out: int, *, operations: Operations = INITLESS) -> None:
        super().__init__()
        self.conv = torch.nn.Sequential(
            operations.conv2d(inc, out, 3, padding=1),
            torch.nn.ReLU(),
            operations.conv2d(out, out, 3, padding=1),
            torch.nn.ReLU(),
            operations.conv2d(out, out, 3, padding=1),
        )
        self.skip = operations.conv2d(inc, out, 1) if inc != out else torch.nn.Identity()
        if isinstance(self.skip, torch.nn.Conv2d) and inc != out:
            self.skip.bias = None
        self.fuse = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(self.conv(x) + self.skip(x))


def _conv(
    inc: int, out: int, operations: Operations, *, stride: int = 1, bias: bool = True
) -> torch.nn.Conv2d:
    layer = operations.conv2d(inc, out, 3, stride=stride, padding=1)
    if not bias:
        layer.bias = None
    return layer


class TAESDEncoder(torch.nn.Sequential):
    def __init__(self, *, operations: Operations = INITLESS) -> None:
        layers: list[torch.nn.Module] = [
            _conv(3, 64, operations),
            Block(64, 64, operations=operations),
        ]
        for _ in range(3):
            layers += [
                _conv(64, 64, operations, stride=2, bias=False),
                *(Block(64, 64, operations=operations) for _ in range(3)),
            ]
        layers.append(_conv(64, 4, operations))
        super().__init__(*layers)

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        # TAESD.encode's own mapping. The outer VAE facade maps [0, 1]
        # to [-1, 1], so the composition presents the original pixels.
        return self(content * 0.5 + 0.5)


class TAESDDecoder(torch.nn.Sequential):
    def __init__(self, *, operations: Operations = INITLESS) -> None:
        layers: list[torch.nn.Module] = [Clamp(), _conv(4, 64, operations), torch.nn.ReLU()]
        for _ in range(3):
            layers += [
                *(Block(64, 64, operations=operations) for _ in range(3)),
                torch.nn.Upsample(scale_factor=2),
                _conv(64, 64, operations, bias=False),
            ]
        layers += [Block(64, 64, operations=operations), _conv(64, 3, operations)]
        super().__init__(*layers)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4 or latent.shape[1] != 4:
            raise ValueError(
                f"TAESD decode requires NCHW latent width 4, got {tuple(latent.shape)}"
            )
        # TAESD.decode returns the VAE's [-1, 1] content convention.
        return self(latent).sub(0.5).mul(2.0)


class TAESD(torch.nn.Module):
    """Loaded TAESD facade owning latent scaling and compute dtype."""

    def __init__(
        self,
        config: TAESDConfig,
        encoder: TAESDEncoder,
        decoder: TAESDDecoder,
        *,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder
        self.compute_dtype = compute_dtype

    def encode(self, content: torch.Tensor) -> torch.Tensor:
        scale = self.config.vae_scale
        assert scale is not None
        return (
            self.encoder.encode(content.to(dtype=self.compute_dtype))
            .div(scale)
            .add(self.config.vae_shift)
            .float()
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        scale = self.config.vae_scale
        assert scale is not None
        scaled = latent.to(dtype=self.compute_dtype).sub(self.config.vae_shift).mul(scale)
        return self.decoder.decode(scaled).float()


def taesd_codec_plugin(model: TAESD) -> CodecPlugin:
    def crop(content: torch.Tensor) -> torch.Tensor:
        return crop_to_multiple(content, model.config.spatial_downscale)

    return CodecPlugin(
        descriptor=taesd_descriptor(model.config.family),
        encoder=model,
        decoder=model,
        memory=TAESDMemoryEstimator(),
        content_crop=crop,
        content_in=process_input,
        content_out=process_output,
    )


__all__ = [
    "Block",
    "Clamp",
    "TAESD",
    "TAESDDecoder",
    "TAESDEncoder",
    "taesd_codec_plugin",
]
