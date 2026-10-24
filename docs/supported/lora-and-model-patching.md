## LoRA and model patching

Native model patches select discrete, continuous EDM, continuous V, Stable
Cascade, SD3, AuraFlow, and Flux sampling behavior; set flow noise scale;
rescale classifier-free guidance; select compute dtype; and select a supported
dense attention backend. They use the pinned `dinkster_inference` sampler and
model patcher.

`ScaleROPE` scales and shifts spatial and temporal rotary coordinates on a cloned
model. `DifferentialDiffusion` applies timestep-dependent denoise masks, with
optional blending strength, without changing the source model.

`TomePatchModel` merges self-attention tokens, `HyperTile` tiles compatible
self-attention layers, and `PatchModelAddDownscale` applies sigma-windowed U-Net
block downscaling with a matching skip-path resize. Each creates a cloned model.

The native attention multiplier nodes scale self-attention, cross-attention,
CLIP attention, and temporal-attention projection weights through the model
patcher. LoRA loading, scheduled patches, context-window patches, and
model-family patch formats are not currently exposed by Dinkster.

ControlNet values use the native `CONTROL_NET` type and stay resident in their
owning worker, with the same governed lifetime as other loaded models.
