## LoRA and model patching

`ScaleROPE` scales and shifts spatial and temporal rotary coordinates on a cloned
model. `DifferentialDiffusion` applies timestep-dependent denoise masks, with
optional blending strength, without changing the source model.

The native attention multiplier nodes scale self-attention, cross-attention,
CLIP attention, and temporal-attention projection weights through the model
patcher. LoRA loading, scheduled patches, other model patch nodes,
context-window patches, and model-family patch formats are not currently
exposed by Dinkster.

ControlNet values use the native `CONTROL_NET` type and stay resident in their
owning worker, with the same governed lifetime as other loaded models.
