## LoRA and model patching

Native model patches select discrete, continuous EDM, continuous V, Stable
Cascade, SD3, AuraFlow, and Flux sampling behavior; set flow noise scale;
rescale classifier-free guidance; select compute dtype; and select a supported
dense attention backend. They use the pinned `dinkster_inference` sampler and
model patcher.

`MultiGPU_WorkUnits` creates reusable model deep clones for the visible devices.
`SelectModelDevice`, `SelectCLIPDevice`, and `SelectVAEDevice` route their
resident values to the selected default, CPU, or portable `gpu:N` device where
supported. Unavailable devices leave the value unchanged; VAE routing does not
offer CPU.

`ScaleROPE` scales and shifts spatial and temporal rotary coordinates on a cloned
model. `DifferentialDiffusion` applies timestep-dependent denoise masks, with
optional blending strength, without changing the source model.

`TorchCompileModel` clones a model, compiles its diffusion model with the
selected backend, and applies the compiled module only for model execution.
`wanBlockSwap` is retained as a deprecated no-op for existing workflows.

`TomePatchModel` merges self-attention tokens, `HyperTile` tiles compatible
self-attention layers, and `PatchModelAddDownscale` applies sigma-windowed U-Net
block downscaling with a matching skip-path resize. Each creates a cloned model.

`BlockSparseAttention` uses the pinned Comfy Kitchen sparse kernels for eligible
self-attention calls and preserves the selected dense backend for ineligible
calls. MiniMax-H3 models use the runtime's chunked sparse producer path.

The native attention multiplier nodes scale self-attention, cross-attention,
CLIP attention, and temporal-attention projection weights through the model
patcher. LoRA loading, scheduled patches, context-window patches, and
model-family patch formats are not currently exposed by Dinkster.

Self-Attention Guidance, Perturbed Attention Guidance, and both Skip Layer
Guidance DiT nodes clone a model and install sampler or attention callbacks.
They use the inference sampler's conditional-routing and model-patch mechanisms;
they do not load or apply LoRA weights.

ControlNet values use the native `CONTROL_NET` type and stay resident in their
owning worker, with the same governed lifetime as other loaded models.

Hook scheduling creates and interpolates keyframes, assigns cloned keyframes to
hook groups, and combines up to eight hook groups. Hook values remain governed
resident objects; float-strength tensors are materialized only by the scheduling
consumer while retaining their dtype and ordering.

Resident CLIP values retain their conditioning and scheduled hook dependencies.
Releasing a retained hook group or keyframe group invalidates dependent cached
CLIP outputs without invalidating unrelated encoders.
