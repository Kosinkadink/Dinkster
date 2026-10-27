## LoRA and model patching

Native model patches select discrete, continuous EDM, continuous V, Stable
Cascade, SD3, AuraFlow, and Flux sampling behavior; set flow noise scale;
rescale classifier-free guidance; select compute dtype; and select a supported
dense attention backend. They use the pinned `dinkster_inference` sampler and
model patcher.

`LoraSave` extracts standard low-rank or full-difference LoRA checkpoints from
model and text-encoder differences, optionally includes bias differences, and
saves a safetensors file to the selected writable mount.

`ChromaRadianceOptions` applies Chroma Radiance transformer options over an
inclusive sigma range while optionally retaining an existing model wrapper.
`FluxKVCache` reuses reference-image attention keys and values independently
for each model block, including changes in prompt batch size.
`HiDreamO1PatchSeamSmoothing` blends shifted patch-grid predictions over a
sampling window, with fixed or ramped pass counts and average, Hann-window,
or median blending while preserving the image edges.

`MultiGPU_WorkUnits` creates reusable model deep clones for the visible devices.
`SelectModelDevice`, `SelectCLIPDevice`, and `SelectVAEDevice` route their
resident values to the selected default, CPU, or portable `gpu:N` device where
supported. Unavailable devices leave the value unchanged; VAE routing does not
offer CPU.

`HypernetworkLoader` loads a bound hypernetwork asset and applies its key/value
networks to self- and cross-attention on a cloned model, with adjustable strength.

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
patcher. LoRA loading, scheduled patches, and context-window patches are not
currently exposed by Dinkster.

`ModelPatchLoader` loads bound model-patch assets as governed `MODEL_PATCH`
values. Supported checkpoint formats include Anima LLLite, Qwen blockwise and
Qwen Image 2.1 control, SigLIP feature projection, ZImage control, MiniMax H3
control, Wan Uni3C and MultiTalk, LTX duration heads, and SUPIR. Loading a patch
does not apply it to a model.

Self-Attention Guidance, Perturbed Attention Guidance, and both Skip Layer
Guidance DiT nodes clone a model and install sampler or attention callbacks.
They use the inference sampler's conditional-routing and model-patch mechanisms;
they do not load or apply LoRA weights.
`SkipLayerGuidanceSD3` applies the double-block skip-layer guidance controls used
by SD3 workflows.

ControlNet values use the native `CONTROL_NET` type and stay resident in their
owning worker, with the same governed lifetime as other loaded models.

Hook scheduling creates and interpolates keyframes, assigns cloned keyframes to
hook groups, and combines up to eight hook groups. Hook values remain governed
resident objects; float-strength tensors are materialized only by the scheduling
consumer while retaining their dtype and ordering.

Resident CLIP values retain their conditioning and scheduled hook dependencies.
Releasing a retained hook group or keyframe group invalidates dependent cached
CLIP outputs without invalidating unrelated encoders.

Hook scheduling, combinations, CLIP schedules, and conditioning preserve retained
weight dependencies. Releasing those weights invalidates dependent outputs;
replacing a schedule releases the output's dependency on the old keyframes.
