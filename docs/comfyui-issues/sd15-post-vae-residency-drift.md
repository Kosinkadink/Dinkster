# SD1.5 output changes after VAE residency transition

Status: verified at ComfyUI `b5cc8830279eae909a59de030af1e50761c36751` with
Torch 2.13.0+cu130 on an RTX 4090; Dinkster normalizes model residency before
sampling.

## Symptom

An SD1.5 text-to-image graph produces different decoded bytes on its first and
second execution in one process. A fresh process reproduces the first output.
The graph uses the same checkpoint, prompt, seed, five Euler steps, normal
scheduler, and 256x256 latent on every execution.

## Localization

The drift is localized to the residency transition after the first VAE decode.
The next sampling load starts from a partially offloaded `ModelPatcher` state
and produces different latent bytes. Calling `unload_model_and_clones(model)`
before sampling restores the complete offload state and makes both executions
match the fresh process byte for byte. Unloading only the VAE does not fix the
drift. The exact differing field or operation inside the reload path has not
been isolated.

Torch's lazy CUDA SDPA priority mutation is a separate known source of cold/warm
drift. Initializing that chooser and restoring the declared priority did not
remove this residency-dependent difference on the tested runtime.

## Reproduction

Load `v1-5-pruned-emaonly-fp16.safetensors`, encode the prompts, sample, and
decode twice without recreating the process. Hash the contiguous float32 image
bytes. The first and second hashes differ. Repeat with
`unload_model_and_clones(model)` immediately before each sampling call; both
hashes and a fresh-process hash are identical.

## Suggested upstream fix

Make partial unload and reload preserve the same weights, casts, and execution
path as a complete model offload and reload. A narrower fix should replace the
full-offload normalization once the differing residency field is isolated.
