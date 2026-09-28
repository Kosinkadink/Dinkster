# uni_pc samplers run without any grad guard (commented-out no_grad)

Status: found 2026-07-30; not reported upstream; Dinkster immune (parity
adapter fixed to reproduce the server's inference_mode environment).

Baseline: ComfyUI f4b99bc62389af315013dda85f24f2bbd262b686.

## Symptom

Calling `nodes.common_ksampler` (or the sampler layer) directly with
`sampler_name="uni_pc"` / `"uni_pc_bh2"` outside ComfyUI's server
process retains the full denoiser autograd graph across every sampling
step. Memory grows linearly per step (~2.5 GiB/step for SD1.5 fp16 at
64x64 latents, batch 2 cond+uncond) until CUDA OOM. Observed in the
W0-SD15-INPAINT parity gate: 22.42 GiB allocated by step 9/20 on a
24 GB RTX 4090, OOM inside attention `to_k` / `F.linear` (incidental
allocation site).

## Root cause

Three facts combine:

1. `comfy/extra_samplers/uni_pc.py` has no grad guard anywhere on its
   sampling path: `sample_unipc` / `sample_unipc_bh2` (lines 846-873)
   are undecorated, and the internal guard in `UniPC.sample` is
   literally commented out at line 711 (`# with torch.no_grad():`).
   `x` and `model_prev_list` (lines 715, 725, 744-747) chain `grad_fn`
   across steps, so the entire graph from step 0 is retained.
2. Every k-diffusion sampler is `@torch.no_grad()` decorated
   (`comfy/k_diffusion/sampling.py:189, 215, 239, ...`), so only the
   uni_pc family is exposed.
3. On Linux, model weights require grad by default:
   `comfy.ops.disable_weight_init.Linear` falls through to plain
   `torch.nn.Linear.__init__` (`comfy/ops.py:299-301`; the
   `requires_grad=False` construction at ops.py:329-344 is the
   Windows/dynamic-VRAM path only), and nothing in `comfy/sd.py`
   flips `requires_grad` after load.

Inside ComfyUI's own server the bug is masked because all node
execution is wrapped in `torch.inference_mode()`
(`execution.py:709`). Any consumer that imports comfy and calls the
sampler layer directly (scripts, tests, embedders, parity harnesses)
hits it.

## Repro

On the pinned checkout, load any SD1.5 checkpoint via
`nodes.CheckpointLoaderSimple`, then call `nodes.common_ksampler(...,
sampler_name="uni_pc_bh2", ...)` with no surrounding
`torch.no_grad()` / `torch.inference_mode()`. Watch
`torch.cuda.memory_allocated()` grow linearly per step; the same call
with `sampler_name="euler"` stays flat.

## Suggested upstream fix

Restore the guard: decorate `sample_unipc` (and thereby
`sample_unipc_bh2`) with `@torch.no_grad()`, matching the k-diffusion
samplers, or un-comment the `with torch.no_grad():` in
`UniPC.sample`. Either is a one-line change with no numerics impact.

## Dinkster handling

Dinkster's own samplers do not have this hazard. The parity ComfyUI
adapter wraps its request handling in `torch.inference_mode()` to reproduce
the executed server environment (`execution.py:709`), which both
fixes the OOM and is the faithful reference behavior.
