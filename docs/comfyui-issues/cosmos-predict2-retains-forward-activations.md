# Cosmos Predict2 / Anima retains forward activations on the module

- **Area:** ComfyUI `comfy/ldm/cosmos/predict2.py` at
  `b78cec879b9460d5cb25228a83a942fb78d2cd24` (inherited by
  `comfy/ldm/anima/model.py`)
- **Status:** found 2026-08 (Dinkster issue #841); worked around in the
  Dinkster ComfyUI benchmark shim

## Symptom

After a prompt finishes and `comfy.model_management.unload_all_models()`
runs, the CUDA caching allocator still holds the last forward pass's
conditioning tensors. For an Anima 1024x1024 run with 512 context tokens
that is 2,105,348 bytes: `crossattn_emb` (2, 512, 1024) bf16 = 2,097,152 B,
`t_embedding_B_T_D` (2, 1, 2048) bf16 = 8,192 B, plus a 4 B fp32
`logit_scale` scalar.

## Root cause

`MiniTrainDIT._forward` (predict2.py lines 886-890, invoked by `forward`)
stashes activations on the module, annotated "for logging purpose":

```python
self.affline_scale_log_info = affline_scale_log_info
self.affline_emb = t_embedding_B_T_D
self.crossattn_emb = crossattn_emb
```

Nothing ever reads these attributes in ComfyUI. `unload_all_models()`
offloads weights but the module object stays alive through the executor
caches, so these plain attributes keep the last forward's CUDA tensors
allocated until the executor cache drops the model or the next forward
overwrites them.

## Repro

Run any Anima (or Cosmos Predict2) prompt, wait for completion, call
`unload_all_models()` plus `gc.collect()` and `torch.cuda.empty_cache()`,
then read `torch.cuda.memory_allocated()`: the conditioning tensors remain.
Walking `gc.get_referrers` over live CUDA tensors shows the owner is the
`Anima` module's `__dict__`.

## Suggested upstream fix

Delete the three assignments; they are dead stores in inference. If the
logging hook is wanted, detach to CPU or guard behind an explicit debug
flag.

## How Dinkster handles it meanwhile

The benchmark shim's unload endpoint releases memory through ComfyUI's own
`free_memory` prompt-queue flag (what `POST /free` sets), which drops the
executor caches and with them the module, instead of calling
`unload_all_models()` directly. The residual ceiling stays strict.
