# RenormCFG crashes on batch > 1 when renorm_cfg > 0

Status: found 2026-08-27; not reported upstream; Dinkster mirrors the
batch-1 contract and documents it at the factory.

Baseline: ComfyUI b78cec879b9460d5cb25228a83a942fb78d2cd24.

## Symptom

Sampling with the `RenormCFG` node patched onto a model errors with
`RuntimeError: Boolean value of Tensor with more than one value is
ambiguous` whenever the latent batch size is greater than 1 and
`renorm_cfg` is positive (its default is 1.0).

## Root cause

`comfy_extras/nodes_lumina2.py` (`renorm_cfg_func`) computes per-sample
norms with `keepdim=True`, giving `ori_pos_norm` / `new_pos_norm` shape
`(B, 1, 1, 1)`, then branches on them directly:

```python
if new_pos_norm >= max_new_norm:
    half_eps = half_eps * (max_new_norm / new_pos_norm)
```

For `B == 1` the comparison collapses to a single element and Python's
truthiness works; for `B > 1` the elementwise comparison yields a
multi-element bool tensor and raising is torch's defined behavior. The
truncation gate `if timestep[0] < cfg_trunc:` also reads only element 0,
so a per-sample decision is silently global, but the norm branch crashes
before that matters.

## Repro

On the pinned checkout, register the `RenormCFG` cfg function on any
model (`cfg_trunc=100.0`, `renorm_cfg=1.0`) and run
`comfy.samplers.sampling_function` with a `(2, 4, 8, 8)` latent.
Executed on 2026-08-27 via the Dinkster golden generator harness
(`tools/gen_cfg_transform_goldens.py` `run_case` with
`shape=(2, 4, 8, 8)`): crashes at `if new_pos_norm >= max_new_norm:`.

## Suggested upstream fix

Make the renorm branch per-sample: replace the scalar branch with a
`torch.where` on the `(B, 1, 1, 1)` comparison, e.g.
`scale = torch.where(new_pos_norm >= max_new_norm, max_new_norm /
new_pos_norm, torch.ones_like(new_pos_norm)); half_eps = half_eps *
scale`. The truncation gate needs a per-sample treatment too if batched
sigmas can differ.

## Dinkster handling

`renorm_cfg` in `dinkster_inference_torch/guidance_transforms.py` mirrors
the reference math faithfully, including the batch-1-only contract when
`renorm > 0`; the limitation is documented in its docstring and goldens
cover batch 1. Fixing beyond the reference would break bit-exact parity.
