# Duplicate window indices lose fuse contributions (advanced += does not accumulate)

- **Area:** ComfyUI `comfy/context_windows.py` at
  `0d80858061b511bd38c8cef4c235ef8e01040822` (verified unchanged at
  `c65f9f169cd04637540026f8e4e506715c3c76f0`)
- **Status:** found 2026-08-17; Dinkster's windowed-evaluation contract
  specifies per-occurrence accumulation instead
  (`docs/typed-execution-contracts.md` chapters 9.2/9.3)

## Symptom

Looped context schedules can produce a window whose index list
repeats a frame index. For the flat/pyramid/overlap-linear fuse
profiles, the duplicate positions' contributions are not reliably
accumulated: on the tested torch 2.13 CPU path only one duplicate
position contributes to the fused output, and the other
occurrence's model output and weight are silently dropped.

## Root cause

`create_windows_uniform_looped` builds
`[e % num_frames for e in range(j, j + context_length * context_step, context_step)]`
with no uniqueness guard (`context_windows.py:816-833`), so a strided
modular walk whose cycle is shorter than the requested length
revisits an index: `num_frames=6`, `context_step=2`,
`context_length=4` starting at 0 yields `[0, 2, 4, 0]`.

`IndexListContextWindow.add_window` then accumulates with
advanced-index augmented assignment, `full[idx] += to_add`
(`context_windows.py:85-90`). Advanced assignment is
non-accumulating, and PyTorch documents its behavior as undefined
when the indices contain duplicates (`Tensor.index_put_` with
`accumulate=False`, to which it is equivalent): no backend
guarantees per-occurrence accumulation, and CUDA or other backends
may be nondeterministic. The tested torch 2.13 CPU path observed
last-write behavior, dropping every duplicate occurrence but one.
Both the weighted conds accumulator and the counts
denominator flow through `add_window`
(`combine_context_window_results`, `:729-735`), so the lost weight
partially cancels in the final normalization, but the duplicated
position's model output is still discarded rather than averaged in.
The RELATIVE (running-average) fuse branch iterates positions
explicitly (`:713-728`) and does not have this defect.

## Repro

```python
import torch
x = torch.zeros(3)
x[[0, 2, 0]] += torch.tensor([1.0, 2.0, 3.0])
# x == [3, 0, 2] observed on torch 2.13 CPU: the first write to
# index 0 is lost; undefined for duplicates in general
y = torch.zeros(3)
y.index_put_((torch.tensor([0, 2, 0]),),
             torch.tensor([1.0, 2.0, 3.0]), accumulate=True)
# y == [4, 0, 2]: true per-occurrence accumulation
```

Executed on torch 2.13 CPU. Any looped uniform schedule whose
modular stride cycle is shorter than the window length (e.g. the
`[0, 2, 4, 0]` window above) hits the defective path with the
flat, pyramid, or overlap-linear fuse method.

## Suggested upstream fix

Accumulate with `index_put_(..., accumulate=True)` (or an explicit
per-position loop, as the RELATIVE branch already does) in
`add_window`, so every occurrence of a repeated index contributes
its weighted output and its weight - or deliberately deduplicate
indices at window generation and document that choice.

## How Dinkster handles it

The windowed-evaluation contract defines merge weights and
accumulation per ordered occurrence
(`docs/typed-execution-contracts.md` chapters 9.2/9.3): repeated
indices are first-class occurrences and every occurrence contributes
exactly once. Chapter 9.7 records that core's duplicate-index
advanced update is not the reference behavior for
accumulate-then-normalize profiles.
