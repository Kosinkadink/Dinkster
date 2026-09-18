# Native checkpoint RAM calibration

Date: 2026-07-27.

## Method

`tools/measure_native_checkpoint_ram.py` computes the same runtime identity
as the native host policy, exposes the source checkpoint to a fresh child by
temporary symlink, imports the production native arm and torch, and records
the child's Linux `VmHWM` before and after the production `_load_runtime`
seam plus construction of the production `NativeRuntimeHandle`. Handle
construction calls the lazily imported `enroll_assembled`, so these revised
numbers include safetensors header parsing, CPU module assembly, and per-unit
residency enrollment. The tool deliberately stops before any residency stage
placement. Measurements used `.venv-gpu` (torch 2.13.0+cu130); no module was
moved to CUDA.

The combined Flux file was first verified from its safetensors header: its
17,246,524,772 bytes contain 780 diffusion tensors, 418 text-encoder tensors,
and 244 VAE tensors. The revised calibration runs that combined file twice in
fresh children and rechecks the documented SD1.5 file once:

```text
.venv-gpu/bin/python tools/measure_native_checkpoint_ram.py /home/kosin/ComfyUI/models/diffusion_models/flux1-dev-fp8.safetensors
.venv-gpu/bin/python tools/measure_native_checkpoint_ram.py /home/kosin/ComfyUI/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors
```

## Results

| Checkpoint | File bytes | Run | Baseline VmHWM bytes | Peak VmHWM bytes | Peak/file factor |
|---|---:|---:|---:|---:|---:|
| `flux1-dev-fp8.safetensors` (combined) | 17,246,524,772 | 1 | 542,330,880 | 18,043,236,352 | 1.046195 |
| `flux1-dev-fp8.safetensors` (combined) | 17,246,524,772 | 2 | 542,285,824 | 18,047,025,152 | 1.046415 |
| `v1-5-pruned-emaonly-fp16.safetensors` | 2,132,696,762 | 1 | 542,457,856 | 2,889,359,360 | 1.354791 |

The two combined-Flux fresh-child factors differ by 0.000220. The
handle-inclusive SD1.5 result is within the prior pre-enrollment fresh-child
range of 1.354560-1.358514. The prior runs remain useful conservative evidence,
but only the table above includes production handle construction and per-unit
enrollment.

## Recommendation

The Comfy-path factor 2.6 was calibrated against a 2.487724 measured peak/file
factor because that loader's child reached a 5,305,561,088-byte peak. Native
CPU assembly peaked at 1.358514 for these SD1.5 files, about 45 percent below
the Comfy measured factor. Use **1.5** as the conservative native checkpoint
RAM factor: the new handle-inclusive maximum of 1.354791 leaves 10.7 percent
headroom, while the highest accepted current or prior measurement (1.358514)
still leaves 10.4 percent. The combined Flux fp8 factors are lower, not a reason
to reduce the shared policy. The factor remains an interim static admission
safeguard and fallback for devices outside future dynamic offload coverage,
not a claim of dynamic memory parity.
