# CPU and Apple Silicon execution

`dinkster-serve` supports GPU-less CPU workers and Apple Silicon MPS workers.
Device selection belongs to each worker, not the engine host. Use
`DINKSTER_ACCELERATOR=cpu` to force CPU, `mps` to require MPS, or leave it unset
for automatic selection. See [worker setup](serve-cli.md#cpu-and-apple-silicon-workers).

Environments and downloads are clone-local; neither a system Python install
nor a CUDA runtime is required. The engine environment is torch-free.

## Precision

For a precision-matched comparison, select FP32 text computation in Dinkster.
ComfyUI's BF16 text flag controls stored weights; its CLIP and Qwen text
forward paths compute in FP32. Dinkster's text setting controls computation.
BF16 text remains an explicitly lower-precision option, not a pixel-parity
promise. The reference for these ComfyUI findings is the unmodified
[ComfyUI master checkout](https://github.com/Comfy-Org/ComfyUI/commit/15eb748b3ec5f8a0a2d470b7fb280e2d7579f916).

FP16 MPS attention uses the existing FP32 attention workaround for macOS;
the reference's PyTorch attention path uses FP16. The ComfyUI schedule
finding is a difference of at most 2.38e-7 between CPU interpolation and MPS
interpolation. These findings do not widen golden-test tolerances.
Comfy-aimdo residency is Linux and Windows CUDA only; macOS CPU and MPS use
eager residency without it.

## Artifact pins

Files are downloaded read-only from Hugging Face; no gated models or paid
providers are needed. Revisions below are immutable repository revisions.

| File | Repository / revision | Bytes | SHA256 |
| --- | --- | --- | --- |
| `v1-5-pruned-emaonly-fp16.safetensors` | `Comfy-Org/stable-diffusion-v1-5-archive` / `9cfd069101959ca3828bf9c04a4419870832b74f` | 2132696762 | `e9476a13728cd75d8279f6ec8bad753a66a1957ca375a1464dc63b37db6e3916` |
| `sd_xl_base_1.0.safetensors` | `stabilityai/stable-diffusion-xl-base-1.0` / `462165984030d82259a11f4367a4eed129e94a7b` | 6938078334 | `31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b` |
| `split_files/diffusion_models/flux-2-klein-4b.safetensors` | `Comfy-Org/flux2-klein-4B` / `5f526678002e43af5551dadb73ce2e8c91b43afe` | 7751105712 | `ec3d4e733a771f61c052fb4856c48b336c55eaf2c65487c2a1faeb9bbda7a343` |
| `split_files/text_encoders/qwen_3_4b.safetensors` | Same Klein repository / revision | 8044982048 | `6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a` |
| `split_files/vae/flux2-vae.safetensors` | Same Klein repository / revision | 336211292 | `868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3` |

Place the checkpoints, diffusion model, text encoder, and VAE in their
respective model directories and expose them through ComfyUI's
`--extra-model-paths-config`. Give Dinkster a writable output library mount.
Select precision with `--diffusion-dtype`, `--text-encoder-dtype`, and
`--vae-dtype`.

## Transport and verification limits

Noise seeds, sampler choices/options, and sigma schedules are portable values.
Model-bearing guiders remain resident handles. In-process, shared-memory,
and network transports use the same value contracts and retain worker-owner
checks for models. Remote and cloud workers use their own device selection;
the controller does not substitute a host-local CPU or MPS path. Shared-memory
and inline/network codec tests cover these contracts.

Execution-receipt warnings name combinations without separately minted
determinism receipts; they do not prevent execution.
