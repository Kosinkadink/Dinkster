# ComfyUI capability evidence

Canonical data: [comfy-capability-evidence.json](comfy-capability-evidence.json).

Workflow templates: `d3b4a9e89573162b005961865164c18c8ae2206b`. ComfyUI source: `15eb748b3ec5f8a0a2d470b7fb280e2d7579f916` (896 statically registered node IDs).

## Evidence tiers

| Tier | Counts as supported | Requirement |
| --- | --- | --- |
| T0 | no | Schema translation only. |
| T1 | no | Compatibility alias demonstrably lowers to a native operation. |
| T2 | yes | Native execution with focused tests. |
| T3 | yes | Pinned authoritative-source correctness using an official checkpoint. |
| T4 | yes | Matched source-versus-Dinkster official-weight same-GPU end-to-end evidence covering correctness, cold and warm performance, GPU and host memory, residual memory, and template LoRA and adapter paths. |

T0 and T1 are translation evidence only and never establish support. Missing evidence remains `absent`, `refused`, or `unverified`; the census does not infer support from a translated schema.

## Current ledger

935 capabilities: proven 345, unverified 0, absent 587, refused 3. Supported capability claims: 18.

| Native model family | Exposure | State | Highest tier | Evidence |
| --- | --- | --- | --- | ---: |
| `dinkster.anima` | implemented | proven | T2 | 1 |
| `dinkster.chroma` | supported | proven | T3 | 3 |
| `dinkster.chroma_radiance` | supported | proven | T3 | 3 |
| `dinkster.flux2_dev` | implemented | proven | T2 | 1 |
| `dinkster.flux2_klein_4b` | implemented | proven | T2 | 1 |
| `dinkster.flux2_klein_9b` | implemented | proven | T2 | 1 |
| `dinkster.flux_dev` | supported | proven | T2 | 1 |
| `dinkster.flux_schnell` | supported | proven | T2 | 1 |
| `dinkster.ideogram4` | supported | proven | T3 | 3 |
| `dinkster.krea2` | implemented | proven | T2 | 1 |
| `dinkster.ltxav` | implemented | proven | T2 | 1 |
| `dinkster.ltxv` | implemented | proven | T2 | 1 |
| `dinkster.lumina2` | supported | proven | T3 | 3 |
| `dinkster.minimax_h3` | supported | proven | T2 | 1 |
| `dinkster.minimax_music3` | supported | proven | T3 | 3 |
| `dinkster.qwen_image` | supported | proven | T2 | 1 |
| `dinkster.sd15` | supported | proven | T2 | 1 |
| `dinkster.sdxl` | supported | proven | T2 | 1 |
| `dinkster.sdxl_refiner` | supported | proven | T2 | 1 |
| `dinkster.seedvr2` | supported | proven | T3 | 3 |
| `dinkster.trellis2` | supported | proven | T3 | 3 |
| `dinkster.triposplat` | implemented | proven | T2 | 1 |
| `dinkster.wan21` | supported | proven | T2 | 1 |
| `dinkster.wan22` | supported | proven | T2 | 1 |
| `dinkster.z_image` | supported | proven | T2 | 1 |
| `dinkster.z_image_pixel_space` | supported | proven | T2 | 1 |

## Official template feature paths

LoRA paths: 115. Adapter paths: 16. Explicit low-step LoRA paths: 4.

| Low-step LoRA template | Steps | LoRA node | Sampler node |
| --- | ---: | --- | --- |
| `image_krea2_turbo_int8_image_style_reference.json` | 8 | `LoraLoaderModelOnly` | `BasicScheduler` |
| `image_krea2_turbo_t2i.json` | 8 | `LoraLoaderModelOnly` | `KSampler` |
| `image_krea2_turbo_t2i_int8.json` | 8 | `LoraLoaderModelOnly` | `KSampler` |
| `image_qwen_image_2512_with_2steps_lora.json` | 2 | `LoraLoaderModelOnly` | `KSampler` |
