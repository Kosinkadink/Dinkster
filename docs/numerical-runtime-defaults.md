# Numerical runtime defaults

This inventory pins ComfyUI commit
`b5cc8830279eae909a59de030af1e50761c36751`. Line numbers refer to that
commit. "Default" means behavior without an explicit CLI, workflow, or
checkpoint override. Optional providers are listed because their presence can
change arithmetic, even when they are absent from both projects' managed
environments.

## ComfyUI baseline

| Area | ComfyUI default | Pinned source |
| --- | --- | --- |
| Global attention route | Explicit Sage, Flash, and Kitchen requests win. Otherwise use xFormers when its optional package is usable, then PyTorch SDPA, then split or subquadratic attention. CPU uses subquadratic attention. Sage and Kitchen are not automatic defaults. | `comfy/ldm/modules/attention.py:891-953`; `comfy/model_management.py:403-424,462-475` |
| Attention dtype eligibility | PyTorch SDPA accepts fp32, fp16, and bf16. xFormers and explicit Flash defer dtype eligibility to their providers. Sage falls back to SDPA when low-precision attention is disabled; Sage 3 additionally requires CUDA fp16/bf16. Kitchen INT8 falls back to SDPA for fp32 when low-precision attention is disabled. | `comfy/ldm/modules/attention.py:511-620,639-686,738-750,848-880` |
| Checkpoint-selected Kitchen attention | A layer whose checkpoint metadata names `comfy_kitchen_int8` uses Kitchen INT8 when available. This is independent of the global default. | `comfy/ldm/modules/attention.py:76-100` |
| SDPA priority | For queries with at least 131072 elements: Flash, cuDNN, efficient, math. Smaller queries use PyTorch's chooser without an imposed order. | `comfy/ops.py:65-96` |
| SDPA enable flags | When PyTorch attention is selected, math, Flash, and memory-efficient SDPA are enabled. cuDNN is not explicitly changed. | `comfy/model_management.py:462-475,547-550` |
| Reduced-precision SDPA reduction | fp16/bf16 math-SDPA reduction is enabled when the Torch API exists. | `comfy/model_management.py:567-571` |
| TF32 and fp16 matmul accumulation | No global TF32 value is set. One cuDNN convolution workaround explicitly allows TF32. fp16 accumulation is off unless the `fp16_accumulation` fast option is requested. | `comfy/ops.py:613-622`; `comfy/model_management.py:553-560` |
| Attention upcast | fp16 attention is upcast to fp32 only when explicitly forced or on macOS 14.5 and newer. `--dont-upcast-attention` overrides it. | `comfy/model_management.py:1770-1780`; `comfy/ldm/modules/attention.py:103-115` |
| Plain linear, norm, and embedding weights | Parameters retain checkpoint storage and are cast at use to the input/compute dtype. Linear, LayerNorm, RMSNorm, and Embedding all use the shared cast route. | `comfy/ops.py:339-440,523-577,667-696,742-805` |
| Quantized linear and embedding weights | Quantized storage and scales remain in their encoded layouts. Native quantized kernels are used when eligible; weight-only or full-precision fallback dequantizes to the input/compute dtype. Unquantized weights in a mixed-precision module materialize directly in compute dtype. | `comfy/ops.py:1158-1278,1347-1484` |
| Kitchen Triton backend | Off by default; `--enable-triton-backend` opts in and `--disable-triton-backend` wins. | `comfy/cli_args.py:118-120`; `comfy/quant_ops.py:21-43` |
| `torch.compile` | Model execution is eager by default. The model-core compile branch is disabled. | `comfy/ldm/modules/diffusionmodules/mmdit.py:885-890` |
| VAE dtype | An explicit fp16/bf16/fp32 override wins. Otherwise choose the first family-allowed, device-supported fp16 or bf16 dtype, then fp32. | `comfy/model_management.py:1290-1305`; `comfy/sd.py:1097-1123` |
| VAE direct/tiled mode | Encode and decode are direct and untiled by default, then retry tiled only after an accelerator OOM. | `comfy/sd.py:1251-1327,1392-1444` |
| Generic 2D VAE tiles | Decode tile 64x64 with overlap 16x16; encode tile 512x512 with overlap 64x64. Three aspect ratios are averaged. | `comfy/sd.py:1164-1176,1196-1207` |
| Generic 3D VAE tiles | Decode tile 999x32x32 with overlap 1x8x8; encode tile 9999x512x512 with overlap 1x64x64. | `comfy/sd.py:1188-1190,1228-1230` |

## Dinkster correspondence

`Aligned (changed)` identifies defaults that differed before this inventory.
All other rows already matched. There are no deliberately retained numerical
default differences.

| Area | Dinkster default | Status | Source |
| --- | --- | --- | --- |
| Global attention route | Automatic routing uses SDPA (bounded attention on ROCm VAE and as VAE OOM fallback). Optional Sage remains explicit and never promotes itself merely because it is installed. The managed environment declares neither xFormers nor Flash Attention, matching the effective ComfyUI baseline environment. | **Aligned (changed)** | `packages/dinkster-protocol/src/dinkster_protocol/attention.py:47-62`; `packages/dinkster-inference-torch/src/dinkster_inference_torch/attention.py:1818-1876` |
| Attention dtype eligibility | SDPA supports fp32/fp16/bf16. Explicit provider routes validate their own eligibility and fall back to SDPA. Kitchen INT8 admits fp32/fp16/bf16 and returns the input dtype. | **Aligned (changed)** | `packages/dinkster-inference-torch/src/dinkster_inference_torch/attention.py:741-865,1067-1129,1198-1230,1880-1970` |
| Checkpoint-selected Kitchen attention | Quantized checkpoint metadata selects Kitchen INT8 only for the layers and model integrations that declare it; it does not alter the global route. | Aligned | `packages/dinkster-inference/src/dinkster_inference/quantization.py:123-133,254-330`; `packages/dinkster-inference-torch/src/dinkster_inference_torch/attention.py:1940-1964` |
| SDPA priority | At 131072 query elements or more: Flash, cuDNN, efficient, math. Smaller queries retain the Torch chooser. | Aligned | `packages/dinkster-inference-torch/src/dinkster_inference_torch/attention.py:61-96,373-470` |
| SDPA enable flags | The priority context enables the listed SDPA backends for the scoped call and restores prior process state. | Aligned | `packages/dinkster-inference-torch/src/dinkster_inference_torch/attention.py:119-139,410-425` |
| Reduced-precision SDPA reduction | fp16/bf16 math-SDPA reduction is enabled when the Torch API exists. | Aligned | `packages/dinkster-inference-torch/src/dinkster_inference_torch/attention.py:428-446` |
| TF32 and fp16 matmul accumulation | No runtime-global TF32 or fp16-accumulation override is applied; Torch defaults remain in force. | Aligned | `packages/dinkster-inference-torch/src/dinkster_inference_torch/attention.py:428-449` |
| Attention upcast | fp16 SDPA upcasts on the same macOS 14.5+ host rule and returns fp16 output; no other automatic upcast is added. | Aligned | `packages/dinkster-inference-torch/src/dinkster_inference_torch/attention.py:447-449,774-865`; `packages/dinkster-inference-torch/src/dinkster_inference_torch/dtype_policy.py:135-155` |
| Plain linear, norm, and embedding weights | Cast-at-use modules retain storage dtype and materialize linear, embedding, LayerNorm, and RMSNorm state in their bound compute dtype. | Aligned | `packages/dinkster-inference-torch/src/dinkster_inference_torch/operations.py:769-798,801-895,898-916,954-1054` |
| Quantized linear and embedding weights | Encoded storage remains encoded. Native Kitchen kernels run when eligible; fallback dequantizes weights or selected embeddings to the bound compute/output dtype. | Aligned | `packages/dinkster-inference-torch/src/dinkster_inference_torch/quant_linear.py:115-216,721-742,800-878` |
| Kitchen Triton backend | Worker attention capability discovery disables Kitchen's Triton backend before model execution; accelerated CUDA/HIP and eager backends remain eligible. | **Aligned (changed)** | `packages/dinkster-inference-torch/src/dinkster_inference_torch/attention.py:998-1045`; `packages/dinkster-workers/src/dinkster_workers/host.py:235-280` |
| `torch.compile` | Normal worker execution is eager. Compile is an explicitly tested alternate mode, not a runtime default. | Aligned | `packages/dinkster-workers/src/dinkster_workers/backend_env.py:25-32` |
| VAE dtype | Each family records the same ordered allowed dtypes; runtime selection chooses its first device-supported value. | Aligned | `packages/dinkster-inference/src/dinkster_inference/identity.py:317-345`; `packages/dinkster-inference-torch/src/dinkster_inference_torch/wiring.py:2483-2573` |
| VAE direct/tiled mode | Native encode/decode starts direct and retries tiled once only for accelerator OOM. | Aligned | `packages/dinkster-native/src/dinkster_native/native_arm_core.py:980-1107`; `packages/dinkster-native/src/dinkster_native/nodes_sampling_runtime.py:1019-1094` |
| Generic 2D VAE tiles | Decode 64x64/16x16 and encode 512x512/64x64; the same three aspect ratios are averaged in reference order. | Aligned | `packages/dinkster-inference/src/dinkster_inference/autoencoder_kl.py:710-731`; `packages/dinkster-inference-torch/src/dinkster_inference_torch/codecs.py:45-58,163-196` |
| Generic 3D VAE tiles | Decode 999x32x32/1x8x8 and encode 9999x512x512/1x64x64. | Aligned | `packages/dinkster-inference/src/dinkster_inference/wan21.py:1280-1308` |
