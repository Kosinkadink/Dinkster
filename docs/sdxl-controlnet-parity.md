# Classic SDXL ControlNet parity

Dinkster follows ComfyUI commit
`de6b062fb5ed1c9b471a3ebcd614705d93d67560` for classic SDXL ControlNet.
Detection accepts the three layouts handled by ComfyUI: canonical native keys,
native keys under `control_model.`, and Diffusers keys normalized through the
SDXL UNet conversion table. Hint channels come from the artifact's first hint
convolution. The provider emits nine input residuals and one middle residual.

The acceptance artifact is Diffusers
`diffusers/controlnet-canny-sdxl-1.0` revision
`eb115a19a10d14909256db740ed109532ab1483c`, file
`diffusion_pytorch_model.fp16.safetensors` (2,502,139,136 bytes, SHA-256
`b2e7d3921058a442cc80430d1ec8847f42599c705e2451c95e77cf4dcf8d6c25`,
BLAKE3 `38b8924a1949fd2dea0edf12fde939f25ec3da3eafb690f9baf5dc2e461f7486`).
For the deterministic FP16 CUDA workload recorded in the platform golden,
Dinkster and ComfyUI produced the same ten residual tensors bit for bit, with
combined SHA-256
`37bb48f6e7a5baf12f2ef481bfe53f82fc2a80dc5ca0b67796ddd9d11209d893`.
