# SDXL Control-LoRA comparison

Reference: ComfyUI `origin/master` commit
`de6b062fb5ed1c9b471a3ebcd614705d93d67560`, `comfy/controlnet.py`
`ControlLoraOps` and `ControlLora.pre_run`.

ComfyUI constructs the control branch from the attached SDXL UNet geometry,
copies matching base UNet state, installs artifact weights and biases, and
evaluates each factorized Linear or Conv2d as `base_weight + up @ down`.
Dinkster uses the same operation order and requires assembly-proven BLAKE3
identities for both the Control-LoRA and the copied SDXL base. The exact
rank-128 header is admitted before payload reads; missing, extra, dtype-drifted,
and geometry-drifted tensors refuse.

The official acceptance artifacts are:

- `stabilityai/control-lora` revision
  `75590eb0e7868d3d0f1581b5126d487b33c7fdb4`, file
  `control-LoRAs-rank128/control-lora-canny-rank128.safetensors`, 395733680
  bytes, SHA-256
  `56389dbb245ca44de91d662529bd4298abc55ce2318f60bc19454fb72ff68247`,
  BLAKE3 `4d688185e7d1f21c5e3d7f52b5d337d839ec6ae9c5b54fd90343c43a2b68599b`.
- `stabilityai/stable-diffusion-xl-base-1.0` revision
  `462165984030d82259a11f4367a4eed129e94a7b`, file
  `sd_xl_base_1.0.safetensors`, 6938078334 bytes, SHA-256
  `31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b`,
  BLAKE3 `5aee6bdac81d43958c3613070593d99dc649cf16b1bc868dd1ac5180868a07d0`.

The official Control-LoRA repository contains SDXL artifacts and workflows;
it provides no official SD1.5 Control-LoRA acceptance artifact. Native support
therefore admits the exact SDXL rank-128 layout only.
