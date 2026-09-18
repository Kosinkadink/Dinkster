# SDXL ControlNet Union comparison

Reference: ComfyUI `origin/master` commit
`de6b062fb5ed1c9b471a3ebcd614705d93d67560`, `comfy/controlnet.py`
`ControlNet` Union handling and `comfy/ldm/modules/diffusionmodules/controlnet.py`
`ControlNet`.

ComfyUI detects the Union task capacity from `task_embedding`, creates one-hot
control-type timesteps, applies the learned control-add embedding, task
embedding, transformer, and spatial channel projection, then executes the SDXL
ControlNet branch. Dinkster preserves that operation order and admits only exact
6-mode or 8-mode xinsir layouts before payload reads. The semantic selector is
declared independently from the provider and validated against detected
capacity before execution.

The official acceptance artifact is `xinsir/controlnet-union-sdxl-1.0`
revision `801a4a3fa3d4c936f4feea95b98607bc6726f80c`, file
`xinsir-controlnet-union-sdxl-1.0-promax.safetensors`, 2513342408 bytes,
SHA-256 `9fae2e50cb431bfcbe05822b59ec2228df545ef27f711dea8949e9f4ed9f7cdc`,
and BLAKE3 `4a3243750f74b2acf56edb1a40d408fa3be8249c49f1fc9b5d2f3946ae55b9e1`.

On an RTX 4090 with torch 2.13.0+cu130, a fixed float16 mode-3 workload
produced ten residual tensors bit-exactly equal to ComfyUI. The platform-keyed
acceptance golden records the reproducible Dinkster residual digest and artifact
provenance.
