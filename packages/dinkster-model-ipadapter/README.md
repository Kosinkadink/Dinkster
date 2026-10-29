# dinkster-model-ipadapter

First-party native nodes for the standard SD1.5 IP-Adapter. The pack loads the
adapter and CLIP ViT-H/14 image encoder as independently residency-owned
components, projects one reference image once, and applies its immutable
attention contribution through the ordinary SD denoiser and custom-sampling
engine.

## Pinned sources and artifacts

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI/tree/b78cec879b9460d5cb25228a83a942fb78d2cd24)
  behavior at commit `b78cec879b9460d5cb25228a83a942fb78d2cd24`
- [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus/tree/a0f451a5113cf9becb0847b92884cb10cbdec0ef)
  behavior at commit `a0f451a5113cf9becb0847b92884cb10cbdec0ef`
- [Standard SD1.5 adapter](https://huggingface.co/h94/IP-Adapter/resolve/9bf28b38530e55ffa91c6d82e5161a982c22f284/models/ip-adapter_sd15.safetensors):
  44,642,768 bytes; SHA-256
  `289b45f16d043d0bf542e45831f971dcdaabe18b656f11e86d9dfba7e9ee3369`;
  BLAKE3
  `7f0a43a48969f0e17963995676df5dab9849bf1051a50c24685348f5a8960f33`
- [CLIP ViT-H/14 image encoder](https://huggingface.co/h94/IP-Adapter/resolve/0859e809306db97aa2338e370587ab284e8a754f/models/image_encoder/model.safetensors):
  2,528,373,448 bytes; SHA-256
  `6ca9667da1ca9e0b0f75e46bb030f7e011f44f86cbfb8d5a36590fcd7507b030`;
  BLAKE3
  `4649ee2cccf3b579a716035ba57d199aaad1b090217516632ccc1df004e0291a`

The pinned custom node registers adapter index 31 against the legacy ComfyUI
patch key `("middle", 1)`, while the pinned ComfyUI runtime declares that block
as `("middle", 0)`. Reference parity remaps that locator so all 16 official
projections execute. Dinkster binds index 31 to the canonical module path
`middle_block.1.transformer_blocks.0.attn2` instead of a patch key.

Only the standard four-token SD1.5 artifact is accepted. Light, Plus, face,
ViT-G, SDXL, multiple-reference, and per-layer weighting variants are not
claimed. Scheduled prompt or patch sampling is refused; ordinary KSampler and
decomposed custom sampling use the same implementation.
