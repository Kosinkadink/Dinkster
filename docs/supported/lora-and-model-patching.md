## LoRA and model patching

Native model patches select discrete, continuous EDM, continuous V, Stable
Cascade, SD3, AuraFlow, and Flux sampling behavior; set flow noise scale;
rescale classifier-free guidance; select compute dtype; and select a supported
dense attention backend. They use the pinned `dinkster_inference` sampler and
model patcher.

LoRA loading, scheduled patches, context-window patches, and model-family patch
formats are not currently exposed by Dinkster.
