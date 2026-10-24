## LoRA and model patching

The native attention multiplier nodes scale self-attention, cross-attention,
CLIP attention, and temporal-attention projection weights through the model
patcher. LoRA loading, scheduled patches, other model patch nodes,
context-window patches, and model-family patch formats are not currently
exposed by Dinkster.
