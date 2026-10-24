## LoRA and model patching

LoRA loading, scheduled patches, model patch nodes, context-window patches, and
model-family patch formats are not currently exposed by Dinkster. They will be
added only as explicit seams into the single `dinkster_inference` sampling engine.

ControlNet values use the native `CONTROL_NET` type and stay resident in their
owning worker, with the same governed lifetime as other loaded models.
