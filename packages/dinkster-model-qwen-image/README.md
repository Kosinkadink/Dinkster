# dinkster-model-qwen-image

First-party native nodes for Qwen Image operations that are not universal
generation operations. The pack provides Edit reference conditioning and
Layered latent geometry, plus typed loaders and model applications for
maintained InstantX, Qwen Fun, and DiffSynth controls.

Model, text, and codec resources load independently through universal generation
schemas; helper loaders are optional. Text-only conditioning, sampling, LoRA,
and codec operations use the same universal schemas. This pack consumes those
ids without importing their owning package.
