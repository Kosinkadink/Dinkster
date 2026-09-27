# dinkster-inference

`dinkster-inference` contains torch-free value and wire contracts shared by
Dinkster's engine, workers, and node packs. Model loading, sampling, and model
execution live in the `dinkster_comfy` package.

The retained contracts cover conditioning carriers, generation providers,
multi-stream and sparse values, safetensors header inspection, device
descriptors, and portable sampling selections. The package does not contain a
model runtime or an inference registry.

This package is a uv workspace member. From the repository root:

```sh
uv sync --all-packages
```
