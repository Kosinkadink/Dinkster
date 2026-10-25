# dinkster-inference-wire

`dinkster-inference-wire` contains torch-free value and wire contracts shared by
Dinkster's engine, workers, and node packs. Model loading, sampling, and model
execution live in the `dinkster_inference` package.

The retained contracts cover conditioning carriers, generation providers,
multi-stream and sparse values, safetensors header inspection, device
descriptors, and portable sampling selections. Runtime SAMPLER and NOISE values
use the governed process-resident pool: only an identity crosses the boundary,
and only the holding worker can resolve it. Portable selection bytes and
fingerprints are unchanged. Resident envelopes have distinct type-specific
framing and structural validation that does not import Torch or inference;
expired or foreign-worker identities raise the existing residency error.
The package does not contain a model runtime or an inference registry.

This package is a uv workspace member. From the repository root:

```sh
uv sync --all-packages
```
