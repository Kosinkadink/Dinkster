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

`BOUNDING_BOXES`, `IMAGECOMPARE`, and `COMPOSITOR` retain ordinary structured
state, including pixel coordinates, metadata, and authored list order. `LAYERS`
uses the structural tensor codec without requiring `samples`: nested raster
images and masks, layer order, canvas tuples, optional transforms, and supported
extra metadata round-trip without geometry conversion or invented defaults.
`materialize_latent` reconstructs nested tensors at the consuming node; live
tensors retain their storage, dtype, and device. Registration is runtime-free
and preserves an existing authority. This tensor-layer document is distinct
from asset-backed video and image editing documents.

This package is a uv workspace member. From the repository root:

```sh
uv sync --all-packages
```
