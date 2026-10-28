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

`POSE_KEYPOINT` preserves a list of frame mappings with integer `canvas_width`
and `canvas_height`, a `people` list and flat numeric keypoint triples. Optional
body, foot, face and hand fields retain absence, empty lists and `None`; numeric
coordinates and scores are not normalized or range-limited. Supported structural
metadata also round-trips. Its Torch-free structural codec has a type-specific
fingerprint and preserves existing registration authority. This format is not
converted to the distinct skeleton/edge representation of `POSE`.

`INTERP_MODEL` is a canonical builtin exported as `INTERP_MODEL_TYPE` through
`dinkster_api.v1`. Shared inference registration uses the existing default
resident pool and preserves an already registered type spec. Interpolation
model objects stay in their holding process; normal value transport carries
their resident identity, not model contents.

`MOGE_MODEL` uses that same holding-process resident pool. `MOGE_GEOMETRY`
uses the structural latent codec without requiring a `samples` entry. Geometry
tensor fields (`points`, `depth`, `intrinsics`, `mask`, `normal`, and `image`)
retain their shapes, dtypes, bytes, and optional-key presence. Supported extra
metadata also round-trips; mapping insertion order does not affect identity.
The fingerprint includes the geometry type, and encoded tensor records are
validated without loading Torch. Existing registration authority is preserved.

`VOXEL` preserves source-shaped `data`, optional `voxel_colors` and `resolution`,
and supported extra metadata on the existing structural tensor wire. Dense
occupancy and sparse coordinate rows retain their exact layouts; color channels
are not truncated and optional-key omission survives encoding. Public
`materialize_voxel` reconstructs tensors lazily into an attribute view, using
`None` for absent optional attributes without modifying the wire mapping.
Live tensors keep their dtype, device and storage. Registration is cold and
preserves existing authority; malformed tensor records are rejected.

This package is a uv workspace member. From the repository root:

```sh
uv sync --all-packages
```
