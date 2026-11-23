# dinkster-inference-wire

Torch-free value and wire contracts shared by the engine, workers and node
packs. Model loading, sampling and execution live in `dinkster_inference`.

## Develop locally

This is a workspace library, not a standalone service. From the repository root:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_geometry_wire.py tests/test_sampling_wire.py tests/test_vision_output_codec.py tests/test_api_v1.py
```

See [contributor setup](../../README.md) and [testing](../../docs/testing.md).

## Reference

- [Public API](src/dinkster_inference_wire/__init__.py) and [shared registrations](src/dinkster_inference_wire/types.py)
- [Sampling values](src/dinkster_inference_wire/sampling_wire.py) and [pose frames](src/dinkster_inference_wire/pose_keypoint.py)
- [Vision packets](src/dinkster_inference_wire/vision_packet_wire.py), [voxels](src/dinkster_inference_wire/voxel_wire.py) and [sparse subdivisions](src/dinkster_inference_wire/shape_subdivides_wire.py)
- [Pack-facing API](../dinkster-api/src/dinkster_api/v1.py) and [architecture](../../DESIGN.md)
