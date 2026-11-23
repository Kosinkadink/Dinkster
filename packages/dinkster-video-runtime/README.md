# dinkster-video-runtime

Shared decoding, filtering and encoding of portable VIDEO values without a
full decoded frame batch. Includes the opt-in video-preview pack.

## Develop locally

From the repository root:

```sh
python scripts/develop.py setup --yes
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_video_formats.py tests/test_video_runtime.py
```

See [contributor setup](../../README.md) and [testing](../../docs/testing.md).

## Reference

- [Public runtime exports](src/dinkster_video_runtime/__init__.py)
- [Encoder, conformance and preview operations](../../docs/video-runtime-operations.md)
- [Media consumption](../../docs/media-consumption.md)
- [Video design reference](../../docs/video-io-design.md)
- [Extension integration](../../docs/extension-integration.md)
