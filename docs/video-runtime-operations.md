# Video runtime operations

## Encoder callers

The caller owns bounded spooling and atomic publication. Publish only after a
successful encode and use the returned suffix and MIME, not the originally
requested format. Native savers use a local spool that rolls to disk after
512 KiB and pass encoded bytes to the asset writer, including remote targets.
Do not materialize a full frame batch to save a lazy VIDEO.

Frame-record adapters supply straight-alpha RGB/RGBA, canonical `image_color`
facts and source precision. Bound raw and normalized chunks before conversion.
Pass a JSON metadata object, not Pillow `PngInfo`. Chunk adapters map `png` to
`apng` and `gif` to `gif_pillow`; individual still assets retain their separate
per-image publication and quantization path. Do not confuse still-image
truncation with animation's nearest-integer packing.

When speaker assignment or downmix is intended, supply an explicit canonical
AUDIO `channel_map` matrix before assembly. Selecting `stereo` does not invent
a matrix for discrete inputs, and a PCM codec fallback cannot make an unsafe
PyAV frame boundary safe.

Direct callers outside worker execution can supply `on_diagnostic`; there is
no ambient worker collector there. Native node adapters attach the actual
output port as `outputId` and publish through
`report_value_diagnostic('media_format_fallback', data)` rather than logging
away requested/effective substitutions. See the encoder implementations for
format defaults, limits and callback records:
[runtime](../packages/dinkster-video-runtime/src/dinkster_video_runtime/runtime.py)
and [frame formats](../packages/dinkster-video-runtime/src/dinkster_video_runtime/formats.py).

## Conformance

The [video design reference](video-io-design.md#reference-behavior) records
source revisions and explains why RGB-only upstream output is not an alpha
golden. Alpha goldens use independently constructed coverage ramps, not
stream-copy outputs. Run the CPU format/runtime checks from the repository root:

```sh
python scripts/develop.py run -- python -m pytest -q -n0 tests/test_video_formats.py tests/test_video_runtime.py
python scripts/develop.py run -- python tools/video_conformance.py --rss
```

The RSS exercise measures the 1080p/10-second load/trim/transcode path in a fresh
process against the design's 200 MiB growth bound. Record the source and corpus
alongside measurements; encoded output and decoded-pixel parity are different
checks.

## Preview initialization

The opt-in `dinkster-video-preview` pack is included in this package. After
[contributor setup](../README.md), start it from the repository root:

```sh
python scripts/develop.py run -- dinkster-serve --pack packages/dinkster-video-runtime/preview/dinkster-pack.toml --port 8199
```

Connect a VIDEO-producing node to `video-preview.initialize`. Use an enabled,
compatible frontend to inspect the policy response, connection/snapshot identity
and latest preview dimensions, rate and count. Check the unavailable-policy
state on a failed query. Disabling frontend contributions must not be mistaken
for disabling the node. Do not construct module URLs in pack code: use the
snapshot-selected module and follow the
[frontend consumer integration cautions](extension-integration.md#frontend-consumers-and-route-handlers).
The executable route/event/module exercise is
[`test_video_preview_pack_route_event_and_module_end_to_end`](../tests/test_serve.py).
