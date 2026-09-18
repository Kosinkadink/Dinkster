# dinkster-nodes-media-io

`dinkster-nodes-media-io` supplies Dinkster's asset-backed image, mask, animation,
video, audio, 3D model, and gaussian-splat I/O nodes, bounded UTF-8 text
saving, server-side audio recording and webcam frame capture through
injected providers, plus the typed save-target prefix constructor. Media
dependencies and codecs stay out of the lightweight foundation pack.

Save Video accepts a VIDEO value and returns that value unchanged plus the
saved video asset. Container and codec default to preserving the source;
omitting CRF permits stream copy when the edit plan allows it. Explicit CRF
requests encoding, with a 0-51 range for H.264 and 0-63 for VP9/AV1. Metadata
is a JSON object of container tags. Deferred components carry bit depth and
color space into the shared frame-at-a-time encoder. Persisted frame-input
savers migrate to an Assemble Video node feeding Save Video, preserving
codec settings and reconnecting saved-asset consumers to the asset output.

Capture nodes never open a device during schema discovery. A deployment
injects capture backends through `DINKSTER_AUDIO_CAPTURE_PROVIDER` and
`DINKSTER_VIDEO_CAPTURE_PROVIDER` (`module:attribute` naming a zero-argument
factory); without one configured, capture executions fail closed with a
typed error, and tests inject fake providers instead of hardware. Device
inputs are remote combos backed by lazy choice providers: the server
enumerates the provider's devices only when the editor fetches or refreshes
the dropdown, and an empty value still selects the first available device
at execution time.

The pack owns the native `dinkster.image` and `dinkster.mask` array contracts.
Images use BHWC layout; masks use BHW layout. Both cross worker boundaries as
NumPy arrays and expose PNG renditions without importing a tensor framework.
Image loading applies EXIF orientation and emits ComfyUI-compatible
`1 - alpha` masks. Animated and multipage files produce ordered batches,
omitting pages whose dimensions differ from the first page. The native asset
decoder preserves embedded alpha; explicit Load Image separates RGB and mask.
Paint Mask accepts one still-image asset and a strict, source-digest-bound JSON
recipe. It derives an image-sized transparency mask from source alpha and
replays bounded paint, erase, clear, and invert commands in source-pixel
coordinates. Animated sources, stale digests, malformed recipes, and dimension
mismatches fail instead of applying edits to different pixels.
Saving publishes ordered typed assets through mounted save
targets; PNG metadata inspection exposes bounded, data-only ComfyUI and A1111
representations. Render Image Document consumes an adopted ImageDocument
asset and resolves its raster children through the same asset resolver. Its
selector accepts `composite`, `layer:<id>`, or `mask:<id>` and returns the
deterministic CPU reference result as `dinkster.image`.

`dinkster-pack.toml` is the public composition boundary. `MEDIA_IO_NODES` and
`register_media_types` are host wiring, not a stable Python API for other
packs. Packs compose through registered node, capability, and type contracts
rather than importing this implementation.

The distribution embeds a complete replayable pack artifact. A source checkout
uses the adjacent manifest; an installed wheel uses that embedded artifact.
