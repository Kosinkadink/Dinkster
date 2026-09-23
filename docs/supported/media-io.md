## Media I/O

- Typed PNG, JPEG, and WebP image loading with EXIF orientation, ICC-to-sRGB
  conversion, alpha masks, bounded metadata inspection, and input- or
  output-category asset selection. Media outputs retain compact pixel or PCM
  storage with actual host-RAM byte accounting; ordinary typed consumers
  receive normalized float32 values at invocation, while inputs that declare
  compact-storage support receive the retained dtype unchanged
- Ordered animated and multipage image loading; pages with different dimensions
  from the first page are omitted from the batch
- Ordered mounted image saving to PNG, JPEG, or WebP; Save Image defaults to a
  selectable readwrite output mount under the local library, publishes the
  saved asset immediately, and returns its mounted virtual path; 8-bit and
  16-bit PNG mask I/O with explicit channel and polarity; non-publishing typed
  image previews
- Source-bound still-image mask painting with bounded pressure-sensitive paint
  and erase strokes, clear/invert commands, and transparency-alpha output
- Unified asset-backed ImageDocument v2 and `dinkster.layers`, with structural v1
  migration, complete/layer/mask rendering, isolated/pass-through groups,
  clipping, affine transforms, flips, z-order, canvas color, and
  enabled/inverted/combined raster masks;
  native, OpenRaster, and raster-subset PSD load/save, exact native OpenRaster
  reimport, editable standard layer visibility/opacity/integer offsets/blends,
  PSD groups and raster masks, and shared graph/workspace flattening with
  transparency masks. PSD export preserves 8-bit RGBA raster layers, names,
  order, visibility, rounded 8-bit opacity, integer offsets, supported blends,
  raster clipping, isolated/pass-through groups, and one raster mask per raster
  layer. Mask channel, inversion, and opacity are baked into grayscale coverage.
  Unsupported transforms, group geometry/masks/clipping, combined masks, canvas
  backgrounds, linear compositing, and grain extract/merge export as one merged
  layer with a warning. PSD text and smart objects are rasterized as ordinary
  layers on import; 16/32-bit channels and non-RGB color are converted to 8-bit
  RGBA. Vector masks are not preserved, effects are not rendered, and adjustment
  layers without raster pixels are refused. Each gap emits a warning. OpenRaster
  uses nearest 8-bit portable opacity values that Krita imports without downward
  truncation; the native extension preserves exact Dinkster 16-bit opacity
- Bounded animated PNG and WebP saving with frame-order, timing, loop, quality,
  compression, lossless, and alpha controls
- Bounded still and animated AVIF saving with 8/10-bit YUV420, sRGB, HLG or PQ
  color signaling, CRF quality, loop and frame-rate controls, and ComfyUI-compatible
  EXIF workflow metadata. AVIF loading and metadata inspection include that EXIF.
- Asset-backed native video loading to IMAGE, optional AUDIO, fps, frame count,
  and duration, with rate, resize, start, frame-cap, and frame-selection controls
- Native video saving through PyAV: MP4/H.264 at 8 or 10 bits, WebM/VP9 at
  8 bits with optional alpha, WebM/AV1 and MP4/H.265 at 8 or 10 bits,
  ProRes including 4444 alpha, and lossless RGB/RGBA FFV1/MKV at 8 or 16 bits;
  AAC audio for MP4, Opus for WebM, PCM for ProRes, and FLAC for FFV1.
  Save Video accepts VIDEO with container, codec, ProRes profile, declared
  audio-layout preservation or explicit mono/stereo downmix, trim-to-audio
  or silence padding, optional CRF, and JSON metadata controls. It outputs the unchanged VIDEO
  plus the saved asset. Persisted frame-input savers migrate through deferred
  assembly while preserving codec settings and saved-asset output links.
  Component audio without proven sample coverage preserves finite VIDEO length
  and reports that shortest-audio trimming was not applied exactly.
  NVENC requests are refused until a hardware-encoder policy exists.
- Value-level video operations on IMAGE batches and VIDEO values: frame
  windowing with skip, stride, and cap controls; exact nearest-tick frame-rate
  resampling matching native video loading; deferred assembly with optional
  audio, codec preference, 8/10-bit precision, sRGB, HLG, and HDR PQ;
  concatenation of up to 100 videos with optional complete soundtrack override;
  disassembly to frames, optional audio, fps, duration, bit depth, and color space
- VIDEO values retain an encoded source or deferred components, rational
  source probe facts, and ordered lazy trim, crop, scale, and concat edits.
  Compatible encoded concatenation uses packet copy; incompatible or
  component-backed inputs share one encoding. Info and lazy edits do not decode frames.
  Encoded sources can be MP4, MKV, MOV, WebM, AVI, or GIF. Source-copy saves
  preserve original bytes; exact closed packet cuts use stream copy and
  other edits transcode frame-at-a-time. Alpha and source precision are
  retained through a preserving CPU format fallback with requested/effective
  diagnostics when a format preference is incompatible. Actual encoder
  inability remains a capability error; alpha is never silently discarded.
- Source-only video timeline documents with lazy `comfy.VIDEO` rendering,
  lossless OTIO JSON import/export, declared asset bindings, clip-level
  VIDEO_EDIT preservation, track composition, transitions, retiming, bounded
  audio mixing, CURVE parameters, and deterministic CPU text effects
- Streamed media uploads and video encoding with explicit upload, decoded
  frame, decoded audio, and encoded output limits
- Bounded frame-record export to GIF (Pillow or FFmpeg dithering), animated
  WebP, 8/16-bit APNG with alpha, and ZIP PNG sequences; variable frame
  durations, animation loops, workflow metadata, and declared-count checks
- PNG/APNG workflow metadata readable by ordinary image metadata readers
- Native saver format fallback reports requested/effective choices through
  nonblocking value diagnostics, including isolated-worker execution
- Asset-backed native audio loading to AUDIO (first audio stream of audio or
  video containers), including audio-only Opus/Vorbis WebM and AAC M4A uploads
  classified from their track metadata; silent-audio construction with declared
  duration, sample rate, and channel count; and typed audio preview. Value discovery
  advertises versioned waveform PNG and bounded WAV window renditions with
  explicit batch selection, normalized selectors in cache identity, and
  structured invalid or unavailable responses. Waveforms use fixed colors
  without antialiasing or labels
- Native audio saving through PyAV: FLAC, MP3 with V0/128k/320k quality, and
  Opus with 64k-320k bit rates and 48 kHz coercion of unsupported rates; one
  ordered published asset per batch element under a mounted save target
- Value-level audio operations on AUDIO values: duration trimming with
  negative-from-end starts, stereo channel split and mono join, before/after
  concatenation, add/subtract/multiply/mean merging with peak normalization,
  decibel gain, fade in/out with linear or cosine curves, and a three-band
  shelf/peaking equalizer; mixed sample rates are matched through
  libswresample
- Audio amplitude-envelope extraction to a per-video-frame float list and
  editable time-aligned CURVE (the audio-reactive replacement path):
  band-limited FFT magnitude with avg/max/sum aggregation at a chosen frame
  rate, optional peak normalization (silence yields zeros), inversion, and
  deterministic peak retention within the CURVE point budget; linked editors
  can retrieve the exact executed interpolation and points within that budget
- Native text saving to mounted targets as UTF-8 without platform newline
  translation: txt, csv, md, and json formats with ComfyUI-matching JSON
  pretty-printing and a bounded output size
- Server-side audio recording and single-frame webcam capture through
  deployment-injected capture providers with device discovery, declared
  rate/channel/resolution capabilities, typed permission/not-found/
  cancellation/timeout failures, bounded duration and byte budgets, and no
  device access during schema discovery; device inputs are refreshable
  dropdowns that enumerate the provider's devices on demand, with empty
  still selecting the first available device
- Lazy per-request choice providers as a pack capability: a pack's choices
  entry can name a callable that the server invokes only when the editor
  fetches or refreshes that combo route, never at composition or startup
- GLB 3D model assets through Load 3D Model, Save 3D Model, and Preview 3D
  nodes carrying validated raw GLB bytes as `dinkster.model3d`
- Gaussian splat assets through Load Gaussian Splat and Save Gaussian Splat
  nodes: `dinkster.splat` batches of activated world-space gaussians with
  binary little-endian PLY interchange byte-compatible with ComfyUI's
  splat PLY files
