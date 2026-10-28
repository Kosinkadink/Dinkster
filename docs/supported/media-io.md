## Media I/O

- Explicit mounted Save attachments reach execution results after host validation
  of the live writable mount, writer publication record, and exact file bytes.
  Custom targets and non-media files, including empty text files, are supported
  through both ordinary hosts. Failed invocations do not return kept artifacts;
  returning an ASSET without attaching it does not advertise a saved file.
- Indexed video file selections remain references to their granted files,
  without input copying or an encoded-file size cap. Unindexed selections
  must be scanned before VIDEO admission. Decode working-set and produced
  output limits still apply.
- Shared editor values preserve plural pixel boxes with metadata, image-comparison
  and compositor state, and nested tensor-layer documents including optional
  masks, transforms, authored layer order, and canvas dimensions
- Portable SAM3 track values preserve packed uint8 masks, empty object axes or
  absent masks, original dimensions, frame counts, optional scores, and structural
  metadata across worker boundaries
- Portable camera embeddings, motion tracks with optional visibility, and complete
  audio-encoder outputs preserve tensor dtype, shape and bytes, ordered layer
  sequences, sample counts and structural metadata across execution hosts and caches
- Portable `SVG` values retain immutable ordered raw byte streams, including empty
  and duplicate members, across execution hosts and caches without XML rewriting;
  consumers can open independent seekable streams without mutating cached bytes
- Portable IC-LoRA parameter mappings preserve source fields, ordering, and
  structural extras across execution hosts and caches
- Portable LOAD3D_CAMERA values preserve perspective and orthographic camera
  dictionaries, including position, target, zoom, camera type, optional camera
  controls, and structural extensions across execution hosts and caches
- Typed PNG, JPEG, and WebP image loading with EXIF orientation, ICC-to-sRGB
  conversion, alpha masks, bounded metadata inspection, and input- or
  output-category asset selection. Media outputs retain compact pixel or PCM
  storage with actual host-RAM byte accounting; ordinary typed consumers
  receive normalized float32 values at invocation, while inputs that declare
  compact-storage support receive the retained dtype unchanged
- Public bound-image component decoding returns first-frame RGB float32 pixels
  or ordered frames with optional straight alpha under explicit source ownership,
  using the same PyAV precision, channel, alignment, and display-orientation
  handling as the ComfyUI reference without admitting still images as VIDEO;
  source bytes, dimensions and estimated aligned conversion working sets are
  admitted before conversion, with source length checked before opening the
  decoder. FFmpeg probing is not a hard allocation sandbox. PyAV's non-byte
  alignment graph rejects some small asymmetric inputs. Frame iteration does not
  accumulate a batch; consumers admit the memory of frames they retain separately
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
- Asset-backed native video loading to IMAGE, optional AUDIO, fps, frame count,
  and duration, with rate, resize, start, frame-cap, and frame-selection controls
- Native video saving through PyAV: MP4/H.264 at 8 or 10 bits, WebM/VP9 at
  8 bits with optional alpha, WebM/AV1 and MP4/H.265 at 8 or 10 bits,
  ProRes including 4444 alpha, and lossless RGB/RGBA FFV1/MKV at 8 or 16 bits;
  AAC audio for MP4, Opus for WebM, PCM for ProRes, and FLAC for FFV1.
  Save Video accepts VIDEO with container, codec, ProRes profile, declared
  audio-layout preservation or explicit mono/stereo downmix, trim-to-audio
  or silence padding, optional finite integer or fractional CRF within the
  codec's range without rounding, and JSON metadata controls. It outputs the unchanged VIDEO
  plus the saved asset. Persisted frame-input savers migrate through deferred
  assembly while preserving codec settings and saved-asset output links.
  Component audio without proven sample coverage preserves finite VIDEO length
  and reports that shortest-audio trimming was not applied exactly.
  NVENC requests are refused until a hardware-encoder policy exists.
- Opt-in unedited component-video encoding preserves source H.264/AV1
  quantization, color signaling, CRF defaults, optional H.264 preset, and
  thousandth-rounded encoded frame rates without changing component values.
  Explicit codec failures propagate without FFV1 substitution or resizing.
  Small AV1 inputs may succeed when the encoder supports them. Automatic
  PyAV conversion threading can produce nondeterministic pixels on small
  10-bit inputs; this path retains that source behavior.
- Value-level video operations on IMAGE batches and VIDEO values: frame
  windowing with skip, stride, and cap controls; exact nearest-tick frame-rate
  resampling matching native video loading; deferred assembly with optional
  audio, 8/10-bit precision, sRGB, HLG, and HDR PQ; disassembly to frames,
  optional audio, fps, duration, bit depth, and color space
- Native Trim Video and Trim Video (Advanced) select presentation-time
  windows, including tail-relative starts and strict duration checks.
  Component inputs encode with source defaults before selection; the
  advanced editor's zero window keeps its input unchanged.
  Empty selections fail immediately rather than producing an empty VIDEO;
  see [video behavior differences](video-deliberate-differences.md).
- Native Crop Video clamps and even-aligns a pixel rectangle, retaining the
  full frame for empty or full-frame selections. Component crops copy pixels
  without changing their precision, frame rate, or complete audio, but reset
  encoding color space to sRGB as in the fixed source. No-op selections keep
  their original color space;
  encoded inputs retain their source with composable lazy crops.
- Public `iter_video_pixels` yields edited, display-oriented float32 pixels
  with rational timestamps without accumulating the complete frame batch.
  Consumers that stop early close the iterator to release the decoder and
  source handle, and copy any selected arrays they retain. Iteration is forward
  decoding, not random access; it yields pixels only, not audio.
- VIDEO values retain an encoded source or deferred components, rational
  source probe facts, and ordered lazy trim, crop, scale, and restricted
  identical-codec concat edits. Info and lazy edits do not decode frames.
  Public `concatenate_video` accepts one to 100 ordered VIDEO values, encodes
  component inputs to MKV, preserves compatible encoded inputs, and supports
  one optional complete AUDIO soundtrack. Later trims select the matching
  soundtrack window; explicit audio replaces nested inherited soundtracks
  in both playback and serialized identity.
  Native Concatenate Video accepts one to 100 ordered VIDEO list family
  members with an advanced codec choice and optional complete AUDIO soundtrack.
  Decoded source trims select half-open stream-tick windows while retaining
  rational timeline bounds and independent audio sample windows.
  Encoded sources can be MP4, MKV, MOV, WebM, AVI, or GIF. Source-copy saves
  preserve original bytes; exact closed packet cuts use stream copy and
  other edits transcode frame-at-a-time. Alpha and source precision are
  retained through a preserving CPU format fallback with requested/effective
  diagnostics when a format preference is incompatible. Actual encoder
  inability remains a capability error; alpha is never silently discarded.
- Source-only video timeline documents with lazy `VIDEO` rendering,
  lossless OTIO JSON import/export, declared asset bindings, clip-level
  VIDEO_EDIT preservation, track composition, transitions, retiming, bounded
  audio mixing, CURVE parameters available in core-only registries, and
  deterministic CPU text effects
- Streamed media uploads and video encoding with explicit upload, decoded
  frame, decoded audio, and encoded output limits
- Native VOID optical-flow-warped noise with 16 latent channels, even temporal
  lengths, spatial resizing and batch replication. Controlled CPU coverage does
  not certify real-model optical-flow quality or GPU parity.
- Native Comfy packs exchange portable VIDEO and VIDEO_EDIT values without
  requiring the media I/O pack. Video admission accepts verified assets and
  granted local files, rejecting revoked or changed local bindings before
  probing or publishing; admitted local bytes use bounded inline or published
  asset storage without decoding video frames.
- Bounded frame-record export to GIF (Pillow or FFmpeg dithering), animated
  WebP, 8/16-bit APNG with alpha, and ZIP PNG sequences; variable frame
  durations, animation loops, workflow metadata, and declared-count checks
- PNG/APNG workflow metadata readable by ordinary image metadata readers
- Native optical-flow loading from granted RAFT-large checkpoint assets, with
  safe state-dict loading, float32 evaluation and ordinary resident model transport;
  controlled CPU loader coverage does not certify real-model optical-flow quality
- Native Save WEBM exports IMAGE sequences as VP9 or 10-bit AV1, retaining
  VP9 RGBA transparency, fractional frame rates and submitted workflow metadata,
  with default or selected writable output mounts and unchanged image output
- Native saver format fallback reports requested/effective choices through
  nonblocking value diagnostics, including isolated-worker execution
- Asset-backed native audio loading to AUDIO (first audio stream of audio or
  video containers), including audio-only Opus/Vorbis WebM and AAC M4A uploads
  classified from their track metadata; Record Audio decodes a canonical recorded
  file identity to AUDIO through ordinary in-process and isolated hosts;
  silent-audio construction with declared
  duration, sample rate, and channel count; and typed audio preview. Value discovery
  advertises versioned waveform PNG and bounded WAV window renditions with
  explicit batch selection, normalized selectors in cache identity, and
  structured invalid or unavailable responses. Waveforms use fixed colors
  without antialiasing or labels
- Native audio saving through PyAV: FLAC, MP3 with V0/128k/320k quality, and
  Opus with 64k-320k bit rates and 48 kHz coercion of unsupported rates; one
  ordered published asset per batch element under a mounted save target
- Native ComfyUI audio nodes include Trim Audio Duration, Split Audio Channels,
  Adjust Audio Volume, Audio Equalizer (3-Band), Empty Audio with rounded sample
  counts, Empty MiniMax Music3 Latent Audio with frame-aligned duration and batching,
  VAE Encode Audio, VAE Decode Audio, tiled VAE Decode Audio, LTXV Audio VAE Encode,
  LTXV Audio VAE Decode, Empty LTXV Latent Audio, and Get Audio Latent Size for
  audio and joint AV latents, plus LTXV audio/video latent concatenation and separation.
  Value-level audio operations on AUDIO values include duration trimming with
  negative-from-end starts, stereo channel split and mono join, before/after
  concatenation, add/subtract/multiply/mean merging with peak normalization,
  decibel gain, fade in/out with linear or cosine curves, and a three-band
  shelf/peaking equalizer; mixed sample rates are matched through
  libswresample. Native ComfyUI join, concatenate, and merge nodes
  materialize bounded AUDIO values, match the lower sample rate to the higher
  rate, and preserve ComfyUI channel, trimming, padding, and normalization
  behavior
- Native bound-checkpoint Audio Encoder loading, audio conditioning through
  AUDIO_ENCODER_OUTPUT, and SheetSage2 music transcription to ordered ABC
  strings in melody-only or full melody-and-chords mode
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
- Native ImageOnlyCheckpointSave exports model, vision encoder and VAE weights
  to a selected writable mount, preserving checkpoint metadata and reserved
  counter names. Controlled CPU weights exercise serialization; real-model
  loading and GPU parity are not certified.
- Native VOIDSampler supplies the node-defined alpha-space DDIM sampler for
  the existing custom-sampling path, without ordinary KSampler noise scaling.
  Controlled CPU schedules cover the numerical contract, not real-model or
  GPU output parity.
- Native VOID quadmask quantization and spatial dilation, inpainting
  conditioning with even temporal latent lengths, and precomputed warped
  noise sources. Controlled CPU encoders cover mask/video conditioning and
  source preservation; real-model and GPU output parity are not certified.
- Native Create Video assembles IMAGE frames, rational frame rate and optional
  AUDIO as a VIDEO, retaining component storage until explicit encoding.
  Bit depth and sRGB/HLG/PQ color choices are retained; optional immediate
  H.264 or AV1 encoding uses source quantization, color and encoder defaults,
  rounding only the encoded frame rate to a thousandth. Get Video Components
  extracts bounded frames, audio and presentation facts from component or
  encoded VIDEO values through ordinary in-process and isolated hosts.
- Native Save Video publishes a mounted MP4, MKV, or WebM asset while
  returning its input VIDEO unchanged. Nested container, codec and CRF
  controls select the public writer; component inputs use source encoding
  defaults. Filename-prefix formatting, output mount selection and submitted
  workflow metadata use the common mounted-output boundary. Encoded-input
  re-encoding retains the canonical writer policy, not an export-equivalence
  guarantee against the source encoder.
