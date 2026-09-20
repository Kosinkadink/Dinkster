## Graph, node, and widget vocabulary

- Input types: concrete, union, wildcard, variable, list, asset;
  scalar/list cardinality; required/advanced/hidden/lazy input flags
- Required closed dynamic slots can bind their selected variant type to shared
  input and output variables, including list and asset element types
- Widgets: asset, save-target, combo and multi-combo with optional labels,
  per-option info, and folder paths; combo options derived from dynamic input
  family members with stable stored IDs and separately editable labels;
  boolean; number (number/slider/knob/gradientslider displays), string, color,
  curve editor and graph-native image compositor,
  pack-declared custom widget types with JSON parameters for frontend extensions;
  representations with schema defaults and optional user switching;
  schema-declared static and dynamic-family text completions; conditional
  visibility groups for top-level widgets;
  after-generate modes fixed/increment/decrement/randomize;
  exact integer bounds from signed 64-bit minimum through unsigned 64-bit maximum
- Node schemas can declare presentation-only editor roles and JSON-safe custom
  widget descriptors. Clients use these declarations without matching node IDs;
  native video trim and crop declare their exact `VIDEO_EDIT` features.
- Values: core.int, core.float, core.string, core.combo, core.boolean,
  core.absent, immutable linear or monotone-cubic `dinkster.curve`, ordered
  `dinkster.layers`, strict versioned `dinkster.compositor`, parametric `list<T>` and
  `asset<T>`
- Integer, float, text, and boolean primitive nodes declare their exact output
  values before execution for downstream presentation. Text switches between
  single-line and multiline views; the legacy multiline type remains loadable
  but is hidden from node search.
- Foundation value operations: versioned scalar math expressions; comparison
  with numeric epsilon; lazy typed selection;
  variadic boolean logic; clamping; range remapping with easing curves; and
  deterministic seeded integer or float random values
- User-declared expression outputs with stable IDs, editable names, and
  independently selected integer, float, boolean, or string outputs
- Entry-list fan-out with edit-time output count and types. Each entry carries
  a literal value or selects an index from an optional input list; runtime
  list length does not change the output interface
- Foundation list operations: create, append, concatenate, repeat, reverse,
  slice, integer range, index, length, and cross product. Create and append
  preserve list-valued items as nested elements; concatenate flattens exactly
  one level; slice and range use Python bounds, ordering, and step behavior
- Foundation routing operations: lazy document-ordered or stable-name N-way
  selection and a typed-absence gate; named selection uses an ordinary
  promotable combo while links and execution retain stable family member IDs;
  unselected linked branches do not execute across local, isolated, remote,
  and region boundaries
- Foundation text operations: bounded scalar formatting; case, whitespace,
  slicing, replacement, padding, line, and section transforms; text tests and
  length; regular-expression matching, extraction, and replacement; typed and
  variadic joining; bounded splitting; strict and legacy JSON modes; and CSV
  parsing and emission
- Foundation conversion and curve operations: explicit scalar conversion with
  guarded lossy modes; curve construction, interpolation, uniform sampling,
  native curve editing with optional histogram previews,
  and strict numeric schedule parsing
- Role-labeled multi-stream values inside ordinary `comfy.LATENT`, with generic
  audio/video concat, separate, visual preview, and audio preview nodes
- Native temporal video-latent trimming with metadata-preserving
  `TrimVideoLatent` workflow compatibility
- Native latent operation library with exact pinned-ComfyUI parity: tensor
  math (combine, mix, multiply, rotate, flip, crop, resize, composite,
  concat, cut), batch/metadata operations (from-batch, repeat, seed
  behavior, batching, rebatching, noise masks, video frame replacement),
  and reusable latent-operation values (tonemap reinhard, sharpen) applied
  through `LatentApplyOperation` or per-step before CFG combination through
  `LatentApplyOperationCFG`, each with comfy workflow alias records
- Native tensor resizing without a ComfyUI installation: nearest-exact,
  bilinear, bicubic, area, bislerp, and Lanczos interpolation, center cropping,
  video-frame resizing, and batch resampling. Empty Hunyuan video latents
  also work without ComfyUI.
- Native latent noise tooling with exact pinned-KJNodes parity: standalone
  noise generation (`GenerateNoise` compatibility: seeded CPU noise in BCHW,
  BCTHW, or BTCHW shapes with 4 or 16 channels, optional sigma-range scaling
  from a model's latent scale factor, multiplier, normalization, and
  constant-batch repetition) and latent noise injection (`InjectNoiseToLatent`
  compatibility: strength or averaged blending, normalization, bilinear mask
  gating, and seeded randn mixing), each with comfy workflow alias records
- Native `EmptyARVideoLatent`, `SamplerARVideo`, and `ARVideoI2V` workflow
  nodes for Wan 2.1 CausalAR generation
- Native `AudioEncoderLoader`, `AudioEncoderEncode`, `WanSoundImageToVideo`, and
  `WanSoundImageToVideoExtend` workflow nodes for Wan 2.2 S2V generation
- Native `WanDancerEncodeAudio`, `WanDancerVideo`, `WanDancerPadKeyframes`, and
  `WanDancerPadKeyframesList` workflow nodes for Wan 2.2 WanDancer generation
- Native `TRACKS` values and the `WanMoveTracksFromCoords`, `GenerateTracks`,
  `WanMoveConcatTrack`, `WanMoveVisualizeTracks`, and `WanMoveTrackToVideo`
  workflow nodes
- Asset-backed native Save/Load Latent with deterministic `.latent`
  safetensors, ordered multi-stream roles, VAE provenance hints, and safe
  stock ComfyUI single-latent import; legacy `LoadLatent` prompts resolve
  output-directory filenames to native assets and preserve the samples output
- Assets as identity (digest-addressed, not paths); media/image,
  media/audio, media/video, media/model3d, and data/latent source
  bindings; conventional model/checkpoint assets with ordered declarative
  component manifests; input/output/temp categories; deterministic single-file
  BitTorrent v2 descriptor derivation and validation with no networking
- Regions: map, fold, while; bindings zip/cross/broadcast; outputs
  gather/compact/state/flatten, including generic body outputs whose type is
  fixed by linked concrete inputs. Gather and flatten preserve whole-output
  absence; compact gathers only present iteration values in iteration order.
  Lazy routing resolves independently for each iteration; unselected branch
  nodes and nested regions neither prepare nor execute.
  Each body exposes its immediate zero-based `$region.index`; an explicitly
  declared `index` port keeps its declared meaning for graph compatibility.
