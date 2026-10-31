## Graph, node, and widget vocabulary

- Input types: concrete, union, wildcard, variable, list, asset;
  scalar/list cardinality; required/advanced/hidden/lazy input flags
- Backend dynamic choices can declare hidden presentation while retaining
  optionality, their complete option trees and omitted defaults. The flag
  round-trips through schema declarations without changing execution choices.
  Frontend descendant visibility is not claimed here.
- Backend input families can require only the initial declared positions,
  leaving later members optional. Missing initial required slots remain
  validation errors even when later slots are connected. Ordinary families
  retain their stored order and uniform requiredness. Frontend support is
  not claimed here.
- Replacement input-family copies can declare historical template input IDs
  and a flat member-port path without retaining a predecessor schema. Existing
  mappings without that metadata retain their wire shape. Historical same-type
  templates are distinct from the current node interface. This describes the
  backend declaration contract, not frontend migration support.
- Backend replacement input-family copies can explicitly map authored source
  order to target naming positions instead of preserving suffix IDs. The option
  composes with historical-template metadata; omitted options retain their wire
  shape. This is a declaration contract, not a claim of frontend migration support.
- Batch Images, Batch Masks, Batch Latents, And, Or, Create List and Math
  Expression publish same-type rules for historical family member identities.
  Rules preserve the math expression and output identities, and identify the
  old flat boolean-family socket path. Node execution is unchanged. Automatic
  application requires a frontend that supports positional family migrations.
- Backend `list<*>` values retain independently typed child envelopes through
  ordinary worker and persistent-cache boundaries. Outputs collect existing
  typed values rather than guessing types from Python objects. Nested lists,
  child input conversion and resident-resource discovery retain each child's
  concrete identity; concrete `list<T>` remains homogeneous. This does not
  claim native CreateList or frontend editor/importer support.
- Backend schemas describe fixed positional tuples, derive tuple outputs from
  independently typed input families in document order, and select a position's
  type using a stored integer index. Tuple declarations round-trip through the
  schema wire contract; ordinary list families retain homogeneous binding.
  Tuple values retain independently typed child envelopes across worker and
  persistent-cache boundaries, including nested lists and tuples. Resource
  invalidation and device metadata projection inspect every position; persistent
  caches refuse tuples containing live resident references. Foundation tuple
  nodes collect independently typed connected values and select a position with
  a document-stored integer index, including negative indices. Graph validation
  rejects non-tuple sources and out-of-range indices at any region depth. Host
  preparation and COMBO checks inspect every position. This does not claim
  frontend editor or importer support.
- Required closed dynamic slots can bind their selected variant type to shared
  input and output variables, including list and asset element types
- Widgets: asset, save-target, combo and multi-combo with optional labels,
  per-option info, and folder paths; static combo choices preserve strings,
  booleans, JSON-safe integers and finite floats without conversion. Integer
  choices outside -(2^53 - 1) to 2^53 - 1 are rejected. Multi-combo choices
  remain strings. Combo options derived from dynamic input
  family members with stable stored IDs and separately editable labels;
  boolean; number (number/slider/knob/gradientslider displays), string, color,
  curve editor and graph-native image compositor,
  pack-declared custom widget types with JSON parameters for frontend extensions;
  representations with schema defaults and optional user switching;
  schema-declared static and dynamic-family text completions; conditional
  visibility groups for top-level widgets;
  after-generate modes fixed/increment/decrement/randomize;
  exact integer bounds from signed 64-bit minimum through unsigned 64-bit maximum
- Backend schemas accept number, string, boolean, and combo widgets on one-level
  lists of their matching primitive type, retaining numeric constraints and wire
  defaults unchanged. This schema support does not provide frontend list-value
  submission or scalar-to-list execution coercion.
- Backend color widgets accept scalar COLOR or STRING sockets, retaining the
  declared type and stored color text. Palettes, lists, and unions are rejected.
  This does not claim frontend COLOR editor support.
- Backend `CAMERA_INFO_STATE` values retain nested JSON camera-view state across
  pack boundaries. This does not claim a camera editor or CreateCameraInfo node.
- Inputs can declare socketless presentation while retaining their type,
  widget, default and interface order, including nested combos and families.
  The flag defaults to false, conflicts with forced socket presentation, and
  does not change execution signatures. Frontend pin rendering is not claimed.
- Backend mount widgets present one scalar STRING mount id. Input adjacency
  hints name a sibling input without changing interface order or execution
  identity. This does not claim frontend rendering or native save-node support.
- Backend asset-selection widgets retain one source-named COMBO input; VAE
  selection also permits the existing ASSET carrier for real files. The public
  asset authority enumerates only exact granted kinds for VAE, Diffusers,
  input subdirectories and saved datasets. It resolves literals, same-root
  file pairs and contained directory subtrees without filesystem or library
  fallback. The options API returns names and shapes, not host paths.
  Explicit mount purposes persist in config and runtime grants; omitted kinds
  remain untyped and do not authorize source selections.
  This does not claim frontend picker support.
- Ordinary text-encoder ASSET inputs accept contained source-relative filenames
  through model-name preflight, with exact-kind root priority and existing
  verified or unverified file references. Traversal and escaping links refuse.
- FLOAT number descriptors, including one-level FLOAT lists, accept `round: false`
  for explicit no-round presentation, or a positive finite numeric precision.
  Omission leaves precision unspecified; false is distinct from numeric zero.
  This backend schema support does not claim frontend editor support.
- Backend number descriptors also accept the scalar union of exactly INT and
  FLOAT, in either order, with FLOAT constraint and rounding semantics. Socket
  types and numeric defaults remain unchanged, including with named widget
  representations. This does not claim frontend numeric-union editor support.
- Forced-input FLOATS sockets accept numeric presentation with FLOAT constraint
  semantics, retaining their canonical socket type and default on the wire,
  including named representations. This does not change runtime values or
  provide frontend FLOATS editing.
- Node schemas can declare presentation-only editor roles and JSON-safe custom
  widget descriptors. Clients use these declarations without matching node IDs;
  native video trim and crop declare their exact `VIDEO_EDIT` features.
- Values: INT, FLOAT, STRING, COMBO, BOOLEAN,
  core.absent, immutable `CURVE` with monotone-cubic default or explicit linear
  interpolation, stable-sorted points including duplicate positions and empty
  curves that evaluate to zero; the 4096-point limit applies only to curve
  preview renditions, not value transport. HISTOGRAM is a plain ordered integer
  list with no length or positivity restriction through shared structural
  transport. Ordered `dinkster.layers`, strict versioned `dinkster.compositor`,
  parametric `list<T>` and `asset<T>`
- Primitive values reject mismatched runtime types and noncanonical encoded
  payloads. INT retains large integer seeds; FLOAT accepts integers and finite
  floats but not booleans. Primitive values, HISTOGRAM and core.absent retain
  declared auxiliary asset dependencies without hiding assets in their payloads.
- CURVE transport validates its numeric structure before exposing received
  values, preserves shorthand source records and declared auxiliary asset
  dependencies, and does not impose the preview rendition's point limit.
- Integer, float, text, and boolean primitive nodes declare their exact output
  values before execution for downstream presentation. Text switches between
  single-line and multiline views; the legacy multiline type remains loadable
  but is hidden from node search.
- Foundation value operations: versioned scalar math expressions; comparison
  with numeric epsilon; lazy typed selection;
  variadic boolean logic; clamping; range remapping with easing curves; and
  deterministic seeded integer or float random values
- Native `ComfySwitchNode` uses ComfyUI's `switch`, `on_false`, and `on_true`
  inputs. Only the selected branch executes; an unconnected selected input
  produces an absent output. The two arms share one type template.
- Loop boundary validation rejects unmatched starts or ends, ambiguous nesting,
  body escape paths, and accumulation controls fed from their own body,
  including boundaries inside region bodies.
- Native loop boundaries expand simple counts, For ranges, and typed lists
  into run-local body occurrences. They carry values between iterations,
  collect final or accumulated results, and preserve nested occurrence identity.
  Absent termination dependencies still complete the loop; an absent collected
  output propagates absence instead of leaving the closing boundary waiting.
  Loops can read enclosing region ports, and pending regions can be inhibited.
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
- Empty LATENT mappings remain present values through host transport and cache;
  nonempty LATENT mappings still require a samples entry
- Packs can import `LatentStream` and `MultiStreamLatent` from `dinkster_api.v1`
  without loading Torch or inference. These are the same ordered, role-labeled
  values returned by `materialize_latent`, with no wrapper or wire conversion.
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
  Declared output nodes in a body execute each iteration, including nested
  regions containing output nodes. A region needs an exported output or a
  declared output node; sink-only maps need no dummy exported output.
  Lazy routing resolves independently for each iteration; unselected branch
  nodes and nested regions neither prepare nor execute.
  Each body exposes its immediate zero-based `$region.index`; an explicitly
  declared `index` port keeps its declared meaning for graph compatibility.
  Images, latents, conditioning, masks, audio, video, assets, strings,
  integers, floats, and pack-defined values use the same generic region path.
  Expansion and per-iteration start/completion events expose real occurrence
  progress. The foundation template catalog includes executable map, gather,
  fold/scan, while/until, and per-item image-spawn workflows. See
  [Loop regions](../loops.md).
