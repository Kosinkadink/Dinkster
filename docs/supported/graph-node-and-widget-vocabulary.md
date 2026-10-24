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
- Backend schemas accept number, string, boolean, and combo widgets on one-level
  lists of their matching primitive type, retaining numeric constraints and wire
  defaults unchanged. This schema support does not provide frontend list-value
  submission or scalar-to-list execution coercion.
- FLOAT number descriptors, including one-level FLOAT lists, accept `round: false`
  for explicit no-round presentation, or a positive finite numeric precision.
  Omission leaves precision unspecified; false is distinct from numeric zero.
  This backend schema support does not claim frontend editor support.
- Backend number descriptors also accept the scalar union of exactly INT and
  FLOAT, in either order, with FLOAT constraint and rounding semantics. Socket
  types and numeric defaults remain unchanged, including with named widget
  representations. This does not claim frontend numeric-union editor support.
- Node schemas can declare presentation-only editor roles and JSON-safe custom
  widget descriptors. Clients use these declarations without matching node IDs;
  native video trim and crop declare their exact `VIDEO_EDIT` features.
- Values: INT, FLOAT, STRING, COMBO, BOOLEAN,
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
