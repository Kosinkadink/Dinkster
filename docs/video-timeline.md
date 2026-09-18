# Timeline document and VIDEO v3 contract

`dinkster.video_document` is source-only JSON, version 1. The root has exactly
`version`, `timeline`, `sources`, and `settings`. `timeline` is OTIO Timeline.1
containing Stack.1, Track.1 (`kind: Video | Audio`), Clip.2, Gap.1, and sibling
Transition.1 children. Nested stacks/tracks are retained. `settings` has
`width`, `height`, and numeric `rate` (frames/second). JSON must be below 1 MiB,
finite, and duplicate-key-free; limits are 4096 items, 16 composition levels,
256 bindings, 8192 pixels per dimension, and at most 240 fps.

## References and authority

Clips select `sources[clip.metadata.dinkster.source]`. A binding is either:

- `{type: "comfy.VIDEO", video: <reference-only VIDEO v2 packed record>}`,
  produced by `dinkster_values.video_document.video_reference(video)`. It retains
  the entire ordered base edit plan, not just the encoded asset. Inline
  sources and deferred component payloads must be published before binding.
- `{type: "comfy.VIDEO" | "dinkster.image" | "comfy.AUDIO" | "dinkster.layers",
  asset: <declared asset wire>}`. Assets contain a canonical blake3 digest,
  size, and optional name/mediaType/virtualPath, never host paths.
- A `dinkster.layers` binding may also contain `resources: [<asset wire>, ...]`.
  Every referenced raster must be declared, including its correct byte size.

The codec neither opens nor resolves assets. Imported ExternalReference
`target_url` is opaque text, even for `file:` and `https:` URLs. Import does
not create bindings from it. Host asset resolution only receives declared
digest references. OTIO media-reference maps, active reference keys, proxy
identities, markers (including comments), effects, and unknown metadata survive
interchange. `metadata.dinkster.proxy` is preserved but is not selected implicitly.

Each clip has exactly one authored trim authority:

- With `metadata.dinkster.video_edit`, that raw JSON object is authoritative.
  The sibling `metadata.dinkster.strict_duration` is the authoritative boolean
  (default false). All unknown widget keys, including a widget key named
  `strict_duration`, are uninterpreted data. Optional sections (including null)
  and original numeric values are retained. Codec roundtrips do not execute widget edits.
  Execution/export derives `source_range` from the widget; any stored
  `source_range` is a shadow projection and is never a second trim.
- Without the widget, OTIO `source_range` is authoritative. Source coordinates
  are relative to the active reference's `available_range.start_time`, or
  zero if absent. Editing this clip with the `trim` node's `video_edit` field
  converts the OTIO selection once into widget seconds, unless that submitted
  widget already supplies its own `trim` section.

VIDEO_EDIT execution uses the same VIDEO trim/crop arithmetic as the ordinary
graph nodes: seconds, duration 0 meaning remaining duration, negative starts,
clamping, separate strict duration, crop in source-display pixels after
rotation, full-frame sentinel, and even alignment. Raw JSON is never replaced
with normalized execution values. A selection-changing split/roll/ripple
updates the selected trim fields, preserving unrelated widget fields.

## Graph ports and editor operations

Node IDs are `dinkster.video_document.<command>`. No HTTP timeline endpoint is
implied. Every mutation takes `document: dinkster.video_document` and
`params: core.string` containing JSON, and returns `document` of the same type.
Indices are zero-based OTIO child indices, including gaps/transitions, not
indices of clips with those items filtered out. `track` defaults to 0 and
`clip` defaults to 0; track selection addresses top-level tracks.

| Command | JSON parameters or distinct ports |
| --- | --- |
| `make` | Optional `params` (`{}` default): width, height, rate, name, clips. Returns document. |
| `add_track` | kind (`Video` default), name, blend (`normal`), opacity (1). |
| `add_clip` | track, item (complete OTIO Clip/Gap), optional insertion index. |
| `bind_source` | source (binding ID), reference; optional track/clip also assigns that ID to a clip. Optional typed `video: comfy.VIDEO` port replaces reference and carries base edits. |
| `set_effect` | track, clip, index (append default), effect `{node_type, parameters}`, or remove:true. |
| `transition` | track, clip (outgoing clip), in_offset/out_offset in seconds (0.5 each). Inserts an SMPTE_Dissolve between adjacent clips. |
| `retime` | track, clip, scalar; replaces LinearTimeWarp/FreezeFrame without changing declared item duration. |
| `mix_audio` | track, audio_mix `{gain: scalar-or-CURVE}`. |
| `split` | track, clip, position (clip-local seconds); creates two selections. |
| `move` | track, clip, to_track (same default), index (final child index). Adjacent transitions must be removed first. |
| `trim` | track, clip, start_time/duration/strict_duration; leaves gaps to preserve surrounding timeline positions. Alternatively video_edit plus sibling strict_duration sets the raw VIDEO_EDIT selection. |
| `ripple` | track, clip, start_time/duration/strict_duration; closes removed time. |
| `roll` | track, clip, delta seconds; moves an adjacent cut without changing total duration. |
| `render` | document -> video: comfy.VIDEO. Lazy, no encoding. |
| `import_otio` | otio: core.string -> document. No source resolution. |
| `export_otio` | document -> otio: core.string. |

Effects remain ordered in `clip.metadata.dinkster.effects`. A parameter may be
`{type: "dinkster.curve", value: {points: [{position, value}], interpolation}}`.
Positions are clip-local seconds, evaluated by the canonical CURVE kernel.
Tracks use `metadata.dinkster.blend`, `opacity`, and `audio_mix.gain`. Dissolves
are track siblings with in_offset/out_offset around the cut, using source
handles; they do not shorten the declared duration. Audio fades follow these
same intervals. LinearTimeWarp changes sampling speed, not duration;
FreezeFrame samples one frame. Reverse uses bounded per-frame seeks.

## Execution and wire admission

A single uneffected video clip lowers to ordinary VIDEO v2, preserving the
untouched-source and safe packet-trim stream-copy paths. General renders use
the explicit VIDEO v3 variant: magic `DINKSTER-VIDEO` followed by byte 3, followed
by the canonical document bytes. Runtime shape is `{timeline: <document>}`.
Its local asset factory is never serialized or fingerprinted. There is no
invented encoded-source probe and no inline frame/audio chunk. Admission
accepts v2 and v3; v2 encoding and fingerprints are unchanged. v3 metadata
declares `codec_version: 3`, `representation: timeline`, effective canvas/time
facts, and the transitive declared asset references. Ordinary v2 edit nodes
refuse v3; edit its document instead.

The CPU profile renders frame/window-at-a-time to the existing container sink:
RGB8 sRGB against black at settings.rate, exact canvas geometry, shared image
blend/source-over/transition/text kernels, canonical CURVE, and shared
ImageDocument rendering. The visual effect interpreter admits
`dinkster.image.draw_text`; other node types and opaque third-party effects are
preserved but refuse rendering. Composition time effects also refuse rendering.
HDR inputs and implicit geometry changes are not silently converted. Decoders
own at most current/lookahead frames and close on cancellation. Active-frame
allocation is budgeted before opening sources. VIDEO sources require a known
duration; sampling beyond it fails instead of holding the last frame.
Premultiplied inputs are converted once before straight-alpha composition.
Audio tracks use bounded windows, linear gains/dissolves, and explicit
silence, not whole waveforms.
Audio retiming requires a declared resampler and is refused by this profile.
Standalone AUDIO sources require the canonical bounded AUDIO runtime;
`SourceMedia.audio_window` is the integration point for that runtime.

`TimelineError` carries `code`, `path`, and a readable message. Stable codes
include `invalid_document`, `document_limit`, `unsupported_version`,
`unsupported_schema`, `invalid_time`, `invalid_edit`, `invalid_selection`,
`unbound_source`, `invalid_source`, `invalid_transition`,
`unsupported_transition`, `unsupported_effect`, `unsupported_time_effect`,
`source_range_unavailable`, `unknown_duration`, `geometry_mismatch`, `color_space_mismatch`,
`audio_rate_mismatch`, `audio_layout_mismatch`, and `invalid_opacity`.
`dinkster_video.timeline.diagnostics(document)` reports unbound clips and opaque
effects without resolving media. Render errors retain their diagnostic code.

## Interchange fidelity and conformance

OTIO RationalTime value/rate and TimeRange fields are retained, not reduced
to equivalent decimal seconds during foreign-document roundtrips. Import
uses the OTIO core JSON parser to preserve its float semantics (Python and
OTIO parse some Resolve metadata decimals differently). Dinkster bindings/settings
are embedded under `timeline.metadata.dinkster.document` only on bound exports;
other editors may retain this extension without interpreting it, or discard
it. Dinkster-specific widget/effect/CURVE/blend/audio metadata has the same loss
risk outside Dinkster. Opaque effects are not represented as supported effects.

Run `python -m pytest -q tests/test_video_document.py` for codec, actual OTIO
`is_equivalent_to`, graph/shared-memory, widget, layer, and window tests.
The loopback TCP sink test stages declared sources, refuses producer decoding,
and returns only an encoded asset reference; it is not a multi-machine LAN test.
`python tools/timeline_conformance.py` generates two synthetic 31-second
720p sources and renders a 60-second timeline with a 1-second dissolve and
title. It checks 1440 frames, source-only JSON size, and absolute peak RSS
below 2 GiB in a fresh subprocess. It runs on Linux/macOS; a Linux result is
not Mac evidence. Corpus provenance is in `tests/fixtures/otio/README.md`.
