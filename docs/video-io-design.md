# First-class VIDEO values

This is the replacement for the frame-batch value model in PR #108. It
specifies the VIDEO v2 contract owned by #1250, not a claim that the runtime
already implements it. All eleven acceptance criteria in #1250 remain
required. IMAGE/AUDIO materialization is an explicit conversion, not the
representation of an ordinary loaded, trimmed, or saved video.

## Reference behavior

Compared source revisions:

- [ComfyUI master 15eb748b3ec5](https://github.com/Comfy-Org/ComfyUI/commit/15eb748b3ec5f8a0a2d470b7fb280e2d7579f916),
  `comfy_api/latest/_input/video_types.py`,
  `comfy_api/latest/_input_impl/video_types.py`, and
  `comfy_extras/nodes_video.py`.
- [VideoHelperSuite main 4d907bee61e9](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/commit/4d907bee61e92c2e65af3bd6383a4e4d356126d1),
  `videohelpersuite/load_video_nodes.py`, `videohelpersuite/nodes.py`, and
  `video_formats/`.

ComfyUI supplies lazy file-backed VIDEO, VIDEO_EDIT, HDR inputs, and
frame-at-a-time transcoding. Its `save_to` disables stream reuse for every
trim or crop. Stream-copy trimming in Dinkster therefore exceeds ComfyUI; it is
not a parity claim. Its `GetVideoComponents` emits RGB separately from the
internal alpha component. Dinkster emits RGBA when present; ComfyUI is not the
alpha golden.

VHS supplies source/loaded information fields and useful frame-window
controls. Its frame-batch saver always encodes, except for copying the encoded
video during subsequent audio muxing. Its OpenCV source duration is frame
count divided by fps; its FFmpeg path estimates source count from duration
times fps. Neither establishes exact VFR frame counts. Dinkster exposes the
provenance of estimates rather than presenting them as exact counts.

## Value and ownership

The schema atom remains `comfy.VIDEO`. Version 2 belongs to the value codec,
not a second media atom. The source-backed runtime form is:

```text
{
  source: AssetRef | bytes,
  probe: VideoProbe,
  edits: [VideoEditOperation, ...]
}
```

The deferred alternative replaces `source` with:

```text
components: {images: IMAGE, audio: AUDIO | absent, fps: Fraction,
             bit_depth: 8 | 10, color_space: sRGB | HDR | HDR PQ,
             color: {primaries, transfer, matrix, range}}
```

Exactly one of `source` and `components` is present. Components are existing
typed values, not compressed video or a hidden eager encode. A runtime may
retain references to immutable component storage; crossing a boundary uses
their canonical codecs, never pickle, tensor addresses, or producer paths.
For components, probe dimensions/count/rate/duration come from component
metadata; container, encoded codec, pixel format, and stream time base are
null until save. Bit depth is intended encoding depth, not float storage
precision. The components variant cannot participate in packet-copy concat.

`dinkster-values` owns the data validation, serialization, identity, probe, and
pure edit arithmetic. The non-pack `dinkster-video` library owns decoding,
filtering, and encoding shared by native and compatibility nodes. Media I/O
owns node schemas and mount-authorized saving. The asset layer owns local
source binding and publication; the worker transport owns discovery and staging of referenced
assets. The compat package only translates upstream runtime forms. No node
pack imports another node pack's implementation, and values do not import
the asset package that already depends on them.

## Probe

Probe once at source admission with PyAV, without materializing frame arrays.
The demux allowlist is `mp4`, `mkv`, `mov`, `webm`, `avi`, and `gif`.
Container identification must distinguish Matroska from WebM and QuickTime
from MP4; a comma-separated demuxer name is not a container identity.
Unsupported or malformed media fails before publication as VIDEO.
Received probe claims must match the authorized source bytes at admission
or first source resolution. A content digest alone does not validate probe
claims. Local verification evidence may be retained across lazy edits but
must not be accepted from serialized metadata.

`VideoProbe` contains these stable fields:

| Field | Meaning |
| --- | --- |
| `container`, `video_codec`, `pix_fmt` | Canonical container, encoded codec name, source pixel format |
| `bit_depth`, `alpha` | Component precision and presence of alpha, including palette/side-stream alpha |
| `color_space` | `sRGB`, `HDR`, `HDR PQ`, or `unknown`; a label, not a conversion |
| `primaries`, `transfer`, `matrix`, `range` | FFmpeg enum integers, retaining unspecified values |
| `width`, `height` | Coded source pixel dimensions |
| `rotation` | Display rotation in degrees; applied once on pixel materialization |
| `fps`, `time_base` | Reduced positive rational average rate and stream time base, or null when unknown |
| `start_time` | Source video stream origin in seconds, as a rational |
| `frame_count`, `frame_count_kind` | Nonnegative count or null; `header`, `estimated`, or `unknown` |
| `duration`, `duration_kind` | Video duration in seconds or null; `stream`, `frames`, `container`, or `unknown` |
| `audio` | Ordered list of `{index, codec, sample_rate, channels, layout, time_base, start_time, duration}` |

Video and audio codec names identify the encoded format, not a decoder
implementation: for example, `av1`, never `libdav1d`.

Prefer stream duration times time base, then header frame count divided by
fps, then container duration. Keep unknown facts unknown; do not decode the
entire source to answer an info request. A count inferred from duration and
average fps is explicitly estimated. All durations and timestamps are
rational internally. Finite widget seconds convert through their decimal
representation, not by accumulating binary floating-point offsets.

`video.info` returns the source fields above and effective width, height,
duration, fps, frame count, and count provenance after edits. Source facts
never change. Effective facts are a pure fold over edits: no re-probe, decode,
or asset fetch. Unknown duration propagates until a consumer can determine it;
an end-relative trim or strict duration check requiring that fact fails
explicitly rather than assuming zero. VHS source-field comparisons use the
same CFR corpus and separately identify VFR/container-duration differences.

## Ordered lazy edits and VIDEO_EDIT

`edits` is the sole execution-authoritative ordered list. Each entry has
exactly one of `trim`, `crop`, `scale`, or `concat`; a trim entry also carries
`strict_duration`. This makes crop-before-scale distinct from
scale-before-crop and avoids a second output-affecting summary of the list.
Empty lists are legal. Append and compose without mutating prior values.

The UI/import widget is preserved verbatim as ordinary graph data:

```json
{"trim":{"start_time":1.25,"duration":3.5},"crop":{"x":100,"y":40,"width":1280,"height":720}}
```

Both sections are optional; `{}` is legal. A trim node reads only `trim`, a
crop node reads only `crop`; unrelated sections survive workflow save/load.
`features` chooses visible widget sections and has no execution semantics.
`strict_duration` is a separate node input, never inserted into the imported
widget. The execution normalizer does not rewrite the stored widget.
Explicit scalar node inputs remain linkable. When the optional widget input
is supplied, its relevant section is authoritative; omission of that section
is a no-op, not a fallback to hidden scalar defaults.

### Trim

`{"trim":{"start_time":seconds,"duration":seconds},"strict_duration":bool}`
operates on the current effective clip. Negative start is relative to its
end and clamps to zero. Duration zero means the remaining clip. Positive
duration is limited to the available interval unless `strict_duration` is
true, in which case an overrun is an error. Empty selections fail explicitly.
Selected frame presentation timestamps lie in the half-open interval
`[start, end)`. Audio selects the same interval with sample-accurate clipping
when decoded. Stream origins are accounted for before selection.

Successive trims intersect the current clip; they cannot restore frames
discarded by an earlier trim. For example, a 10-second source trimmed to
`(2, 3)`, then `(1, 0)`, selects source `[3, 5)`, not `[3, 10)`.
This is deliberately different from ComfyUI's file-backed `as_trimmed`, which
adds starts but replaces duration. Aliases and the Dinkster-side `VideoFromFile`
adapter use native intersection. Importing an upstream value takes its
already-composed active window as one edit, preserving that result without
replaying its history. Imported widgets remain lossless graph data; the
following execution differences require explicit alias tests, not an exact
parity or general superset label:

| Operation on a 10-second source | ComfyUI | Dinkster |
| --- | --- | --- |
| File `(2,3)` then `(1,0)` | `[3,10)` | `[3,5)` |
| File `(2,3)` then `(1,5)`, non-strict | `[3,8)` | `[3,5)` |
| File `(2,3)` then `(1,5)`, strict | `[3,8)` | Strict-duration error |
| File `(2,3)` then `(-1,0)` | `[1,10)` | `[4,5)` |
| Components `(3,8)`, non-strict | Error | `[3,10)` |

### Crop

`{"crop":{"x":int,"y":int,"width":int,"height":int}}` uses pixels in the
current input's display-oriented frame, not normalized coordinates or CSS
pixels. Either nonpositive extent is a full-frame no-op. Clamp x/y to
`[0, dimension - 1]`, then floor each origin to an even coordinate. Limit
width/height to the remaining frame extent without shifting the requested
extent when a negative origin is clamped. An exact full-frame rectangle is
a no-op, including odd source dimensions. Otherwise floor extents to even
sizes; either zero extent is a no-op. These steps match ComfyUI's
`normalize_crop_rect`. Crops compose relative to the preceding effective
view; an implementation may fuse adjacent crops without changing that
meaning. Rotation is interpreted before display-space crops and not applied
twice.

### Scale

`{"scale":{"width":int,"height":int,"fit":"stretch|crop|pad",
"interpolation":"nearest|bilinear|area|bicubic|lanczos",
"pad_color":[r,g,b,a]}}` declares positive even output dimensions. `stretch`
uses those dimensions; `crop` covers then center-crops; `pad` fits then
center-pads, assigning an odd remainder to right/bottom. Resized dimensions
round up to even for cover and down to positive even for fit. Color samples
are finite in `[0,1]` in the carried transfer function; omitted pad alpha is opaque.
Alpha and source color metadata survive scaling. Identity scale is a no-op;
any actual scaling requires pixel processing at save. The shared geometric
arithmetic must also serve frame materialization; there is no editor-only
resize interpretation. Extending the existing image resize node is separate
from defining and executing this VIDEO operation.

### Concat

`{"concat":[VIDEO, ...]}` appends clips to the current value in list order.
The list is nonempty and acyclic. Every participating clip must have matching
video codec parameters: codec, extradata, profile, level, coded dimensions,
pixel format, color fields, alpha, rotation, sample aspect ratio, and time
base. Audio stream count, order, codec parameters, layout, and rate must also
match. Probe display fields alone are insufficient proof of compatibility;
packet-copy preflight compares the actual stream templates.

Each clip's effective display dimensions after its own prior edits must also
match. Mismatched edited dimensions fail without implicit resizing, padding,
or applying one clip's crop to another. Matching spatial edits can therefore
require transcode while still producing one rectangular output frame shape.

Concatenation rebases timestamps onto one continuous timeline while preserving
within-clip presentation/decode offsets and audio/video synchronization.
Incompatible clips fail clearly; general timeline composition, transitions,
retiming, and mixed-codec concat belong to #1260. Compatible concat remains
legal when an earlier spatial edit requires streaming transcode rather than
packet copy. Later trims may span concat boundaries without decoding at the
edit-producing node.

## Save and materialization

`dinkster.save_video` accepts VIDEO, an authorized save target, container, codec,
optional CRF, and bounded data-only metadata. It returns the input VIDEO
unchanged as `video` and the published `asset<comfy.VIDEO>` as `asset`.
Original-source rendition/download remains distinct from rendering edits;
returning original bytes must never masquerade as an edited result.

Three paths share one normalized edit plan:

1. Unchanged source, same container/codec, no metadata overlay or CRF: stream
   the verified original bytes to the writer. Byte identity is exact.
2. Compatible encoded streams: remux packets without decoding. Preserve
   stream side data, color/rotation tags, extradata, and container metadata;
   requested metadata overlays container tags. Byte identity is not promised
   for a rewritten container; encoded packet payload identity is.
3. Otherwise: decode, transform, and encode frame-at-a-time into one bounded
   disk spool. Do not pass through a full IMAGE batch. Preserve source codec,
   precision, alpha, and color unless the user explicitly selected a change.

Trim copy is allowed only when the requested interval is exactly representable
by independently decodable packets, with no discarded reference dependencies,
and audio boundaries are exactly representable too. A keyframe flag alone is
not proof for open GOPs or B-frame dependencies. When copy safety cannot be
proved, transcode rather than snap the requested edit to a nearby keyframe,
emit preroll as visible frames, or shorten the interval. Strict duration does
not authorize inaccurate packet cuts. Encode thread counts, decoder reorder
buffers, and mux interleave buffering are bounded as part of the memory gate.

Native `container=auto` preserves the source container; component sources
default to MP4. Native `codec=auto` preserves a compatible source codec;
component sources choose H.264 for MP4/MKV or AV1 for WebM (VP9 for supported
alpha). Explicit H.264, AV1, and VP9 map to fixed PyAV encoders, never command
strings. MP4/MKV support H.264 and AV1; WebM supports AV1 and VP9, not H.264.
All allowed source containers support unchanged export/remux when their
streams permit it. Missing encoders or unsupported explicit combinations
fail before writing. No implicit codec fallback or alpha drop is allowed.

Metadata is bounded JSON data, not executable workflow instructions. Saving
uses existing mount authorization, streamed atomic publication, and the
1 GiB encoded-output bound. Failure or cancellation closes the spool and
publishes nothing. Audio tracks remain ordered and preserved during remux;
transcode uses supported explicit layouts, not silent stereo downmix.

### Precision and color

Assemble accepts `bit_depth=auto|8|10` and
`color_space=sRGB|HDR|HDR PQ`. An omitted color selection inherits IMAGE
color metadata; only untagged inputs default to sRGB. Explicit selection
declares the input sample interpretation, not tone mapping. Auto depth keeps
carried source precision and otherwise chooses 10 for either HDR label, or 8.
Decoded IMAGE range is full RGB; its optional matrix is coded-form provenance,
not a claim that the RGB array contains YUV. Assemble targets limited-range
YUV. Save derives destination conversion and written tags together; an RGB
destination uses the identity matrix and full range.

| Label | Primaries | Transfer | Matrix | Range |
| --- | --- | --- | --- | --- |
| sRGB | BT.709 | IEC 61966-2-1 | BT.709 | limited/MPEG |
| HDR | BT.2020 | ARIB STD-B67 (HLG) | BT.2020 NCL | limited/MPEG |
| HDR PQ | BT.2020 | SMPTE ST 2084 (PQ) | BT.2020 NCL | limited/MPEG |

H.264/AV1 use `yuv420p` at 8 bits and `yuv420p10le` at 10 bits. The 10-bit
RGB conversion scales normalized float samples to uint16 through `rgb48le`.
Eight-bit decode retains uint8; higher precision retains uint16, using
`gbrpf32le`, or `gbrapf32le` for alpha, without intermediate uint8 quantization.
Ordinary typed consumers normalize integer storage on demand. Conversion must
not read uninitialized alignment padding.
This normalization is not tone mapping. Transcode copies raw primaries,
transfer, matrix, and range, including unknown values rather than inventing
sRGB. Alpha-capable source export preserves alpha; an explicit incompatible
encoder fails. HDR and alpha conformance uses independently generated media.

Disassemble emits images, optional audio, fps, bit depth, color space, frame
count, and duration. Its IMAGE carries color and source-depth provenance
through serialization so reassembly can inherit them after a worker hop.
This is a narrow metadata requirement, not the broader storage-class or
alpha-policy redesign in #1251/#1258. Materializing a full IMAGE/AUDIO value
retains the existing bounded-array limits; long frame-tensor streaming is
#1252. Ordinary load/trim/save must not pay those full-array costs.

## Compatibility and persisted workflows

File-backed ComfyUI values use the real `VideoFromFile` source without calling
the base `VideoInput.get_stream_source`, which may encode. Preserve its active
trim and crop separately. The pinned source has no public crop getter:
isolate the version-specific private crop extraction in the compat adapter
and test it against the pinned class. Unknown subclasses must not silently
drop edits or encode at the boundary. Components-backed values use component
getters without encoding and retain separate upstream alpha as RGBA.

Resolve a local backing file to an authorized asset, or ingest it into the
worker's publishable value/asset store. Inline only genuinely small bytes.
On return to ComfyUI, a `VideoFromFile` subclass delegates metadata, edit,
components, and save behavior to the same VIDEO contract. A worker-local
resolved path may exist inside that adapter, never in the serialized value.

Aliases preserve all source outputs and these controls:

| Source | Native expression |
| --- | --- |
| LoadVideo | Asset-backed `dinkster.load_video_value` |
| Video Slice | Lazy `dinkster.video.trim` scalar inputs |
| VideoTrim / VideoCrop | Lazy trim/crop with verbatim VIDEO_EDIT |
| CreateVideo | Deferred assemble including auto/8/10 and color space |
| GetVideoComponents | Disassemble including bit depth/color space; RGBA superset |
| SaveVideo | VIDEO save, all flat and nested dynamic format/codec/encoding shapes, metadata and passthrough |
| SaveWEBM | Deferred assemble plus VIDEO save; preserve IMAGE passthrough, VP9/AV1, fps, CRF |

ComfyUI SaveVideo's node-level auto container rule differs from native auto:
auto plus AV1 means WebM; other auto cases mean MP4. The alias lowers this
explicitly, including nested `codec.encoding.crf`. Invalid WebM/H.264 is an
execution error, not a blanket translation refusal. Old save schemas migrate
with links, target, output identities, literals, and dynamic selections
intact; an IMAGE saver lowers to assemble plus the VIDEO saver.

Persisted byte-only VIDEO payloads are recognized as v1, validated against
their original metadata, then probed and represented as source plus no edits.
Large payloads are promoted by the hosting asset layer without re-encoding;
pure codec decode may hold the legacy bytes until that publication boundary.
New writes always use v2. Old cache fingerprints are not reinterpreted as v2
fingerprints; replay either decodes through the migration or misses safely.

## Execution lanes and evidence

The acceptance suite must cover every allowed container; unknown metadata;
composed trims/crops/scale/concat; empty and partial widgets; HDR and alpha;
deferred components; v1 migration; all official template video shapes; and
passthrough VIDEO plus saved-asset outputs. Concat coverage includes matching
edited dimensions and rejection of mismatched edited dimensions. Source
fixtures record generator, source revision, size, SHA-256, and ffprobe facts.
Deterministic paths compare exact bytes or packet payloads as appropriate.
Re-encodes declare measured codec tolerances; they are never silently widened
to pass.

CI runs in-process and worker/shared-memory conformance and measures peak RSS
growth below 200 MiB for a 1080p 10-second LoadVideo -> Video Slice -> SaveVideo
chain, including the materializing save process, not only its parent.
Linux 4-GPU-host checks cover 10-bit HLG remux, HDR PQ component round-trip,
AV1 retention, and alpha. A second LAN host measures zero producer frame
decodes and one destination source fetch for cross-host trim/save. The RunPod
execution-pool lane exercises the same saved workflow and source staging.
Single-job multi-GPU is applicable to a graph containing a distributed
producer; standalone CPU PyAV edit/save has no GPU sharding and must be
recorded as inapplicable, not reported as a GPU pass.

Hardware/cloud lanes are evidence obligations, not alternate value formats.
Missing lanes are requested explicitly with runnable commands. Frame-tensor
streaming (#1252), preview renditions (#1254), and NVENC are outside #1250.
The single-clip widget is not a second timeline type: #1260 consumes the same
VIDEO source and trim/crop meanings at a clip boundary. Editors only author
node-expressible values; no editor state changes output behind the graph.
