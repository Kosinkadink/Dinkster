# dinkster-video

`save_video_stream` consumes the shared lazy VIDEO edit plan and writes one
encoded container. `save_video_frames` writes GIF, animated WebP, or a ZIP of
numbered PNG files. The latter returns an ordinary asset, not a new VIDEO
source type. VIDEO inputs remain MP4, MKV, MOV, WebM, AVI, and GIF only.
Native savers encode at the frame-owning worker into a bounded local spool;
the spool rolls to disk after 512 KiB. The asset writer receives encoded
bytes, including for remote save targets.

`save_frame_records(records, destination, *, format, bit_depth, color,
frame_count=None, loop=0, dither="sierra2_4a", lossless=True, quality=80,
method=4, compression=6, metadata=None, on_diagnostic=None)` is the same
animation/sequence encoder without a VIDEO source. Records are
`(Fraction start_seconds, Fraction duration_seconds, frame)` with normalized
float32 HWC RGB/RGBA arrays or PyAV RGB frames. The adapter supplies straight
alpha, canonical `image_color` facts and source precision, and bounds input
chunk sizes before iteration. Dimensions/channels must stay fixed; timestamps
must be contiguous from zero. The optional declared frame count is
checked. The sink closes the iterable/iterator on success and failure, but
leaves the caller-owned seekable binary spool open. Publish only after success.
It returns the actual suffix/MIME after preserving defaults. Formats are
`gif_pillow`, `gif_ffmpeg`, `webp`, `png8`, `png16`, and `apng`; APNG is an
animated PNG asset, not a ZIP or a new allowed VIDEO source. High-depth/HDR
record streams select APNG when an 8-bit animation cannot preserve them.
PNG compression is 0-9; WebP method is 0-6. APNG retains 8/16-bit alpha,
loop count, rational frame durations, and workflow tags. APNG frames replace
the canvas rather than requiring decoder support for 16-bit alpha blending.
Chunk adapters map `png` to `apng` and `gif` to `gif_pillow`, check raw and
normalized chunk budgets before conversion, and publish the returned suffix
and MIME. They pass a JSON metadata object, not Pillow `PngInfo`.
Individual still assets keep their existing encoder and per-image publication;
they are not APNG or PNG ZIP outputs. In particular, their legacy 8-bit
truncation differs from the animation sink's nearest-integer packing.

## CPU formats and controls

Source comparison pins (resolved with git, not abbreviated revisions):

- VHS: `4d907bee61e92c2e65af3bd6383a4e4d356126d1`.
- ComfyUI: `15eb748b3ec5f8a0a2d470b7fb280e2d7579f916`.

The following maps every format in the pinned VHS `video_formats` directory
and the two Pillow choices in `VideoCombine.INPUT_TYPES`. These are native
controls, not permission to execute VHS format JSON.

| VHS format | Native controls | Encoding details / unsupported knobs |
| --- | --- | --- |
| `image/gif` | Save Animation or PNG Sequence: `format=gif_pillow`, `loop` | Pillow per-frame quantization, disposal 2; no full decoded frame list |
| `ffmpeg-gif` | `format=gif_ffmpeg`, `loop`, `dither` | All nine pinned paletteuse dithers; per-frame palettegen rather than an unbounded whole-video palette |
| `image/webp` | `format=webp`, `loop`, `lossless`, `quality` | Independent compressed frames in ANMF chunks; straight alpha retained |
| `h264-mp4` | Save Video: `container=mp4`, `codec=h264`, `crf` | 8/10-bit YUV420 selected by VIDEO precision; fixed fast/zerolatency preset |
| `h265-mp4` | `container=mp4`, `codec=hevc` (`h265` also accepted), `crf` | hvc1 tag, 8/10-bit YUV420, medium/zerolatency preset, bounded CPU threads |
| `av1-webm` | `container=webm`, `codec=av1`, `crf` | 8/10-bit YUV420; alpha uses preserving FFV1 fallback because the configured CPU encoder has no alpha side-data support; fixed SVT preset 8, one logical processor, low-delay prediction, no lookahead/temporal filter |
| `webm` | `container=webm`, `codec=vp9` or `vp8`, `crf` | VP9 8-bit alpha uses yuva420p; zero encoder lag; Opus audio, not VHS's libvorbis |
| `ProRes` | `container=mov`, `codec=prores`, `profile=auto/lt/standard/hq/4444/4444xq` | 10-bit YUV422 or YUV444; alpha uses yuva444p10le, 16-bit ProRes alpha coding, PCM audio |
| `ffv1-mkv` | `container=mkv`, `codec=ffv1` | Lossless RGB/RGBA at 8 or 16 bits, FLAC audio; fixed level 3, coder 1, context 0 and four slices (bounded probability-model memory), GOP 1, slice CRC 1 |
| `8bit-png` | `format=png8` | ZIP of RGB/RGBA PNGs; one-based six-digit filenames, not a printf path supplied by the user |
| `16bit-png` | `format=png16` | 16-bit RGB/RGBA PNGs; no Pillow 8-bit RGB downconversion |
| `gifski` | Use `gif_ffmpeg` or `gif_pillow` | External gifski executable and its quality pass are not run; a named unknown format uses preserving CPU defaults with a diagnostic |
| `nvenc_h264-mp4`, `nvenc_h265-mp4`, `nvenc_av1-mp4` | Refused | NVENC execution and requests are refused until a hardware-encoder policy exists; hardware bitrate/megabit controls are not used |

`bit_depth=auto/8/10` and `color_space=sRGB/HDR/HDR PQ` belong to Assemble
Video, not a second conversion implementation in the saver. H.264/HEVC CRF
is 0-51; VP8/VP9/AV1 is 0-63. VHS's UI maximum 100 does not define a valid
codec range. Arbitrary pixel formats, FFV1 coder/context/slices/GOP changes,
fake transfer tags, environment overrides, filters, and custom FFmpeg
command JSON are not an executable interface. Fixed settings above replace
those knobs; metadata is data only, never a command or environment source.

## Shared VHS options

| VHS option | Native mapping or boundary |
| --- | --- |
| `images`, `frame_rate`, `audio` | Assemble Video `images`, `fps`, `audio`; the saver consumes VIDEO |
| `filename_prefix`, `save_output` | Save target mount and prefix select output/temporary storage; no producer-local paths cross workers |
| `loop_count` | Animation saver `loop`, 0 means infinite; ordinary video containers do not have a playback-loop flag |
| `save_metadata`, `prompt`, `extra_pnginfo` | Saver `metadata` JSON object; omit tags with `{}`; read back with `dinkster.read_video_metadata` |
| `trim_to_audio` | Requests shortest audio with proven coverage; otherwise preserves finite VIDEO length with a diagnostic. False pads short audio with silence to video length |
| `pix_fmt`, `input_color_depth`, `has_alpha` | Derived from source precision, color, and channel facts, not untrusted FFmpeg arguments |
| `pingpong` | Not a saver control; frame-order edits must be explicit before assembly |
| `meta_batch` | VHS BatchManager is not a native input; native saving iterates the VIDEO source without materializing a frame batch |
| `vae` / latent images | Decode latents through the existing VAE node before assembly |
| `unique_id` | Execution identity, not a codec option |
| `VHS_MetadataImage`, `VHS_KeepIntermediate`, `manual_format_widgets` | No implicit intermediate sidecar files or arbitrary widgets; PNG-sequence metadata is in its first PNG |

## Preservation and defaults

Format/layout choices are preferences, not admission gates. When the
requested format cannot preserve channels or precision, a preserving CPU
default is selected with a nonblocking requested/effective diagnostic.
For example, H.264 RGBA and 10-bit VP9 RGBA use FFV1/MKV; ProRes alpha
selects 4444; GIF RGBA selects lossless animated WebP; HDR animations and
image-only requests carrying audio use FFV1/MKV; high-depth PNG8 selects
PNG16. Unknown audio-output layouts preserve the source; `mono` and
`stereo` are explicit downmix requests. No implicit tone mapping or alpha
flattening occurs. Failure of the preserving encoder remains a capability
error. JSON command execution and asset/path integrity checks remain strict.

Component AUDIO's numeric frame bound does not certify physical source
coverage. `trim_to_audio` uses exact PCM shape or revalidated `pcm_npy`
lengths, including ordered edits whose concat children also have exact PCM
coverage. Other component AUDIO, including finite encoded-duration estimates
and trims of unknown sources, preserves finite VIDEO length with the
`component_audio_endpoint_unproven` diagnostic reason and explicit requested
true/effective false `trim_to_audio` substitution. It does not claim exact
shortest-audio completion. Reader exhaustion or zero-filled windows are not
native decoder EOF. Unknown component AUDIO and VIDEO read bounds reject
before decoder admission. Encoded VIDEO's native audio decoder retains its
own termination behavior; no component reader internals are used.

Declared speaker layouts, including 5.1, retain their channel positions.
Seven-channel source PCM without speaker labels also round-trips through
the PCM preserving fallback. More than eight channels cannot pass the
PyAV frame boundary safely; count-only canonical layouts such as `10c`
must not be reinterpreted as named speaker positions. Transcoding these
fails before frame conversion; unchanged safe stream copies remain available.
Supply an explicit canonical
AUDIO `channel_map` matrix before assembly when speaker assignment or downmix
is intended. Merely selecting `stereo` does not invent a matrix for discrete
inputs. A PCM codec fallback does not make this frame boundary safe.

The runtime callback receives one `media_format_fallback` record after a
successful encode. Encoded headers supply the final container, codec, pixel
format and first audio stream's layout; `audioStreams` describes every audio
stream. `requested` and `effective` each contain `container`, `codec`,
`pixelFormat` and `channelLayout`. Requested null means unspecified (including
auto/preserve); effective null means no applicable stream. A stable `reason`
describes the fallback, with individual substitutions retained as details.
Header inspection never materializes frames or an entire animation.

Native savers remove `code` from the callback record, add their actual asset
output port as `outputId`, then call
`dinkster_api.v1.report_value_diagnostic('media_format_fallback', data)`.
The worker snapshots the record into output `valueDiagnostics` metadata;
the engine emits `value_diagnostics` and supplies `nodeId`.
This uses the same transport and cache replay as other media diagnostics,
not a logger-only warning. Direct calls outside worker execution have no
ambient collector; library callers can supply their own `on_diagnostic`.

Container metadata keeps a typed JSON envelope so case-sensitive keys and
JSON-looking strings round-trip exactly. PNG sequences store workflow tags
in `000001.png`; each PNG carries full-range RGB cICP color facts. PNG/APNG
also expose compatible metadata keys as ordinary PNG text fields, including
`prompt` and `workflow`. Reserved or non-PNG keys stay in the typed envelope.
GIF stores a comment; WebP stores XMP. No new metadata side channel is
attached to IMAGE.
GIF time is quantized to centiseconds, WebP time to milliseconds. Alpha
goldens use independently constructed coverage ramps, not ComfyUI's
RGB-only component saver and not stream-copy outputs.

Run `python -m pytest -q tests/test_video_formats.py tests/test_video_runtime.py`
for CPU facts, pixel, alpha, precision, metadata, and duration checks.
`python tools/video_conformance.py --rss` measures the 1080p/10-second
load/trim/transcode path in a fresh process (less than 200 MiB RSS growth).

## Preview initialization

The opt-in `dinkster-video-preview` pack is included in this package. Its
code is isolated under `preview/`: the pack imports the API, while the video
library remains a lower-layer API dependency and does not import the pack.
The Python wheel includes the complete pack under `dinkster_video/preview`.
The `video-preview.initialize` node accepts and returns `comfy.VIDEO` unchanged.
It folds portable video metadata, bounds preview size to 512 pixels wide and
120 frames, and emits `video-preview.initialized` with `fps`, `frameCount`,
`height`, and `width`. It neither decodes frames nor imports Torch or ComfyUI.

From a workspace installed with `uv sync --all-packages`:

```console
uv run dinkster-serve --pack packages/dinkster-video/preview/dinkster-pack.toml --port 8199
```

Connect a VIDEO-producing node to `video-preview.initialize`. The host-owned
GET `/api/extensions/dinkster-video-preview/routes/preview-policy` returns
`{ "defaultFps": 24.0, "maxFrames": 120, "maxWidth": 512 }` from the same
worker policy used to initialize the event. The route requires `jobs:read`.

The snapshot-selected self-contained module `dinkster-video-preview.frontend.preview`
declares `event-consumer` for its typed event subscription and `app-workflow`
for querying its own route and presenting a host-rendered metadata status.
An enabled compatible frontend shows the policy response, connection/snapshot
identity, and latest node's preview dimensions, frame rate, and frame count.
It shows an unavailable-policy status on query failure. Disabling frontend
contributions does not disable the node.

The module never imports legacy ComfyUI JavaScript, calls PromptServer, accesses
global extension state, or fetches arbitrary URLs. Its immutable module URL is
chosen by `/api/extensions/snapshot`, not constructed by pack code.

See [the API contract](../dinkster-api/README.md#json-routes-and-events).
`tests/test_serve.py::test_video_preview_pack_route_event_and_module_end_to_end`
boots the isolated server and checks a real MP4 source through this node,
the route result, module bytes, and the correlated JSON WebSocket event.
