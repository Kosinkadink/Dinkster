# dinkster-nodes-image

`dinkster-nodes-image` supplies Dinkster's deterministic first-party image and mask
operations. Implementations use channels-last NumPy arrays on CPU and do not
import a tensor framework. `dinkster.image` values use BHWC layout and
`dinkster.mask` values use BHW layout.

The pack owns `dinkster.region`, the shared rectangular spatial value used by
image, mask, detection, and segmentation operations, along with two typed
values built on it: `dinkster.pose`, an immutable per-frame keypoint and skeleton
value whose nodes import and export OpenPose/controlnet_aux JSON and render
human or AP10K animal control images, and `dinkster.detection`, a labeled and
scored region with an optional full-frame soft mask. Detections batch as
`list<dinkster.detection>`; a segmentation result is a list of detections
carrying masks, so no separate segmentation type exists. The region and
detection value classes and byte codecs live in `dinkster-values` so detection
provider packs can produce them in isolated workers; this pack registers the
type ids and owns their node surface. Image and mask array
codecs remain owned by `dinkster-nodes-media-io` and compose
through their registered type ids.

The pack provides consolidated families for image creation and drawing; mask
creation, text rasterization, morphology and component cleanup, composition,
conversion, and inspection; image geometry, composition, adjustment, filtering,
and channel operations; image and mask batch editing and combining; image list conversion
and rebatching; grid composition and decomposition; stitching and overlapping
tiles; image transitions; detection creation, inspection, filtering, sorting,
and mask conversion; and checkpoint-free control-hint preprocessing.
Small-component cleanup classifies pixels strictly above its threshold and
processes each mask frame independently. Components use selectable 4- or
8-neighbor connectivity; areas below `minimum_area` are removed, areas equal to
the boundary are retained, and all unremoved float values stay unchanged. Hole
fill treats only 4-neighbor background connected to the frame border as outside.
Preprocessors cover Canny and pyramid edges, standard line art, scribble and
XDoG, binary thresholding, palette and grayscale hints, content shuffle, tile
hints, inpaint sentinels, and pixel-perfect resolution calculation. Operation
selectors keep related behavior on one node, while adaptive control-hint
resizing preserves categorical maps, binary edges, and continuous images. The
pack also owns stable schemas for model-backed HED, realistic/anime/manga line
art, AnyLine, TEED, M-LSD, and depth preprocessing. Separately installed model
packs provide their execution and populate their provider selectors. The same
owner-schema pattern covers the model-backed vision capabilities:
`dinkster.detection.detect` (prompted object detection producing detections;
`max_results` limits each frame independently, with default count and
source-compatible slice-stop modes), `dinkster.detection.segment` (detection boxes
prompting a segmentation model, returning the detections carrying masks),
`dinkster.detection.segment_text` (text-prompted detection and segmentation in one
operation),
`dinkster.image.matte` (foreground alpha matting), and `dinkster.detection.track`
(video object tracking producing one per-frame-batched mask per tracked
object). Each refuses execution until a provider pack is installed.

`dinkster.image.resize` describes the requested result rather than a source node
pack's mechanics. It accepts images or masks, optional companion masks, and
image-or-mask references; exposes dimensional, scale, total-pixel, fit, fill,
padding, anchor, apply-condition, and divisibility intents; and provides
constant, edge-average, edge-pixel, and blurred-background padding. Legacy
native, KJNodes, and Essentials workflows migrate to this single schema.

`Tracked Crop` and `Tracked Uncrop` process one mask or Region/Detection per
IMAGE or VIDEO frame without temporal broadcasting. Crops use the largest
visible padded selection as a fixed pixel-preserving window, retain empty
frames, and reinsert processed crops with masks and opacity. VIDEO frame rate,
audio, bit depth, and numeric color metadata are retained. Encoded or edited
VIDEO is materialized without re-encoding; inputs whose color space is unknown
or whose bit depth is not 8 or 10 are rejected because component VIDEO cannot
represent that metadata.

Graph-native composition uses `dinkster.layers` as the ordered runtime value and
keeps source pixels in graph topology. Add Layer appends an image batch and its
native placement, Layers From Bounding Boxes maps image frames to regions or
detections, and Create Layered Image renders up to 50 expanded layers on a
bounded canvas. Its strict `dinkster.compositor` value stores only a versioned edit
recipe. The recipe is replayed when its ordered source fingerprints match;
otherwise execution reports stale state and renders the native placements.
Rendering supports per-layer previews, transparency masks, transforms, 20
blend modes, and perceptual or linear-light color math.

The `dinkster.compositor.state` event's `documentDigest` identifies the canonical
input document, before commands or color-space overrides. Its `document` is
the post-edit preview. Apply sends the input digest with commands relative to
that input; preview edits do not change the replay identity.

GLSL Shader executes GLSL ES 3.00 fragment shaders in a fresh ANGLE child
process so shader compilation and driver failures cannot corrupt the engine
process. It accepts up to five image inputs, float, integer, boolean, and curve
uniform families, up to four image outputs, custom dimensions, and 1-32 passes.
On passes after the first, `u_image0` reads the previous pass's `fragColor0`.
`u_resolution` and `u_pass` are provided automatically. Native execution is
available on Linux and Windows x86-64/ARM64 and macOS ARM64; unsupported ANGLE
platforms refuse with an explicit runtime error.

Preprocessor and pose-rendering parity are pinned to comfyui_controlnet_aux commit
`59b1fc411ede8623b2997855b8018f0b3b6cf49f`. Content shuffle deliberately uses
its stored seed for every value, including zero, so cached executions remain
deterministic instead of inheriting the reference's process-global RNG state.

`dinkster.image.adjust` declares a GLSL mirror covering all four adjustment
operations (see the GLSL mirror binding contract in the dinkster-schema README).
The parity corpus is `tests/fixtures/mirror-parity/image_adjust_v1.json`,
regenerated with `uv run python tools/generate_image_adjust_mirror_vector.py`.

`dinkster.image.filter` declares a GLSL mirror scoped by `applies` to its
`gaussian_blur` and `sharpen` operations; the remaining filter operations are
outside the one-shader single-pass contract and render no estimate. The
parity corpus is
`tests/fixtures/mirror-parity/image_filter_v1.json`, regenerated with
`uv run python tools/generate_image_filter_mirror_vector.py`.

`dinkster-pack.toml` is the public composition boundary. `IMAGE_NODES`,
`register_image_types`, and `comfy-aliases.json` are host wiring and
translation data, not stable Python APIs for other packs.
