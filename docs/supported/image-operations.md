## Image operations

- Typed `dinkster.region` rectangle values for shared crop and placement geometry
- Typed per-frame `dinkster.pose` keypoint and skeleton values, including
  OpenPose/controlnet_aux JSON import and export and human/AP10K rendering
- Image or mask resize by dimensions, one side, longest or shortest side, scale
  factor, total pixels, or an image-or-mask reference; stretch, fit, fill, and
  pad modes with apply conditions and anchors; constant, edge-average,
  edge-pixel, and blurred-background padding; explicit nearest, crop, or pad
  divisibility finalization; optional synchronized companion masks; nearest,
  nearest-exact, bilinear, bicubic, Lanczos, and area interpolation
- Image quarter-turn and arbitrary rotation, horizontal and vertical flip,
  pixel or fractional translation and shear, and outpaint padding with a
  feather mask
- Rectangular, typed-region, and mask-derived crop with padding and explicit
  rounding/outside policies; mask-aware uncrop with opacity, clipping, resize,
  and singleton batch broadcast
- Per-frame tracked crop and uncrop for IMAGE and VIDEO from masks or
  Region/Detection lists, with fixed largest-visible pixel windows, empty-frame
  preservation, soft-mask and opacity compositing, and processed-crop resizing.
  VIDEO retains frame rate, audio, bit depth, and color metadata; encoded or
  edited inputs require 8/10-bit sRGB, HDR, or HDR PQ metadata.
- Image compositing with placement, mask, common blend modes, Porter-Duff
  operations, explicit alpha polarity, and configurable batch reconciliation
- Graph-native image composition with ordered `dinkster.layers` values, native
  placement from image batches or regions/detections, digest-bound compositor
  edit commands, 26 blend modes plus dissolve, perceptual or linear-light
  rendering, per-layer previews, flatten/split, structural editing nodes, and
  safe native-placement fallback for stale edits; CPU reference rendering is
  bounded to 4,194,304 canvas pixels and 67,108,864 layer/mask work-pixels
- Isolated GLSL ES 3.00 image rendering through ANGLE with up to five image
  inputs, typed scalar and curve uniforms, four render targets, custom output
  dimensions, and 1-32 passes; Linux and Windows x86-64/ARM64 and macOS ARM64
  are supported, while unavailable platforms refuse explicitly
- Pointwise invert, normalize, brightness, and contrast adjustment; Gaussian
  blur, sharpen, quantize, seeded noise, and bounded image morphology
- sRGB, linear Rec.709, Rec.2020 HLG, and Rec.2020 PQ conversion with
  straight-alpha preservation, plus RGB and YCbCr channel split/merge and
  explicit alpha extraction and joining
- Explicit coverage/transparency mask polarity, with migration of saved polarity
  controls; mask polarity inversion and RGBA premultiply/unpremultiply nodes
- ComfyUI BatchImagesNode, ResizeAndPadImage, and ImageCompare translations;
  paired comparison previews accept either, both, or neither image
- Alpha premultiply/unpremultiply and mask polarity conversion are native-only
  operations without fabricated ComfyUI aliases
- Image operations declare alpha preservation, required alpha, creation, or
  intentional loss; channel extraction and RGB control hints declare their drops
- Solid, linear-gradient, radial-gradient, and checkerboard image creation;
  text, typed-region, and mask drawing with RGB or RGBA composition
- Solid, geometric, polygon, typed-region, gradient, transition, checkerboard,
  deterministic noise, and text mask creation with batched growth and rotation
- Mask threshold, grow, erode, grow-and-blur, open, close, hole fill,
  small-component removal with 4- or 8-neighbor connectivity, feather, blur,
  offset, remap, round, block, invert, and crop operations
- Mask composition with signed placement; image/mask conversion by channels or
  color distance; mask dimensions, batch count, area, and typed bounds
- Image width, height, batch count, channel count, min, max, mean, and histogram
- Image and mask batch range, repeat, reverse, shuffle, ping-pong loop, count
  expansion, concatenation, insertion, and replacement; image batch/list
  conversion and rebatching
- Row-major image grid composition and decomposition; image stitching; split
  and merge with overlapping tiles
- Image transitions between inputs, within a batch, or while joining batches,
  with slide, box, circle, door, and fade shapes plus easing and blur controls
- Checkpoint-free control-hint preprocessing: Canny and pyramid edges, standard
  line art, scribble and XDoG, binary thresholding, palette/luminance/intensity
  hints, deterministic content shuffle, pyramid/guided/simple tile hints,
  inpaint sentinels, pixel-perfect hint resolution, and adaptive control-hint
  resizing for categorical maps, binary edges, and continuous images
- Digest-pinned learned ControlNet line and edge preprocessing through the
  standard float32 CPU/CUDA vision components:
  realistic and coarse realistic, anime, and manga line art; AnyLine with
  standard, realistic, anime, or manga merging; HED soft edge and fake
  scribble; TEED; and M-LSD. TEED weights are CC-BY-NC-SA-4.0 and AnyLine
  MTEED weights are CreativeML Open RAIL++-M. PiDiNet and Scribble PiDiNet
  compatibility imports fail closed because their model license requires
  separate commercial permission;
  Diffusion Edge compatibility imports fail closed because its three-model
  runtime-installed dependency path is not implemented. No support is implied
  for other `comfyui_controlnet_aux` preprocessors.
- Model-based image upscaling (`UpscaleWithModel`) through the standard,
  digest-pinned float32 CPU vision components:
  ESRGAN/RRDBNet checkpoints in old and new key layouts (including plus/
  conv1x1 and pixel-unshuffle variants) and Real-ESRGAN Compact
  (SRVGGNetCompact), with overlapping-tile execution and feathered seam
  blending; requested tile size is honored exactly (no out-of-memory
  halving retry)
- Depth Anything V2 Large relative-depth control hints through the standard,
  digest-pinned CPU vision components;
  its model weights are restricted to non-commercial use by CC-BY-NC-4.0
- Depth Anything 3 Mono Large relative-depth control hints through the
  standard, digest-pinned float32 CPU vision components, using Apache-2.0
  model weights
- Model Depth offers Automatic, Depth Anything V3, and Depth Anything V2
  Large as human-labeled model choices. Automatic deterministically selects
  a compatible installed implementation.
- Typed `dinkster.detection` values (label, score, typed-region box, optional
  full-frame soft mask) batched as plain lists; detection creation,
  inspection, label/score/area filtering with result limits, stable
  multi-key sorting, and conversion to per-detection plus combined union
  masks with box rasterization and signed dilation
- COCO object detection (`Detect Objects`) through the standard, digest-pinned
  float32 CPU vision components
  (DETR ResNet-50); the prompt filters the fixed COCO class set by
  case-insensitive name, and an empty prompt keeps every class
- COCO object detection through the standard, digest-pinned
  `dinkster-vision-rtdetr` float32 CPU vision component (RT-DETR v4 x-HGNet), with
  strict fp16 checkpoint loading, unrestricted finite score thresholds,
  source-compatible per-frame slice-stop limits, and source-sized boxes
- Text-prompted object detection through the standard, digest-pinned
  `dinkster-vision-sam31` float32 CPU vision component (SAM 3.1 Multiplex);
  prompts can be interpreted as comma-separated phrases or one literal phrase,
  producing scored boxes with source-sized binary masks. The model weights use
  the SAM License
- Detect Objects offers Automatic, SAM 3.1, DETR ResNet-50, and RT-DETR v4
  x-HGNet model choices.
- Text-prompted segmentation (`Segment Objects by Text`) through the same SAM
  3.1 provider; one operation returns both the labeled detections and their
  source-sized binary masks
- Box-prompt segmentation (`Segment Detections`) through the standard,
  digest-pinned float32 CPU vision components (EfficientSAM-Ti); it preserves
  each detection and attaches its
  highest-predicted-IoU full-frame soft mask
- Box-prompt segmentation through the standard, digest-pinned float32 CPU
  vision components (SAM 3.1 Multiplex); it preserves
  each detection and attaches a source-sized binary mask. The model weights
  use the SAM License
- Segment Detections offers Automatic, SAM 3.1, and EfficientSAM-Ti model
  choices.
- Video object tracking (`Track Objects`) through the same SAM 3.1 provider;
  first-frame detection boxes or masks identify a fixed set of objects. Each
  object returns one source-sized mask batched across all input frames, along
  with their combined union
- Foreground alpha matting (`Image Matte`) through the standard, digest-pinned
  `dinkster-vision-birefnet` float32 CPU vision component (BiRefNet general)
