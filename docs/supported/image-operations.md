## Image operations

- Native `RenderMesh` produces a single image and coverage mask from the first
  counted mesh, with texture, vertex-color, clay, normal and depth modes,
  authored model transforms, supplied cameras and automatic framing.
- Native `Preview3DAdvanced`, `PreviewGaussianSplat` and `PreviewPointCloud`
  preserve original-file assets and sidecars while returning viewport camera,
  model placement and dimensions, with explicit metadata inputs taking priority.
- `DA3_GEOMETRY`, `FACE_LANDMARKS`, `MHR_POSE_DATA` and `KIMODO_POSE_DATA`
  transport depth, source images and cameras, per-frame face and body detections,
  face topology, masks, canonical colors and external rig state without dropping
  optional fields or converting NumPy arrays to tensors. Numeric dtypes, scalar
  arrays, frame order and unsigned rig indices survive transport.
- `BOUNDING_BOX` transports single, flat or per-frame pixel boxes; `comfy.ARRAY`
  transports flat or per-frame pixel boxes and normalized prompt elements.
  Both accept JSON only, retaining frame order, integer/float coordinates,
  negative origins, zero dimensions, labels, scores and declared metadata.
  Normalized elements retain thousandths in ymin/xmin/ymax/xmax order, text
  and palettes. Unknown fields, embedded references, nonfinite coordinates
  and pickle payloads are refused before reconstruction. Global `ARRAY`
  remains unrestricted by the placement contract. Layer placement retains
  first-frame selection, source-size zero fallback and canvas-based conversion.
  `CropByBBoxes` broadcasts a single box or flat box list across image frames;
  per-frame lists retain last-frame repetition and per-frame union cropping.
  Box and placement payloads have no external resource fields; declared
  auxiliary asset references remain retained across worker boundaries.
  Conflicting sizes for one identity refuse reference accounting. Cleanup
  remains disabled.
- Native `PadImageNode` pads image batches by explicit dimensions, a multiplier,
  or a size multiple, using solid-color feathering or repeated mirrored edges.
- Native `ImageUpscaleWithModel` runs tiled single-image model upscaling with
  smaller-tile memory retries and separately resized alpha channels.
- Native `ImageHistogram` reports first-frame RGB and luminance histograms,
  retaining independent red, green, and blue bin counts.
- Native `TextOverlay` draws wrapped, aligned text on image batches with optional
  outlines and straight-alpha compositing for transparent images.
- Native `DecimateMesh` reduces face counts using midpoint or QEM placement,
  with optional line and feature-edge quadrics and retained vertex colors.
- Native `RemeshMesh` re-extracts meshes using unsigned, oriented or graph-cut
  solid distance fields, with optional sharp-feature placement, component
  filtering, smoothing and retained vertex colors and per-item batch counts.
- Native `Get3DComponents` imports GLB, glTF, OBJ and STL into editable meshes,
  merging transformed scene geometry and retaining the first material's textures
  and factors. External glTF data must be included in the FILE_3D resource map.
- `FILE_3D` and its format-specific types preserve a null file value when a loader
  has no selected model. Null files retain their declared type and have no assets;
  downstream consumers still execute rather than receiving engine absence.
- Native `FillHoles` fills small boundary components under perimeter and vertex
  limits, with optional pre-welding and open-chain filling.
- Native `ApplyTextureToMesh` attaches base color and optional ORM/normal maps,
  fitting out-of-range UVs uniformly and retaining the normal-map tangent basis.
- Native `RenderUVAtlas` renders the first mesh's UV charts with deterministic
  colors and black boundary outlines.
- Native `UnwrapMesh` generates UV atlases with PEC or adaptive charts, retaining
  seam vertex colors, source textures and variable batch lengths.
- Native `MeshSmoothNormals` computes area-weighted smooth normals, welding
  coincident positions or splitting hard creases while retaining mesh attributes.
- Native `MeshTextureToImage` extracts baked base color, metallic, roughness,
  occlusion and normal maps as images, with neutral occlusion and normal fallbacks.
- Native `WeldVertices` merges grid-coincident mesh vertices, averaging positions
  and colors while retaining faces and remapping their vertex indices.
- Native `RotateMesh` rotates mesh vertices, normals and tangents using Euler
  angles or a normalized quaternion while retaining material attributes.
  `GetMeshInfo` reports padded-batch counts and available mesh attributes.
- Native `MergeMeshes` concatenates the first batch item of each mesh, excluding
  padding and offsetting face indices. Missing UVs become zero, missing color
  channels become white, and only the first available texture is retained.
- Native `PaintMesh` colors mesh vertices from their nearest sparse voxel,
  converting clamped RGB colors to linear space and respecting sparse batch IDs.
- Portable mesh values retain all fifteen geometry and material fields; voxel
  values retain dense or sparse data, optional colors and resolution.
- Portable shape-subdivision values retain ordered sparse stages, feature and
  coordinate tensors, explicit shapes and exact rational scales, without caches.
- Native `VaeDecodeStructureTrellis2` decodes batched structure latents to dense
  occupancy voxels at 32 or 64 resolution for the Trellis shape stage.
- Native `VaeDecodeShapeTrellis` decodes sparse shape latents into Y-up meshes
  and portable subdivision stages, retaining per-item geometry lengths.
- Native `VAEDecodeHunyuan3D` decodes latents to voxel grids; `VoxelToMesh`
  extracts basic or surface-net meshes, retaining variable per-item lengths
  in padded batches. The deprecated `VoxelToMeshBasic` remains available.
- Native `Trellis2ShapeStage` converts dense structure occupancy to batch-grouped
  sparse coordinates and an empty shape latent, with stage-specific conditioning.
- Native `Trellis2TextureStage` prepares texture conditioning and sparse noise
  targets from the batch-grouped shape latents emitted by native Trellis stages.
  Nonstandard custom latents with interleaved coordinate rows retain the source
  limitation: padded features flatten in batch order without reordering coordinates.
- Native `VaeDecodeTextureTrellis` decodes sparse texture latents with portable
  shape subdivisions, retaining all six material channels and batch IDs in
  sparse voxels, with resolution-aware Z-up to Y-up coordinate conversion.
- Native `Trellis2Conditioning` encodes per-image vision tokens at 512 and 1024
  resolution, with independent zero negative conditioning at both sizes.
- Native `UpscaleModelLoader` loads granted single-image upscale checkpoints
  into the shared `UPSCALE_MODEL` value.
- Native `FrameInterpolationModelLoader` loads granted FILM and five-block
  RIFE checkpoints into the shared resident `INTERP_MODEL` value.
  Native `FrameInterpolate` inserts intermediate frames using that model,
  retaining source frame order and dimensions.
- Native `CurveEditor` passes canonical curves through the curve editor and
  displays an optional histogram without restricting its bin count.
- Native `AddLayer` appends image and mask batches to a layer document with
  placement, rotation, opacity, blend mode, and stacking metadata.
- Native `CameraAngle` produces an orbit-camera dictionary and shot description
  from horizontal angle, elevation, and zoom.
- Native `Pixal3DConditioning` and `Pixal3DMultiViewConditioning` combine DINO
  vision tokens, optional NAF features, and per-view camera projection features
  for single images or a fixed front/left/back/right orbit rig.
- Native `CreateCameraInfo` builds cameras from orbit angles, a look-at position,
  or an explicit quaternion, with roll, field of view, zoom, and projection.
- Native `RenderSplat` renders gaussian splats as images and coverage masks,
  with camera-controlled turntables, background plates, and color, clay,
  depth, or normal shading.
- Native `MoGeGeometryToFOV` derives horizontal, vertical, or diagonal field of
  view and focal length in pixels from MoGe geometry intrinsics.
- Native `TripoSplatPreprocessImage` erodes alpha, crops foreground to a padded
  square, resizes with Lanczos, and composites the image over black.
- Native `TripoSplatSamplingPreview` patches a cloned model to report decoded
  gaussian splat previews during sampling without replacing the sampler callback.
- Native `VAEDecodeTripoSplat` decodes shape latents to portable gaussian splats
  with density selection and seeded point sampling.
- Native `TripoSplatConditioning` combines normalized vision tokens and VAE
  reference latents with zero negative conditioning and fixed shape/camera noise targets.
- Native `MoGeRender` visualizes geometry as normalized inverse depth, Turbo-colored
  depth, OpenGL or DirectX normals, or an RGB mask.
- Native `Trellis2UpsampleStage` cascades sparse shape latents through a VAE
  and prepares second-pass conditioning, projection features, and padded sparse latents.
- Native `MoGePanoramaInference` estimates spherical geometry for a single
  equirectangular image by merging perspective-view depth predictions.
- Native `MoGeInference` predicts optional depth, points, camera intrinsics,
  masks, and normals from image batches with configurable chunking and camera hints.
- Native `LoadDA3Model` loads one bound Depth Anything 3 model file with optional
  weight dtype selection. `DA3Inference` predicts monocular or multi-view depth,
  optional confidence and sky, and multi-view cameras at the source image size.
- Native `DA3Render` visualizes raw or normalized depth, confidence and sky.
  `DA3GeometryToMesh` unprojects one frame with confidence/sky filtering,
  stride and discontinuity selection, UVs and optional source-image texture.
- Native `MediaPipeFaceMeshVisualize` draws selected face connections, points
  or a filled outline on an optional image. `MediaPipeFaceMask` unions selected
  face regions into one binary mask per frame, including empty frames.
- DA3, MediaPipe face-detection and SAM3D Body model values retain their
  complete model wrappers in the holding worker when passed between nodes.
- Native `SAM3DBody_Smooth` reduces temporal pose jitter with Gaussian or
  Savitzky-Golay filtering, wrapped-angle handling and fast-spin geometry
  protection while camera motion remains smoothed.
- Structural tensor values preserve unsigned 16-bit joint indices and unsigned
  32-bit mesh faces without narrowing their values to a signed integer range.
- Native `MoGePointMapToMesh` triangulates one geometry sample with UVs and
  optional source-image texture, converting perspective or spherical axes for mesh output.
- Color strings, ordered color palettes, bounding boxes, and optional
  resolution-preview metadata pass between node packs without rewriting their values.
- Native `PrimitiveBoundingBox` constructs pixel bounding boxes; `ImageCropV2`
  crops each image in a batch to that box and reports a PNG preview.
- Native `ImageCropToMask` crops images around foreground masks with adjustable
  margin, mask growth, background color and output dimensions.
- Native `CropByBBoxes` crops each frame to its bounding-box union and resizes
  with stretching or aspect-preserving black padding.
- Native `SDPoseFaceBBoxes` converts canonical pose face keypoints into per-frame
  pixel boxes with area scaling and optional square expansion for cropping.
- Native `SDPoseDrawKeypoints` renders body, head, hand, foot and face keypoints
  as pose images with confidence thresholds and adjustable marker widths.
- Native `SDPoseKeypointExtractor` extracts OpenPose-format body, foot, face and
  hand keypoints from SDPose models, with batched images or per-person crop boxes.
- Native `RTDETR_detect` returns per-frame pixel bounding boxes filtered by
  confidence and COCO class, ordered by score and limited per frame.
  `DrawBBoxes` overlays boxes and labels on images or a black canvas.
- Native `LoadBackgroundRemovalModel` loads an admitted background-removal
  checkpoint; `RemoveBackground` generates its foreground mask for an image batch.
- Native `SAM3_VideoTrack` tracks initial masks or text-prompted objects across
  video frames. `SAM3_TrackToMask` selects tracked objects and resizes their union
  to the original frame dimensions using the portable packed-track value.
- Native `SAM3_TrackPreview` publishes an MP4 preview of packed tracks with
  colored overlays, object numbers and confidence labels, optionally over images.
  Its H264 `yuv420p` encoding requires even width and height; odd dimensions
  retain the [reference encoding failure](../comfyui-issues/sam3-preview-odd-dimensions.md).
- Native `SAM3_Detect` segments text, box or point prompts with optional mask
  refinement, returning per-object or union masks and scored text detections.
- Native `ColorToRGBInt` preserves RGB letter casing and separates optional alpha;
  `ResolutionSelector` rounds aspect-ratio and megapixel dimensions to a chosen multiple.
- Marigold V2 decoded predictions convert to per-image normalized depth,
  unit surface normals, or sRGB albedo.
- Native list construction preserves member order and nested values across
  independently typed elements; indexed selection supports positive and negative
  indices and retains the selected element's type.
- Native image tile lists split batches with overlap and merge tiles using
  feathered weights; mask previews retain their mask output, and arbitrary
  values can be inspected as text.
- Native `Canny` edge detection with low/high hysteresis thresholds and RGB output
- Native Stable Cascade stage C/B latent creation, stage C VAE encoding,
  stage B prior conditioning, and super-resolution control inputs
- Typed `dinkster.region` rectangle values for shared crop and placement geometry
- Typed per-frame `dinkster.pose` keypoint and skeleton values, including
  OpenPose/controlnet_aux JSON import and export and human/AP10K rendering;
  received pose values reject malformed data before exposure and retain
  declared source dependencies
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
  ImageDocument storage distinct from structural tensor `LAYERS`, and
  placement from image batches or regions/detections, digest-bound compositor
  edit commands, 26 blend modes plus dissolve, perceptual or linear-light
  rendering, per-layer previews, flatten/split, structural editing nodes, and
  safe native-placement fallback for stale edits; CPU reference rendering is
  bounded to 4,194,304 canvas pixels and 67,108,864 layer/mask work-pixels
- Received image documents validate every non-inline resource against declared
  source references without reading or publishing raster files; auxiliary
  references are retained. Legacy layer stacks validate without publishing,
  but their migrated resource lifetimes do not qualify for cleanup.
- Isolated document retrieval retains newly derived external rasters in the
  serving host vault before returning, without changing the original layer
  value or treating caller-owned inputs as newly produced assets. Missing,
  truncated, or corrupt external resources refuse retrieval.
- Native document saves retain every external resource through their catalog
  dependency manifest, including unused resources. Isolated saves transfer
  newly derived resources into the host vault before producer acknowledgment,
  without changing the saved file's identity or copying caller-owned inputs.
  Missing, corrupt, or
  size-mismatched resources refuse catalog acknowledgment and remain recoverable
  when the declared bytes become available; retention does not grant media access.
- Isolated GLSL ES 3.00 image rendering through ANGLE with up to five image
  inputs, typed scalar and curve uniforms, four render targets, custom output
  dimensions, and 1-32 passes; Linux and Windows x86-64/ARM64 and macOS ARM64
  are supported, while unavailable platforms refuse explicitly
- Pointwise invert, normalize, brightness, and contrast adjustment; Gaussian
  blur, sharpen, quantize, seeded noise, and bounded image morphology
- ComfyUI SDR-to-HDR and image color-space conversion with Rec.2020 HLG/PQ,
  Rec.709 linear, ACEScg/ACEScct, and LogC3 transforms; straight alpha is
  preserved and results use the inference intermediate device and dtype
- RGB and YCbCr channel split/merge plus explicit alpha extraction and joining
- Explicit coverage/transparency mask polarity, with migration of saved polarity
  controls; mask polarity inversion and RGBA premultiply/unpremultiply nodes
- ComfyUI BatchImagesNode, ResizeAndPadImage, and ImageCompare translations;
  paired comparison previews accept either, both, or neither image
- ComfyUI ResizeImageMaskNode, BatchImagesNode, BatchMasksNode, BatchLatentsNode,
  and ColorTransfer native nodes
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
- ComfyUI image and image/video latent width, height, frame, and batch-size
  inspection with node-attributed dimension progress text
- ComfyUI rebatching of image and latent lists into target-size batches,
  retaining frame order and latent batch indices
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
- Region and detection transport rejects malformed numeric records, framed
  bytes and masks before exposing received values. Negative region origins,
  zero dimensions, optional full-frame soft masks and declared auxiliary
  asset dependencies remain supported across image and vision providers.
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
- SAM 3.1 prompt tokenization works offline with a digest-verified plain-text
  vocabulary that is readable during pack admission.
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
