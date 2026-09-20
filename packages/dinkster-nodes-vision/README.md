# dinkster-nodes-vision

`dinkster-nodes-vision` contains the first-party model-backed vision providers.
Each model remains an independently composed pack while sharing one Python
distribution, dependency set, and test directory.

The model modules are `birefnet`, `depth_anything_v2`, `depth_anything_v3`,
`detr`, `efficient_sam`, `hed`, `rtdetr`, `sam31`, and `upscale` under the
`dinkster_nodes_vision` namespace.

## Providers

- BiRefNet implements `dinkster.image.matte` with the declared general-use
  checkpoint. It evaluates frames independently at 1024x1024 and resizes the
  resulting soft foreground mattes to their source dimensions.
- Depth Anything V2 Large and Depth Anything 3 Mono Large implement
  `dinkster.preprocess.model_depth`. Both produce normalized relative-depth
  control hints; their manifests retain the model-specific licenses and exact
  checkpoint provenance.
- DETR ResNet-50 and RT-DETR v4 x-HGNet implement
  `dinkster.detection.detect` for the 80 COCO classes. Prompts filter class
  names, results remain score ordered per frame, and result limits preserve
  both count and imported slice-stop behavior.
- EfficientSAM-Ti implements `dinkster.detection.segment` on CPU. It accepts
  one image frame, uses each detection region as a box prompt, and preserves
  detection metadata while attaching source-sized soft masks.
- HED implements the learned ControlNet line and edge preprocessors on CPU or
  CUDA, including line-art variants, AnyLine, HED, TEED, and M-LSD. Its cache
  participates in host pressure handling, and M-LSD retries on CPU after CUDA
  allocation failure. The bundled weight licenses remain in the HED pack
  sidecar.
- SAM 3.1 implements text detection, box and text segmentation, and tracking.
  Its combined checkpoint, tokenizer vocabulary, SAM license, and CLIP license
  remain declared and bundled with the SAM 3.1 pack sidecar.
- Upscale implements `dinkster.image.upscale_model` for ESRGAN-family and
  RealESRGAN Compact checkpoints. Architecture detection and state-dict loading
  fail closed, while tiled execution uses deterministic overlap feathering and
  no out-of-memory tile-size fallback.

Execution reads model files only through declared assets and never downloads
weights. The logical pack manifests record immutable model sources, digests,
licenses, and runtime requirements. The provider tests retain their pinned
reference commits and exact parity vectors.
