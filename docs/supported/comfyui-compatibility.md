## ComfyUI compatibility

- Opt-in legacy on-ramp: `dinkster-serve --comfy-root ... --legacy-pack PATH`
  runs unmodified v1 custom-node packs in quarantined isolated workers,
  namespaced `comfy.<pack>.<node>`
- v1 `NODE_CLASS_MAPPINGS` translation using the selected ComfyUI install
  and interpreter
- `comfy.IMAGE`/`dinkster.image` and `comfy.MASK`/`dinkster.mask` each share one
  image-array byte contract and explicit symmetric type equivalence across
  the compat boundary; native schemas and worker invocations retain native
  type ids, and the torch-free engine renders PNG previews
- Invocation-scoped media staging for Comfy source-filename inputs on POSIX;
  custom nodes requiring source filenames are not supported on Windows
- Pinned core video nodes `LoadVideo`, `Video Slice`, `CreateVideo`,
  `GetVideoComponents`, `SaveVideo`, and `SaveWEBM`, including guarded saved
  output import as execution artifacts
- Official MiniMax H3 image-to-video and reference-to-video source nodes, plus
  Resolution Selector, lower to native nodes from workflows pinned to ComfyUI
  `b5cc8830279eae909a59de030af1e50761c36751`.
- Native aliases for those video nodes plus `VideoTrim` and `VideoCrop`,
  including flat and nested saver selections, numeric depth choices, and
  lossless VIDEO_EDIT widget values. The pinned template corpus replays all
  311 SaveVideo, 167 CreateVideo, 105 GetVideoComponents, and 12 Video Slice
  nodes across 24 distinct widget shapes.
- Worker-created sources declared by `meta.asset_refs`, including nested
  list values, survive producer shutdown. Local workers publish in writable
  scratch; remote workers transfer encoded source blobs through persistent
  CAS. The engine verifies and adopts sources into `DINKSTER_ASSET_VAULT` before
  acknowledgement, retaining transfer pins across resumable disconnects.
- Pinned core vision groups at revision `c67885b1`: RT-DETR fp16 loading and
  detection, BiRefNet background removal, literal-text SAM 3.1 detection, and
  initial-mask SAM 3.1 video tracking with all-object mask output
- Impact Pack SEGS/SAM chains and `comfyui_controlnet_aux`'s stochastic
  `SAMPreprocessor` remain namespaced legacy nodes instead of native group
  translations because their value and mask contracts are not equivalent
- `UpscaleModelLoader` with `ImageUpscaleWithModel`, `UltimateSDUpscale`, or
  `UltimateSDUpscaleNoUpscale` lowers to native model upscaling and sequential
  tiled image-to-image refinement. Redraw supports Linear and Chess ordering;
  seam repair supports Band Pass, Half Tile, and Half Tile + Intersections.
  The upstream `seam_fix_denoise` field is accepted on import, but ordinary
  `denoise` drives both refinement passes. Execution requires batch size 1 and
  non-tiled VAE decode.
- SD1.5 `ControlNetLoader` with `ControlNetApplyAdvanced` or the deprecated
  `ControlNetApply` executes through the native arm. Ordered chains preserve
  each control's hint, strength, and start/end window. The optional VAE input
  is not supported, and other checkpoint families are refused.
- Not supported: V3-only (`comfy_entrypoint`) packs, pack HTTP routes/web
  assets, executor hooks - these are diagnosed, not emulated
- Compat-translated node schemas declare `mayExpandGraph` on the schema wire
  where expansion is provable at translation time (V3 `enable_expand`, v1
  functions that literally return an "expand" dict, and the enumerated core
  expander list pinned to a ComfyUI revision), so importers can refuse
  unsupported runtime graph expansion before import
  (`import.loop.runtimeExpansionUnsupported` on a Generic Loop containing a
  flagged node). Dynamically built or delegated v1 returns are unclassifiable
  and stay unflagged; executing one that expands still refuses loudly. See
  the conversion-limits table in `docs/compat-porting-recipes.md`.
