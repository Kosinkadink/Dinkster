# Legacy pack census: what popular packs hack, and what Dinkster owes them natively

Date: 2026-07-19. Companion data: [legacy-survey.json](legacy-survey.json)
(dependency-provisioned sweep) and
[legacy-survey-bare.json](legacy-survey-bare.json) (bare-venv sweep); both
produced by `tools/legacy_survey.py`.

This is primarily a **hack census**, not a compatibility scoreboard: the goal
is to learn what the most-used ComfyUI custom node packs had to monkey-patch,
import, or reach into so Dinkster's extension API (DESIGN 3.6) makes each of
those needs a first-class, declared capability. Compatibility numbers are the
side benefit. Frontend-dependent behavior is expected to be rewritten for the
Dinkster frontend and is out of scope here.

## Method

Two passes over 17 packs (usage-ranked from the 2026-07 telemetry export:
70,399 users, 29,146 with at least one custom node; Windows 98.2%):

1. **Static census**: read-only grep/read over each checkout for server
   hooks, monkey-patches, core-internal imports, registry mutation, schema
   tricks, install/download behavior, filesystem assumptions,
   ModelPatcher/device internals, custom events, and frontend surface.
2. **Dynamic sweep**: `tools/legacy_survey.py` runs the DESIGN 3.8 legacy
   quarantine loader once per pack, one fresh child process each (a pack that
   poisons its interpreter must not contaminate the next pack's report),
   against the real ComfyUI install at `/home/kosin/ComfyUI` with **no extra
   dependencies installed** - missing-dependency classification is itself
   census data.

## Dynamic sweep results

| Pack (usage where known) | Status | Nodes | Skipped | Routes |
|---|---|---:|---:|---:|
| ComfyUI-KJNodes (18,400 / 26.1%) | loaded | 229 | 3 | 0 |
| rgthree-comfy (18,136 / 25.8%) | loaded | 24 | 0 | 22 |
| ComfyUI-Easy-Use (15,904 / 22.6%) | missing-dependency: cv2 | - | - | 15 registered pre-crash |
| ComfyUI-Impact-Pack (15,108 / 21.5%) | missing-dependency: cv2 | - | - | - |
| ComfyUI-VideoHelperSuite (13,762 / 19.5%) | missing-dependency: cv2 | - | - | - |
| ComfyUI-Custom-Scripts (13,325 / 18.9%) | loaded | 8 | 5 | 14 |
| ComfyUI-GGUF (13,249 / 18.8%) | missing-dependency: gguf | - | - | - |
| ComfyUI_essentials (12,213 / 17.3%) | loaded | 83 | 2 | 0 |
| comfyui_controlnet_aux (12,153 / 17.3%) | missing-dependency: cv2 | - | - | - |
| ComfyUI_LayerStyle | missing-dependency: cv2 | - | - | - |
| was-node-suite-comfyui | missing-dependency: numba | - | - | - |
| ComfyUI_Comfyroll_CustomNodes | loaded | 179 | 20 | 0 |
| cg-use-everywhere | v3-entrypoint (port natively) | - | - | - |
| ComfyUI-WanVideoWrapper | missing-dependency: accelerate | - | - | - |
| RES4LYF | missing-dependency: pywt | - | - | - |
| ComfyUI_UltimateSDUpscale | loaded | 4 | 0 | 0 |
| ComfyUI-Impact-Subpack | missing-dependency: cv2 | - | - | - |

Reading of the skip reasons across the 6 loaded packs (30 skips total):

- **28 of 30 skips are `INPUT_IS_LIST`/`OUTPUT_IS_LIST` nodes** (Comfyroll's
  list suite, Custom-Scripts' Show Text/Repeater, essentials' list<->batch
  converters, KJNodes' ImageTransform). This is direct evidence for DESIGN
  3.13: once native `list<T>` plus the batch/list converters exist, almost
  every remaining skip translates honestly.
- The other 2 were v1-tolerated sloppiness the translator now handles
  (duplicate/wrong-arity `RETURN_NAMES`, duplicate input ids - see
  "translator fixes" below) and one pack/ComfyUI version-skew crash inside
  the pack's own `define_schema` (correctly a per-node skip, not pack-fatal).
- `cv2` alone blocks 7 of 17 packs. For compat-worker environments,
  `opencv-python-headless` is effectively part of the de facto baseline;
  the manifest-first dependency story (DESIGN 3.6) must make such shared
  heavy deps declared and provisioned, not assumed.

## Dependency-provisioned rerun

The bare-venv sweep above classifies honestly but under-reports: most packs
never got past their first missing import. A second sweep provisioned the
missing deps through a reusable **PYTHONPATH overlay**
(`~/.cache/dinkster/survey-deps`, built with the ComfyUI venv's own pip and
`pip install --target ... --no-deps` plus explicitly-listed transitive
deps; a full resolve drags in a second torch/numpy that shadows the venv's
and breaks everything). `tools/legacy_survey.py --overlay` prepends it to
the child PYTHONPATH; the user's venv is never modified.

Overlay contents (3 iterative rounds, following each pack's dep chain):
opencv-python-headless, gguf, accelerate, PyWavelets, numba, color-matcher,
matplotlib, mss, scikit-image, ultralytics, ftfy, piexif, dill,
blend_modes, segment_anything, plus their missing transitive deps.

Result: **15 of 17 packs load; 1,665 nodes translate.**

| Pack | Status | Nodes | Skipped | Routes |
|---|---|---:|---:|---:|
| RES4LYF | loaded | 298 | 2 | 2 |
| ComfyUI-KJNodes | loaded | 230 | 2 | 0 |
| was-node-suite-comfyui | loaded | 218 | 2 | 0 |
| ComfyUI_Comfyroll_CustomNodes | loaded | 180 | 19 | 0 |
| ComfyUI-Easy-Use | loaded | 174 | 33 | 20 |
| ComfyUI-Impact-Pack | loaded | 171 | 26 | 14 |
| ComfyUI_LayerStyle | loaded | 165 | 4 | 0 |
| ComfyUI_essentials | loaded | 83 | 2 | 0 |
| comfyui_controlnet_aux | loaded | 64 | 0 | 0 |
| ComfyUI-VideoHelperSuite | loaded | 39 | 1 | 6 |
| rgthree-comfy | loaded | 24 | 0 | 22 |
| ComfyUI-Custom-Scripts | loaded | 8 | 5 | 14 |
| ComfyUI-GGUF | loaded | 6 | 0 | 0 |
| ComfyUI_UltimateSDUpscale | loaded | 4 | 0 | 0 |
| ComfyUI-Impact-Subpack | loaded | 1 | 0 | 0 |
| cg-use-everywhere | v3-entrypoint (port natively) | - | - | - |
| ComfyUI-WanVideoWrapper | import-error: `cached_download` removed from huggingface_hub (env version skew, honest diagnostic) | - | - | - |

The 96 remaining skips, fully accounted for:

- **84 list-convention nodes** (`OUTPUT_IS_LIST` 59, `INPUT_IS_LIST` 25) -
  the DESIGN 3.13 / M6 acceptance set.
- **10 Easy-Use `KeyError` skips** (`llm`, `instantid`, `pulid`, ...):
  their `INPUT_TYPES()` reads folder categories that the pack's
  `prestartup_script.py` registers. The quarantine deliberately does not
  run prestartup scripts; census finding - prestartup-registered folder
  categories are load-order coupling that Dinkster's manifest-declared asset
  kinds (3.12) eliminate.
- **2 genuine pack bugs/skew** (Easy-Use `saveTextLazy` FUNCTION naming,
  KJNodes Ideogram vs its ComfyUI's `io.BoundingBox`).

This rerun also exposed and fixed a third translator containment bug: the
ecosystem's **wildcard proxy hack** (`AnyType`/`AlwaysEqualProxy`, a str
subclass whose `__eq__` always matches and which defines no `__hash__`)
crashed set/dict lookups in the translator and cost 31 nodes across
LayerStyle and Easy-Use. The translator now collapses str subclasses to
plain strings; `"*"` translates as an honest wildcard TypeExpr. This hack
is itself census evidence for category #15: v1 has no declared wildcard,
so packs forge one with operator overloading.

## Hack census: category -> Dinkster disposition

Categories are ordered by how many top packs need them, weighted by usage.

| # | What packs actually do today | Packs | Dinkster disposition |
|---|---|---|---|
| 1 | Reach into ModelPatcher/CLIP/VAE internals: `add_object_patch`, `model_options["transformer_options"]`, block/attention replacement, sampler callback wrapping | KJ, Easy, Ess, Wan, RES4LYF, USDU, Impact | **Planned, must be designed**: scoped, reversible model-interposition API on the compat model surface (3.10 residency + 3.6 registries). Highest-value gap; RES4LYF alone has 252 strong hits. |
| 2 | Register HTTP routes on `PromptServer.instance` for settings, file serving, model metadata, interactive tools (SAM), video streaming | rgthree, Easy, Impact, Custom-Scripts, RES4LYF, VHS | **Covered by design**: namespaced route registry (3.6). Quarantine already counts-not-serves these (22 routes in rgthree alone). |
| 3 | Send ad hoc websocket events (`impact-preview`, `easyuse-toast`, `kj_preview_override`, `VHS_latentpreview`, binary `PREVIEW_IMAGE`) | KJ, Easy, Impact, Wan, VHS, Custom-Scripts, CR, RES4LYF | **Covered by design, needs schema**: typed event registry (3.5/3.6) with declared event schemas + binary channel; preview/progress is the dominant use and belongs to the progress service (3.9). |
| 4 | Mutate `folder_paths.folder_names_and_paths` / `add_model_folder_path` to invent model categories (`gguf`, `nlf`, `kjnodes_fonts`, `VHS_video_formats`, LUTs) | GGUF, KJ, Easy, Impact, Subpack, Ess, Wan, VHS | **Covered by design**: assets system (3.12) - declared asset kinds/folders in the pack manifest; registry conflicts are diagnostics. |
| 5 | Direct `unload_all_models()`, `soft_empty_cache()`, `.to(device)` on shared models, interrupt processing | LayerStyle, Wan, Easy, Impact, Aux, WAS, CR | **Covered by design**: memory governor (3.10) owns eviction/placement; packs request pressure relief via API, never issue global unloads. |
| 6 | Runtime `pip install` / import-time installs / HF downloads into models dirs | Easy, Impact, CR, KJ, LayerStyle, Wan, WAS | **Covered by design**: manifest-declared deps + per-pack venvs (3.6); model fetch goes through asset resolvers with digests (3.12). No import-time side effects, period. |
| 7 | `INPUT_IS_LIST`/`OUTPUT_IS_LIST` list processing | CR, Custom-Scripts, Ess, Impact, Easy, KJ, LayerStyle, WAS | **Covered by design**: native `list<T>` + combinators + regions (3.13, planned M6). |
| 8 | Hidden inputs (PROMPT, EXTRA_PNGINFO, UNIQUE_ID, DYNPROMPT) for workflow metadata, node identity, queue control | Easy(167), Impact(43), rgthree(28), Wan(27), KJ(16), VHS(11) | **Covered by design**: typed execution-context object in the native schema instead of stringly hidden slots; embedding workflow-in-image is an export concern, not an execution input. |
| 9 | Monkey-patch core functions (`comfy.samplers.sample` swap, `setattr(comfy.ops, ...)`, `torch.load` wrap) | Easy (brushnet/kolors), KJ, Subpack | **Intentional non-support**: isolation makes host patching impossible. Each instance maps to a needed API: sampler interposition (#1), safe weight loading (asset/codec layer). |
| 10 | Prestartup scripts, import-time threads, atexit, `sys.path` mutation, cross-pack probing (KJ probing ComfyUI-GGUF, Impact installing CLIPSeg) | rgthree, Easy, Impact, WAS, LayerStyle, CR, Custom-Scripts | **Intentional non-support + replacement**: manifest lifecycle phases and declared optional capabilities/inter-pack deps (3.6). Cross-pack probing becomes a registry query. |
| 11 | Direct queue manipulation (`PromptServer.instance.prompt_queue`, `.number`, stop-iteration, node muting) | Impact, VHS, Easy | **Planned, needs API**: queue/run-control endpoints in the native protocol (3.5); Impact's feedback loops overlap with regions (3.13 While). |
| 12 | Custom disk caches under `custom_nodes/` (Wan's text_embed_cache) and pack-local config files | Wan, rgthree, Custom-Scripts | **Covered by design**: per-pack namespaced storage dirs (config/cache/artifacts) provided by the host; cacheable results belong in the tiered cache (3.4). |
| 13 | `IS_CHANGED` (WAS 22, VHS 9, KJ 9) and `VALIDATE_INPUTS` (VHS 10) | WAS, VHS, KJ, Custom-Scripts, Impact | **Covered by design**: idempotence declaration + asset-identity fingerprints (3.12) replace `IS_CHANGED` hashing; typed schema validation replaces most `VALIDATE_INPUTS`. |
| 14 | GraphBuilder node expansion | Easy (for-loops), controlnet_aux (AIO preprocessor) | **Covered by design**: regions are the engine primitive (3.13); AIO-style type-dispatch expansion becomes a schema-level dynamic-output case. |
| 15 | `**kwargs` catch-all node signatures (Easy 180, RES4LYF 151, Wan 107, KJ 76) | most big packs | **Covered by design**: dynamic inputs are declared in the native schema, not smuggled through kwargs; compat translator already passes v1 kwargs through. |

The static census (17 packs, full category matrix with file:line evidence)
is summarized above; the raw matrix lives in the survey conversation and can
be regenerated - the pack checkouts are disposable (`/tmp/dinkster-legacy-packs`).

## Translator fixes that came out of the sweep

The sweep's two hard crashes were containment bugs in our translator, both
fixed in this commit:

- One bad node killed its whole pack: `translate_mappings` only caught
  `CompatError`, but pack `INPUT_TYPES()`/`define_schema` runs arbitrary
  code and can raise anything (KJNodes: `AttributeError` from
  ComfyUI-version skew). Sweep mode now records any exception as a per-node
  skip; explicit `only=` requests still raise.
- v1-tolerated schema sloppiness was pack-fatal: duplicate input ids across
  required/optional (Comfyroll) now collapse first-declaration-wins, and
  duplicate/wrong-arity `RETURN_NAMES` (Comfyroll, KJNodes) now derive
  tolerant positional output ids - in v1 these are display labels over
  positional outputs, so packs never notice.

Result: 0 crashes across all 17 packs; every pack yields a classified
report.

## Follow-ups (priority order)

1. ~~Model-interposition API design (#1)~~ - designed as DESIGN 3.14 (patch
   programs as values); RES4LYF and WanVideoWrapper remain the validation
   stress tests when it is implemented. Feeds M7 API freeze.
2. Typed event/preview schema (#3) - fold into the progress service work;
   KJ's AIMDO-style memory visualization is the flagship consumer.
3. Queue/run-control endpoints (#11) - small, unblocks Impact-style UX.
4. ~~Dependency-provisioned compat workers~~ - done (see the rerun above);
   the overlay at `~/.cache/dinkster/survey-deps` is reusable.
5. When native `list<T>` lands (M6), re-translate the 84 list-node skips as
   the acceptance test.
