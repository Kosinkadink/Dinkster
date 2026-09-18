# ComfyUI Custom-Node Ecosystem Survey - Aggregate Results

Date: 2026-07-28. Scope: top-90% cumulative local usage (829 packs; 747 resolved+scanned,
82 unresolved, 12 dead repos). Usage weights from 2-week local install data
(584,916 user-installs total; roster covers 526,467).

## Coverage and data quality

- 747/747 resolved packs scanned exactly once; all 25 batch JSONL files validate.
- 12 clone failures (dead/disabled repos) = 3,693 users = 0.70% of roster usage.
  Largest: comfyui-reactor-node (1,779u, disabled by GitHub staff).
- Known scan caveats:
  - frontend_features corruption in batch-16 + 15 scattered rows was repaired
    (see results/REPAIR-LOG.md); 4 flagged anomalies manually verified genuine.
  - conditioning_manipulation and runtime_pip over-detect in packs that vendor
    model code (e.g. 3d-pack, sam3, bagel); treat both as upper bounds.
  - node_count_approx is heuristic; dynamic registration undercounts
    (impact-subpack, controlnetaux) and dict-parsing overcounts (xb_toolbox,
    djz-nodes) are flagged in batch reports.
  - "comfyui" rank 638 is ComfyUI core itself appearing in usage data; excluded
    from pack-level conclusions.

## Schema format spread (V1 vs V3)

| format | packs | pack % | usage % |
|--------|-------|--------|---------|
| v1     | 600   | 81.6%  | 75.9%   |
| mixed  | 102   | 13.9%  | 18.6%   |
| v3     | 33    | 4.5%   | 5.4%    |

V1 remains overwhelmingly dominant. Pure-V3 packs skew newer/smaller, but several
major packs are mixed (kjnodes, easy-use, advanced-controlnet is v3-classified).

## Feature prevalence (usage-weighted, descending)

| feature                    | packs | pack % | usage % |
|----------------------------|-------|--------|---------|
| web_directory              | 364   | 49.5%  | 56.2%   |
| conditioning_manipulation* | 300   | 40.8%  | 55.2%   |
| model_patcher_use          | 203   | 27.6%  | 39.2%   |
| server_routes              | 173   | 23.5%  | 33.6%   |
| sampler_registration       | 118   | 16.1%  | 32.3%   |
| ws_messages                | 103   | 14.0%  | 32.2%   |
| model_folder_registration  | 145   | 19.7%  | 29.8%   |
| transformer_options        | 115   | 15.6%  | 23.5%   |
| comfy_ops_override         | 61    | 8.3%   | 21.1%   |
| sampling_wrappers          | 85    | 11.6%  | 18.3%   |
| custom_model_arch          | 91    | 12.4%  | 16.3%   |
| custom_noise               | 63    | 8.6%   | 15.8%   |
| attention_patch            | 55    | 7.5%   | 15.5%   |
| monkeypatch_core           | 40    | 5.4%   | 9.3%    |
| runtime_pip*               | 126   | 17.1%  | 6.3%    |
| scheduler_registration     | 27    | 3.7%   | 5.3%    |
| hooks_api                  | 9     | 1.2%   | 1.8%    |
| custom_latent_format       | 6     | 0.8%   | 0.6%    |

(* = upper bound, see caveats)

Frontend: 341 packs (46.4%, 54.9% usage) ship JS extensions. Spread:
registerExtension 46.0% of packs, beforeRegisterNodeDef 37.0%, api client 26.9%,
canvas 9.5%, menu 9.5%, settings 8.7%, getCustomWidgets 3.9%.

19 packs are extension-only (zero nodes; frontend and/or server routes only),
e.g. crystools-monitoronly, autocomplete-plus, workspace-manager, agent-panel,
quick-connections, workflow-models-downloader.

## Key readings

1. Half the ecosystem's usage involves frontend extension code. A Dinkster extension
   story that is backend-only misses ~55% of what packs actually do.
2. Sampler/scheduler/noise/guider surface is the single biggest backend seam:
   sampler_registration alone is 32% usage-weighted; sampling_wrappers +
   custom_noise + attention_patch + transformer_options overlap heavily with it.
3. ModelPatcher is the de-facto extension currency (39% usage): packs speak
   "patches on a model clone" rather than editing model code.
4. comfy.ops override (21% usage) is how quantization/casting packs (GGUF etc.)
   interpose on weight layers without touching model defs.
5. ComfyUI's official hooks API has near-zero adoption (9 packs, 1.8% usage)
   despite being designed for exactly these use cases - the ecosystem votes with
   monkeypatches and transformer_options instead. Lesson: seams must be there
   from day one and be the easiest path, or packs will patch around the host.
6. monkeypatch_core at 9.3% usage is the "missing seam" indicator: what packs
   patch is what the host failed to expose.
7. Server routes + websocket events (33%/32% usage) are mainstream, not niche:
   packs assume they can add HTTP endpoints and push events to their frontend.
8. Custom model architectures (16% usage) arrive via wrapper packs
   (wanvideowrapper, ltxvideo, cogvideox) long before core support exists.

## Deepest core-integration packs (core-touchpoint count)

res4lyf 11; comfyui-dazzle-ksampler 11; comfyui-easy-use 10;
comfyui-advanced-controlnet 10; comfyui-animatediff-evolved 10;
comfyui-umeairt-toolkit 10; xb_toolbox 10; tbg-etur 10; comfyui-ltxvideo 9;
comfyui-ppm 9; comfyui-nag 9; comfyui-apt_preset 9; efficiency-nodes 8;
comfyui-kjnodes 8 (rank 1 by usage).

Top-15 usage packs by core touch: kjnodes 8, easy-use 10, impact-pack 3,
ultimatesdupscale 5, essentials 4, gguf 3 (ops override), rgthree 1 (frontend-
heavy), videohelpersuite/custom-scripts/layerstyle 0 (pure node/frontend packs).
