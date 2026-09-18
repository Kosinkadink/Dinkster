# Research index

Evidence base for the extension, frontend, and training architecture
programs. Extensive deep dives live in the private
[Kosinkadink/dinkster-research](https://github.com/Kosinkadink/dinkster-research)
repository; this directory holds the directly-referenceable conclusions
plus earlier standalone studies.

## In this directory

- `founding-design.md` - archived initial design and milestone document from
  commit `2038ad61`; the root `DESIGN.md` is the current architecture rationale.
- `inference-parity-conclusions.md` - accepted native inference parity scope,
  porting waves, shared dependency and adapter contracts, and benchmark gates.
- `latest-comfy-inference-parity-2026-08-06.md` - current-source capability
  matrix against pinned ComfyUI `2eb60976`, exact native family/proof levels,
  performance evidence boundary, ordered implementation roadmap, and the
  selected NVFP4 foundational tranche.
- `ecosystem-scan-conclusions.md` - aggregate results of the usage-weighted
  ComfyUI custom-node ecosystem survey (829 packs, top-90% cumulative local
  usage) that selected the deep-dive targets.
- `lora-masking-regional-audit.md` - pinned ComfyUI and custom-node behavior
  matrix for conditioning-scoped regional LoRA execution and adjacent mask
  mechanisms that require separate contracts.
- `legacy-pack-census.md`, `legacy-survey.json`, `legacy-survey-bare.json` -
  earlier legacy pack census.
- `native-checkpoint-ram-calibration.md` - native checkpoint RAM
  calibration study.
- `frontend-decision-index.md` - dated index of cross-repo design
  decisions whose documents of record live in Dinkster-Frontend/docs
  (references only, no duplication).
- `lazy-acceptall-conclusions.md` - conclusions of the 747-pack
  lazy + arbitrary-input usage census (full data in dinkster-research
  feature-usage/).
- `comfy-angle-glsl.md` - study of ComfyUI's GLSL shader node and its
  comfy-angle (ANGLE binary wheel) dependency: replicability verdict
  for Dinkster and the native-GL-dependency node class it represents.
- `listexp-conclusions.md` - conclusions of the 747-pack list
  execution (INPUT_IS_LIST/OUTPUT_IS_LIST and v3 equivalents) +
  graph expansion (GraphBuilder/expand/ExecutionBlocker) usage
  census, with the Dinkster capability/gap matrix and recommended
  compat porting slices (full data in dinkster-research
  feature-usage/).
- `partner-nodes-census.md` - conclusions of the comfy_api_nodes
  partner-node census (37 providers, 231 nodes, remote-definition
  classification; full report + dataset in dinkster-research dives/
  report-09-api-nodes.*).
- `template-workflows-census.md` - conclusions of the 580-template
  official workflow_templates census vs the native compat surface
  (GREEN/YELLOW/RED/API split, ranked blockers, construct prevalence,
  model families; full report + dataset in dinkster-research dives/
  report-10-template-workflows.* ).

## Deep dives (private repo: Kosinkadink/dinkster-research, dives/)

Cited throughout docs/extension-design.md as "report-NN" and
docs/training-design.md as "report-tN". Each report pins the exact commit
it analyzed and carries validated file:line citations.

| Report | Subject |
|--------|---------|
| report-01-acn-ade.md | Advanced-ControlNet + AnimateDiff-Evolved |
| report-02-res4lyf.md | RES4LYF samplers |
| report-03-kjnodes-easyuse.md | KJNodes + Easy-Use |
| report-04-impact-usdu.md | Impact Pack + UltimateSDUpscale |
| report-05-gguf-wrappers.md | GGUF + model wrappers (WanVideoWrapper etc.) |
| report-06-guidance-attention.md | Guidance/attention packs (NAG, PAG, ppm) |
| report-07-prompt-control.md | prompt-control |
| report-08-frontend-server.md | Frontend/server extension surfaces |
| report-t1-comfyui-training.md | ComfyUI built-in trainer |
| report-t2-kohya.md | kohya sd-scripts |
| report-t3-ai-toolkit.md | ostris ai-toolkit |

The private repo also archives the full scan data (roster, batch results,
node maps), the original design drafts at hand-off time, and the
coordinator spec drafts for the implementation slices.

## Programs of record (authoritative documents)

- Extension architecture: `docs/extension-design.md` (this repo)
- Training architecture: `docs/training-design.md` (this repo)
- Frontend extension RFC: Dinkster-Frontend `docs/frontend-extension-rfc.md`
