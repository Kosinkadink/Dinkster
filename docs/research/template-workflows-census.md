# Template-workflow gap census - conclusions

Census of every active official ComfyUI template workflow against the
live Dinkster native compat translation surface (user goal 2026-07-29:
templates must run natively in Dinkster, including subgraph translation).
Full report + per-template JSONL dataset live in the private
dinkster-research repo: `dives/report-10-template-workflows.md` and
`dives/report-10-template-census.jsonl` (commit ed1654d). Delegate
T-019fae1f-9767-7098-a16e-035841cf23ee; coordinator-verified.

Pins: workflow_templates `aa3661d9`, Dinkster `7b37cab`, reference
ComfyUI `f4b99bc` (DINKSTER_COMFYUI_ROOT catalog: 461 upstream core
types, 458 translated, exactly 3 classified skips).

## Headline

Of 580 active templates: **193 GREEN** (every executable node
translates and no unhandled document construct), **164 YELLOW**
(translatable nodes but document constructs needing import-time
handling), **223 RED** (at least one executable node type that does
not translate), 0 unclear. **299 templates use partner/API nodes**
(overlapping flag: 157 otherwise GREEN, 78 YELLOW, 64 RED) - the
partner-node pack (M4 phase 1) gates over half the corpus.

## What blocks RED templates

Counts are templates blocked, not occurrences. Only 3 blockers are
compat SKIPS of upstream-core types; every other blocker is a node
type absent from the pinned upstream core catalog (custom packs the
official templates embed, or core types newer than the f4b99bc pin -
distinguish before scheduling):

| Blocker | Templates | Status |
| --- | ---: | --- |
| ComfySwitchNode | 70 | compat-skipped (lazy) - document-time pruning design approved |
| ResizeImageMaskNode | 46 | compat-skipped (combo option keys contain spaces) - grammar/lowering decision |
| ComfyMathExpression | 42 | not in pinned core catalog |
| SimpleMath+ (essentials) | 26 | custom pack |
| ImageResizeKJv2 (KJNodes) | 24 | custom pack |
| VHS_VideoCombine / VHS_LoadVideo / VHS_VideoInfo | 21 / 17 / 11 | custom pack (VideoHelperSuite) |
| ImageBatchMulti (KJNodes) | 18 | custom pack |
| ResolutionSelector | 17 | not in pinned core catalog |
| SaveImageAdvanced | 16 | not in pinned core catalog |
| CustomCombo | 14 | compat-skipped (accept_all_inputs) - wire-15 names-form candidate |
| TextGenerate / SaveText / GetImageSizeAndCount | 14 / 11 / 9 | not in pinned core catalog |
| WanVideoWrapper family (Decode/ModelLoader/VAELoader/Sampler/LoraSelect/...) | 6-8 each | custom pack |

The three compat skips alone account for the top-2 and rank-11
blockers (130 blocked-template counts). Restoring
ComfySwitchNode (lazy document-time pruning), ResizeImageMaskNode
(option-key grammar), and CustomCombo converts the largest RED
cohorts that are purely our decision to fix.

## What makes YELLOW templates

Document constructs forcing YELLOW (no RED node blocker): subgraphs
249 templates, PrimitiveNode family 173, muted/bypassed nodes 144,
Reroute 44, SetNode/GetNode 0. Notes (407) and groups (330) are
cosmetic and never block. Subgraph translation - explicitly in the
user goal - is the single largest document-level gap.

## Model families

Best-effort static labels: wan 67, qwen 55, flux 39, 3d 35, ltxv 28,
z-image 21, audio 18, hidream 7, sd15 6, sd3/sdxl 4 each, smaller
tails; 275 have no inferable family label (many are API-only or
utility templates). Forwarded to the inference thread as Wave 0+
workload-pin evidence: wan/qwen/ltxv/z-image dominate the template
corpus far beyond the current Wave 0 (SD1.5/SDXL/Flux) scope.

## Program implications (recorded in ROADMAP)

1. Lazy document-time pruning (ComfySwitchNode) is the highest-value
   single compat restoration: 70 templates.
2. ResizeImageMaskNode grammar decision is second: 46 templates for
   what is likely a small combo-option-key lowering rule.
3. Partner-node pack phase 1 gates 299 templates (157 otherwise
   GREEN today).
4. Subgraph + PrimitiveNode + muted/bypassed import handling converts
   the YELLOW cohort; overlaps the frontend paste/import program and
   the approved bypass/mute Option A direction.
5. Custom-pack blockers (math/essentials/KJ/VHS/Wan wrapper) need a
   separate decision: native equivalents, 1->N replacement mappings,
   or an explicit out-of-scope line. The "not in pinned core catalog"
   rows must first be re-checked against current ComfyUI master to
   separate newer-core types from true custom nodes.

## Blocker disposition recheck (report-11, 2026-07-29)

Follow-up recheck of all 265 blocker types (847 template-blocker
memberships) against current ComfyUI master `e651b7be` (verified
still the origin/master head at review time). Full artifacts:
dinkster-research `dives/report-11-blocker-disposition.*` (addc8c0).
Delegate T-019fae35-3e7b-74ad-bf90-8e2da038c79c; coordinator-verified
(counts reconcile exactly against the report-10 JSONL; module paths
exist at the master pin and are absent at f4b99bc; cnr_id evidence
resolves in the cited templates).

| Disposition | Types | Memberships | Unique RED templates touched |
| --- | ---: | ---: | ---: |
| NEWER-CORE (core added after f4b99bc) | 79 | 248 | 115 |
| CUSTOM-PACK | 181 | 467 | 84 |
| COMPAT-SKIP (our 3 skips) | 3 | 130 | 105 |
| OTHER (template-data oddities) | 2 | 2 | 2 |

The 79 NEWER-CORE types span 41 comfy_extras modules (math,
resolution, textgen, text/string, images/audio advanced-save,
seedvr, sam3, mediapipe, moge, gaussian-splat, wandancer, void, ...).
Custom packs are dominated by comfyui-kjnodes (97 memberships),
ComfyUI-WanVideoWrapper (93), comfyui-videohelpersuite (53),
comfyui_essentials (47), ComfyUI-LTXVideo (20), AnimateDiff-Evolved
(18); a long tail of 32 more packs. OTHER: `AudioStemSeparate`
(unresolvable ID) and `RecraftV4ImageToVectorNode` (stale partner ID)
look like template-repo data defects worth reporting upstream.

Actionable overlap math (coordinator-computed from both JSONLs):
templates whose blockers are ALL of one kind - NEWER-CORE only 49,
COMPAT-SKIP only 35, CUSTOM-PACK only 63, plus combos. A compat
catalog refresh to current core PLUS the three skip restorations
converts 137 of 223 RED templates (61%); custom packs gate the
remaining 84. So the compat catalog refresh (translate the 79 new
core types from a NEW reference checkout - /home/kosin/ComfyUI must
stay at f4b99bc as the pinned W0-HARNESS baseline) is the
second-highest-value backend slice after lazy pruning.

## Catalog refresh validation (report-12, 2026-07-29)

Controlled re-run of the full compat catalog translation against a
fresh ComfyUI checkout at master `e651b7be` (dedicated clone + venv
under /home/kosin/node-analysis/clones; /home/kosin/ComfyUI untouched
at f4b99bc). Full artifacts: dinkster-research
`dives/report-12-catalog-refresh.*` (d149f0f). Delegate
T-019fae42-afb4-723e-a85c-a633d4943a0d; coordinator-verified (sha256s,
all 587 JSONL rows parse with exact schema, outcome Counter, blocker
set and per-type counts reconcile exactly against report-11, RED
conversion math recomputed independently from the report-10 JSONL).

- New pin catalog: 587 registered = 584 translated + 3 skips + 0
  errors. The skip list is UNCHANGED (ComfySwitchNode lazy,
  CustomCombo accept_all_inputs, ResizeImageMaskNode option-key
  grammar) - no new skip category appears at the new pin.
- All 79 NEWER-CORE report-11 blockers translate cleanly (0 skips,
  0 errors); no comfy_extras import failures.
- Diff vs f4b99bc: +127 registrations, -1 (MultiGPU_Options,
  deliberately unregistered upstream in current master), and zero
  shared types changed outcome or reason.
- RED conversion confirmed: refresh alone converts 49 of 223;
  refresh + the three skip restorations converts 137 (61%).

Conclusion: the compat translator needs NO code work for the newer
core surface - adopting a newer reference checkout is purely a pin
policy decision (ROADMAP 6a). The ResizeImageMaskNode skip is being
removed separately by the option-key grammar widening slice.
