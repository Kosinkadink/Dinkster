# Partner (API) node census - conclusions

Date: 2026-07-29. Full report: dinkster-research `dives/report-09-api-nodes.md`
plus per-node dataset `dives/report-09-api-nodes-census.jsonl` (231 rows,
one per registered node; committed at dinkster-research a8e65c2). Read-only
dive over upstream ComfyUI pinned at
`e651b7bef55a5376343dcb1c0edb79f0142c985e`. This grounds the two-phase
partner-node plan in DESIGN 3.7 and the ROADMAP M4 entry.

## Exact census

- 37 `comfy_api_nodes/nodes_*.py` provider modules; 34 dedicated
  `apis/*.py` contract modules; no dead provider module.
- 231 registered node classes, ALL V3 schema style. 211 set
  `is_api_node`; 20 do not (selectors, local value constructors, and
  the two ByteDance asset nodes whose `is_api_node` is commented out).
- 11 deprecated registrations, zero experimental.
- Two defined but unregistered classes: `MinimaxSubjectToVideoNode`
  (commented out of the registration list) and
  `RecraftStyleV3VectorIllustrationNode` (absent from the list).

## CORRECTION (2026-07-30): Kling misclassification

The Kling rows in this census are WRONG. Report-09 classified all 24
networked `nodes_kling.py` nodes as class 2 using schema-line-only
evidence (the `"submit+poll"` pattern on the node's schema surface); it
never inspected function bodies. The wave 3.2 delegate's per-node audit
against the pin found, and the backend orchestrator
(T-019fb0c7-9208-7109-900d-ea1a13471bb5) independently re-verified,
body logic that no declarative adapter grammar expresses. Load-bearing
counter-evidence sites at `e651b7be`:

- `nodes_kling.py:248-273` - `normalize_omni_prompt_references` regex
  rewrite of `@image[n]`/`@video[n]` placeholders, applied by
  registered nodes at :1117, :1315, :1450, :1558, :1676.
- `nodes_kling.py:946-973` - storyboard imperative construction:
  count parsed from the combo key, per-item validation, variable-length
  `multi_prompt` assembly, duration-sum == global-duration check,
  conditional `multi_shot`/`shot_type` derivation (recurs at
  :1033-1047/:1114-1155, :1232-1246/:1312-1340, :2928-2953/:3074-3106).
- `nodes_kling.py:3105-3156` - `KlingVideoNode` protocol switch routing
  kling-3.0-turbo to `execute_kling_turbo` vs the image2video/
  text2video endpoint choice, plus the Turbo `contents` construction at
  :2856-2904; the same region shows boolean `sound` on/off projection
  and conditional shaping such as `multi_shot=True if shot_type else
  None`.
- Smaller conditionals ruled class 3: the Image2Video mode mutation
  (:486-488) and MotionControl's orientation-dependent max duration
  (:2824-2828).

Consequence: the class split table below overstates class 2 and
understates class 3 for Kling; wave 3.2 was re-scoped into 3.2a (the
genuinely class-2 subset under the adjudicated closed grammar) and 3.2b
(class-3 named helpers + helper registry). See
`docs/DELEGATION-LEDGER-PARTNER.md` wave 3.2/3.2b rows. The
dinkster-research report-09 dataset errata is owned by the backend
orchestrator, not this repo.

### Corrected per-node Kling classification (3.2a delegate audit, 2026-07-30)

Per-node pinned body audit at e651b7be (line numbers are
`comfy_api_nodes/nodes_kling.py`; API model lines are
`comfy_api_nodes/apis/__init__.py`). Result: 2 nodes class 1/2 (shipped
in 3.2a at Dinkster a79924a), 23 nodes class 3 (deferred to 3.2b). The
recurring class-3 causes are: dynamic prompt max-length refusals via
`validate_prompts` (:320-329, reached through :435 by most video
nodes), media dimension/duration probing beyond MediaConstraints,
storyboard/list construction, conditional request mutation, the Omni
prompt regex (:248-273), and scalar type conversion.

| Node | Class | Body evidence |
|---|---|---|
| KlingCameraControls | 1 (3.2a) | Pure value constructor :737-759; all-zero validation :713-734 expressible by CheckInputs. |
| KlingVirtualTryOnNode | 2 (3.2a) | Pure encode/submit/task-id poll/download body :2446-2476; pinned 2048*2048 downscale on both inputs. |
| KlingTextToVideoNode | 3 | Lookup fanout :838 expressible, but shared body calls validate_prompts :435 with dynamic max-length errors :320-329. |
| KlingImage2VideoNode | 3 | Image dim/aspect + prompt validation :481-482, nested camera-control mutation :484-486, conditional std->pro rewrite :488-489. |
| KlingCameraControlI2VNode | 3 | Delegates :1937-1948 into the image2video helper (:481-489). |
| KlingCameraControlT2VNode | 3 | Delegates :1771-1781 into text2video incl dynamic prompt-length :435/:320-329. |
| KlingStartEndFrameNode | 3 | Fanout :2035 expressible; delegation :2036-2047 enters image2video checks/mutation :481-489. |
| KlingVideoExtendNode | 3 | Dynamic prompt maximum/refusal :2100 via :320-329. |
| KlingLipSyncAudioToVideoNode | 3 | Video dimension and duration probes :593-594 before upload :596-604; dimension probing outside MediaConstraints. |
| KlingLipSyncTextToVideoNode | 3 | VOICES_CONFIG fanout :2400 expressible; shared helper needs dynamic text max + video probes :591-594. |
| KlingImageGenerationNode | 3 | Exact prompt/negative-prompt maximum validation :2582-2583 outside CheckInputs. |
| KlingSingleImageVideoEffectNode | 3 | Reachable pinned schema/model mismatch: schema offers durations 5 and 10 :2243-2245, request enum KlingSingleImageEffectDuration (API 5154-5160) accepts only 5 (API 1290-1292). See docs/comfyui-issues/kling-single-image-effect-duration-enum-mismatch.md. |
| KlingDualCharacterVideoEffectNode | 3 | Conditional request union + plain scalar-string image list :538-553; segments emit mapping items, not this shape. |
| OmniProTextToVideoNode | 3 | Model-dependent refusal matrix :939-948, dynamic prompt validation :949, storyboard parsing/summed-duration :951-973. |
| OmniProFirstLastFrameNode | 3 | Model/storyboard matrix :1107-1155, prompt regex :1117, image probes + typed list construction :1157-1181. |
| OmniProImageToVideoNode | 3 | Model/storyboard logic :1305-1340, prompt regex :1315, reference probes + typed lists :1342-1349. |
| OmniProVideoToVideoNode | 3 | Prompt regex/max + video probes :1450-1453, image probes :1455-1462, typed video state :1463-1468. |
| OmniProEditVideoNode | 3 | Prompt regex/max + video probes :1558-1561, optional image probe/list :1562-1570, typed base-video state :1571-1576. |
| OmniProImageNode | 3 | Model-dependent refusals :1674-1675/:1687-1689, prompt regex/max :1676-1677, image probes/list :1678-1686, conditional int(series_amount) :1701; response fallback :1714 now expressible but insufficient alone. |
| TextToVideoWithAudio | 3 | Dynamic prompt maximum :2658, int-to-string duration conversion :2668. |
| ImageToVideoWithAudio | 3 | Dynamic prompt + image probes :2726-2728, int-to-string duration conversion :2738. |
| MotionControl | 3 | Prompt/image probes :2821-2823, orientation-dependent video max duration :2824-2827, video dimension probe :2828. |
| KlingVideoNode | 3 | Storyboard parsing/list construction :3074-3103, Turbo prompt/protocol branch :3105-3114, image-vs-text submit/poll protocol branch :3116-3156. |
| KlingFirstLastFrameNode | 3 | Dynamic prompt + both-frame probes :3256-3260, int-to-string duration conversion :3279. |
| KlingAvatarNode | 3 | Image/audio probes :3363-3365, explicit MP3/libmp3lame upload construction :3371-3374, custom 800-attempt polling :3383-3388. |

Corrected Kling split: class 1 = 1, class 2 = 1, class 3 = 23 (census
originally recorded class 2 = 24). Global split correction is deferred
until other providers' bodies are audited wave by wave.

METHOD DEFECT (for future censuses): classification from schema-level
pattern matching alone is insufficient. A node's class is a property of
its execute BODY; every census row must cite body evidence (request
construction, validation, response handling), not just the schema
surface. Any future census that does not inspect bodies must say so and
mark its class splits as provisional.

## Classification split (the core result)

NOTE (2026-07-30): the Kling contribution to these counts is wrong -
see the correction section above. The split is retained as originally
reported pending the corrected per-node audit from wave 3.2a.

| Class | Meaning | Count | Share |
|-------|---------|-------|-------|
| 1 | pure schema / local value constructor | 17 | 7.4% |
| 2 | declarative + standard generic adapter | 166 | 71.9% |
| 3 | declarative + named provider-specific helper | 35 | 15.2% |
| 4 | imperative escape hatch required | 13 | 5.6% |

79.2% (classes 1+2) are directly interpretable as pure
RemoteNodeDefinition data with the standard adapter vocabulary; 94.4%
(1+2+3) can remote their schema/contract while retaining named local
helpers. Only 13 nodes genuinely need imperative bodies: 7 Rodin
(custom multipart/mode mapping, multi-job status, result fanout +
filesystem-compatibility outputs), 2 Topaz video (source geometry and
filter math + provider signed PUT/ETag completion), 2 Sonilo
(long-lived NDJSON event stream + base64 audio chunk assembly), 2
ByteDance personal-asset (conditional interactive H5 browser
verification + upload/register/poll).

## Generic adapter vocabulary (14 items)

`value_construct`, `http_sync_json`, `http_sync_binary`, `submit_poll`,
`multi_stage`, `proxy_upload`, `encode_media`, `media_constraints`,
`mask_prepare`, `multipart_map`, `download_decode`, `batch_map_join`,
`response_select`, `local_progress`. This list defines the required
expressiveness of the phase-2 remote-definition format.

## Auth model

All initial provider service requests go through comfy.org-relative
`/proxy/<provider>/` paths resolved against `--comfy-api-base` (default
`https://api.comfy.org`); no provider embeds its own API host. Hidden
login token is preferred as `Authorization: Bearer ...`, else hidden
comfy API key as `X-API-KEY` (`comfy_api_nodes/util/_helpers.py:38-40`).
Response-controlled ABSOLUTE URLs are used for some polling, signed raw
PUT uploads, and result media downloads, WITHOUT Comfy auth - these are
the SSRF/redirect trust boundaries a Dinkster runtime must police. Price
badges are client-side estimates; actual credits may arrive after
success in `X-Comfy-Credits-Used`.

## Consequences for the two-phase plan

- Phase 1 (normal Dinkster pack, ComfyUI-equivalent): ship all 37
  providers' schemas/contracts plus one shared runtime reproducing the
  util/ conventions: proxy auth, retries/rate limits, poll
  states/defaults, progress/cancellation, storage upload, media
  codecs/download, friendly errors, credits surfacing. Snapshot object
  info and contract fixtures against the pinned commit.
- Phase 2 (remote definitions): remote the schema/flags/badges, typed
  wire contracts, operation DAGs over the 14-item vocabulary, relative
  endpoint templates, status sets/timeouts, declarable input
  constraints, response/output mappings, and REFERENCES to a closed
  local helper-id registry. Remote data, never remote code.
- Keep local forever: credentials/proxy policy, HTTP
  trust/idempotency/quota, retries/cancellation/progress/actual
  credits, tensor/media codecs, upload/download + SSRF policy, archive
  and model-file handling, the 35 named helpers (while
  provider-specific), and the 13 imperative bodies unless the provider
  APIs normalize.
- The obstacle to "partner nodes as remote data" is NOT schema volume:
  it is that current upstream classes conflate presentation, billing
  estimates, secrets, HTTP, codecs, filesystem, and interaction.
  Separating those layers is what makes the split above achievable.
- No provider frontend JS, extra server routes, or websockets exist;
  frontend coupling is limited to the API badge, client price
  expressions, dynamic inputs, hidden auth inputs, and progress text.
