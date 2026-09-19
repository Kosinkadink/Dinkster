# Partner (API) node pack: phase 1 structure

Status: PROPOSED 2026-07-29 (backend coordinator design pass; fulfils
the ROADMAP M4 "pack-structure spec" commitment queued 2026-07-29).
Grounded on the report-09 partner-node census
([partner-node research](https://github.com/Kosinkadink/comfy-vibe-station/blob/main/notes/research/partner-nodes-census.md); upstream pinned e651b7be:
37 provider modules, 231 registered V3 nodes, 79.2% class 1+2
standard-adapter, 13 imperative holdouts) and the report-10 template
census (299 of 580 official templates carry partner/API nodes).
DESIGN 3.7 is the plan of record: phase 1 ships partner nodes as a
NORMAL Dinkster pack with ComfyUI-equivalent behavior; phase 2 migrates
definitions to remote data. This document pins the phase-1 structure
so nothing built now forecloses phase 2.

## 1. Scope and non-goals

In scope for phase 1:

- One new in-repo pack shipping partner nodes with behavior
  equivalent to upstream comfy_api_nodes (same operations, same
  proxy auth model, same polling/upload conventions, same media
  handling), authored exclusively through dinkster_api.v1.
- A shared in-pack runtime reproducing the upstream util/ client
  conventions, driven by pure-data operation descriptors.
- A closed SSRF/download trust policy (upstream has none; census
  section 3.3: no byte cap, no MIME allowlist, no redirect policy).
- Upstream-id aliasing so official templates referencing upstream
  class ids resolve to pack nodes.

Non-goals (phase 2 or separately triggered; see section 12):

- Remote-fetched definitions, definition signing, hot reload.
- Price badges / client price expressions (frontend coordination).
- comfy.org login-token (OAuth) flows; phase 1 is API-key only.
- The 13 class-4 imperative nodes (Rodin 7, Topaz video 2, Sonilo 2,
  ByteDance personal-asset 2) and class-3 named-helper providers,
  except where a first-wave provider needs a specific named helper.

## 2. Pack identity and layout

- Directory: `packages/dinkster-nodes-partner/` (sibling of
  dinkster-nodes-foundation / dinkster-nodes-dev, same shape: `dinkster-pack.toml`,
  module tree, `tests/`, `pyproject.toml`). DESIGN 3.7 requires that
  splitting it into its own repo later is `git mv`, not surgery: the
  pack must not import anything outside `dinkster_api.v1` and its own
  modules.
- Pack name / namespace claim: `partner`. Node types are
  `partner.<provider>.<operation>`, lowercase closed name grammar,
  e.g. `partner.kling.text-to-video`, `partner.bfl.flux-pro-ultra`.
  The provider segment matches the census `provider` key; the
  operation segment is a mechanical lowering of the upstream class
  name minus the provider prefix and `Node` suffix.
- Isolated worker with its OWN venv (packs default to isolation;
  this pack pins network/media deps the host venv must not carry):
  `httpx` (HTTP client), `pillow` (image codecs), `av` (MP4/H.264
  video encode), all version-pinned in `requires`.
- Presentation: display name "Partner Nodes", API-badge-like mark;
  category strings mirror upstream (`partner/video/Kling` etc.) so
  search feels identical.

## 3. Operation descriptors: data now, remote later

The single most important structural rule: a standard (class 1/2)
node's behavior is expressed as a PURE DATA descriptor - an `OpSpec`
built from the 14-item adapter vocabulary pinned by the census
(value_construct, http_sync_json, http_sync_binary, submit_poll,
multi_stage, proxy_upload, encode_media, media_constraints,
mask_prepare, multipart_map, download_decode, batch_map_join,
response_select, local_progress). `execute()` on a standard node is
a one-line shim: `return run_op(OP_SPEC, inputs, ctx)`.

- `OpSpec` is frozen dataclasses (typed, validated at import), NOT
  ad-hoc dicts, but must remain losslessly serializable to JSON -
  that property IS the phase-2 migration path: remote definitions
  are serialized OpSpecs plus schemas, and the phase-1 runtime is
  already their interpreter. Anything not expressible as an OpSpec
  is a named helper (class 3) or an imperative body (class 4), both
  quarantined in per-provider modules with an explicit registry of
  helper ids, mirroring the census classification.
- Request/response contracts: pydantic-free typed models authored as
  plain frozen dataclasses with explicit field validation (the pack
  venv could carry pydantic, but phase 2 needs a closed validation
  grammar anyway; do not inherit upstream's pydantic surface).
  Contract shapes are transcribed from the upstream apis/ modules at
  the pinned commit and snapshotted as fixtures (section 10).

## 4. Shared runtime (partner_runtime module)

One module tree inside the pack reproduces the upstream util/
conventions with the SAME constants (census section 3.2, all
line-cited in report-09):

- Retry statuses 408/500/502/503/504; rate-limit 429 (or provider
  predicate) with its separate default budget of 16; backoff honors
  Retry-After seconds or HTTP-date, capped at 150 s.
- Poll defaults: 5 s interval, 480 non-queued attempts, broad
  queued/terminal/failed status vocabularies; queued polls do not
  consume attempts; optional cancel endpoint invoked on local
  interruption.
- Friendly 401/402/409/429 error mapping (login, credits/account,
  conflict, limits); exhausted-network errors distinguish local
  connectivity from API-server failure. Errors surface as normal
  node errors (the error-hints interpreter registry may gain
  partner-specific interpreters later; not in this slice).
- Upload: POST UploadRequest(file_name, content_type) to
  /customers/storage via the proxy, raw PUT (no Comfy auth) to the
  signed URL, interruption-safe, then hand back the download URL.
- Download: streaming 1 MiB chunks, cancellation-checked, decode to
  IMAGE/VIDEO/AUDIO/SVG/bytes per download_decode.
- Progress: elapsed/status text through the normal Dinkster progress
  events (upstream send_progress_text equivalent); cancellation
  checks at every await point per the upstream once-per-second
  discipline.
- Credits: surface actual X-Comfy-Credits-Used in the job's progress
  text and worker log in phase 1; a structured credits event is
  contract-visible and deferred (section 12).

Async httpx client, one per worker, connection-pooled. All requests
race against cancellation like upstream.

## 5. Auth threading (comfy.org proxy)

- All initial provider calls are RELATIVE paths joined to the
  comfy.org API base and carry Comfy headers, exactly like upstream;
  no provider host is ever embedded in a definition. Absolute URLs
  never receive Comfy headers or credentials (upstream rule,
  preserved verbatim).
- Credential source (v1): `dinkster-serve --comfy-api-key KEY` (or env
  `DINKSTER_COMFY_API_KEY`), plus optional `--comfy-api-base` (default
  https://api.comfy.org). The supervisor injects both into the
  partner pack worker's environment at spawn. Credentials are HOST
  configuration: never document state, never schema-visible, never
  in cache identity, never logged, never in /api/settings responses
  (settings categories are a closed list; this is deliberately not
  one). Rotation = worker restart, which the worker boundary already
  makes cheap.
- No hidden auth inputs on schemas (upstream V3 appends hidden
  token/key inputs per node; Dinkster holds auth entirely host-side, so
  schemas stay clean and documents stay portable).
- Shared headers mirror upstream where meaningful: Dinkster version,
  usage source, job id when available.
- Missing credentials fail at execution with the friendly 401-class
  error, not at load; the pack loads and its nodes are browsable
  without a key.

## 6. SSRF/download trust policy (closed rules, Dinkster addition)

Response-controlled ABSOLUTE URLs occur in three places: provider
polling URLs (e.g. BFL), signed upload PUT targets, and result media
downloads. Upstream applies no policy; Dinkster pins one:

- https only; explicit ports 443 only unless the URL came from the
  comfy.org proxy response itself.
- Resolve-and-deny private, loopback, link-local, and metadata
  ranges (RFC1918, 127/8, 169.254/16, ::1, fc00::/7, fe80::/10) at
  connect time (post-DNS, not string matching).
- Redirects: followed at most 3 hops, each hop re-validated against
  the same rules, never forwarding any credential or Comfy header.
- Download caps: per-response byte cap (default 1 GiB, descriptor
  can lower, never raise past the cap) enforced while streaming;
  content-type must match the descriptor's expected media family.
- No archive auto-extraction anywhere in phase 1 (the Tencent OBJ
  ZIP helper is class 3 and out of first wave; when it lands it
  extracts with explicit entry allowlists and size caps).
- Violations are loud node errors naming the rejected URL's host.

## 7. Template/compat aliasing

Official templates reference upstream class ids (KlingTextToVideoNode
et al). Those ids are NOT in the compat catalog (the 587-type catalog
at e651b7be excludes comfy_api_nodes; the census tracks the 231
separately). To make the 299 API-flagged templates resolve:

- The pack ships an alias table as pack data: upstream class id ->
  pack node type, exact strings, one-to-one, validated at load
  (every target must be a type the pack itself claims).
- The compat prompt path (and the frontend's document import, which
  already consumes schema-carried replacement data) consults the
  alias table so a prompt or document naming an upstream id lands on
  the pack node. Exact host-side mechanism (a new optional
  `[pack.entry]` alias hook vs. compat-layer consumption of
  replacement-rule data) is decided in the implementation slice
  after grounding how dinkster-compat-comfy resolves unknown type ids
  today - STOP-RULE material, not improvisation.
- Input id parity matters for aliasing to be useful: pack schemas
  keep upstream input ids verbatim (snake_case survives the closed
  grammar) so template stored values map 1:1. Where an upstream
  input is dropped (hidden auth inputs), the alias mapping notes it.

## 8. Execution semantics

- Caching stays ON (default idempotent): ComfyUI re-uses API results
  when inputs are unchanged and re-runs on seed/input change; Dinkster's
  schema-signature + input-fingerprint cache reproduces that exactly.
  No idempotent=False except where upstream is genuinely
  side-effecting (asset registration nodes - class 4, out of scope).
- occupies: no gpu claim; remote calls are network-bound. If queue
  pressure from long polls becomes real, an abstract "network" kind
  is a one-line addition later; not pinned now.
- Cancellation: local interruption cancels in-flight HTTP, invokes
  the provider cancel endpoint when the descriptor declares one, and
  stops polling; matches upstream.
- media_constraints validate locally before any credit-burning call
  (fast-fail courtesy; the remote answer stays authoritative).

## 9. First wave (evidence-ranked)

Providers whose ENTIRE module is class <= 2 (pure standard adapter),
ranked by unique official templates touched (computed from report-09
x report-10 JSONLs, 2026-07-29):

| Provider | Nodes | Templates | Notes |
|----------|-------|-----------|-------|
| kling    | 25    | 30        | L constructors + U/P/D |
| wan      | 14    | 17        | U/P/D incl. HappyHorse |
| grok     | 7     | 14        | S/P/U/D |
| vidu     | 13    | 11        | U/P/D |
| bfl      | 10    | 9         | S/P/D; response-controlled absolute poll URL - exercises section 6 |
| luma     | 15    | 9         | L + U/P/D |
| bria     | 6     | 8         | U/P/D |

Wave A = kling + wan + grok + bfl (56 nodes, ~64 unique templates,
and bfl forces the absolute-URL trust path early). Wave B = vidu +
luma + bria + the remaining small class-2 providers (wavespeed,
hitpaw, magnific, sync_so, beeble, krea, ltxv, quiver, veo2, reve,
heygen, runway, meshy, pixverse, tripo, elevenlabs, openrouter,
anthropic, bytedance_llm). Class-3 providers (gemini - 75 templates,
the single biggest API driver - openai, recraft, hunyuan3d, ideogram,
minimax, bytedance non-asset) follow once their named helpers are
audited into the helper registry; gemini is the priority class-3
target because of its template weight.

## 10. Testing (no live provider calls in CI)

- The runtime takes an injectable transport; CI uses a fixture
  transport replaying contract fixtures snapshotted from the pinned
  upstream apis/ modules (request shape, response shape, status
  sequences, credit headers). Every OpSpec gets: request-mapping
  proof, response-select proof, poll state-machine proof (queued /
  running / terminal / failed / timeout), cancellation proof, and
  trust-policy proofs (private-IP deny, redirect cap, byte cap).
- Schema parity: golden object-info comparison against the pinned
  upstream for each shipped node (input ids, types, defaults,
  optionality), with documented deltas only for dropped hidden
  inputs.
- One optional live smoke (env-gated, never CI): a single cheap
  provider call end-to-end, run manually when credentials exist.

## 11. Slice plan

1. Pack skeleton + runtime core: manifest, venv deps, OpSpec model,
   http_sync_json/response_select/local_progress, auth threading,
   trust policy, fixture transport, friendly errors. No providers.
2. submit_poll + proxy_upload + encode_media + download_decode +
   media_constraints + mask_prepare; first provider = bfl (small,
   exercises sync, poll-via-absolute-URL, masks, download).
3. Wave A providers (kling, wan, grok) + alias table + compat/import
   aliasing mechanism (with its grounding stop-rule).
4. Wave B class-2 providers (mechanical, parallelizable).
5. Helper registry + first class-3 provider (gemini).
6. Each slice: PROMISES rows with named proofs, ROADMAP flips, live
   catalog/template-census delta where applicable.

Slices 1-2 are one delegate each; 3-4 fan out per-provider once the
runtime is frozen. No compose.py / native_arm.py contact anywhere in
this program; the pack boundary keeps it clear of inference S3.

## 12. Deferrals (ledgered in ROADMAP with triggers)

- Price badges / client price expressions: frontend coordination;
  trigger = frontend asks or first user demand.
- Structured credits event on the wire: trigger = frontend billing
  surface work.
- comfy.org login-token flow: trigger = Dinkster gaining a comfy.org
  account association story.
- Remote-fetched definitions, signing, hot reload: phase 2 (DESIGN
  3.7), trigger = phase-1 waves A+B shipped and stable.
- Class-4 imperative nodes: trigger = per-provider user demand;
  ByteDance personal-asset additionally needs an interactive
  verification UX design.
- Advertise-before-install: M8 registry metadata; unchanged.
