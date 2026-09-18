# Federated Asset DTO V1

Status: jointly frozen on 2026-08-08 by backend amendment A3 and explicit
frontend concurrence. This document freezes JSON shapes only. Route paths
remain frontend-injected through `paths.catalog`, `paths.candidates`, and
`paths.resolve`. Pure private backend models/codecs are implemented in
`dinkster_server.federated_assets_v1`; read-only catalog/candidates routes, auth,
and cursor signing are implemented at injected paths. Resolve/repair behavior,
A2 transfer, frontend adoption/live calls, deployment, and live proof remain
separate gated slices.

## Global rules

- Every operation uses POST JSON. Every request, 2xx response, and non-2xx
  error is a closed plain object with its own numeric `contractVersion: 1`.
  Unknown or inherited fields reject.
- The exact five-field `AssetRef` remains
  `{digest,name,size,mediaType,virtualPath}`. Digest is lowercase `blake3:`
  plus exactly 64 lowercase hexadecimal characters. Size is a safe
  nonnegative integer.
- The server derives authorized providers from the authenticated principal and
  provider policy. The caller never supplies an authorized-provider set.
  Optional `scope` only disambiguates among already authorized scopes under
  existing resolve-scope semantics and grants no authority.
- MIME accepts are exact case-sensitive `type/subtype` or `type/*`; `*/*` is
  invalid. An empty accepts list is unrestricted. At most 32 accepts are
  allowed.

## Shared objects

`SourceIdentityV1`:

```text
{providerId:string, sourceId:string}
```

`CandidateSelectionV1`:

```text
{logicalId:string, variantId:string, digest:string, source?:SourceIdentityV1}
```

The first three fields are required. `source` is an optional explicit provider
selection.

`ResolveContextV1`:

```text
{
  assetKind:string,
  schema:{nodeType:string,inputId:string,typeId?:string},
  accept:string[]
}
```

`ExpectedV1` is `{digest?:string,size?:number}` and contains at least one
field when present.

`HintsV1` has only the optional string keys `source`, `reference`,
`loaderPath`, `modelType`, and `displayName`.

`CandidateV1`:

```text
{
  logicalId:string, family:string, assetKind:string,
  variantId:string, dtype:string, quantization:string, format:string,
  role:string,
  requirements:{loaders:string[],runtimes:string[],hardware:string[]},
  digest:string, size?:number, mediaType?:string,
  availability:{
    status:'local'|'downloadable'|'unavailable',
    reason:string
  },
  compatibility:{
    status:'compatible'|'incompatible'|'unknown',
    reason:string
  },
  assetRef?:AssetRef,
  providerSources:[{
    source:SourceIdentityV1,
    status:'available'|'unavailable',
    reason:string,
    requires:{
      credential?:string,
      license?:string,
      cost?:string,
      policyOverride?:string
    }
  }]
}
```

`assetRef` is present if and only if availability is `local`; it is the exact
admitted execution reference and its digest equals `candidate.digest`.
Compatible has an empty compatibility reason; non-compatible has a nonempty
reason. Same-digest mirrors remain one logical/variant/digest artifact row with
multiple provider sources. No locator, provider metadata, secret, raw host
path, filename authority, or price value appears. Candidate arrays are ordered
deterministically by `(logicalId, variantId, digest)` and provider sources by
`(providerId, sourceId)`.

## Catalog

Request:

```text
{
  contractVersion:1,
  scope?:string,
  query?:string,
  assetKind?:string,
  context?:ResolveContextV1,
  availability?:('local'|'downloadable'|'unavailable')[],
  compatibility?:('compatible'|'incompatible'|'unknown')[],
  cursor?:string,
  limit?:number
}
```

If `assetKind` and `context.assetKind` both exist, they agree. `query` is only
a casefold substring filter over logical id, family, and advisory aliases. It
never ranks or selects; filenames never participate. State filters are
no-duplicate closed sets. Omitted filters mean all.

Response:

```text
{contractVersion:1,items:CandidateV1[],nextCursor?:string}
```

There is one item per `(logicalId, variantId, digest)`.

## Candidates

This operation is read-only.

Request:

```text
{
  contractVersion:1,
  scope?:string,
  context:ResolveContextV1,
  expected?:ExpectedV1,
  hints?:HintsV1,
  selection?:CandidateSelectionV1,
  cursor?:string,
  limit?:number
}
```

There is deliberately no mode. The operation explains existing S2/A1 facts
and performs no mapping, acquisition, repair, or mutation. Hints remain
advisory and never rank or establish digest or variant authority. `expected`
and `selection` agree when both are supplied.

Response:

```text
{
  contractVersion:1,
  status:'resolved'|'missing'|'ambiguous'|'incompatible',
  items:CandidateV1[],
  selectedCandidate?:{
    logicalId:string,
    variantId:string,
    digest:string,
    reason:'reference-mapping'|'explicit-selection'|'trusted-source'|
           'expected-digest'|'compatible-only',
    source?:SourceIdentityV1
  },
  nextCursor?:string
}
```

`selectedCandidate` is present if and only if backend resolution selected a
candidate. Status `resolved` always means compatible and local.

## Resolve

Request:

```text
{
  contractVersion:1,
  mode:'existing'|'acquire-managed',
  scope?:string,
  context:ResolveContextV1,
  expected?:ExpectedV1,
  hints?:HintsV1,
  selection?:CandidateSelectionV1,
  consents?:{
    license?:string[],
    cost?:string[],
    policyOverride?:string[]
  },
  document?:{
    digest:string,
    occurrences:[{pointer:string,current:AssetRef}]
  }
}
```

Consent values are opaque printable server-issued requirement IDs returned by
a prior typed error. They are at most 256 characters, at most 16 per category,
and at most 32 total. The server revalidates current provider policy before
treating them as held. They are not bearer credentials or authority.
Credential setup is server/provider-adapter state and has no client grant
field. `expected` and `selection` agree.

2xx response:

```text
{
  contractVersion:1,
  status:'resolved-existing'|'resolved-remapped'|'acquired',
  selected:AssetRef,
  selectedCandidate:{
    logicalId:string,
    variantId:string,
    digest:string,
    reason:'reference-mapping'|'explicit-selection'|'trusted-source'|
           'expected-digest'|'compatible-only',
    source?:SourceIdentityV1
  },
  repair?:AssetRefRepairSuggestionV1
}
```

Every 2xx response carries a compatible local exact `AssetRef`. `acquired` is
reachable only after A2 verifies expected/trusted size, media type, and digest,
atomically publishes, and reruns resolution. Until A2 is enabled, an
`acquire-managed` request requiring bytes fails with typed
`service-unavailable` and reason `acquisition-not-enabled`; it never falsely
returns `acquired`. Routine ready authorized same-digest remap is
`resolved-remapped` and silent/nonblocking.

## Repair suggestion

`AssetRefRepairSuggestionV1`:

```text
{
  type:'asset-ref-repair',
  version:1,
  atomic:true,
  documentDigest:string,
  preconditions:[{pointer:string,equals:AssetRef}],
  replacements:[{pointer:string,value:AssetRef}]
}
```

The backend echoes `document.digest` verbatim as `documentDigest`. RFC 6901
and list-element pointers, target-set equality, overlap rejection, deep exact
five-field preconditions, and all-or-none semantics apply. The backend only
suggests; the frontend validates and applies one consolidated command. Repair
is omitted when `document` is absent or no change exists.

## Error envelope

Every non-2xx response has the exact top-level shape:

```text
{contractVersion:1,error:<one closed variant>}
```

Existing auth middleware emits this envelope for these registered operations;
other routes stay unchanged. Catalog and candidates require `assets:read`
despite POST. Resolve requires `assets:write`. Configured paths are registered
with explicit capability metadata, never inferred only from method or prefix.

Closed variants and statuses:

- 400 `{code:'invalid-request',reason:string,field?:string}`
- 400 `{code:'cursor-invalid',reason:'malformed'|'query-mismatch'|'stale-snapshot'|'expired'}`
- 401 `{code:'authentication-required',reason:string}`
- 403 `{code:'forbidden',reason:string}`
- 404 `{code:'not-found',reason:string}`
- 409 `{code:'selection-required',reason:'digestless'|'different-digest'|'multiple-variants'|'ambiguous-alias',candidates:CandidateV1[],truncated:boolean}`
- 409 `{code:'not-available'|'incompatible',reason:string,candidates:CandidateV1[],truncated:boolean}`
- 409 `{code:'credential-required'|'license-required'|'cost-required'|'policy-override-required',requirementId:string,selection:CandidateSelectionV1}`
- 409 `{code:'source-unavailable'|'no-compatible-destination'|'mapping-conflict',reason:string,selection?:CandidateSelectionV1}`
- 422 `{code:'wrong-kind',expectedKind:string,actualKind?:string}`
- 422 `{code:'integrity-mismatch',expectedDigest?:string,observedDigest?:string,expectedSize?:number,observedSize?:number}`
- 502 `{code:'acquisition-failed',reason:string,selection?:CandidateSelectionV1}`
- 503 `{code:'service-unavailable',reason:string,selection?:CandidateSelectionV1}`

`integrity-mismatch` carries at least one expected/observed digest or size
mismatch pair. There is no free `message`; the frontend localizes from code and
reason. Candidate-bearing errors cap candidates at 100 and set `truncated`
truthfully.

## Outcome and prompt mapping

- `selection-required` with `digestless` or `different-digest` are the only
  ambiguity prompts. One variant with multiple artifact digests is
  `different-digest`. Multiple variants at one digest and ambiguous aliases
  use the ordinary explicit candidate picker, not an exceptional modal.
- S2 mapping/expected-digest disagreement is
  `selection-required/different-digest`; mapping/selection disagreement is
  `mapping-conflict`; kind mismatch is `wrong-kind`; compatibility failures
  are `incompatible`.
- Credential, license, cost, and policy errors are the only other prompt/setup
  classes. Wrong kind and integrity mismatch are terminal errors, never
  choices.
- A1 unauthorized maps to `forbidden`. Conflicting source/destination
  authority facts map to `service-unavailable`, not a user choice. A1
  source/no-destination/grant statuses map to their matching codes.

## Cursors and bounds

- Cursor is opaque, signed, query/filter/scope/catalog-generation-bound, and at
  most 4,096 characters. A later page on a changed generation fails
  `cursor-invalid/stale-snapshot` instead of mixing generations. A cursor is
  never authority and may expire.
- Limit is 1 through 100 and defaults to 50. Query is at most 512 characters.
  Context identifiers are nonempty and at most 512 characters. Hint strings
  are at most 1,024 characters. The document token is opaque, control-free,
  and at most 4,096 characters. Occurrences cap at 256. Provider sources cap
  at 64. Reason strings cap at 256. Requirement and consent IDs are printable
  and at most 256 characters.
- Results never silently truncate except for candidate-bearing errors with the
  explicit `truncated` field.

## Authority and non-goals

There is no `AssetRef` mutation, filename/path/hint/provider-order authority,
implicit sibling-variant preference, or caller-supplied provider, locator,
credential, or secret. Catalog and candidates are read-only. This contract
does not itself implement backend routes, A2 transfer, frontend UI/live calls,
deployment, or live proof.
