# Native account API

These routes control the native install's `AccountSession` and bind its verified account identity to the browser that starts sign-in, as described in [inbound authentication](auth.md#native-browser-accounts). They do not control collaboration sessions. They require the engine's normal API authentication and a nondelegated human operator with settings authority. Host/origin checks still apply. All responses use `Cache-Control: no-store` and `Referrer-Policy: no-referrer`.

## Status

`GET /api/account` returns:

```json
{
  "available": true,
  "canStartSignIn": true,
  "signInUnavailableReason": null,
  "signedIn": true,
  "principalRevision": "non-secret-browser-binding-revision",
  "profile": {
    "displayName": "Example Person",
    "email": "person@example.test"
  },
  "signIn": {"state": "idle"}
}
```

`available` is false when hosted account services are not configured. `canStartSignIn` describes this request's ability to start local sign-in; the app must not infer it from its own URL. When false, `signInUnavailableReason` is `account-unavailable` or `account-sign-in-requires-engine-machine`; otherwise the reason is null. `signedIn` describes the native account session, not whether a remote-node project is selected or usable. `profile` is null when signed out or when an existing session predates profile consent. The only profile fields are `displayName` and `email`: trimmed text or null, with blank/nontext values becoming null. Neither field is synthesized from the other. No picture URL is returned and no third-party avatar request is needed. No identity response is passed through: tokens, refresh tokens, ID tokens, secret state, PKCE verifiers, session paths and session file contents are never returned.

`principalRevision` identifies this request's browser-bound identity without conferring authority; it is `local` when unbound. Account-bound caches and pending operations must be invalidated when it changes. A different browser polling status does not acquire the install's account principal.

`signIn.state` is `idle`, `pending` or `failed`. A failed attempt includes the stable `error` code `account-sign-in-failed`, without provider errors or callback contents. Pending status does not block behind the native callback listener. A status refresh failure returns503 `{"error":"account-status-unavailable"}` rather than claiming the user is signed out.

## Project links

`GET /api/account/projects` lists the signed-in account's choices as
`{issuer, projects: [{id, displayName}]}`. The issuer is the configured identity
service, IDs are UUIDs, and display names are trimmed text or null. No other
identity project fields are returned. A service/session failure returns503
`account-projects-unavailable`; an unconfigured account returns409
`account-unavailable`.

`PUT /api/account/projects/{project}` links an existing library project using
`{"revision": 1, "accountProject": "account-project-uuid"}`. The revision is the
library project's current revision. The selected account project must appear in
the signed-in account's list. The library stores only `{issuer, id}` in
`accountProject`, and the response contains the updated library project. A null
`accountProject` removes the link even when signed out or account services are
unconfigured. Links are metadata, not OAuth credentials or cloud synchronization;
saving one does not mint a bearer. Account-backed serving selects gateway
credentials from the actual run's library-project link, never the install-wide
selection. A link change applies to subsequent runs; an active run keeps its
original account project and cannot retarget. Unlinked projects retain local
execution but have no account-backed remote authority. Explicit external-token
overrides remain independent of project links.
The CLI provides the same validation through `account link-project UUID` and
`account unlink-project`; both require `--library-root PATH --project ID --revision N`
for an existing library project at its current revision. Unlinking needs no
signed-in account. Neither command creates projects or changes another project's
link. At engine startup, a legacy install-wide selection migrates only when the
saved environment identity and library binding match the running host and exactly
one nondeleted project belongs to that environment. Existing links are never
overwritten. Unknown or shared ownership leaves the account choice in private
migration metadata without linking any project. The old bearer is removed and
never reused. Interrupted writes resume only for their original project and
revision; later unlink, deletion or environment changes cannot retarget migration.

An unavailable/deleted library project returns404 `project-unavailable`, a stale
revision returns409 `project-changed`, an unlisted account project returns400
`account-project-unavailable`, and invalid request fields return400
`invalid-account-request`. A helper save failure returns `account-project-save-failed`
with400/404/409 for invalid/missing/stale records, otherwise503. An unavailable
library returns503 `project-library-unavailable`. Both routes require the same
nondelegated human operator as sign-in and sign-out. While sign-in is pending,
they return409 `account-sign-in-pending` without waiting for browser consent.
The generic project patch route refuses `accountProject` with400
`use-account-project-link-route`, so asset-edit authority cannot bypass account
selection validation.

## Start and cancel

`POST /api/account/sign-in` with `{}` returns202 with the status fields above and an `authorizationUrl`. Open that URL in a browser on the engine's own machine, then poll status. It uses the existing native loopback PKCE listener, issuer/state checks and one-use code exchange; no engine HTTP callback bypass exists. Account consent requests `openid profile email projects:gateway offline_access`; profile is read from the configured identity service's authenticated `/userinfo`, never by decoding an unverified token.

Only same-machine loopback requests may start this flow. Other hosts/peers return409 `{"error":"account-sign-in-requires-engine-machine"}`. A browser running on a different machine must start sign-in on the engine machine until the paired browser authentication bridge is available. Proxy headers do not grant loopback status.

There is one active attempt per install API. Repeating start with the initiating browser cookie and operator authority returns the same pending authorization URL. An explicitly authenticated non-local initiator can also control its attempt with the same credentials. An unbound browser sharing the implicit local principal cannot retrieve or cancel another browser's attempt. Another principal cannot replace or cancel that attempt and receives403 `account-sign-in-owner-required`. The attempt expires after180 seconds; no automatic token-exchange retry occurs. Missing service configuration returns409 `account-unavailable`.

`DELETE /api/account/sign-in` cancels the caller's pending attempt and returns status. It closes the native listener and fences a late exchange from writing a session. Cancellation does not remove an older signed-in session. If completion won the race, status reports the completed session; use sign-out to clear it. An authenticated remote browser sees the same pending state and can cancel as the same principal; another principal cannot cancel it. The account belongs to the engine serving these routes, not a tab's selected execution backend.

## Sign out

`POST /api/account/sign-out` with `{}` cancels the caller's pending attempt, removes the existing local account session and gateway bearer, then returns signed-out status. This is install-local sign-out, not identity-provider global logout or revocation. Failure returns503 `account-sign-out-failed`, without filesystem paths. Malformed/nonempty start or sign-out bodies return400 `invalid-account-request`.

Sign-out removes active run-token files and stops their refresh. It does not cancel engine runs, send a gateway job cancellation, or revoke an already issued bearer. An account-backed remote operation requires the run's current token file, so a later upload, submission, poll or cancellation is refused after sign-out; an HTTP request already in flight may still complete. New account-backed runs require sign-in again. Explicit external token overrides are not account sessions and are not removed by these routes.

A missing/invalid engine credential returns401 through normal authentication. Agents and nonoperator callers receive403 `operator-required` after authentication; middleware capability/delegation errors retain their normal codes. Service/network/exchange failures do not reveal identity bodies, callbacks or credentials in API errors.
