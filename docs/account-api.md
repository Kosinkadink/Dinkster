# Native account API

These routes control the native install's `AccountSession`, not collaboration sessions or engine access credentials. They require the engine's normal API authentication and a nondelegated human operator with settings authority. Host/origin checks still apply. All responses use `Cache-Control: no-store` and `Referrer-Policy: no-referrer`.

## Status

`GET /api/account` returns:

```json
{
  "available": true,
  "canStartSignIn": true,
  "signInUnavailableReason": null,
  "signedIn": true,
  "profile": {
    "displayName": "Example Person",
    "email": "person@example.test"
  },
  "signIn": {"state": "idle"}
}
```

`available` is false when hosted account services are not configured. `canStartSignIn` describes this request's ability to start local sign-in; the app must not infer it from its own URL. When false, `signInUnavailableReason` is `account-unavailable` or `account-sign-in-requires-engine-machine`; otherwise the reason is null. `signedIn` describes the native account session, not whether a remote-node project is selected or usable. `profile` is null when signed out or when an existing session predates profile consent. The only profile fields are `displayName` and `email`: trimmed text or null, with blank/nontext values becoming null. Neither field is synthesized from the other. No picture URL is returned and no third-party avatar request is needed. No identity response is passed through: tokens, refresh tokens, ID tokens, secret state, PKCE verifiers, session paths and session file contents are never returned.

`signIn.state` is `idle`, `pending` or `failed`. A failed attempt includes the stable `error` code `account-sign-in-failed`, without provider errors or callback contents. Pending status does not block behind the native callback listener. A status refresh failure returns503 `{"error":"account-status-unavailable"}` rather than claiming the user is signed out.

## Start and cancel

`POST /api/account/sign-in` with `{}` returns202 with the status fields above and an `authorizationUrl`. Open that URL in a browser on the engine's own machine, then poll status. It uses the existing native loopback PKCE listener, issuer/state checks and one-use code exchange; no engine HTTP callback bypass exists. Account consent requests `openid profile email projects:gateway offline_access`; profile is read from the configured identity service's authenticated `/userinfo`, never by decoding an unverified token.

Only same-machine loopback requests may start this flow. Other hosts/peers return409 `{"error":"account-sign-in-requires-engine-machine"}`. A browser running on a different machine must start sign-in on the engine machine until the paired browser authentication bridge is available. Proxy headers do not grant loopback status.

There is one active attempt per install API. Repeating start as the same engine principal returns the same pending authorization URL. Another principal cannot replace or cancel that attempt and receives403 `account-sign-in-owner-required`. The attempt expires after180 seconds; no automatic token-exchange retry occurs. Missing service configuration returns409 `account-unavailable`.

`DELETE /api/account/sign-in` cancels the caller's pending attempt and returns status. It closes the native listener and fences a late exchange from writing a session. Cancellation does not remove an older signed-in session. If completion won the race, status reports the completed session; use sign-out to clear it. An authenticated remote browser sees the same pending state and can cancel as the same principal; another principal cannot cancel it. The account belongs to the engine serving these routes, not a tab's selected execution backend.

## Sign out

`POST /api/account/sign-out` with `{}` cancels the caller's pending attempt, removes the existing local account session and gateway bearer, then returns signed-out status. This is install-local sign-out, not identity-provider global logout or revocation. Failure returns503 `account-sign-out-failed`, without filesystem paths. Malformed/nonempty start or sign-out bodies return400 `invalid-account-request`.

Sign-out does not cancel engine runs, send a gateway job cancellation, or revoke an already issued bearer. A remote invocation that already loaded a bearer can keep using it until it expires or the gateway refuses it; completion is not guaranteed. New invocations using the removed account token file require sign-in again. Explicit external token overrides are not account sessions and are not removed by these routes.

A missing/invalid engine credential returns401 through normal authentication. Agents and nonoperator callers receive403 `operator-required` after authentication; middleware capability/delegation errors retain their normal codes. Service/network/exchange failures do not reveal identity bodies, callbacks or credentials in API errors.
