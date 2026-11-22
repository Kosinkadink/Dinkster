# Native account API

These routes control the native install's `AccountSession`, not collaboration sessions or engine access credentials. They require the engine's normal API authentication and a nondelegated human operator with settings authority. Host/origin checks still apply. All responses use `Cache-Control: no-store` and `Referrer-Policy: no-referrer`.

## Status

`GET /api/account` returns:

```json
{
  "available": true,
  "signedIn": true,
  "profile": {
    "displayName": "Example Person",
    "email": "person@example.test",
    "avatarUrl": null
  },
  "signIn": {"state": "idle"}
}
```

`available` is false when hosted account services are not configured. `signedIn` describes the native account session, not whether a remote-node project is selected or usable. `profile` is null when signed out or when an existing session predates profile consent; sign in again to grant that consent. Each projected profile field can be null. `avatarUrl` is null: provider-hosted pictures are not same-origin app images, so the account menu uses its initial fallback without making a third-party request. No identity response is passed through: tokens, refresh tokens, ID tokens, secret state, PKCE verifiers, session paths and session file contents are never returned.

`signIn.state` is `idle`, `pending` or `failed`. A failed attempt includes the stable `error` code `account-sign-in-failed`, without provider errors or callback contents. Pending status does not block behind the native callback listener. A status refresh failure returns503 `{"error":"account-status-unavailable"}` rather than claiming the user is signed out.

## Start and cancel

`POST /api/account/sign-in` with `{}` returns202 with the status fields above and an `authorizationUrl`. Open that URL in a browser on the engine's own machine, then poll status. It uses the existing native loopback PKCE listener, issuer/state checks and one-use code exchange; no engine HTTP callback bypass exists. Account consent requests `openid profile email projects:gateway offline_access`; profile is read from the configured identity service's authenticated `/userinfo`, never by decoding an unverified token.

Only same-machine loopback requests may start this flow. Other hosts/peers return409 `{"error":"account-sign-in-requires-engine-machine"}`. A browser running on a different machine must start sign-in on the engine machine until the paired browser authentication bridge is available. Proxy headers do not grant loopback status.

There is one active attempt per install API. Repeating start as the same engine principal returns the same pending authorization URL. Another principal cannot replace or cancel that attempt and receives403 `account-sign-in-owner-required`. The attempt expires after180 seconds; no automatic token-exchange retry occurs. Missing service configuration returns409 `account-unavailable`.

`DELETE /api/account/sign-in` cancels the caller's pending attempt and returns status. It closes the native listener and fences a late exchange from writing a session. Cancellation does not remove an older signed-in session. If completion won the race, status reports the completed session; use sign-out to clear it.

## Sign out

`POST /api/account/sign-out` with `{}` cancels the caller's pending attempt, removes the existing local account session and gateway bearer, then returns signed-out status. This is install-local sign-out, not identity-provider global logout or revocation. Failure returns503 `account-sign-out-failed`, without filesystem paths. Malformed/nonempty start or sign-out bodies return400 `invalid-account-request`.

A missing/invalid engine credential returns401 through normal authentication. Agents and nonoperator callers receive403. Service/network/exchange failures do not reveal identity bodies, callbacks or credentials in API errors.
