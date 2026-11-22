## Remote nodes

- Default-installed `dinkster.remote.*` nodes are discovered from the Dinkster
  catalog and execute through its authenticated job gateway. Asset inputs,
  progress, previews, cancellation, and verified asset outputs are supported.
  With a matching gateway environment document, the last valid public catalog
  remains available during catalog outages. A missing or mismatched environment
  document disables remote nodes, including cached nodes. Live catalog epoch
  changes replace the remote node surface atomically; startup uses live remote
  discovery rather than precomputed pack schemas.
- The default remote pack requests its supported schema wire and uses the
  current session bearer for catalog discovery. Authenticated schemas stay
  memory-only; public cache is isolated by catalog endpoint. Session changes
  clear prior schemas, and signed-in policy changes can reload nodes without
  changing the public catalog epoch.
- The initial catalog provides `dinkster.remote.nanobanana.image` for image
  generation and editing and `dinkster.remote.seedance.video` for Seedance 2.5
  text-to-video and first-frame image-to-video.
- Engine serving resolves account, catalog, gateway and gateway-audience defaults
  from one production/QA service-address table. Production is the default for
  both release channels. `--services qa` or `DINKSTER_SERVICES=qa` selects QA for
  the process only; the flag takes precedence, and other values refuse startup.
  Service selection does not change project metadata. The startup log and
  `/api/nodes` header report `services.name`, `services.identityBase`, and
  `services.gatewayBase`. `DINKSTER_REMOTE_PARTNER_MODE` and
  `DINKSTER_REMOTE_FAKE_OUTCOME` are process-local remote-pack inputs, forwarded
  only to that pack without saving them in project metadata. Explicit
  `--remote-catalog-base` and `--remote-gateway-base` values (or their
  `DINKSTER_REMOTE_*_BASE` variables) remain custom overrides; an explicitly empty
  catalog base disables discovery. Without a separate gateway override, a custom
  catalog also supplies the job endpoint. Default-pack doctor and catalog
  preparation keep remote discovery disabled by default. An explicit
  `DINKSTER_SERVICES` selection supplies its hosted endpoints; an explicit empty
  catalog still disables discovery.
- `dinkster account login` opens native PKCE consent in the system browser;
  `account projects` lists accessible projects, and `account select-project UUID`
  requests gateway authority for exactly that project. `account refresh` refreshes
  OAuth if needed and requests a fresh gateway token; `account logout` removes
  this install's local credentials. The identity service must support the
  `projects:gateway` consent scope and project discovery/mint routes.
- The operator-only [account API](../account-api.md) reports sign-in status and
  consented name/email fields, starts or cancels native sign-in from a
  same-machine browser, and signs this install out without exposing credentials.
  Remote browsers must start sign-in on the engine's machine; the loopback
  callback is not a remote browser bridge.
- Account commands accept `--root` for the install and explicit custom service
  bases. Credentials are kept outside projects, isolated by install path and
  identity/catalog/gateway/audience. OAuth secrets are separate from the
  gateway-only token file. Remote workers never receive OAuth refresh secrets.
- Serving uses the selected install account by default, refreshes its gateway
  token before remote pack composition and before expiry, and keeps explicit
  `--remote-auth-token-file` overrides. Managed snapshots use their environment
  root; direct serving accepts `--account-root` (`DINKSTER_ACCOUNT_ROOT`) and
  `--account-identity-base` (`DINKSTER_ACCOUNT_IDENTITY_BASE`) for custom services.
  Sandboxed account-backed workers see token replacements without access to
  the OAuth session. Explicit token-file replacement and removal also reach
  sandboxed workers without a restart, exposing neighboring files, or falling
  back to the account; the source is checked every second.
  A failed or ambiguous OAuth refresh requires sign-in again;
  no refresh exchange is automatically replayed.
