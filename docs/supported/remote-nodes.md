## Remote nodes

- Default-installed `dinkster.remote.*` nodes are discovered from the Dinkster
  catalog and execute through its authenticated job gateway. Asset inputs,
  progress, previews, cancellation, and verified asset outputs are supported.
  The last valid public catalog remains available during startup outages, and live
  catalog epoch changes replace the remote node surface atomically.
- The default remote pack requests its supported schema wire and uses the
  current session bearer for catalog discovery. Authenticated schemas stay
  memory-only; public cache is isolated by catalog endpoint. Session changes
  clear prior schemas, and signed-in policy changes can reload nodes without
  changing the public catalog epoch.
- The initial catalog provides `dinkster.remote.nanobanana.image` for image
  generation and editing and `dinkster.remote.seedance.video` for Seedance 2.5
  text-to-video and first-frame image-to-video.
- Engine serving resolves account, catalog, gateway and gateway-audience defaults
  from one production service-address table. Release channels do not select
  services, and service selection does not change project metadata. Explicit
  `--remote-catalog-base` and `--remote-gateway-base` values (or their
  `DINKSTER_REMOTE_*_BASE` variables) remain custom overrides; an explicitly empty
  catalog base disables discovery. Without a separate gateway override, a custom
  catalog also supplies the job endpoint.
- `dinkster account login` opens native PKCE consent in the system browser;
  `account projects` lists accessible projects, and `account select-project UUID`
  requests gateway authority for exactly that project. `account refresh` refreshes
  OAuth if needed and requests a fresh gateway token; `account logout` removes
  this install's local credentials. The identity service must support the
  `projects:gateway` consent scope and project discovery/mint routes.
- Account commands accept `--root` for the install and explicit custom service
  bases. Credentials are kept outside projects, isolated by install path and
  identity/catalog/gateway/audience. OAuth secrets are separate from the
  gateway-only token file. Remote workers never receive OAuth refresh secrets.
- Serving uses the selected install account by default, refreshes its gateway
  token before remote pack composition and before expiry, and keeps explicit
  `--remote-auth-token-file` overrides. Managed generations use their install
  root; direct serving accepts `--account-root` (`DINKSTER_ACCOUNT_ROOT`) and
  `--account-identity-base` (`DINKSTER_ACCOUNT_IDENTITY_BASE`) for custom services.
  Sandboxed account-backed workers see token replacements without access to
  the OAuth session. Explicit token-file replacement and removal also reach
  sandboxed workers without a restart, exposing neighboring files, or falling
  back to the account; the source is checked every second.
  A failed or ambiguous OAuth refresh requires sign-in again;
  no refresh exchange is automatically replayed.
