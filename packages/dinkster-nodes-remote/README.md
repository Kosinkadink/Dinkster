# dinkster-nodes-remote

Default-installed client pack for catalog-driven remote nodes. It loads the
current `NodeSchema` catalog, exposes valid `dinkster.remote.*` nodes, uploads
asset inputs, submits authenticated jobs, forwards progress and previews, and
verifies downloaded output digests before adding them to Dinkster's asset vault.

The last valid catalog is cached under the pack scratch directory. Startup
continues with that cache, or an empty remote pack, if the catalog is
unavailable. Catalog epoch changes replace the isolated worker atomically.
Malformed catalog entries are skipped without hiding valid entries.

`dinkster-serve` accepts:

```text
--remote-catalog-base URL
--remote-gateway-base URL
--remote-auth-token-file PATH
--remote-catalog-poll-interval SECONDS
```

The equivalent environment variables are `DINKSTER_REMOTE_CATALOG_BASE`,
`DINKSTER_REMOTE_GATEWAY_BASE`, `DINKSTER_REMOTE_AUTH_TOKEN_FILE`, and
`DINKSTER_REMOTE_CATALOG_POLL_INTERVAL`. The gateway base defaults to the catalog
base. No network request is made when the catalog base is unset. Job polling
defaults to one second for image nodes and five seconds for
video nodes; `DINKSTER_REMOTE_IMAGE_POLL_INTERVAL` and
`DINKSTER_REMOTE_VIDEO_POLL_INTERVAL` override those intervals. The bearer token
file is read for each invocation so session rotation does not require
restarting the pack. Without a token, schemas remain visible and invocation
fails with a sign-in error.
