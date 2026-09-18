# Dinkster acceptance

This package contains the installable ModelSamplingFlux remote-worker acceptance node,
its pinned ComfyUI golden, and the controller that executes it through `RemoteWorker`
and `Engine`. It does not import repository test modules or pytest.

Install the worker closure with the `cloud` extra, then resolve the packaged manifest:

```sh
uv sync --package dinkster-acceptance --extra cloud --no-dev --locked
uv run --no-sync dinkster-acceptance-import-check
manifest=$(uv run --no-sync dinkster-acceptance-manifest)
DINKSTER_ACCEPTANCE_COMMIT=$(git rev-parse HEAD) DINKSTER_ACCEPTANCE_DEVICE=cuda:0 \
  uv run --no-sync python -m dinkster_workers.service \
  --listen 0.0.0.0:52146 --token-file remote-token.txt --manifest "$manifest"
```

Before creating a paid pod, run `scripts/prepare_cloud_acceptance.py` at the exact
candidate commit. It installs the closure into a clean local environment, rejects any
test-only import, and emits the commit-pinned source archive only after that check passes.
Run the same archive and commands on the LAN worker before cloud validation. The paid-pod
runbook must also stop before pod creation unless comfy-runner's
`is_tailscale_configured()` returns true and the controller host's tailnet FQDN resolves.
These checks keep an otherwise unreachable pod from starting billing.
