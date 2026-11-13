# Submit a node pack

Start with the [node pack template](https://github.com/Kosinkadink/dinkster-pack-template)
and its rename instructions, or build the small pack below from an empty directory.
Only single-pack submission is available. A submission is not a publication until
the scanner accepts it and its release is `active`.

Use Python 3.12+, [uv](https://docs.astral.sh/uv/), and a POSIX shell.
Open [publisher accounts](https://id.dinkster.org/registry), sign in with your
Dinkster account, select your account or an organization you own or administer,
and choose **Claim publisher**. The name is your Dinkster handle or organization
slug, bound to its immutable id, not proof of a matching GitHub name. Existing
registry names and `dinkster` are reserved. A claim grants no official tier.
Choose an expiry at most 90 days away and your pack name for single-pack scope,
then **Create token**. Save its one-time value in a secret manager. Refresh lists
metadata only; **Revoke token** refuses subsequent publishing requests. No server
access or operator command is needed to register or obtain a publish credential.

For this example, use a **disposable registry**, its origin URL, and its identity
service's `/registry` page to obtain a token scoped to `maple-text`. Do not submit
example packs to production. The registry service and scanner must already run.
Set `DINKSTER_REGISTRY_URL` and `DINKSTER_REGISTRY_TOKEN` privately in your shell.
Never commit tokens, pass them as command arguments, or paste them into support reports.
The operator runs the service and its scanner; no private engine source is needed
to build or test your pack.

## Build and test

Run from an empty working directory. These commands create `maple-text/` and
build `maple_text-0.1.0-py3-none-any.whl` using published packages only.

<!-- registry-command: build -->
```sh
mkdir maple-text
cd maple-text
mkdir -p src/maple_text tests
cat > pyproject.toml <<'TOML'
[project]
name = "maple-text"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[build-system]
requires = ["hatchling==1.30.0"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/maple_text"]

[tool.hatch.build.targets.wheel.force-include]
"dinkster-pack.toml" = "maple_text/dinkster-pack.toml"

[dependency-groups]
dev = ["dinkster-api==0.0.2", "pytest>=8,<10"]
TOML
cat > dinkster-pack.toml <<'TOML'
[pack]
name = "maple-text"
version = "0.1.0"
engine = ">=0.0.2,<0.1"
namespaces = ["maple-text"]
requires = []

[pack.contracts]
host = "dinkster-pack-host/1"
api = "dinkster-api/v1"

[pack.entry]
nodes = "maple_text:NODES"

[pack.sandbox]
gpu = false
network = false
writable-mounts = false
TOML
cat > src/maple_text/__init__.py <<'PY'
from dinkster_api.v1 import InputSpec, Node, NodeSchema, OutputSpec, TypeExpr


class Shout(Node):
    @classmethod
    def define_schema(cls):
        return NodeSchema(
            node_type="maple-text.shout",
            display_name="Shout",
            inputs=(InputSpec("text", TypeExpr.concrete("STRING")),),
            outputs=(OutputSpec("shouted", TypeExpr.concrete("STRING")),),
        )

    @classmethod
    def execute(cls, *, text):
        return cls.outputs(shouted=text.upper() + "!")


NODES = [Shout]
PY
cat > tests/test_pack.py <<'PY'
from maple_text import Shout


def test_shout():
    assert Shout.execute(text="Hello there") == {"shouted": "HELLO THERE!"}
PY
uv sync --python 3.12
uv build --wheel
uv pip install --no-deps --reinstall dist/*.whl
uv run --no-project --python .venv python -m pytest -q
uv run --no-project --python .venv dinkster-doctor .
```

Expect one passing test and doctor `healthy: 0 error(s), 0 warning(s)`.
Stop on a failure. Test `execute` directly on plain values; run doctor separately,
not through private host imports in tests. Doctor also imports and checks node schemas.

### Manifest and wheel rules

For publication, provide `pack.name`, paired `version` and `engine`, the host/API
contracts and the entry declaring your nodes. The wheel's Name and Version must
match the submitted name/version and manifest. Its one top-level import package
is the pack name with `-` and `.` replaced by `_`; put the manifest and every
relative asset inside that package. Only pure `py3-none-any` wheels are admitted,
not sdists, legacy ZIPs, native binaries, startup `.pth` files or entry-point scripts.

Pack names are 1-64 lowercase ASCII characters, start with a letter, and have
alphanumeric segments separated by one `-`, `_` or `.`. Those separators are
equivalent for identity; use canonical hyphen spelling for hosted coordinates.
Versions are normalized public PEP 440, without a local `+suffix`.
Published bytes are immutable; changing code means choosing a new version.

`namespaces` defaults to your pack name. Each node id must be covered by a claim;
here `maple-text.shout` belongs to `maple-text`. Use your own namespace, not
reserved roots such as `dinkster`, `std`, `comfy`, `core` or `dev`.
Namespace claims do not grant ownership: the authenticated publisher and registry
grants decide it. Omit `publisher` to use token authority; if declared, it must match.

`requires` is for Python **libraries**, never other node packs. Match it to the
wheel's Requires-Dist with exact pins, for example `pillow==13.0.0` in both
manifest `requires` and project dependencies when you actually import Pillow.
The host supplies `dinkster_api.v1`. Shared helpers belong in library wheels.
Pack composition dependencies use `[pack.dependencies]` with a pack name and
version range instead; importing a dependency's public helpers requires that
declaration, and private imports are forbidden. Its satisfying release must
already be active. Dependencies cannot form a cycle.

For engine compatibility and pack dependency ranges, **the lower bound is the
first version that has what you import; raise it only on a new import**.
Do not copy the newest installed version into that bound. Keep the development
API pin current and rerun doctor, tests and admission when updating it.
See [pack authoring](pack-authoring.md) for optional docs, assets, presentation,
value-type entries, permissions, platform markers and accelerator requirements.
The sandbox table requests capabilities; it does not grant them.

## Submit and wait for activation

Continue inside `maple-text/`. This example deliberately refuses non-loopback
origins so it cannot publish a trial pack to a hosted registry. Real release
automation uses your registry's HTTPS origin and publisher token.

<!-- registry-command: submit -->
```sh
uv run --no-project --python .venv python - <<'PY'
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

base = os.environ["DINKSTER_REGISTRY_URL"].rstrip("/")
origin = urllib.parse.urlsplit(base)
if (origin.scheme != "http" or origin.hostname not in {"localhost", "127.0.0.1", "::1"}
        or origin.username or origin.password or origin.path or origin.query or origin.fragment):
    raise SystemExit("This example requires a disposable loopback HTTP registry")
token = os.environ["DINKSTER_REGISTRY_TOKEN"]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Use the registry's direct origin; redirects are refused")


opener = urllib.request.build_opener(NoRedirect())


def request(path, data=None):
    headers = {"Authorization": "Bearer " + token}
    if data is not None:
        headers["Content-Type"] = "application/octet-stream"
    with opener.open(urllib.request.Request(base + path, data=data, headers=headers), timeout=30) as r:
        return json.load(r)


candidate = request("/v1/publish/maple-text/versions/0.1.0",
                    Path("dist/maple_text-0.1.0-py3-none-any.whl").read_bytes())["candidateId"]
print("Candidate:", candidate, flush=True)
deadline = time.monotonic() + 300
while time.monotonic() < deadline:
    status = request("/v1/publish/" + candidate)
    print("Candidate state:", status["status"], flush=True)
    if status["status"] == "accepted":
        release = request("/v1/packs/maple-text/versions/0.1.0")
        assert release["state"] == "active", release["state"]
        assert [n["nodeType"] for n in release["nodes"]] == ["maple-text.shout"]
        print("Active: maple-text 0.1.0")
        break
    if status["status"] not in {"queued", "scanning", "promoting"}:
        raise SystemExit(json.dumps(status))
    time.sleep(1)
else:
    raise SystemExit("Timed out; inspect the candidate before retrying")
PY
```

POST returns HTTP 202 with `candidateId`; GET `/v1/publish/{candidateId}` reports
publisher-scoped status and findings. The usual path is `queued` -> `scanning` ->
`promoting` -> `accepted`, with the release then `active`. `review` needs an
operator decision; `rejected` is a failed admission. Neither is publicly listed.
Promotion retries can leave a candidate `promoting` until storage recovers.
An HTTP failure or timeout is not proof that submission failed: inspect the
candidate before retrying. Do not assume a repeated POST returns the same
candidate. A version already published cannot be replaced with different bytes.

## What the scanner checks

Checks run in order: wheel integrity (ZIP paths, size, METADATA/WHEEL/RECORD and
hashes), manifest and publisher/namespace/dependency rules, doctor's static import
checks, an isolated executable schema probe, then fast security scanning.
The probe has no network and read-only binds; it provisions declared CPU library
requirements from an index without source builds or startup `.pth` execution.
Its wall limit defaults to 30 seconds (operator-configurable, 1-600 seconds),
address-space limit to 512 MiB, and stdout/stderr limits to 2 MiB each.
Passing this probe does not validate your GPU execution or other accelerator environments.
Confirmed secrets or known-bad hashes block; some security findings require review.

Findings include a stable code and a message. Common exact messages are:

| Code | Message |
| --- | --- |
| `wheel.required` | `Only wheel artifacts are accepted.` |
| `wheel.distribution-name` | `Wheel Name must equal the submitted name.` |
| `wheel.entry-point-forbidden` | `Wheel entry points and scripts are forbidden.` |
| `manifest.publisher-mismatch` | `Manifest publisher differs from the authenticated publisher.` |
| `entry.unresolvable` | `The pack entry point could not be imported.` |
| `namespace.uncovered` | `A node type is outside the pack's declared namespaces.` |
| `probe.isolation-unavailable` | `A full Linux probe jail is unavailable.` |

Other messages include the offending path, requirement, claim or configured limit;
read the candidate's actual findings rather than matching prose fragments.
An isolation-unavailable finding is an operator setup problem, not permission
to bypass the scanner. Authentication errors occur before candidate creation.

## Withdraw a release

Ask the operator to yank a bad or superseded version with a reason. There is no
publisher HTTP yank endpoint. The operator runs the following with the service's
database/object-store settings; use only the disposable settings for this example.

<!-- registry-command: yank -->
```sh
dinkster-registry admin yank maple-text 0.1.0 --reason "Replaced by a corrected release."
```

Only an active release can be yanked. It disappears from install indexes, but
existing artifact downloads remain available and download tickets warn about the
yank. Publish a new version rather than replacing the yanked wheel. Public indexes
may be cached for 60 seconds. If index rebuilding fails after the database change,
retry the operator's index rebuild, not the yank.

For help, open a [Dinkster issue](https://github.com/Kosinkadink/Dinkster/issues)
with your pack name/version, candidate id, finding codes/messages and doctor output.
For a disputed or reserved publisher name, isolation or storage problems, contact
registry support. Create, replace and revoke your own publish tokens on the
identity service's publisher page. Include no token, environment dump or secret-bearing log.
