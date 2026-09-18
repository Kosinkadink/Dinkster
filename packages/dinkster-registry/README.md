# dinkster-registry

`dinkster-registry` is the pure registry and install model: identities,
publisher memberships, namespace grants, immutable releases, review
lifecycle, artifacts, lockfiles, and deterministic admission. It contains no
server, storage, network, or clock, so CI, publishers, and services can apply
the same invariants on either side of the trust boundary.

## Setup

This package is a uv workspace member and is not published separately yet.
From the repository root:

```sh
uv sync --all-packages
```

It has no console script. `uv run dinkster-registry` belongs to the umbrella
package and runs the durable registry service that composes this model.

## Use

The central publish predicate is a pure function:

```python
from dinkster_registry import DoctorEvidence, GrantTable, ReleaseIndex, Submission, admit

evidence = DoctorEvidence.from_report_json(report_json)
submission = Submission(
    publisher="example-org",
    pack_name="example-pack",
    namespaces=("example",),
    version="1.0.0",
    artifact_digest="sha256:" + "0" * 64,
    evidence=evidence,
)
verdict = admit(submission, GrantTable(), ReleaseIndex())
```

`DoctorEvidence` consumes the doctor's versioned JSON report. It understands
`reportVersion` 1 and refuses any other version rather than guessing.
`GrantTable` enforces namespace ownership, `ReleaseIndex` enforces immutable
pack versions, and `ReviewLog` records actor-attributed review transitions.
Artifact helpers build, verify, and unpack deterministic archives; install
models provide lockfiles, plans, and generation-based activation.

The caller must obtain doctor evidence from the artifact being admitted.
Publisher claims alone are not evidence.

## Learn more

See the "Design contract" section of `../dinkster-registry-service/README.md`
for the distribution design (execution-based indexing, doctor as the shared
admission predicate). Focused coverage is in `tests/test_registry.py`,
`tests/test_registry_principals.py`, `tests/test_artifact.py`,
`tests/test_install.py`, and `tests/test_installer.py`.
