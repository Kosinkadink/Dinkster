# Registry client operations

## Admission callers

Archive checks do not establish registry ownership, namespace grants, catalog
versions, dependency availability or publication state. The service caller must
check those separately. A queued submission or review finding is not acceptance;
wait for admission before treating a version as published.

Keep full Linux probe isolation for submitted code. `require_isolation=False`
is only for controlled tests, not a recovery path for an unavailable jail.
Keep bounded probe tails as operator data, not public findings.

For a staged dependency index, supply the complete compatible CPU wheel set,
including transitive dependencies. Select index mode explicitly when that index
must also supply torch and torchvision; do not rely on public-index fallback.
Local `dependency_wheels` must be selected and verified by the service caller,
never accepted as filesystem paths from submitted manifests.

Official pack sources, native locks and publication belong to
[dinkster-nodes](https://github.com/Kosinkadink/dinkster-nodes), not the engine
repository. Follow its trusted-main release procedure. For publisher setup and
submission recovery, see [submit a node pack](registry-submission.md).

Before serving a template body, call `ReleaseTemplate.verify_body()`. Consumers
must reject unknown catalog versions and treat versioned template bodies and
thumbnails as immutable; cached browse metadata is not a substitute for body
verification.

## Prepare source inputs

Run preparation with the target engine's interpreter. From a contributor checkout,
use its selected runtime:

```sh
python scripts/develop.py run -- python -m dinkster.manager --registry https://registry.dinkster.org prepare-registry-sources --catalog-lock /scratch/dinkster.lock --artifacts /scratch/artifacts
```

Use owned output paths. Source preparation is not dependency-lock preparation or
snapshot activation; follow [installation](install.md) for those separate steps.
For an engine update, use `--installed-lock /retained/dinkster.lock` rather than
fresh `--minimum` selections, and choose a candidate output different from the
retained input. Do not use `--allow-missing` to drop unavailable installed packs.
Official-pack source attribution is not authenticated publisher authority.
See [environment updates](environments-and-updates.md#versions-and-engine-compatibility).

## Validate default pack admission

From the repository root, select the target engine version explicitly:

```sh
python scripts/develop.py run -- python -m scripts.check_default_pack_admission --engine-version VERSION
```

Prepare the full Linux probe jail and satisfying pack dependencies before this
check. A review-required scan is not a passing admission. This validation is not
publication; official pack submission remains in the pack repository.

## Reference

- [Probe configuration and stages](../packages/dinkster-registry/src/dinkster_registry/archive_admission.py)
- [Registry resolution](../src/dinkster/registries.py) and [manager commands](../src/dinkster/manager.py)
- [Release templates](../packages/dinkster-registry/src/dinkster_registry/model.py)
