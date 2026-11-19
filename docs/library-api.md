# Library API versions

The [version register](library-api.toml) records each API's first Stable,
capabilities and adapter source. An empty `first-stable` means no Stable has
shipped that API. API 1 is current; there is no older library API to adapt.
The latest Stable, 0.0.16, predates the helper. Set `first-stable` to the exact
release version before tagging the first Stable that contains an API.

The Stable release builder rejects an engine whose `library-api` is absent
from the register. The helper contract tests compare the registered
capabilities with the real helper's greeting. Additive operations, optional
fields and capabilities keep the same API number. A breaking request or reply
shape requires a new number and an adapter over current storage for every
older served API. Keep each older Stable API's real engine fixture and run
its existing operations against the current helper, including revision
conflicts and preservation of fields the older engine does not know.
Storage format versions are separate and never determine API retirement.

## Removing an old API

Every API shipped in a Stable remains served until the user rules otherwise.
Age, a newer engine release and a lack of recent attachments are not permission
to remove it. Keep the user's ruling linked from the removed version's entry;
keep that entry and its first Stable in the register after retirement.

Before selecting tools that remove an API, compute the effect on each machine
using that machine's existing library. The helper's `compatibility.affected`
operation takes the replacement's `apiMin` and `apiMax`, plus optional `cursor`
and `limit`. It returns a paged union of recorded snapshots outside that range
and live projects whose environment and last snapshot refer to those engines.
It does not guess from a package version, inspect workflow documents or open
`library.sqlite`. A project bound to another snapshot and a deleted project
are not affected by this check.

The tools expose the same check through:

```sh
python -m dinkster_supervisor.environment --root PATH library-api-impact --api-min N --api-max M
```

`--json` returns the normal environment result with the notices in `plan`.
With no affected records, that list is empty and no notice is shown. With
records, show the plain-language by-the-way before selection, naming each
snapshot and project that will no longer open. Do not ask a storage question
or show API/format tables to the person. A failed check must not be treated as
an empty result. Keep the existing tools selected until the check succeeds.
The software update keeps its normal confirmation; this adds no approval.

The removal implementation must check the ruling and present this computed
list before replacing tools, and retain tests for a machine with no affected
records as well as one with affected snapshots and projects. This release
removes no API; installed tools still reject a replacement that loses their
served API range.
