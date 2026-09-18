"""Pack tests: plain calls on plain values.

Nodes are classmethods over plain Python values - no engine, no server,
no fixtures. report_* calls are silent no-ops outside an execution
context, so execute() is directly callable here.

The doctor gate is NOT a test import: `dinkster-doctor .` runs as its own CI
step (see .github/workflows/ci.yml). Importing the doctor here would mean
importing host machinery (dinkster_workers) from pack code - exactly what
doctor flags. Tests live in the pack directory, so they hold to the same
one-door rule: dinkster_api.v1 only."""

from __future__ import annotations

from dinkster_api.v1 import TypeRegistry

from my_pack_nodes import Shout, Tally, register_types


def test_shout() -> None:
    out = Shout.execute(text="hey", times=2)
    assert out == {"shouted": "HEY!HEY!"}


def test_shout_default_times() -> None:
    out = Shout.execute(text="hi")
    assert out == {"shouted": "HI!"}


def test_tally_counts_words() -> None:
    out = Tally.execute(text="a b a")
    assert out == {"tally": {"a": 2, "b": 1}}


def test_registered_type_round_trips() -> None:
    registry = TypeRegistry()
    register_types(registry)
    # The declared codec is the boundary contract: encode/decode must
    # round-trip, or values cannot cross processes or cache to disk.
    spec = registry.spec("my-pack.tally")
    counts = {"a": 2, "b": 1}
    assert spec.decode(spec.encode(counts)) == counts
    # And the default rendition is what frontends preview.
    rendition = registry.render(registry.wrap("my-pack.tally", counts))
    assert rendition.mime == "text/plain"
    assert b'"a": 2' in rendition.data
