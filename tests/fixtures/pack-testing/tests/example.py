"""Test the installed pack with a shared worker or the explicit in-process opt-out."""

import importlib.util
import os
from pathlib import Path

import pytest
from dinkster_engine.testing import load_pack


@pytest.fixture(scope="module")
def pack():
    spec = importlib.util.find_spec("example_tally")
    assert spec is not None and spec.origin is not None
    manifest = Path(spec.origin).with_name("dinkster-pack.toml")
    with load_pack(manifest, in_process=os.environ.get("PACK_TEST_IN_PROCESS") == "1") as loaded:
        yield loaded


def test_graph_values_execution_and_cache(pack):
    graph = {
        "shout": ("example-tally.shout", {"text": "a b a c a b"}),
        "tally": ("example-tally.tally", {"text": pack.link("shout", "text")}),
    }
    first = pack.run(graph)
    assert first.value("shout", "text") == "A B A C A B"
    assert first.value("tally", "counts") == {"A": 3, "B": 2, "C": 1}
    assert first.executed == {"shout", "tally"}
    assert first.cached == set()
    assert (first.value("tally", "pid") == os.getpid()) is (
        os.environ.get("PACK_TEST_IN_PROCESS") == "1"
    )
    again = pack.run(graph)
    assert again.outputs == first.outputs
    assert again.executed == set()
    assert again.cached == {"shout", "tally"}


def test_one_node_and_changed_inputs(pack):
    result = pack.run_node("example-tally.tally", {"text": "lower lower UPPER"})
    assert result.value("node", "counts") == {"lower": 2, "UPPER": 1}
    assert result.executed == {"node"}
    assert result.cached == set()


def test_single_node_values(pack):
    result = pack.run_node("example-tally.shout", {"text": "leave  spacing"})
    assert result.value("node", "text") == "LEAVE  SPACING"
