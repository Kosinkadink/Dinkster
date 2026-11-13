"""The recorded Generic Loops corpus runs through native loop expansion only."""

from __future__ import annotations

import json
from pathlib import Path

from dinkster_nodes_dev.loop_corpus import Observations

from tests.async_scenarios import run_scenario
from tools.run_comfy_loop_corpus import compile_prompt, make_engine

# Recorded issue408 prompts and observations; no sibling evidence checkout is required.
PROMPTS = json.loads(
    (Path(__file__).parent / "fixtures/loop-corpus-prompts.json").read_text(encoding="utf-8")
)


def test_native_loop_compiler_runs_carried_value_fixtures() -> None:
    async def scenario() -> None:
        for name in ("carry-only", "heterogeneous-carry", "list-backed-carry", "for-state"):
            issue408 = PROMPTS[name]
            Observations.reset()
            result = await make_engine().run(
                compile_prompt(issue408["api_prompt"]),
                issue408["outputs"],
                cache_enabled=False,
            )
            assert not [item for item in result.diagnostics if item.severity == "error"]
            assert {key: Observations.values[key] for key in issue408["observed"]} == issue408[
                "observed"
            ]

    run_scenario(scenario())
