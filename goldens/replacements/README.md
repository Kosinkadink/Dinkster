# Replacement-rule golden fixtures

Encoder-authored wire fixtures for the schema wire's lifecycle metadata
(deprecation, searchVisibility, replacement rules), generated through the
REAL `schema_to_wire` / `rule_to_wire` - never hand-written. Dinkster-Frontend
consumes these to lock its decoder against encoder drift; the backend test
`tests/test_goldens.py` regenerates them in-memory on every run and fails if
the committed files differ from what the current encoders produce.

Regenerate after intentional encoder changes:

    uv run python scripts/generate_replacement_goldens.py

Files:

- `vocabulary.json` - one predecessor/successor/alternate trio whose rule
  exercises every union member of the closed vocabulary: all 7 predicate
  kinds (with deep nesting), all 4 mapping kinds, both transforms
  (enumRename, scale with and without offset), multi-successor fan-out
  under guards, and the unconditional fallback case.
- `chain.json` - a replacement chain: the A->B rule rides B's schema and
  the B->C rule rides C's, each hop carrying its own value transform, with
  deprecation pointers (A->B->C) and searchVisibility (deprecated, hidden)
  on the intermediate hops - deprecation + searchVisibility + replacements
  on one schema (B).

Each file is `{"schemas": [schema_wire, ...]}` in dependency-free order.
Both fixture sets validate clean under `validate_replacement_references`.
