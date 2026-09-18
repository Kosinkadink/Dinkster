# Frontend decision index

Dated index of cross-repo design decisions whose documents of record
live in Dinkster-Frontend/docs (user directive 2026-07-29: durably record
cross-repo decisions on the backend side; reference, do not duplicate).
Add a dated entry here whenever the frontend announces a decision that
affects cross-repo awareness, even when the backend contract is
unchanged.

## 2026-07-29

- **Wire-15 bypass/mute routing: Option A approved** -
  structural-interface-stratified routing; liveness derived from
  routing; one shared implementation; Option C (id/type matching) is
  fallback only. Interim: the frontend ships a fail-closed
  compile.wire15.modesUnsupported gate. Record:
  Dinkster-Frontend docs/wire15-bypass-mute-routing-decision.md. Backend
  wire contract unchanged.
- **Regions frontend program of record** -
  Dinkster-Frontend docs/regions-design.md. R1 core landed:
  occurrence-local region block, strict v1 shape,
  compile.region.loweringRequired gate until R3. Backend wire contract
  unchanged.
- **Regions R1.5 contract migration (coordinator Oracle arbitration)** -
  the frontend replaces the R1 outputModes + same-id statePorts
  representation with outputRoles (distinct visible boundary output id
  + statePort indirection) before R2/R3. The backend same-key
  state-chain rule (model.py/validate.py) is unaffected: frontend
  lowering projects visibleOutputId -> statePort via one frozen alias
  map, so the emitted wire stays exactly the existing contract. No
  backend action.
- **Subgraph lifecycle program approved** -
  Dinkster-Frontend docs/subgraph-lifecycle-design.md
  (create/extract/flatten, slices L1-L7); lands before regions R2.
  Frontend-internal; indexed because regions R2/R3 sequencing depends
  on it.
