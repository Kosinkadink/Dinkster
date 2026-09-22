## Installed pack startup

- Installed pack schemas and static choices are available before pack workers
  start. Workers activate on demand, including in-process packs and explicitly
  configured shared-process groups.
- Pack install, update, and doctor refresh the persisted schema catalog.
  Serving fails before binding when an installed catalog is missing or stale.
- The default local launcher prepares missing or stale catalogs before binding,
  then serves the browser application and engine API from one loopback origin.
- Health reports composed and failed counts and is not ready when composition
  is empty or any pack failed.
- System generations retain an engine base and code manifest with their pack
  state. Staging does not activate an update; rollback selects a previously
  activated generation.
- Named projects have independent install roots and supervisor processes while
  sharing downloaded engine objects. Installation and engine garbage collection
  do not change project data roots.
