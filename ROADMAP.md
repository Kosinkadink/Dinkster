# Roadmap

Dinkster has working backend composition, isolated and grouped extension hosting,
model execution through `dinkster_comfy`, governed memory, typed assets, and
versioned schema contracts. The next work is to finish the extension and asset
surfaces already exposed by those foundations and close the remaining execution,
deployment, and operations gaps without weakening compatibility.

## Next

- Complete remote COMBO consumption in the frontend, then define authority, scope, capability, proxying, and cache rules before allowing dynamic or tenant-sensitive providers.
- Enable preview and partial-execution service wiring when deployment is authorized; add visible frontend actions and end-to-end UI proof separately.
- Continue extension composition beyond samplers, patches, and guidance: conditioning, capability enforcement, additional runtime registries, and complete sampling execution context.
- Finish source-filename upload binding and presentation work, including explicit upload authority and the remaining frontend integration.
- Continue federated assets with existing-only resolution and repair suggestions, remote catalog UI, richer metadata and family services, and interruptible scan, hash, and acquisition jobs.
- Add remote or gated asset libraries, authenticated model sources, output provenance embedding, and history-to-asset garbage-collection correlation.
- Finish in-process model-pack serve wiring and add safe reload support only
  after its lifecycle contracts are settled.
- Expand generation schemas and workflow translation without introducing a
  second model-loading or sampling implementation outside `dinkster_comfy`.
- Complete compat coverage for hidden inputs, dynamic schemas, lazy and accept-all behavior, and nested dynamics where ecosystem evidence requires them.
- Add generation-bound worker leases for reloads that must preserve inflight work, plus zero-downtime engine swaps when multi-install operation requires them.
- Continue partner API nodes as an independently updatable pack, then move provider-owned definitions and remote execution behind explicit transport and trust-policy contracts.
- Finish registry and distribution gaps: public doctor reporting, reusable CI integration, marker-aware lock resolution, non-Linux probe isolation, and namespace policy.
- Expand identity and tenancy with users, organizations, projects, persisted scoped settings, inbound OAuth/OIDC, per-asset ACLs, and usage/audit metering.
- Complete fleet ingress durability, mixed-version downgrade support, and collaboration ordering, reconnect, and shared-document semantics as those deployment modes become active.
- Add non-CUDA telemetry and residency only when those devices are targeted, and expose fault, offload, and memory-pressure counters for operator and frontend diagnostics.
- Add supervisor UI and stable third-party API documentation, custom and context-aware error interpretation, and repeatable backend performance records as those surfaces gain consumers.
- Establish the multi-GPU training program after native forward coverage is broad enough, while retaining governed optimizer, gradient, checkpoint, and resumable-job seams.
