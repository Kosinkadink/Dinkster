# Adding a model family

A model family is registration data plus torch-worker adapters. It does not
own sampling orchestration or require a family-id branch in shared engine or
node code.

## Checklist

1. Register a `ModelFamily` with header-only detection evidence, explicit
   specificity, latent and sampling descriptors, component wiring, supported
   dtypes, engine properties, and dotted-path worker callables.
2. Register each loadable component with its geometry planner and dotted-path
   torch-worker loader.
3. Register one assembly planner and dotted-path loader when the family is
   loadable as a complete checkpoint or split component set.
4. Implement a Denoiser adapter for `sampling_execution`. The adapter may
   validate and prepare family conditioning and evaluate the network. Schedule
   construction, noise, guidance, solvers, masks, previews, cancellation, and
   state reporting stay in the shared engine.
5. Register text encoding and latent codecs as dotted-path worker callables
   when the family provides them. Do not add family selection branches to
   shared nodes.
6. Test header detection, registration collisions, malformed dotted paths,
   loader resolution, and identical seeded output through custom sampling and
   KSampler composition on CPU.
7. Return the family, components, and assemblies from one
   `InferenceContribution`, and declare the `model-family-registration`
   capability in the pack manifest.
8. Run `tests/test_family_registration_gates.py`. A family id in a shared
   engine or node policy branch is a failure. The extension-factory allowlist
   ceiling must not increase.

The proof in `packages/dinkster-inference-torch/tests/test_new_family_checklist.py`
materializes the ordinary pack fixture under `tests/fixtures/extension-contract-pack`,
detects its toy family through the merged worker registry, and runs its denoiser
through both sampler surfaces.
