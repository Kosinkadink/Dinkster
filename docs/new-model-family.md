# Adding a model family

A model family is registration data plus torch-worker adapters. It does not
own sampling orchestration or require a family-id branch in shared engine or
node code.

## Checklist

1. Register a `ModelFamily` with header-only detection evidence, explicit
   specificity, latent and sampling descriptors, component wiring, supported
   dtypes, and engine properties.
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
7. Run `tests/test_family_registration_gates.py`. A family id in a shared
   engine policy branch is a failure. The extension-factory allowlist ceiling
   must not increase.

The smallest proof is
`packages/dinkster-inference-torch/tests/test_new_family_checklist.py`. Its
toy family has no production registration and runs only through the same
public registration and sampling contracts available to a real family.
