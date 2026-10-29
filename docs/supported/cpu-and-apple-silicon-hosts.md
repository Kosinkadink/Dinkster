## CPU and Apple Silicon hosts

- GPU-less compatibility workers use CPU automatically. `DINKSTER_ACCELERATOR=cpu`
  also forces CPU on GPU-equipped workers without changing the engine or
  transport. The engine interpreter does not need PyTorch.
- Apple Silicon workers select MPS automatically when available. SD1.5,
  SDXL base, and Flux2 Klein 4B execute translated text-to-image workflows
  through `dinkster-serve` on Apple M4 with macOS 15.7.4 and PyTorch 2.13.0.
- MPS enrollment reports historical parity evidence without using the current
  chip family as an admission key. Recorded results do not verify a new run.
- Component placement accepts torch device types with a diagnostic outside
  CPU/CUDA; allocation and operation support depend on the installed backend.
- Foundation, media-I/O, and image packs execute translated image-only
  workflows on CPU and macOS. Model sampling uses the same native execution
  and worker boundaries as other devices, including decomposed custom sampling.
- Workers without attention capability evidence retain their existing or
  default route under explicit attention requests. Plans, events and durable
  node receipts report fallback reasons and routes without changing cache keys.
- See [worker setup](../serve-cli.md#cpu-and-apple-silicon-workers) and the
  [macOS test matrix](../macos-execution.md) for tested precision settings,
  artifacts, workflow results, and limits. Pack announcement alone is not
  evidence that every node or model in that pack has run on MPS.
