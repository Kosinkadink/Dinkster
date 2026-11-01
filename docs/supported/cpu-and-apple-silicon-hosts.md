## CPU and Apple Silicon hosts

- Apple Silicon on macOS 14+ uses the `mac-arm64` Python/torch bundle with
  MPS for contributor and published environment setup. Native tools provide
  their own Python and uv for setup, doctor, engine start, update and rollback.
  Fresh `dinkster-env setup` selects `mac-arm64` automatically on Apple Silicon;
  explicit variants and existing project selections take precedence.
  Intel macOS and CUDA on macOS are unsupported.
- The Apple Silicon Desktop app is unsigned. Gatekeeper may require
  right-click Open or removal of the quarantine attribute from the verified
  app; [Apple signing and notarization](https://github.com/Kosinkadink/comfy-vibe-station/issues/501)
  are separate from tools and engine support.
- GPU-less compatibility workers use CPU automatically.
  `DINKSTER_ACCELERATOR=cpu` also forces CPU on GPU-equipped workers without
  changing the engine or transport.
- Foundation, media-I/O, image, and retained mesh operations run on CPU when
  their dependencies support it.
- Model-family sampling on CPU or Apple Silicon is unsupported until the
  corresponding `dinkster_inference` runtime path has current execution evidence.
