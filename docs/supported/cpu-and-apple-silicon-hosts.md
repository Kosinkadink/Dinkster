## CPU and Apple Silicon hosts

- GPU-less compatibility workers use CPU automatically.
  `DINKSTER_ACCELERATOR=cpu` also forces CPU on GPU-equipped workers without
  changing the engine or transport.
- Foundation, media-I/O, image, and retained mesh operations run on CPU when
  their dependencies support it.
- Model-family sampling on CPU or Apple Silicon is unsupported until the
  corresponding `dinkster_comfy` runtime path has current execution evidence.
