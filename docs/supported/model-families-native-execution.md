## Model families (native execution)

Dinkster currently exposes two generation families through `dinkster_comfy`:

- Stable Diffusion 1.5 text-to-image through checkpoint loading, CLIP text
  encoding, latent sampling, VAE decoding, and image saving.
- MiniMax H3 text-to-video with audio through the fork's model, conditioning,
  sampler, and VAE implementations.

Other model-family templates and execution paths are not currently supported.
The retained native 3D operation nodes are listed under standalone generation;
they do not imply support for a retired model-family runtime.
