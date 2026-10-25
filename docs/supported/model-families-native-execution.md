## Model families (native execution)

Dinkster currently exposes two generation families through `dinkster_inference`:

- Stable Diffusion 1.5 text-to-image through checkpoint loading, CLIP text
  encoding, latent sampling, VAE decoding, and image saving.
- MiniMax H3 text-to-video with audio through the fork's model, conditioning,
  sampler, and VAE implementations.

The native Empty MiniMax H3 AV Latent node constructs paired video and audio
streams on the model's 17k+5 frame grid at 24 fps.

Other model-family templates and execution paths are not currently supported.
The retained native 3D operation nodes are listed under standalone generation;
they do not imply support for a retired model-family runtime.
