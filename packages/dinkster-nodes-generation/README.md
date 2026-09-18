# dinkster-nodes-generation

First-party schemas for universal generation operations. The pack owns stable
node contracts and the `dinkster.generation.schemas` capability; execution comes
from separately composed providers through `[pack] executes`.

Generate Text and Enhance Prompt expose the optional
`dinkster.generation.providers` choice. Omitting it keeps native execution and
lazily demands the connected text encoder. Selecting an external provider
routes execution without resolving that native input. The package also owns
the shared LTX-2 prompt formatting and cleanup used by both provider paths.

The schemas use inference-owned `dinkster.model`, `dinkster.clip`,
`dinkster.vae`, `dinkster.conditioning`, and `dinkster.latent` values plus the
media-owned `dinkster.image` value. Other node packs consume these ids as opaque
contracts. Execution providers may subclass these schemas to keep their
worker announcement exact; unrelated schema packs must not import this package.

ControlNet schemas load `model/controlnet` assets into the existing
`comfy.CONTROL_NET` carrier, apply that carrier to native conditioning and
image values, and select Union ControlNet modes. Advanced application exposes
the source strength and percentage-range defaults plus an optional native VAE.
Union mode labels preserve ComfyUI's grouped choices; their members correspond
to the mode indices documented by `dinkster_inference.controlnet.SD_CONTROL_MODE_INDEX`:
`openpose=0`, `depth=1`, `hed/pidi/scribble/ted=2`,
`canny/lineart/anime_lineart/mlsd=3`, `normal=4`, `segment=5`, `tile=6`, and
`repaint=7`. The `auto` default leaves mode selection to the loaded ControlNet.

## Metadata-probed model outputs

`dinkster.load_model_profile` uses a required `checkpoint: dinkster.asset` and a
stored `entries: core.string` output descriptor. The catalog fixes semantic
choices `model: dinkster.model`, `clip: dinkster.clip`, and `vae: dinkster.vae`.
`GET /api/output-profiles/model?digest=blake3:<64hex>&revision=1` probes verified
safetensors headers without starting execution workers. It requires the same
`assets:read` capability as asset metadata and returns a non-cached profile.

The profile includes ordered `entries`, `assetDigest`, `detectorRevision`,
`shapeDigest`, compact component identities and mapping digests, and
nonblocking `diagnostics`. A complete checkpoint exposes all three outputs
from the same existing checkpoint handle; a model-only asset exposes MODEL.
Unknown detection keeps MODEL with a diagnostic. Names, types and identity
fields are generated, not user-editable. Host admission verifies the complete
profile before cache lookup, and workers revalidate before projecting the
existing loader result. Asset/probe changes require re-probing, not mutation
of the persisted type-level catalog.
