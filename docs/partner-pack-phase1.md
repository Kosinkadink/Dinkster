# Partner operations use gateway remote nodes

Partner operations are catalog-driven remote nodes. The default-installed
`dinkster-nodes-remote` pack loads `dinkster.remote.*` schemas from a configured
dinkster-gateway deployment, submits authenticated jobs, reports progress and
previews, and verifies downloaded output digests before storing assets. Configure
the service with `--remote-catalog-base`, `--remote-gateway-base`, and
`--remote-auth-token-file`. No public gateway origin is assumed when the catalog
base is unset.

The first gateway catalog exposes:

- `dinkster.remote.nanobanana.image` for text-to-image and one-image editing.
- `dinkster.remote.seedance.video` for Seedance 2.5 text-to-video and first-frame
  image-to-video, with optional generated audio.

The removed provider-specific node ids are not aliases for these remote nodes.
Workflows must select the gateway schema that matches their intended image or
video capability.

## Operations awaiting gateway adapters

These node types have no Dinkster execution path until a gateway adapter is
available:

- `partner.bfl.flux-pro-expand`,
  `partner.bfl.flux-pro-fill`, `partner.bfl.flux-erase`, and
  `partner.bfl.flux-vto`.
- `partner.grok.video-reference`,
  `partner.grok.video-edit`, and `partner.grok.video-extend`.
- `partner.kling.camera-control-i2-v`,
  `partner.kling.camera-control-t2-v`, `partner.kling.start-end-frame`,
  `partner.kling.video-extend`, `partner.kling.lip-sync-audio-to-video`,
  `partner.kling.lip-sync-text-to-video`, `partner.kling.virtual-try-on`,
  `partner.kling.single-image-video-effect`,
  `partner.kling.dual-character-video-effect`,
  `partner.kling.omni-pro-first-last-frame`,
  `partner.kling.omni-pro-video-to-video`,
  `partner.kling.omni-pro-edit-video`, `partner.kling.motion-control`,
  `partner.kling.first-last-frame`, and `partner.kling.avatar`.
- `partner.wan.reference-video`,
  `partner.wan.2-video-continuation`, `partner.wan.2-video-edit`,
  `partner.wan.2-reference-video`, `partner.wan.happy-horse-video-edit`, and
  `partner.wan.happy-horse-reference-video`.
- `dinkster.text_generate` and
  `dinkster.prompt_enhance` no longer have the configured OpenAI-compatible
  execution provider. Their native model execution remains available.

`partner.kling.camera-controls` was a local helper value rather than a remote
operation and is removed without a replacement.
