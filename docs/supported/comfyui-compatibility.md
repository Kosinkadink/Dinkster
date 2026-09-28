## ComfyUI compatibility

- `dinkster-serve --comfy-root ... --legacy-pack PATH` runs unmodified v1 and
  pure V3 custom-node packs in quarantined isolated workers under namespaced
  node ids.
- Comfy image, mask, audio, and video values cross the compatibility boundary
  through explicit codecs. Invocation-scoped media staging supports source-file
  inputs on POSIX; source-filename custom nodes are unsupported on Windows.
- SD1.5 and MiniMax H3 execute through the pinned `dinkster_comfy` package.
  Dinkster owns the graph schemas and worker integration, not a second model or
  sampling implementation.
- Maintained aliases in the foundation, image, and media packs remain active.
  The retired generation and compatibility aliases are preserved as
  `comfy-aliases.inactive.json` evidence and are unavailable until their
  capability is re-expressed against the fork and its acceptance receipt passes.
- The 602-workflow translation census, source-parity baseline, registry source
  snapshot, and pre-retirement capability evidence remain committed as
  historical evidence. The old capability selectors are inactive and make no
  current support claim.
- Pack HTTP routes, web assets, and executor hooks are diagnosed rather than
  emulated.
