## Remote nodes

- Default-installed `dinkster.remote.*` nodes are discovered from the Dinkster
  catalog and execute through its authenticated job gateway. Asset inputs,
  progress, previews, cancellation, and verified asset outputs are supported.
  The last valid catalog remains available during startup outages, and live
  catalog epoch changes replace the remote node surface atomically.
- The initial catalog provides `dinkster.remote.nanobanana.image` for image
  generation and editing and `dinkster.remote.seedance.video` for Seedance 2.5
  text-to-video and first-frame image-to-video.
- Engine serving resolves account, catalog, gateway and gateway-audience defaults
  from one production service-address table. Release channels do not select
  services, and service selection does not change project metadata. Explicit
  `--remote-catalog-base` and `--remote-gateway-base` values (or their
  `DINKSTER_REMOTE_*_BASE` variables) remain custom overrides; an explicitly empty
  catalog base disables discovery. Without a separate gateway override, a custom
  catalog also supplies the job endpoint.
