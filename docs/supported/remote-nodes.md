## Remote nodes

- Default-installed `dinkster.remote.*` nodes are discovered from the Dinkster
  catalog and execute through its authenticated job gateway. Asset inputs,
  progress, previews, cancellation, and verified asset outputs are supported.
  The last valid catalog remains available during startup outages, and live
  catalog epoch changes replace the remote node surface atomically.
- The initial catalog provides `dinkster.remote.nanobanana.image` for image
  generation and editing and `dinkster.remote.seedance.video` for Seedance 2.5
  text-to-video and first-frame image-to-video. The catalog base must be
  configured explicitly; an unset base makes no network request.
