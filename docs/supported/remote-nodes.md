## Remote nodes

- Default-installed `dinkster.remote.*` nodes are discovered from the Dinkster
  catalog and execute through its authenticated job gateway. Asset inputs,
  progress, previews, cancellation, and verified asset outputs are supported.
  The last valid public catalog remains available during startup outages, and live
  catalog epoch changes replace the remote node surface atomically.
- The default remote pack requests its supported schema wire and uses the
  current session bearer for catalog discovery. Authenticated schemas stay
  memory-only; public cache is isolated by catalog endpoint. Session changes
  clear prior schemas, and signed-in policy changes can reload nodes without
  changing the public catalog epoch.
- The initial catalog provides `dinkster.remote.nanobanana.image` for image
  generation and editing and `dinkster.remote.seedance.video` for Seedance 2.5
  text-to-video and first-frame image-to-video. The catalog base must be
  configured explicitly; an unset base makes no network request.
