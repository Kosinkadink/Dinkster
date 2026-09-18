## Remote nodes

- Default-installed `dinkster.remote.*` nodes are discovered from the Dinkster
  catalog and execute through its authenticated job gateway. Asset inputs,
  progress, previews, cancellation, and verified asset outputs are supported.
  The last valid catalog remains available during startup outages, and live
  catalog epoch changes replace the remote node surface atomically.
