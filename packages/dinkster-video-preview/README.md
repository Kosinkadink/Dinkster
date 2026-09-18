# dinkster-video-preview

An opt-in first-party VHS-style video preview initialization pack. The native
`video-preview.initialize` node accepts and returns `comfy.VIDEO` unchanged.
It folds portable video metadata, bounds preview size to 512 pixels wide and
120 frames, and emits `video-preview.initialized` with `fps`, `frameCount`,
`height`, and `width`. It neither decodes frames nor imports Torch or ComfyUI.

From a workspace installed with `uv sync --all-packages`:

```console
uv run dinkster-serve --pack packages/dinkster-video-preview/dinkster-pack.toml --port 8199
```

Connect a VIDEO-producing node to `video-preview.initialize`. The host-owned
GET `/api/extensions/dinkster-video-preview/routes/preview-policy` returns
`{ "defaultFps": 24.0, "maxFrames": 120, "maxWidth": 512 }` from the same
worker policy used to initialize the event. The route requires `jobs:read`.

The snapshot-selected self-contained module `dinkster-video-preview.frontend.preview`
declares two independent privileges: `event-consumer` for its typed event
subscription and `app-workflow` for querying its own route and presenting a
host-rendered metadata status. An enabled compatible frontend shows the actual
policy response, connection/snapshot identity, and latest node's preview
dimensions, frame rate, and frame count. It shows an unavailable-policy status
on query failure. Disabling the frontend contributions does not disable the node.

The module never imports legacy ComfyUI JavaScript, calls PromptServer, accesses
global extension state, or fetches arbitrary URLs. Its immutable module URL is
chosen by `/api/extensions/snapshot`, not constructed by pack code.

See [the API contract](../dinkster-api/README.md#json-routes-and-events).
`tests/test_serve.py::test_video_preview_pack_route_event_and_module_end_to_end`
boots the isolated server and checks a real MP4 source through this node,
the route result, module bytes, and the correlated JSON WebSocket event.
