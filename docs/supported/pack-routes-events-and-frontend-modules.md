## Pack routes, events, and frontend modules

- Packs can declare authenticated GET/POST JSON routes and typed execution
  events without importing server internals. Routes live in host-owned pack
  namespaces; events retain pack, worker, node, and native execution identity.
  Integer payload values use JavaScript's safe integer range; finite floating
  point values remain supported in number fields.
- Installed packs can publish snapshot-selected immutable JavaScript modules.
  Snapshot and module reads do not start execution workers. Frontend privileges
  and controls do not grant or remove backend node execution ownership.
- Frontend modules can declare virtual node kinds for frontend-owned document
  nodes that compatible editors keep out of backend execution requests.
- Doctor warns when a pack declares a frontend contribution kind or extension
  capability for which no runtime consumer exists.
- Isolated packs can publish value renditions whose metadata, parameter
  normalization, MIME selection, and rendering remain pack-owned while being
  available through the host value API.
- Packs can publish validated locale catalogs for node, blueprint, and guide
  text. Their exact digest-addressed JSON bytes are served as immutable pack
  resources.
- The opt-in `dinkster-video-preview` pack initializes bounded VHS-style VIDEO
  metadata, exposes its preview policy, and supplies a declared event consumer
  and host-rendered metadata status for compatible frontends.

### Extension contribution status

The manifest vocabulary is larger than the set of contributions consumed by a
frontend today. `Works` means the generated vocabulary marks the contribution
implemented and the cited frontend test exercises its registration or use.
`Declared and unconsumed` means manifests may declare the contribution, but no
frontend activation door consumes it.

Frontend proof paths were checked at Dinkster-Frontend commit
`50321360d62f855469a9e8401dc1a34a9cd6e7c8`.

| Contribution kind | Status | Proof |
|---|---|---|
| `widgetKind` | Works | `Dinkster-Frontend/packages/core/test/extensions.test.ts` |
| `widgetView` | Works | `Dinkster-Frontend/packages/app/test/extension-world.test.ts` |
| `previewRenderer` | Works | `Dinkster-Frontend/packages/app/test/extension-world.test.ts` |
| `textEditorExtension` | Works | `Dinkster-Frontend/packages/core/test/extensions.test.ts` |
| `menu` | Works | `Dinkster-Frontend/packages/core/test/extensions.test.ts` |
| `command` | Works | `Dinkster-Frontend/packages/app/test/extension-world.test.ts` |
| `keybinding` | Works | World-to-registry: `Dinkster-Frontend/packages/app/test/extension-world.test.ts` (`projects a pack keybinding through the selected extension world`) |
| `setting` | Works | `Dinkster-Frontend/packages/core/test/extensions.test.ts` |
| `canvasLayer` | Declared and unconsumed | - |
| `nodeDecoration` | Declared and unconsumed | - |
| `hostUi` | Works | `Dinkster-Frontend/packages/app/test/host-ui.test.ts` |
| `searchProvider` | Works | `Dinkster-Frontend/packages/core/test/extensions.test.ts` |
| `workflowObserver` | Declared and unconsumed | - |
| `eventConsumer` | Works | `Dinkster-Frontend/packages/app/test/extension-world.test.ts` |
| `workflowImporter` | Declared and unconsumed | - |
| `editor` | Works | `Dinkster-Frontend/packages/app/test/editors.test.ts` |
| `editorBinding` | Works | `Dinkster-Frontend/packages/app/test/editors.test.ts` |
| `panel` | Works | `Dinkster-Frontend/packages/app/test/editors.test.ts` |
| `virtualNode` | Works | `Dinkster-Frontend/packages/core/test/extensions.test.ts` |

### Extension capability status

Capabilities are authorization and audit declarations. A declaration does not
create a runtime door by itself.

| Capability | Status | Proof or limitation |
|---|---|---|
| `accelerator` | Declared and unconsumed | No extension capability consumer |
| `artifacts` | Declared and unconsumed | No extension capability consumer |
| `background-jobs` | Declared and unconsumed | No extension capability consumer |
| `downloads` | Declared and unconsumed | No extension capability consumer |
| `filesystem` | Declared and unconsumed | No extension capability consumer |
| `model-family-registration` | Declared and unconsumed | No pack-facing model-family registration door; adding one requires a core edit |
| `routes` | Works | `tests/test_pack_surfaces.py` |

The generated contribution vocabulary and doctor diagnostics keep declared
doors visible without presenting them as implemented. The broader regression
guards for pack extension contracts are tracked in
[comfy-vibe-station#120](https://github.com/Kosinkadink/comfy-vibe-station/issues/120).
