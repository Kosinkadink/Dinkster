# Try optional node features

These CPU-only examples belong to the development pack, not the default or
published packs. They need no torch, model files, or GPU. From a Dinkster
checkout with the root and dev packages installed, create an empty test library
and enable templates in its `settings.json`:

```json
{"features":{"templates":{"enabled":true}}}
```

Start the server, replacing `../dev-feature-library` with that library path:

```sh
CUDA_VISIBLE_DEVICES="" DINKSTER_ACCELERATOR=cpu .venv/bin/dinkster-serve \
  --host 127.0.0.1 --port 18777 --no-default-packs \
  --pack packages/dinkster-nodes-dev --library-root ../dev-feature-library \
  --allow-settings-changes
```

Point a current Dinkster-Frontend development server's
`DINKSTER_NATIVE_BACKEND` at `http://127.0.0.1:18777`. Keep the backend
loopback-only. Open **Browse starter templates**, select **Node Feature
Examples**, and open it. The node palette also finds these nodes under
`dev/features`. Run individual examples while changing one feature at a time.

| Feature | Node and action | Expected result |
| --- | --- | --- |
| Dynamic slot specialization | **Feature: Dynamic Slot**: right-click the source pin, clear its specialization, then choose text or integer. Connect a compatible output if needed. | The pin changes type. Run to echo the selected variant and value. A compatible connection can select an unambiguous variant. |
| Changeable output count | **Feature: Output Count**: change count from 3 to 5, then back to 2. Try undo. | The number of output pins follows count. Outputs contain start, start + 1, and so on. Undo restores the previous count; stale links are not silently deleted. |
| Conditional rows | **Feature: Conditional Widgets**: edit note, turn details off, then on. | The note row hides and reappears with its text intact. Running while hidden still echoes the stored note. |
| MULTI_COMBO | **Feature: Widget Kinds**, selections: select multiple options. | The editor shows the selected list; running echoes its order. |
| MARKDOWN | Same node, markdown: open the editor and change its source. | Formatted preview and source editing use the same stored text. |
| COLORS | Same node, colors: add, reorder and remove a swatch. | The palette and echoed array follow the change. |
| LIST | Same node, items: enter `["last","first"]`. | The typed list editor accepts the string array and execution preserves its order. Non-string elements are rejected. |
| BUTTON | Select the same node. Click **Open node help**, then its action if a popover opens. | The Node Help panel opens this node's help. No backend run or input-value change is implied. |
| Representations | **Feature: Widget Representations**: right-click text, choose Representation, then Multiline. Edit two lines, switch back to Single line, and undo. | Editor form changes without changing the stored text. The node echoes exactly that text. |
| Output descriptors | **Feature: Output Descriptors**: open entries, add a text row named Caption, apply, rename it, and apply again. | Output pins follow the profile. Integer outputs return their one-based row number; text outputs return the row name. IDs remain stable when names change. |
| Mirror estimates | Run **Feature: Mirror Image Source** and **Feature: Mirror Estimate** once. Change gain without queueing another run, then run again. | Cached source imagery allows an immediate display-only estimate. Running replaces it with authoritative pixels. Gain scales RGB and clips to 0..1; alpha is unchanged. |

## Pack content

| Content | Where and action | Expected result |
| --- | --- | --- |
| Node help | Select **Feature: Widget Kinds** and press F1 or use its help button. | Help explains the five inputs and the frontend-only button action. |
| Guide | Open **Learn**, search for **Try the node features**, then open it. | The pack's guide appears. Its English fallback remains readable in another locale. |
| Blueprint | Double-click empty canvas and search **Node Feature Examples**; choose the blueprint result and place it. | An independent subgraph node contains all eight examples and the source-to-mirror image link. Editing its contents does not change the existing workflow examples. Undo removes the insertion in one step. |
| Pack setting | Open **Settings**, select the **dinkster-nodes-dev** pack category, toggle **Show example tips**, close and reopen. | The saved boolean retains its value. This demonstration preference deliberately does not affect execution. |
| Translation catalog | Change the application locale to Chinese and find **Feature: Widget Kinds** in the node palette. | Its name and description use the pack's Chinese catalog; untranslated nodes fall back to their schema text. |
| Template | Open **Browse starter templates**, then **Node Feature Examples**. | A new workflow contains all examples, including a connected image source and mirror. Templates require the server feature flag above. |
| Extensions panel | Open **Extensions**. | The host's extension snapshot is visible. This pack supplies content, not a frontend module, so it does not claim a loaded code extension. |

An absent estimate, refused command, invalid editor value, or rendering defect
is a result to record, not a reason to change the feature's behavior in the dev
pack. Keep approximate previews separate from backend execution results.
