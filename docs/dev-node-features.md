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
| Colors group | Same node, colors: type or pick a color in the dimmed row at the end, then drag a color to another place. | Each color is its own row with its own pin, and a new dimmed row follows. Running echoes the colors as one list in the order shown. |
| LIST | Same node, items: enter `["last","first"]`. | The typed list editor accepts the string array and execution preserves its order. Non-string elements are rejected. |
| BUTTON | Select the same node. Click **Open node help**, then its action if a popover opens. | The Node Help panel opens this node's help. No backend run or input-value change is implied. |
| Representations | **Feature: Widget Representations**: right-click text, choose Representation, then Multiline. Edit two lines, switch back to Single line, and undo. | Editor form changes without changing the stored text. The node echoes exactly that text. |
| Output descriptors | **Feature: Output Descriptors**: open entries, add a text row named Caption, apply, rename it, and apply again. | Output pins follow the profile. Integer outputs return their one-based row number; text outputs return the row name. IDs remain stable when names change. |
| Automatic values | **Automatic: Frame Rate** (under `dev/automatic`): change rate or divisor, type a frame rate, then empty it and press Enter. Set divisor to 0 and run. | The frame rate row reads "auto" and rate / divisor, and follows its inputs until a value is typed. Emptying the field makes it automatic again. A divisor of 0 runs with the default, 24, and the field and Problems show the warning. A saved workflow stores no frame rate while it is automatic. |
| Automatic from a lookup | **Automatic: Looked-up Frame Rate** (under `dev/automatic`): set rate to 100 and run, then set rate to -4 and run, then set delay to 14 and run. | The frame rate row reads "auto" and 25, and the run uses 25. A rate of -4 makes the lookup fail, and a delay of 14 outlasts the 10 second limit, so queueing waits at most 10 seconds; either way the run uses the default, 24, and the field and Problems show the warning. |
| Automatic in a group | **Automatic: Lora Stack**: turn on separate model and clip strength, add a lora and change its strength. | Each lora's clip strength reads "auto" and follows its own strength until a value is typed there. |
| Mirror estimates | Run **Feature: Mirror Image Source** and **Feature: Mirror Estimate** once. Change gain without queueing another run, then run again. | Cached source imagery allows an immediate display-only estimate. Running replaces it with authoritative pixels. Gain scales RGB and clips to 0..1; alpha is unchanged. |
| Sections and growing groups | **Feature: Layers**: add three layers with **Add layer**, reorder them, remove one, then close and reopen the Layers section. Put the node in a subgraph, forward the whole group to its face, nest that subgraph in another and forward it again, then add a layer and edit its strength on the outermost node. | Layers keep their names and strengths through reorder, close and forwarding. Running echoes the title and each layer's name and strength in the order shown. |
| Fields under an option | **Feature: Fields Under an Option**: wire the amount field's output and the scale field's output to two **Feature: Required Value** nodes. Choose No amount, and plug a text into source. Run. | While its option or variant is not chosen, a field's output gives nothing, and the receiving Required Value node is skipped instead of failing. Choose Fixed amount and plug a number again, and both receiving nodes run. |
| A required value | **Feature: Required Value**: type a number into value and run, then wire value from a field's output that gives nothing and run again. | With a number it echoes `value=` and the number. When its wire gives nothing, the node is skipped and its own output gives nothing in turn, instead of the run failing on a missing input. |
| Curves with their own histograms | **Feature: Curves With Histograms**: wire two **Feature: Histogram Shape** nodes, one rising and one falling, into the histogram pins inside the first two curves' editors. Run, then run again without changes. | Each curve's editor shows the bars of its own histogram, and the third curve shows none. The second run is taken from the cache and still shows both. |

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
