# Core comfy_extras list parity matrix

Reference pin: ComfyUI `947c2749`.

Method: this inventory used only pinned-tree reads (`git grep` and
`git show 947c2749:<path>`). The primary census query was:

```text
git grep -n -E "INPUT_IS_LIST|OUTPUT_IS_LIST|is_input_list|is_output_list" 947c2749 -- comfy_extras/
```

Every exact `is_input_list=True` and `is_output_list=True` hit was then
checked against its enclosing class, schema, execute method, and extension
registration. Variable-valued schema factories and explicit false values are
recorded separately so the exact-true census remains reproducible without
hiding real declaration machinery.

## Verified totals

- Output: **25 live `is_output_list=True` declarations**, matching the
  census's 25.
- Input: **14 live `is_input_list=True` declarations**, not 15. The census's
  fifteenth exact-text hit is the comment `# Extract scalars from lists (due
  to is_input_list=True)` in `TrainLoraNode.execute` at
  `comfy_extras/nodes_train.py:1121`; it is LD, not a declaration.
- All 39 live exact-true declarations are V3 and occur in the seven files
  named by the census. There are also four variable-valued V3 declaration
  sites in the two dataset base-class schema factories and two explicit false
  sites; these are outside the census's exact-true totals and are inventoried
  below.
- `nodes.py` has no V1 or V3 shipping declaration. `comfy_api/` grep hits are
  API field definitions, shim accessors, schema conversion, and explanatory
  text, not shipping node declarations. No non-`comfy_extras` core node file
  at the pin carries a shipping list declaration. Test-pack declarations are
  test infrastructure and are outside this inventory.

`list<T>` below means the compat translator wraps each V3 input or output's
translated element type with `TypeExpr.list_of`. A "fan-out source" is a
`list<T>` output which can drive the implicit map-region lowering shipped in
`1e66e74` when connected to an unflagged scalar consumer.

## Exact-true declaration matrix

| File:line | Class / node (socket) | Flag | Semantic class | What it does | Disposition today |
| --- | --- | --- | --- | --- | --- |
| `nodes_dataset.py:98` | `LoadImageDataSetFromFolderNode` / `LoadImageDataSetFromFolder` (`images`) | output, V3 | L2 | Loads every image in one folder as separate items. | Translates to `list<comfy.IMAGE>`; fan-out source feeds the `1e66e74` lowering; flatten collision G1 applies. |
| `nodes_dataset.py:137` | `LoadImageTextDataSetFromFolderNode` / `LoadImageTextDataSetFromFolder` (`images`) | output, V3 | L2 | Loads aligned image-caption pairs from one folder. | Translates to `list<comfy.IMAGE>`; fan-out source feeds lowering; G1 applies. |
| `nodes_dataset.py:142` | `LoadImageTextDataSetFromFolderNode` / `LoadImageTextDataSetFromFolder` (`texts`) | output, V3 | L2 | Emits the captions aligned with the loaded images. | Translates to `list<core.string>`; fan-out source feeds lowering; G1 applies. |
| `nodes_dataset.py:213` | `LoadVideoDataSetFromFolderNode` / `LoadVideoDataSetFromFolder` (`videos`) | output, V3 | L2 | Loads lazy references for every video in one folder. | Translates to `list<comfy.VIDEO>`; fan-out source feeds lowering; G1 applies. |
| `nodes_dataset.py:255` | `LoadVideoTextDataSetFromFolderNode` / `LoadVideoTextDataSetFromFolder` (`videos`) | output, V3 | L2 | Loads aligned lazy video-caption pairs from one folder. | Translates to `list<comfy.VIDEO>`; fan-out source feeds lowering; G1 applies. |
| `nodes_dataset.py:260` | `LoadVideoTextDataSetFromFolderNode` / `LoadVideoTextDataSetFromFolder` (`texts`) | output, V3 | L2 | Emits the captions aligned with the loaded videos. | Translates to `list<core.string>`; fan-out source feeds lowering; G1 applies. |
| `nodes_dataset.py:366` | `SaveImageDataSetToFolderNode` / `SaveImageDataSetToFolder` | input, V3 | L1 | Saves a complete image list to one folder in one invocation. | Every input becomes a proper `list<T>` socket; no lowering is inserted for this consumer; no matrix gap. |
| `nodes_dataset.py:416` | `SaveImageTextDataSetToFolderNode` / `SaveImageTextDataSetToFolder` | input, V3 | L1 | Saves complete aligned image and caption lists in one invocation. | Every input becomes a proper `list<T>` socket; no lowering is inserted; no matrix gap. |
| `nodes_dataset.py:1067` | `ShuffleImageTextDatasetNode` / `ShuffleImageTextDataset` | input, V3 | L1 | Applies one permutation to complete aligned image and text lists. | Every input becomes a proper `list<T>` socket; no lowering is inserted; no matrix gap. |
| `nodes_dataset.py:1082` | `ShuffleImageTextDatasetNode` / `ShuffleImageTextDataset` (`images`) | output, V3 | L1 | Returns the permuted image list. | Translates to `list<comfy.IMAGE>`; already whole-list consumed, so the node is not implicitly mapped and flatten mode is not needed. |
| `nodes_dataset.py:1086` | `ShuffleImageTextDatasetNode` / `ShuffleImageTextDataset` (`texts`) | output, V3 | L1 | Returns the correspondingly permuted text list. | Translates to `list<core.string>`; already whole-list consumed; no matrix gap. |
| `nodes_dataset.py:1283` | `ShuffleVideoDatasetNode` / `ShuffleVideoDataset` | input, V3 | L1 | Permutes a complete video list. | Every input becomes a proper `list<T>` socket; no lowering is inserted; no matrix gap. |
| `nodes_dataset.py:1293` | `ShuffleVideoDatasetNode` / `ShuffleVideoDataset` (`videos`) | output, V3 | L1 | Returns the permuted video list. | Translates to `list<comfy.VIDEO>`; already whole-list consumed; no matrix gap. |
| `nodes_dataset.py:1319` | `ShuffleVideoTextDatasetNode` / `ShuffleVideoTextDataset` | input, V3 | L1 | Applies one permutation to complete aligned video and text lists. | Every input becomes a proper `list<T>` socket; no lowering is inserted; no matrix gap. |
| `nodes_dataset.py:1334` | `ShuffleVideoTextDatasetNode` / `ShuffleVideoTextDataset` (`videos`) | output, V3 | L1 | Returns the permuted video list. | Translates to `list<comfy.VIDEO>`; already whole-list consumed; no matrix gap. |
| `nodes_dataset.py:1339` | `ShuffleVideoTextDatasetNode` / `ShuffleVideoTextDataset` (`texts`) | output, V3 | L1 | Returns the correspondingly permuted text list. | Translates to `list<core.string>`; already whole-list consumed; no matrix gap. |
| `nodes_dataset.py:1668` | `ResolutionBucket` / `ResolutionBucket` | input, V3 | L1 | Groups complete aligned latent and conditioning lists by resolution. | Every input becomes a proper `list<T>` socket; no lowering is inserted; no matrix gap. |
| `nodes_dataset.py:1682` | `ResolutionBucket` / `ResolutionBucket` (`latents`) | output, V3 | L1 | Emits one batched latent per resolution bucket. | Translates to `list<comfy.LATENT>`; already whole-list consumed; no matrix gap. |
| `nodes_dataset.py:1687` | `ResolutionBucket` / `ResolutionBucket` (`conditioning`) | output, V3 | L1 | Emits one conditioning group per resolution bucket. | Translates to `list<comfy.CONDITIONING>`; already whole-list consumed; no matrix gap. |
| `nodes_dataset.py:1762` | `MakeTrainingDataset` / `MakeTrainingDataset` | input, V3 | L1 | Encodes complete aligned image and text datasets with one VAE and CLIP. | Every input becomes a proper `list<T>` socket; no lowering is inserted; no matrix gap. |
| `nodes_dataset.py:1781` | `MakeTrainingDataset` / `MakeTrainingDataset` (`latents`) | output, V3 | L1 | Emits the encoded latent dataset. | Translates to `list<comfy.LATENT>`; already whole-list consumed; no matrix gap. |
| `nodes_dataset.py:1786` | `MakeTrainingDataset` / `MakeTrainingDataset` (`conditioning`) | output, V3 | L1 | Emits the encoded conditioning dataset. | Translates to `list<comfy.CONDITIONING>`; already whole-list consumed; no matrix gap. |
| `nodes_dataset.py:1851` | `SaveTrainingDataset` / `SaveTrainingDataset` | input, V3 | L1 | Shards and saves complete aligned latent and conditioning lists. | Every input becomes a proper `list<T>` socket; no lowering is inserted; no matrix gap. |
| `nodes_dataset.py:1963` | `LoadTrainingDataset` / `LoadTrainingDataset` (`latents`) | output, V3 | L2 | Loads all latent samples from a named saved dataset. | Translates to `list<comfy.LATENT>`; fan-out source feeds lowering; G1 applies. |
| `nodes_dataset.py:1968` | `LoadTrainingDataset` / `LoadTrainingDataset` (`conditioning`) | output, V3 | L2 | Loads all aligned conditioning samples from a named saved dataset. | Translates to `list<comfy.CONDITIONING>`; fan-out source feeds lowering; G1 applies. |
| `nodes_images.py:721` | `SplitImageToTileList` / `SplitImageToTileList` (`image`) | output, V3 | L2 | Splits one image batch into an ordered list of overlapping tiles. | Translates to `list<comfy.IMAGE>`; fan-out source feeds lowering; G1 applies. |
| `nodes_images.py:774` | `ImageMergeTileList` / `ImageMergeTileList` | input, V3 | L1 | Blends a complete ordered tile list back into one image. | Every input becomes a proper `list<T>` socket; no lowering is inserted; no matrix gap. |
| `nodes_rebatch.py:14` | `LatentRebatch` / `RebatchLatents` | input, V3 | L1 | Repartitions all latent batches into requested batch sizes. | Every input becomes a proper `list<T>` socket; no lowering is inserted; no matrix gap. |
| `nodes_rebatch.py:20` | `LatentRebatch` / `RebatchLatents` (`latent`) | output, V3 | L1 | Emits the repartitioned latent batches as a list. | Translates to `list<comfy.LATENT>`; already whole-list consumed; no matrix gap. |
| `nodes_rebatch.py:117` | `ImageRebatch` / `RebatchImages` | input, V3 | L1 | Flattens and repartitions all image batches into requested batch sizes. | Every input becomes a proper `list<T>` socket; no lowering is inserted; no matrix gap. |
| `nodes_rebatch.py:123` | `ImageRebatch` / `RebatchImages` (`image`) | output, V3 | L1 | Emits the repartitioned image batches as a list. | Translates to `list<comfy.IMAGE>`; already whole-list consumed; no matrix gap. |
| `nodes_seedvr.py:447` | `SeedVR2TemporalChunk` / `SeedVR2TemporalChunk` (`latents`) | output, V3 | L2 | Splits one video latent into ordered overlapping temporal chunks. | Translates to `list<comfy.LATENT>`; fan-out source feeds lowering; G1 applies. |
| `nodes_seedvr.py:524` | `SeedVR2TemporalMerge` / `SeedVR2TemporalMerge` | input, V3 | L1 | Crossfades a complete ordered chunk list back into one video latent. | Every input becomes a proper `list<T>` socket; no lowering is inserted; no matrix gap. |
| `nodes_toolkit.py:17` | `CreateList` / `CreateList` | input, V3 | L4 | Concatenates its autogrow inputs into one list without changing items. | The MatchType autogrow member becomes a proper `list<T>` family member; direct and shim paths are already conformance-tested; no matrix gap. |
| `nodes_toolkit.py:23` | `CreateList` / `CreateList` (`list`) | output, V3 | L4 | Forwards the concatenated items as one typed list. | MatchType output becomes `list<T>` and can feed lowering; the node itself is a whole-list consumer, so no flatten collision. |
| `nodes_train.py:963` | `TrainLoraNode` / `TrainLoraNode` | input, V3 | L1 | Trains once over complete latent and conditioning input lists. | Every input becomes a proper `list<T>` socket; no lowering is inserted; no matrix gap in list representation. |
| `nodes_train.py:1121` | `TrainLoraNode.execute` comment | input text hit, V3 | LD | Explains scalar extraction; it declares nothing. | No schema effect; this is the census's extra fifteenth input hit. |
| `nodes_wandancer.py:948` | `WanDancerPadKeyframesList` / `WanDancerPadKeyframesList` (`keyframes_sequence`) | output, V3 | L2 | Emits one padded keyframe sequence per requested video segment. | Translates to `list<comfy.IMAGE>`; fan-out source feeds lowering; G1 applies. |
| `nodes_wandancer.py:949` | `WanDancerPadKeyframesList` / `WanDancerPadKeyframesList` (`keyframes_mask`) | output, V3 | L2 | Emits one validity mask per requested segment. | Translates to `list<comfy.MASK>`; fan-out source feeds lowering; G1 applies. |
| `nodes_wandancer.py:950` | `WanDancerPadKeyframesList` / `WanDancerPadKeyframesList` (`audio_segment`) | output, V3 | L2 | Emits one aligned audio slice per requested segment. | Translates to `list<comfy.AUDIO>`; fan-out source feeds lowering; G1 applies. |

## Variable and false declaration sites outside the exact-true census

These source sites are real enough to affect shipping schemas, but an
inventory that silently adds them to the exact-true totals would no longer
reproduce the census method.

| File:line | Class / effective nodes | Flag | Semantic class | Evidence and disposition |
| --- | --- | --- | --- | --- |
| `nodes_dataset.py:511` | `ImageProcessingNode` configuration default | output configuration, V3, `None` | LD | This is not itself a schema flag; the schema factory resolves it from processing mode before line 608. It has no independent compat disposition. |
| `nodes_dataset.py:603` | `ImageProcessingNode` schema factory / group-mode subclasses | input, V3, `is_group` | **UNCERTAIN: L1 or L4 by subclass** | True for `ShuffleDataset`, `ImageDeduplication`, `ImageGrid`, `MergeImageLists`; false for individual transforms. When true, compat produces proper `list<T>` inputs. The shared site cannot receive one honest semantic label. |
| `nodes_dataset.py:608` | `ImageProcessingNode` schema factory / group-mode subclasses | output, V3, computed | **UNCERTAIN: L1 or L4 by subclass** | Group transforms normally produce a list; `ImageGrid` overrides the value false. Compat wraps only the effective true schemas. The shared site cannot receive one honest semantic label. |
| `nodes_dataset.py:709` | `TextProcessingNode` configuration default | output configuration, V3, `None` | LD | This is not itself a schema flag; subclasses may override it before the schema factory reads it at line 780. It has no independent compat disposition. |
| `nodes_dataset.py:775` | `TextProcessingNode` schema factory / `MergeTextLists` | input, V3, `is_group` | L4 | True only for the registered group-mode passthrough at this pin; compat produces proper `list<T>` inputs. |
| `nodes_dataset.py:780` | `TextProcessingNode` schema factory / text subclasses | output, V3, class value | **UNCERTAIN: LD or L4 by subclass** | The base default is `None`; `MergeTextLists` does not override it, so this is false at the pin. If a subclass opts in, compat wraps it normally. |
| `nodes_dataset.py:1545` | `ImageGridNode` / `ImageGrid` | output, V3 override false | LD | Deliberately forces the shared image-processing output declaration to scalar; compat correctly leaves `comfy.IMAGE` scalar. |
| `nodes_images.py:782` | `ImageMergeTileList` / `ImageMergeTileList` (`image`) | output, V3 false | LD | Deliberately declares the merged result scalar; compat correctly leaves `comfy.IMAGE` scalar. |

## Adjudicated gaps

### G1 - mapped list-producing nodes require typed flatten (implemented 2026-07-30)

**Affected core nodes:** `LoadImageDataSetFromFolder`,
`LoadImageTextDataSetFromFolder`, `LoadVideoDataSetFromFolder`,
`LoadVideoTextDataSetFromFolder`, `LoadTrainingDataset`,
`SplitImageToTileList`, `SeedVR2TemporalChunk`, and
`WanDancerPadKeyframesList`.

**Triggering workflow shape:** connect any real list-valued source to a scalar
input of one of these nodes (for example, a list of images into
`SplitImageToTileList.image`, or a list of folder names into a dataset loader),
then consume its declared list output. ComfyUI implicitly invokes the node per
input item and flattens each `OUTPUT_IS_LIST` result.

**Disposition:** `RegionOutput(mode="flatten")` concatenates list-typed body
outputs in iteration order while preserving the same `list<T>` type. Compat's
implicit list-map lowering selects flatten for mapped list outputs and gather
for mapped scalar outputs. The seven-candidate fixture wave below proves the
core declaration shapes without executing real model nodes.

### Checks with no discovered gap

- **Input socket shape:** all 14 live exact-true input declarations translate
  to proper `list<T>` sockets. This includes every ordinary input on a
  class-wide declaration and the MatchType member of `CreateList`'s autogrow
  family. No input declaration translated to a scalar or opaque substitute.
- **V1 versus V3 translation:** no semantic difference was found. The direct
  V3 translator reads `schema.is_input_list` and per-output
  `is_output_list`; the runtime V1 shim exposes equivalent
  `INPUT_IS_LIST`/`OUTPUT_IS_LIST` values and the V1 translator applies the
  same `TypeExpr.list_of` wrapping. Existing CreateList conformance tests
  cover the dynamic MatchType case.
- **Region representability:** whole-list aggregation, reordering,
  repartitioning, passthrough, and fan-out are representable as typed node
  bodies plus current list sockets and map regions. The core node bodies are
  not native ports yet, but no additional region kind is required by this
  inventory. G1 is a gather-cardinality operation, not a missing control-flow
  region.

One adjacent compatibility boundary is intentional policy, not a newly
discovered matrix defect: upstream auto-wraps a scalar feeding an
`INPUT_IS_LIST` node, while Dinkster keeps scalar `T` and `list<T>` distinct and
requires an explicit list constructor. That policy is already recorded in
ROADMAP "List/scalar boundary policy".

## Implemented test wave

The conformance wave uses:

1. `SplitImageToTileList` -> ordinary scalar tile consumer ->
   `ImageMergeTileList` for L2 fan-out, implicit mapping, order, and L1 merge.
2. `RebatchImages` and `RebatchLatents` for whole-list repartitioning and
   output-list cardinality.
3. `CreateList` for MatchType autogrow plus L4 passthrough through both direct
   V3 and runtime-shim translation paths.
4. `ShuffleImageTextDataset` for aligned multi-output ordering under one
   whole-list invocation.
5. `SeedVR2TemporalChunk` -> scalar chunk consumer ->
   `SeedVR2TemporalMerge` for the production-shaped G1 adjudication and
   ordered temporal reconstruction.
6. One dataset loader with two list outputs for aligned fan-out and the
   list-of-folder mapped flatten case.
7. `WanDancerPadKeyframesList` for three aligned list outputs from one scalar
   invocation.
