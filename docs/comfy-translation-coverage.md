# ComfyUI translation coverage baseline

Pinned workflow templates: `d3b4a9e89573162b005961865164c18c8ae2206b`. Canonical per-workflow translation data: [comfy-translation-coverage.json](comfy-translation-coverage.json). Canonical capability evidence: [comfy-capability-evidence.json](comfy-capability-evidence.json).
The previous 580-workflow denominator remains in [the historical baseline](research/comfy-translation-coverage-aa3661d9.md).

## Corpus

602 workflows, 484 subgraphs, and 12861 node occurrences were parsed from 618 JSON files; 16 non-workflow JSON files were reported and ignored.

| Status | Node occurrences |
| --- | ---: |
| mapped | 5876 |
| quarantine | 5199 |
| unavailable | 0 |
| unsupported | 217 |
| structural | 1569 |

Translation-ready workflows: 8 / 602.
Evidence-supported workflows (weakest link T2 or stronger): 0 / 602.

`quarantine` means the source pack is known but no maintained native alias exists. `unsupported` means source ownership cannot be resolved safely or a maintained mapping explicitly refuses the source node. `unavailable` means a maintained mapping exists but its native provider is absent.

## Op and family confidence

| Mapping kind | Tier | Declarations | Mapped occurrences | Receipt cases |
| --- | --- | ---: | ---: | ---: |
| op | exact | 156 | 2928 | 68 |
| op | parametric | 99 | 2557 | 27 |
| op | equivalent | 34 | 391 | 3 |
| op | grouped | 26 | 0 | 5 |
| family | exact | 0 | 0 | 0 |
| family | parametric | 0 | 0 | 0 |
| family | equivalent | 0 | 0 | 0 |
| family | grouped | 0 | 0 | 0 |

Receipt results: 103 passing, 0 failing, 103 total. Each canonical receipt and its source/native artifacts are replay-verified; registry evidence alone is not parity evidence.

## Source-parity receipt debt

99 / 312 maintained translation declarations have at least one verified passing source/native receipt; 213 remain unreceipted. The parity unit is one unique maintained mapping, not a receipt-case count.

3 maintained fail-closed mappings are reported as refused and excluded from parity counts; they make no native-equivalence claim.

Mapped workflow occurrences and translation-ready workflow totals are translation coverage only. They are excluded from source-parity counts.

Dinkster-only native-pack exclusions and their non-equivalence dispositions are recorded in `comfy-source-parity-baseline.json`.

## Registry cutlines

Snapshot `2026-08-26` contains 5322 packs and 111293288 downloads. Rank 355 reaches 100228978 downloads (90.058421%); rank 900 reaches 106765708 (95.931848%).

| Source band | Mapped | Quarantine | Unavailable | Unsupported | Structural |
| --- | ---: | ---: | ---: | ---: | ---: |
| core | 5685 | 4536 | 0 | 0 | 0 |
| top-355 | 191 | 622 | 0 | 0 | 0 |
| rank-356-900 | 0 | 0 | 0 | 0 | 0 |
| outside-top-900 | 0 | 41 | 0 | 0 | 0 |
| unresolved | 0 | 0 | 0 | 217 | 1569 |

## Largest unresolved surfaces

| Status | Node type | Source pack | Occurrences | Workflows | Reason |
| --- | --- | --- | ---: | ---: | --- |
| quarantine | `CLIPTextEncode` | `comfy-core` | 440 | 191 | no-maintained-native-alias |
| quarantine | `VAEDecode` | `comfy-core` | 297 | 201 | no-maintained-native-alias |
| quarantine | `SimpleMath+` | `comfyui_essentials` | 228 | 24 | no-maintained-native-alias |
| quarantine | `LoraLoaderModelOnly` | `comfy-core` | 224 | 100 | no-maintained-native-alias |
| quarantine | `CLIPLoader` | `comfy-core` | 218 | 159 | no-maintained-native-alias |
| quarantine | `PreviewAny` | `comfy-core` | 197 | 85 | no-maintained-native-alias |
| quarantine | `PrimitiveFloat` | `comfy-core` | 172 | 61 | no-maintained-native-alias |
| quarantine | `KSamplerSelect` | `comfy-core` | 122 | 77 | no-maintained-native-alias |
| quarantine | `PrimitiveStringMultiline` | `comfy-core` | 122 | 71 | no-maintained-native-alias |
| quarantine | `SamplerCustomAdvanced` | `comfy-core` | 109 | 66 | no-maintained-native-alias |
| quarantine | `RandomNoise` | `comfy-core` | 101 | 62 | no-maintained-native-alias |
| quarantine | `VAEEncode` | `comfy-core` | 98 | 50 | no-maintained-native-alias |
| quarantine | `CFGGuider` | `comfy-core` | 82 | 44 | no-maintained-native-alias |
| quarantine | `TextEncodeQwenImageEditPlus` | `comfy-core` | 77 | 21 | no-maintained-native-alias |
| quarantine | `GeminiImage2Node` | `comfy-core` | 73 | 42 | no-maintained-native-alias |
| quarantine | `CheckpointLoaderSimple` | `comfy-core` | 70 | 62 | no-maintained-native-alias |
| quarantine | `KSamplerAdvanced` | `comfy-core` | 58 | 12 | no-maintained-native-alias |
| quarantine | `LTXVConcatAVLatent` | `comfy-core` | 55 | 27 | no-maintained-native-alias |
| quarantine | `LTXVSeparateAVLatent` | `comfy-core` | 55 | 27 | no-maintained-native-alias |
| quarantine | `ReferenceLatent` | `comfy-core` | 53 | 16 | no-maintained-native-alias |
| quarantine | `CLIPVisionEncode` | `comfy-core` | 51 | 30 | no-maintained-native-alias |
| quarantine | `BasicScheduler` | `comfy-core` | 50 | 41 | no-maintained-native-alias |
| quarantine | `FluxKontextMultiReferenceLatentMethod` | `comfy-core` | 48 | 14 | no-maintained-native-alias |
| quarantine | `CFGNorm` | `comfy-core` | 44 | 26 | no-maintained-native-alias |
| quarantine | `SaveGLB` | `comfy-core` | 43 | 33 | no-maintained-native-alias |
