# ComfyUI translation coverage baseline at aa3661d9

Pinned workflow templates: `aa3661d9fc1a493f8de6b029f5b8af27da3c5d08`.
The complete generated per-workflow JSON is preserved at Dinkster commit
`8d8824052b0c09cb6dd3aca819218c02a0aa82c3` under
`docs/comfy-translation-coverage.json`.

## Corpus

580 workflows, 493 subgraphs, and 12694 node occurrences were parsed from 596
JSON files; 16 non-workflow JSON files were reported and ignored.

| Status | Node occurrences |
| --- | ---: |
| mapped | 3458 |
| quarantine | 7386 |
| unavailable | 0 |
| unsupported | 195 |
| structural | 1655 |

Translation-ready workflows: 0 / 580.

`quarantine` means the source pack is known but no maintained native alias
exists. `unsupported` means source ownership cannot be resolved safely.
`unavailable` means a maintained mapping exists but its native provider is
absent.

## Op and family confidence

| Mapping kind | Tier | Declarations | Mapped occurrences | Receipts |
| --- | --- | ---: | ---: | ---: |
| op | exact | 88 | 1142 | 0 |
| op | parametric | 86 | 2138 | 0 |
| op | equivalent | 31 | 178 | 0 |
| op | grouped | 21 | 0 | 0 |
| family | exact | 0 | 0 | 0 |
| family | parametric | 0 | 0 | 0 |
| family | equivalent | 0 | 0 | 0 |
| family | grouped | 0 | 0 | 0 |

Receipt results: 0 passing, 0 failing, 0 total. Zero means no canonical
receipt files were supplied to this baseline; registry evidence remains in
the preserved JSON report.

## Registry cutlines

Snapshot `2026-08-26` contains 5322 packs and 111293288 downloads. Rank 355
reaches 100228978 downloads (90.058421%); rank 900 reaches 106765708
(95.931848%).

| Source band | Mapped | Quarantine | Unavailable | Unsupported | Structural |
| --- | ---: | ---: | ---: | ---: | ---: |
| core | 3262 | 6709 | 0 | 0 | 0 |
| top-355 | 196 | 636 | 0 | 0 | 0 |
| rank-356-900 | 0 | 0 | 0 | 0 | 0 |
| outside-top-900 | 0 | 41 | 0 | 0 | 0 |
| unresolved | 0 | 0 | 0 | 195 | 1655 |
