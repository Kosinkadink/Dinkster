+++
title = "Loops with real images"
summary = "Run map, gather, fold, scan, while, and per-item image work."
schema_version = 1
+++

# Loops with real images

Regions repeat an ordinary subgraph. Element inputs advance per iteration,
captures stay fixed, and state inputs receive the previous iteration's state.

```dinkster-example
template = "loop-map-images"
caption = "Split a real image batch, map an overlay across each image, and gather the results."
```

```dinkster-example
template = "loop-gather-image-batch"
caption = "Create one image per color, gather in input order, and merge the list into a batch."
```

```dinkster-example
template = "loop-fold-scan-images"
caption = "Fold strips onto a canvas and gather each running canvas state as a scan."
```

```dinkster-example
template = "loop-while-until"
caption = "Carry a count while it is below four; stopping at false gives until behavior."
```

```dinkster-example
template = "loop-per-item-image-spawn"
caption = "Spawn independent image generation for every size in a list."
```

Open any example from Templates. Drill into the region to inspect its body and
use the Queue and Outputs panels to follow each runtime item.
