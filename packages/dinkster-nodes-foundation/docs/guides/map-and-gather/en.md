+++
title = "Map and Gather"
summary = "Map one list input through a subgraph and gather its outputs."
schema_version = 1
+++

# Map and Gather

Use a map region when the same subgraph should process every item in a list. Inputs selected as element ports advance once per item. Other inputs are broadcast unchanged to every invocation.

The example maps `left` over `2`, `4`, and `8`, broadcasts `right = 10`, and runs Add Integers once for each item. The region gathers the `sum` output into `12`, `14`, and `18` in order.

```dinkster-example
template = "map-and-gather"
caption = "Map three integers, broadcast 10, and gather the sums."
```

Open the template from Templates, then inspect the map region and its Add two integers subgraph. Add more values to `left` to create more region invocations without changing the subgraph.
