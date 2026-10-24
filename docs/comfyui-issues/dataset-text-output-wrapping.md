# Dataset text nodes return lists through scalar outputs

Reference: ComfyUI `bc5f880dbdd43e426077f42ccec53bb577ffee9f`,
`comfy_extras/nodes_dataset.py`.

`TextProcessingNode.is_output_list` defaults to `None`, and its schema passes
that value to `io.String.Output`. Execution takes the false branch and wraps
the processed result in a list. The scalar STRING socket therefore carries
`[text]` rather than `text`. Group processing wraps the whole text list again.

Reproduce in the reference CPU environment:

```python
from comfy_extras.nodes_dataset import TextToLowercaseNode, MergeTextListsNode

assert TextToLowercaseNode.OUTPUT_IS_LIST == [None]
assert TextToLowercaseNode.execute("AbC").result == (["abc"],)
assert MergeTextListsNode.execute(["A", "b"]).result == ([["A", "b"]],)
```

A downstream text operation expecting a string cannot directly consume the
wrapped payload. An upstream correction would make the declared output-list
flag and execution shape agree. Native nodes preserve the fixed reference's
schema and wrapping rather than silently changing saved-workflow behavior.

Status: reproduced locally; not reported upstream.
