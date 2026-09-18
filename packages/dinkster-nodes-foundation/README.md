# dinkster-nodes-foundation

`dinkster-nodes-foundation` supplies Dinkster's lightweight first-party logic,
math, text, list, and utility nodes. It depends only on `dinkster-api` and does
not import another node pack.

`dinkster-pack.toml` is the public composition boundary. `FOUNDATION_NODES` is
host wiring, not a stable Python API for other packs. Packs compose through
registered node, capability, and type contracts rather than importing this
implementation.

The distribution embeds a complete replayable pack artifact. A source checkout
uses the adjacent manifest; an installed wheel uses that embedded artifact.

## Lists

`Create List` collects same-typed values in document order. `Append to List`
adds each variadic item as one element after the existing list. A list-valued
item remains nested in both nodes. `Concat Lists` is the explicit operation
that flattens its list inputs by one level; no list operation implicitly
promotes scalars or flattens nested values.

## Math expression grammar v1

`dinkster.math.expression` evaluates lowercase `a` through `z` inputs as finite
`int`, `float`, or `boolean` values and returns required float, int, and boolean
ports.

Grammar v1 supports numeric and bitwise arithmetic, comparisons, boolean
`and`/`or`/`not`, conditional expressions, integer subscripts into `values`,
and direct calls to:

`sum`, `min`, `max`, `clamp`, `floor`, `ceil`, `round`, `abs`, `sign`, `pow`,
`sqrt`, `exp`, `log`, `log2`, `log10`, `sin`, `cos`, `tan`, `asin`, `acos`,
`atan`, `atan2`, `lerp`, `mod`, `int`, and `float`.

Operators are `+`, `-`, `*`, `/`, `//`, `%`, `**`, `<<`, `>>`, `&`, `|`,
`^`, `~`, and the six ordered/equality comparisons. Division and float-valued
functions use float64. Integer division and modulo follow Python's floor and
divisor-sign rules; `round` uses ties-to-even; integer outputs truncate toward
zero. Boolean `and` and `or` short-circuit. N-input `xor` in
`dinkster.bool.logic` is parity XOR: it is true for an odd number of true inputs.
`dinkster.value.random` uses SplitMix64 with unsigned 64-bit seed normalization;
integer ranges are inclusive and float ranges are half-open unless both bounds
are equal.

Attribute access, comprehensions, lambdas, imports, keyword arguments, and
arbitrary calls are rejected. Expressions are limited to 4096 characters and
256 syntax nodes, integers to 16384 bits, exponents to magnitude 4000, and bit
shifts to 1000. Every result must be finite. The mirror vector is
`tests/fixtures/math_expression_v1.json`; regenerate it from pinned ComfyUI
source with `uv run --with simpleeval==1.0.3 python
tools/generate_math_expression_vector.py --comfyui /path/to/ComfyUI`.

## Lazy routing

`dinkster.value.select` lazily evaluates one of two typed values. The integer-indexed
`dinkster.route.switch` selects one member from a document-ordered family of up to
512 typed values, and `dinkster.route.gate` emits a typed absence when closed.
Linked producers on every unselected or closed branch are not executed. Literal
and linked family members retain document order; an out-of-range switch index is
an error.

## String and structured text

The `dinkster.string.*` nodes provide parameterized families for formatting,
case and whitespace transforms, tests, length, regular expressions, joining,
splitting, JSON, and CSV. Formatting accepts scalar slots named `a` through
`z` and a bounded subset of Python format specifications; it rejects traversal,
conversions, nested fields, and custom objects. JSON uses ordinary string and
list values rather than introducing another public type.

String inputs and outputs are limited to 1 MiB of UTF-8. Regular-expression
patterns are limited to 4096 bytes and extraction or replacement to 10000
matches. Python's regular-expression engine does not provide a hard execution
timeout, so these limits do not prevent every catastrophic-backtracking
pattern. JSON and CSV also enforce depth, member, row, column, and cell limits.

## Conversion and curves

`dinkster.value.convert` converts between integer, float, string, and boolean
scalars. Conversions that discard numeric information require `force_lossy`.
Only the output matching the selected target is present, except the `number`
target populates both integer and float outputs. Conversion text is limited to
1 MiB of UTF-8 and integer outputs to 16384 bits.

`dinkster.curve` is an immutable linear or monotone-cubic float curve with strictly
increasing finite positions, endpoint clamping, and at most 4096 points. The
foundation pack can edit a curve with optional histogram preview metadata,
build one from a float list, evaluate or uniformly sample it, and parse strict
comma-separated `position:value` schedules. The
schedule parser accepts plain numeric values with optional parentheses; it does
not evaluate expressions.
