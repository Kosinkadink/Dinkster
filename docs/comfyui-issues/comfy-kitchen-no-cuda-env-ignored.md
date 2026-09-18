# comfy-kitchen: COMFY_KITCHEN_BUILD_NO_CUDA env var is printed but never read

- **Area:** comfy-kitchen `setup.py` (repo `Comfy-Org/comfy-kitchen`,
  observed at 0.2.22, checkout `../comfy-kitchen`)
- **Status:** fixed upstream by `ec72ea1`; verified in 0.2.31

## Symptom

Building a CPU-only comfy-kitchen wheel with
`COMFY_KITCHEN_BUILD_NO_CUDA=1 python setup.py bdist_wheel` still
attempts the CUDA extension build (and fails on machines without a
CUDA toolchain), even though setup.py's own help output names the
environment variable as the way to skip CUDA.

## Root cause

`setup.py` prints `COMFY_KITCHEN_BUILD_NO_CUDA=1` in its usage/help
text but contains no `os.environ` read for it. The only implemented
mechanism is the `--no-cuda` command-line flag, which it strips from
`sys.argv` before invoking setuptools.

## Repro

```bash
grep -n "COMFY_KITCHEN_BUILD_NO_CUDA" setup.py   # only in a printed string
grep -n "no-cuda" setup.py                        # the flag that actually works
```

## Suggested upstream fix

Honor the documented variable next to the flag parse:

```python
no_cuda = "--no-cuda" in sys.argv or os.environ.get("COMFY_KITCHEN_BUILD_NO_CUDA") == "1"
```

(or remove the variable from the help text).

## Dinkster handling

The torch-package README documents the working recipe (`setup.py
bdist_wheel --no-cuda` from a scratch copy of the checkout) for
installing the CPU eager backend into `.venv-torch`, which the
accelerated fp8 stochastic-rounding path uses when present.

In 0.2.31, `setup.py` consistently documents and implements only the
`--no-cuda` flag. The unsupported environment variable is no longer advertised,
so Dinkster retains the flag without a compatibility workaround.
