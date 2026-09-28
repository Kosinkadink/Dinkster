# Workflow benchmark invocation

Run the transcript benchmark from the Dinkster repository root with the
checkout's own project interpreter. The benchmark interpreter and the server
interpreter have different responsibilities: the project interpreter imports
the tracked harness and `psutil`, while `--server-python` starts the prepared
serving runtime.

Record the resolved commands before claiming a GPU. Use a new output path for
every invocation; the benchmark intentionally refuses an existing output
directory.

```bash
repo=/work/Dinkster
python="$repo/.venv/bin/python"
server_python=/models/runtime/dinkster/bin/python
evidence=/evidence/workflow-row
workflow=/evidence/inputs/workflow.json
provenance=/evidence/inputs/workflow.provenance.json
artifacts=/evidence/inputs/artifacts.json
reference_root=/work/ComfyUI
reference_commit=FULL_COMFYUI_COMMIT
gpu_uuid=GPU-UUID

cd "$repo"
PYTHONPATH="$repo" "$python" -c "import tools.workflow_benchmark, psutil"
uv run dinkster-pack prepare-catalogs --defaults

PYTHONPATH="$repo" "$python" tools/workflow_benchmark_transcript.py \
  --workflow "$workflow" \
  --workflow-provenance "$provenance" \
  --artifacts "$artifacts" \
  --repo "$repo" \
  --server-python "$server_python" \
  --reference-root "$reference_root" \
  --reference-commit "$reference_commit" \
  --output "$evidence" \
  --family minimax_h3 \
  --seed-input NODE.INPUT \
  --seed SEED \
  --warm-runs 0 \
  --port PORT \
  --gpu-uuid "$gpu_uuid"
```

The import preflight must pass under the exact interpreter, working directory,
and `PYTHONPATH` used for the benchmark. Do not substitute the serving runtime
interpreter for the harness interpreter. Retain `http-transcript.jsonl`, the
report, server log, terminal history or journal, output files, and cleanup
record for each row. Use a distinct port and output directory for sequential
rows.

The benchmark does not grant a GPU claim. Hold the host's normal GPU lock
around physical runs and verify the server process, port, and lock are released
afterward. `--warm-runs 0` produces one cold run and no warm summary.
