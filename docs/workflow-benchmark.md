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
native_jobs=/evidence/inputs/native-jobs.json
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
  --native-jobs "$native_jobs" \
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

## Native job exports

Dinkster submits to `/api/jobs`. Export the graph and targets through the existing
frontend importer/compiler for every requested seed, retaining the request JSON
and the frontend commit. The benchmark does not convert ComfyUI graphs. It changes
only client/job IDs; seed inputs must already match in each compiled graph under
the source node and input IDs. Native integer wire markers are decoded when
checking seeds, not rewritten. Acquisition requests are not accepted; artifact
provisioning remains the authenticated local manifest's responsibility.

`--native-jobs` is a JSON object with `api_sha256` (the source API file digest),
`frontend_commit`, and `jobs` in cold-then-warm order. Each job has `seed`,
`source_graph_sha256` (canonical sorted compact JSON SHA-256 of the source API
graph with that seed), and `request` (the ordinary native graph/targets request).
The workflow provenance file also records `native_jobs_sha256`, the SHA-256 of
the exact export file bytes. Use the same source provenance for both systems;
ComfyUI still receives the source API graph through `/prompt` and needs no native
export argument.

Reports retain the original export and every submitted native request, with
separate source-graph and native-request hashes. File validation checks the export
bytes, requests, successful histories, journal, outputs and memory samples.
Export provenance and seed validation do not establish numerical parity; only
executed comparisons can do that.
