# Verification record

This document records the workspace, configuration, and run IDs the workshop was tested against. If you fork this repo and re-verify, append your own run here so we don't lose the verification chain.

## Test workspace

| Field | Value |
|---|---|
| Date | 2026-05-30 |
| Run ID | 1013199850453523 |
| Result | TERMINATED / SUCCESS across all 8 task chain steps |
| Workspace | `e2-demo-west.cloud.databricks.com` |
| Compute | Serverless notebooks, client image v4 |
| MLflow version | 3.12.0 |
| Foundation Model endpoint | `databricks-claude-sonnet-4-6` |
| Unity Catalog catalog used by deploy | `main` |
| Unity Catalog schema used by deploy | `default` |
| UC schema used by prompt registry | auto-detected per user, prefers `users.<user>` |

## Verification methodology

A clean experiment was created before each run. All 8 notebooks were submitted as a single dependent Jobs API task chain so each step's output flows into the next. A run is only marked verified when every task returns `SUCCESS`.

The chain is:

```
nb01 -> nb02 -> nb03 -> nb04 -> nb05 -> nb06 -> nb07 -> nb99
```

The Databricks Repos clone path (the recommended onboarding path in the README) was also validated separately. The repo was cloned into `/Workspace/Users/<user>/_scribd_workshop_repos_test/` via the Repos API, then `notebooks/01_agent_app` was submitted from the cloned location. Run ID 185734097551132, TERMINATED/SUCCESS. This confirms the workspace-side import works end to end, not just the local-to-workspace upload path the original chain used.

## PDF-grounded review flow (nb07 end to end)

The nb07 custom annotation app pattern was validated end to end on May 30 2026. Run ID 1024881076868832, TERMINATED/SUCCESS. The flow:

1. Step 1 runs three `parse_document` traces. Each one attaches `source_pdf_uri` as trace metadata via `mlflow.update_current_trace(metadata=...)` and returns parsed markdown.
2. Step 2 calls `MlflowClient.search_traces(filter_string="trace.name = 'parse_document'")` and finds the traces.
3. Step 3 reads `source_pdf_uri` from the trace's `request_metadata` and the parsed markdown from `trace.data.response`, then renders a side-by-side PDF iframe and markdown panel via `displayHTML`.
4. Step 4 writes two annotations back to the same trace via `mlflow.log_feedback` (`markdown_matches_pdf`, `source_attribution_correct`).

The notebook does not require a UC Volume to be configured. The demo uses a public W3C sample PDF so the iframe renders without workspace-specific setup. The same code works against UC Volume paths (`/Volumes/<catalog>/<schema>/<vol>/<file>.pdf`) by changing the `SAMPLE_PDF_URL` constant.

If anything in the chain fails, everything after it gets marked `UPSTREAM_FAILED`. Green at the end means every step was green.

## How to reproduce

```bash
# (Assuming you have the Databricks CLI configured)
RUN_ID=$(databricks api post /api/2.1/jobs/runs/submit --json @e2e_chain.json | jq -r '.run_id')
echo "submitted: $RUN_ID"

# Poll
while true; do
  STATE=$(databricks api get "/api/2.1/jobs/runs/get?run_id=$RUN_ID" | jq -r '.state.life_cycle_state')
  echo "state: $STATE"
  [ "$STATE" = "TERMINATED" ] && break
  sleep 60
done

# Get final result
databricks api get "/api/2.1/jobs/runs/get?run_id=$RUN_ID" | jq '.tasks[] | {task_key, state}'
```

The `e2e_chain.json` payload is in `docs/e2e_chain.json` for convenience.

## Known soft failures (notebook handles gracefully)

| Failure mode | Where | What the notebook does |
|---|---|---|
| Foundation Model API rate limit during MemAlign | nb03 | Falls back to unaligned judge, prints clear diagnostic, continues |
| Insufficient paired traces for align (under 5) | nb03 | Falls back to unaligned judge, prints clear diagnostic, continues |
| OpenAI embedding API key missing (default MemAlign config) | nb03 | Pinned `embedding_model` to `databricks:/databricks-bge-large-en` to avoid this |
| Endpoint already exists | nb06 | Calls `update_config` instead of `create` |
| Calibrated judge already registered for the experiment | nb03 | Skips alignment and uses the registered judge |
| User doesn't own any UC schema | nb99 | Raises clear error pointing at the `PROMPT_SCHEMA` override |

## What "verified" means in practice

Here's what a green verification run produces:

- 25 traces in the experiment (24 from nb01's `DEMO_QUERIES` plus 1 from the single test)
- 1 Review App labeling session with 3 typed schemas
- 1 offline eval run with code + judge scores per row
- 1 calibrated `relevance` scorer registered on the experiment
- 2 scheduled scorers in the Monitoring tab
- 1 Model Serving endpoint `genai-eval-demo-agent` in READY state with the 3 env vars on the served entity
- 1 trace from the nb07 demo annotation with markdown_matches_pdf + downstream_response_correct
- 2 versions of `relevance_judge_instructions` prompt in the Prompt Registry under the auto-detected schema

You can verify all of this via the workspace UI in under 5 minutes after a clean run.
