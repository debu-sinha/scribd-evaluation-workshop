# Screenshots to capture (after nb07 PDF test passes)

These need to be captured from the workspace browser UI by the user (Cmd+Shift+4 region screenshot on macOS). I cannot capture browser screenshots from CLI - they require a real browser session.

For each, navigate to the URL and capture as a PNG.

## Repo placement
Save under `/Users/debu.sinha/Desktop/scribd-evaluation-workshop/notebooks/images/screenshots/`:

| File | What to capture | Workspace URL |
|---|---|---|
| `01_traces_list.png` | Experiment Traces tab showing 25+ answer_question traces with session/user metadata | https://e2-demo-west.cloud.databricks.com/ml/experiments/3508702099289848/traces |
| `02_trace_spans.png` | Click into a single trace, span tree showing answer_question > retrieve > generate | (drill into any trace from above) |
| `03_review_app.png` | Labeling session UI - SME view with typed schemas (groundedness/relevance/rationale) | Experiment > Labeling Sessions tab > click into session > "Open Review App" |
| `04_eval_run.png` | Evaluations tab showing per-row scores from nb03 (code scorer + relevance judge) | Experiment > Evaluations tab > click most recent run |
| `05_monitoring.png` | Monitoring tab showing 2 scheduled scorers from nb04 with sample rates | Experiment > Monitoring tab |
| `06_serving_endpoint.png` | Serving tab showing genai-eval-demo-agent-debu-sinha endpoint READY with 3 env vars | https://e2-demo-west.cloud.databricks.com/ml/endpoints/genai-eval-demo-agent-debu-sinha |
| `07_playground.png` | AI Playground querying the deployed agent, response visible | https://e2-demo-west.cloud.databricks.com/ml/playground |
| `08_prompt_versions.png` | Experiment Prompts tab showing v1 baseline + v2 aligned with commit messages and diff | Experiment > Prompts tab > click into relevance_judge_instructions |
| `09_annotation_app.png` | nb07 cell output showing PDF iframe + parsed markdown side by side | Open nb07 in workspace, scroll to Step 3 render cell |
