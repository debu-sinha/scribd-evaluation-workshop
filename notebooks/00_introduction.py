# Databricks notebook source
# MAGIC %md
# MAGIC # GenAI Evaluation Workshop on Databricks
# MAGIC
# MAGIC This notebook is the framing layer for the workshop. No code runs here. The next eight
# MAGIC notebooks are the working flow: traced agent, human assessments, offline evals, judge
# MAGIC alignment, production monitoring, OTel + Unity Catalog, deployed agent with Playground
# MAGIC routing, custom annotation app for PDF-grounded review, and versioned judges via the
# MAGIC Prompt Registry.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The problem this workshop solves
# MAGIC
# MAGIC ![The problem](./images/hd_problem.png)
# MAGIC
# MAGIC Most teams already have an ad hoc eval setup that's been good enough. It works at a few
# MAGIC hundred traces, then it starts breaking somewhere on the way to production scale.
# MAGIC
# MAGIC Eval data scattered across surfaces. Ad hoc scorers that don't survive a 10x trace-volume
# MAGIC jump. Your existing observability stack lives elsewhere. Custom metrics are one-off
# MAGIC integrations.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The three pillars
# MAGIC
# MAGIC ![Three pillars](./images/hd_three_pillars.png)
# MAGIC
# MAGIC Three principles tie every notebook together. Each capability you build maps back to one of
# MAGIC them.
# MAGIC
# MAGIC Pillar 1, Scale. Trace volume x multi-judge fan-out is where unmanaged eval pipelines hit
# MAGIC their ceiling. This stack keeps pace at that scale.
# MAGIC
# MAGIC Pillar 2, Existing observability integration. OTel-native dual-export means your existing
# MAGIC observability stack (Datadog, Honeycomb, etc.) coexists with Databricks tracing. One span
# MAGIC stream feeds both surfaces.
# MAGIC
# MAGIC Pillar 3, Customizable metrics. Prompt-based judges, statistical and code scorers, third-party
# MAGIC adapters (RAGAS, DeepEval, Phoenix, TruLens, Guardrails) all plug into the same eval call.
# MAGIC
# MAGIC Plus a fourth pillar baked into every notebook: every component you'll touch is called out
# MAGIC as GA, Public Preview, Beta, or not shipped, with operational caveats. The maturity table at
# MAGIC the end of notebook 05 is the full list.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The closed loop
# MAGIC
# MAGIC ![Closed loop](./images/hd_loop_diagram.png)
# MAGIC
# MAGIC The notebooks form a closed loop. Notebook 01 produces traces. Notebook 02 captures human
# MAGIC assessments on those traces. Notebook 03 runs offline evals and calls `judge.align` to
# MAGIC rewrite the judge prompt against SME ground truth. Notebook 04 puts the calibrated judge on
# MAGIC the production stream with sampling. The same UC catalog feeds humans, judges, and
# MAGIC production. New traces flow back into the loop.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What you are about to see
# MAGIC
# MAGIC | Notebook | What it shows | Roughly |
# MAGIC | --- | --- | --- |
# MAGIC | `01_agent_app` | The agent. `@mlflow.trace` decorators, session and user metadata flowing into trace fields. | 2 min |
# MAGIC | `02_capture_assessments` | In-app feedback plus Review App labeling. Real `mlflow.genai.label_schemas` and `mlflow.genai.labeling` APIs. | 3 min |
# MAGIC | `03_offline_evals` | `mlflow.genai.evaluate` over a golden set with a code-based scorer and an LLM judge, plus `judge.align` against SME labels. | 5 min |
# MAGIC | `04_production_monitoring` | Score historical traces, then register a scheduled scorer with sampling. | 2 min |
# MAGIC | `05_otel_uc_integration` | Traces in UC, OTel dual-export pattern, plus the closing maturity statement. | 3 min |
# MAGIC | `06_deploy_agent` | Wrap the agent as a ChatModel, register in UC, deploy to Model Serving. Three env vars route Playground traces back into the experiment. | 8 min |
# MAGIC | `07_custom_annotation_app` | Skeleton for a custom Databricks App that renders source PDFs and parsed markdown side by side for SME review. | 2 min |
# MAGIC | `99_register_judge_prompt_versions` | Push baseline and aligned judge prompts to the UC Prompt Registry with commit messages, tags, and diff view. | 2 min |
# MAGIC
# MAGIC The closing slide at the end of notebook 05 is the explicit maturity statement: every
# MAGIC component in this stack called out as GA, Public Preview, Beta, or not shipped, with the
# MAGIC operational caveats. No paper over the orange rows.
# MAGIC
# MAGIC Continue to `01_agent_app`.
