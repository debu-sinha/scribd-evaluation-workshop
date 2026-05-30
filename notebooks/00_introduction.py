# Databricks notebook source
# MAGIC %md
# MAGIC # GenAI evals on Databricks - opening
# MAGIC
# MAGIC This notebook is the framing layer for the bake-off walkthrough. No code runs here. The next
# MAGIC four notebooks are the working demo - traced agent, human assessments, offline evals, and
# MAGIC production monitoring with Datadog dual-export. The maturity statement is the closing slide
# MAGIC at the end of notebook 05.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The problem this workshop solves
# MAGIC
# MAGIC ![The problem](./images/hd_problem.png)
# MAGIC
# MAGIC Eval data scattered across surfaces. Other tools tested timed out at trace volume x
# MAGIC multi-judge fan-out. Datadog is the observability stack but evals live elsewhere. Custom
# MAGIC metrics are one-off integrations. And Anish asked us directly to tell him what we cannot meet.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What success looks like - the three pillars
# MAGIC
# MAGIC ![Three pillars](./images/hd_three_pillars.png)
# MAGIC
# MAGIC Pillar 1 - a unified data surface where traces and assessments are first-class data in the same
# MAGIC UC catalog, queryable side-by-side.
# MAGIC
# MAGIC Pillar 2 - scale that handles your trace volume with multiple judges in the loop, without
# MAGIC the rate-limit ceilings other tools hit.
# MAGIC
# MAGIC Pillar 3 - a Datadog dual-export pattern that lets you keep your observability stack while
# MAGIC standing up evals on Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The closed loop
# MAGIC
# MAGIC ![Closed loop](./images/hd_loop_diagram.png)
# MAGIC
# MAGIC The four working notebooks form a closed loop. Notebook 01 produces traces. Notebook 02
# MAGIC captures human assessments on those traces. Notebook 03 runs offline evals on a golden set.
# MAGIC Notebook 04 puts a sampled judge on the production stream. Notebook 05 sends the same OTel
# MAGIC spans to Datadog alongside the MLflow path. Disagreement between SMEs and judges feeds
# MAGIC `judge.align`, which rewrites the judge prompt to match the SME ground truth.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What you are about to see
# MAGIC
# MAGIC | Notebook | What it shows | Roughly |
# MAGIC | --- | --- | --- |
# MAGIC | `01_agent_app` | The agent. `@mlflow.trace` decorators, session and user metadata flowing into trace fields. | 2 min |
# MAGIC | `02_capture_assessments` | In-app feedback plus Review App labeling. Real `mlflow.genai.label_schemas` and `mlflow.genai.labeling` APIs. | 3 min |
# MAGIC | `03_offline_evals` | `mlflow.genai.evaluate` over a golden set with a code-based scorer and an LLM judge. | 2 min |
# MAGIC | `04_production_monitoring` | Score historical traces, then register a scheduled scorer with sampling. | 2 min |
# MAGIC | `05_otel_uc_integration` | Traces-in-UC enablement story, Datadog dual-export, plus the closing maturity statement. | 3 min |
# MAGIC
# MAGIC We close at the end of notebook 05 with the explicit maturity statement: every component in
# MAGIC this stack called out as GA, Public Preview, Beta, or not shipped, with the operational caveats.
# MAGIC You asked for that directly. We will not paper over the orange rows.
# MAGIC
# MAGIC Continue to `01_agent_app`.
