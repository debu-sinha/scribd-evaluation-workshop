# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Production monitoring
# MAGIC
# MAGIC Score traces after the fact, then register a sampled scorer that keeps running on the live
# MAGIC stream. This is how you catch drift, prompt regression, and post-deploy quality dips without
# MAGIC paying judge cost on every single request.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why score in production at all
# MAGIC
# MAGIC ![Production monitoring concept](./images/hd_production_monitoring_concept.png)
# MAGIC
# MAGIC Offline evals tell you "this change passes the golden set." They do not tell you what real
# MAGIC users are asking, what new failure modes appeared after a model change, or where the long
# MAGIC tail of bad behavior lives. Production monitoring closes that gap by running judges against
# MAGIC live traces on a schedule.

# COMMAND ----------

# MAGIC %md
# MAGIC ## How sampling keeps the bill bounded
# MAGIC
# MAGIC ![Scale architecture](./images/hd_scale_architecture.png)
# MAGIC
# MAGIC Every registered scorer carries a `ScorerSamplingConfig(sample_rate=...)`. Cheap
# MAGIC deterministic scorers run at 1.0. Expensive LLM judges run at 0.05 to 0.10 so the
# MAGIC cost per million traces stays predictable. Beta surface today (per Databricks docs), with a 20-scorer cap
# MAGIC per experiment and a 15 to 20 minute warm-up after registration.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-sdk databricks-agents
# MAGIC %restart_python

# COMMAND ----------

import mlflow

# --- configuration ----------------------------------------------------------
CURRENT_USER = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
)
EXPERIMENT_PARENT = f"/Workspace/Users/{CURRENT_USER}/genai_evals_demo"
EXPERIMENT_PATH = f"{EXPERIMENT_PARENT}/agent_traces"

# Ensure the parent folder exists before set_experiment - it raises NOT_FOUND
# if the parent path does not exist yet, even though it creates the experiment.
from databricks.sdk import WorkspaceClient

WorkspaceClient().workspace.mkdirs(EXPERIMENT_PARENT)
# ---------------------------------------------------------------------------

mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ### The two scorers
# MAGIC
# MAGIC One non-empty check that runs on every trace (cheap, deterministic). One relevance judge
# MAGIC that runs on 10% of traces (LLM, costs tokens). For the relevance judge we prefer the
# MAGIC **calibrated version registered by notebook 03's `judge.align()` pass** - that is the
# MAGIC closed loop in action: SME labels reshape the judge prompt, the calibrated judge takes over
# MAGIC scheduled monitoring without any code edits here. If no aligned judge has been registered
# MAGIC yet, we fall back to the unaligned definition.

# COMMAND ----------

from mlflow.genai.scorers import scorer, ScorerSamplingConfig, get_scorer
from mlflow.genai.judges import make_judge
from mlflow.client import MlflowClient


@scorer
def answer_non_empty(outputs):
    if outputs is None:
        return 0.0
    answer = ""
    if isinstance(outputs, dict):
        answer = outputs.get("answer") or ""
    elif isinstance(outputs, str):
        answer = outputs
    return 1.0 if str(answer).strip() else 0.0


# Pick the relevance judge: prefer the aligned version from notebook 03 if it exists.
client = MlflowClient()
exp = mlflow.get_experiment_by_name(EXPERIMENT_PATH)


def _resolve_relevance_judge():
    try:
        registered = get_scorer(experiment_id=exp.experiment_id, name="relevance")
        print("Using calibrated 'relevance' judge from notebook 03 (closed loop).")
        return registered
    except Exception:
        print(
            "No calibrated 'relevance' judge registered yet. Falling back to baseline."
        )
        return make_judge(
            name="relevance",
            instructions=(
                "You are evaluating whether an answer is relevant to a user's question.\n\n"
                "Question: {{ inputs }}\n"
                "Answer: {{ outputs }}\n\n"
                "Reply with exactly one word from this set: yes, partial, no."
            ),
            model="databricks:/databricks-claude-sonnet-4-6",
        )


relevance_judge = _resolve_relevance_judge()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step A. Score historical traces in one shot
# MAGIC
# MAGIC Useful when you ship a new judge and want to backfill it against the last N production
# MAGIC traces before turning on the scheduled run. `search_traces` gets the candidates,
# MAGIC `mlflow.genai.evaluate(data=traces, scorers=[...])` scores them synchronously.

# COMMAND ----------

production_traces = client.search_traces(
    experiment_ids=[exp.experiment_id],
    filter_string="trace.name = 'answer_question'",
    max_results=10,
)
print(f"Scoring {len(production_traces)} traces")

scoring_results = mlflow.genai.evaluate(
    data=production_traces,
    scorers=[answer_non_empty, relevance_judge],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step B. Register the scheduled scorers
# MAGIC
# MAGIC `scorer.register(name=...)` writes the scorer definition to the experiment.
# MAGIC `.start(sampling_config=...)` turns it on against the live trace stream. The wrapper below
# MAGIC handles the "already registered" case so re-running the notebook is safe.

# COMMAND ----------


def _register_or_skip(scorer_obj, name, sampling_rate):
    try:
        registered = scorer_obj.register(name=name)
        registered.start(
            sampling_config=ScorerSamplingConfig(sample_rate=sampling_rate)
        )
        print(f"  registered + started '{name}' at sampling={sampling_rate}")
    except Exception as e:
        if "already been registered" in str(e):
            print(f"  '{name}' already registered, leaving running")
        else:
            raise


_register_or_skip(answer_non_empty, "answer_non_empty", 1.0)
_register_or_skip(relevance_judge, "relevance", 0.10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## What it looks like in the workspace
# MAGIC

# MAGIC
# MAGIC The Monitoring tab on the experiment shows registered scorers, their sampling rate, and
# MAGIC the rolling-window score over time. New traces flowing in get sampled and scored without
# MAGIC any further action from the engineer.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Caps to know about (Beta surface)
# MAGIC
# MAGIC Production scorer monitoring is in **Beta** today, per the Databricks docs. Workspace admin
# MAGIC controls access via the Previews page.
# MAGIC
# MAGIC - **20 scorers maximum per experiment**
# MAGIC - **15 to 20 minute warm-up** after `start()` before scores show up
# MAGIC - **Only `@scorer`-decorator (function-based) scorers** are supported in production.
# MAGIC   Class-based `Scorer` subclasses cannot be registered for production monitoring.
# MAGIC - Sampling rate must be between 0.0 and 1.0
# MAGIC - Use 1.0 only for cheap deterministic scorers. LLM judges should be 0.05 to 0.10 to keep
# MAGIC   cost bounded as trace volume grows.
# MAGIC
# MAGIC These are documented limits today, not guesses. We will keep this honest as Beta ramps to GA.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to verify in your workspace
# MAGIC
# MAGIC 1. Click the **Monitoring** tab on the experiment. Both `answer_non_empty` and `relevance`
# MAGIC    should be listed as registered scorers.
# MAGIC 2. The sampling rate column shows 1.0 and 0.10 respectively.
# MAGIC 3. The scoring run from Step A produced one assessment per trace - visible on the trace
# MAGIC    detail page under the Assessments section.
# MAGIC 4. After the warm-up window, new traces flowing in get the same scorers run against them.
# MAGIC
# MAGIC Continue to `05_otel_uc_integration`.
