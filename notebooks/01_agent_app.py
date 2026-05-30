# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - The traced agent
# MAGIC
# MAGIC A small retrieval-augmented Q&A agent that emits hierarchical MLflow traces. This is the
# MAGIC workload everything downstream operates on - assessments, evals, monitoring, dual-export.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Challenges addressed
# MAGIC
# MAGIC 1. **How do you wrap an existing agent so MLflow captures structured traces of every call?**
# MAGIC 2. **How do you tag traces with session and user so the UI filter chips and SQL queries work?**
# MAGIC 3. **How do you produce realistic trace volume that the rest of the notebooks read from?**
# MAGIC
# MAGIC References: [MLflow tracing](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/),
# MAGIC [user / session metadata](https://docs.databricks.com/aws/en/mlflow3/genai/tracing/track-users-sessions),
# MAGIC [Foundation Model APIs](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/).

# COMMAND ----------

# MAGIC %md
# MAGIC ## How session_id and user_id flow into the trace
# MAGIC
# MAGIC ![Session and user metadata](./images/hd_session_user_metadata.png)
# MAGIC
# MAGIC One `mlflow.update_current_trace(metadata=...)` call inside the chain function writes the
# MAGIC canonical `mlflow.trace.user` and `mlflow.trace.session` fields. The agent passes whatever
# MAGIC values the caller supplied (your app's auth + session manager). These become OTel span
# MAGIC attributes, drive the MLflow UI filter chips, and become first-class queryable trace fields
# MAGIC once Traces-in-UC is enabled on the workspace.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The agent shape
# MAGIC
# MAGIC ![Agent shape](./images/hd_agent_explanation.png)
# MAGIC
# MAGIC Three functions, three `@mlflow.trace` decorators. MLflow builds the span tree from the call
# MAGIC graph. The retrieve step is keyword overlap against a small in-memory corpus - replace it
# MAGIC with Vector Search, an external retriever, or anything else without changing the trace shape.
# MAGIC
# MAGIC **Shortcut for 40+ frameworks.** If your agent uses LangChain, LangGraph, OpenAI SDK,
# MAGIC Anthropic SDK, LlamaIndex, DSPy, or any of the 40-odd frameworks MLflow ships native
# MAGIC integrations for, you do not need to hand-decorate every function. A single
# MAGIC `mlflow.langchain.autolog()` (or the equivalent for your stack) auto-instruments the
# MAGIC framework and emits the same trace shape. We use explicit `@mlflow.trace` here for
# MAGIC pedagogical clarity, but production code typically combines the two: auto-log the
# MAGIC framework + decorate the custom orchestration glue between framework calls.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-sdk databricks-agents
# MAGIC %restart_python

# COMMAND ----------

import mlflow

# --- configuration ----------------------------------------------------------
# Derives the current Databricks workspace user automatically. Override
# CURRENT_USER below if you want to write traces to a different user's folder.
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

# MAGIC %run ./_agent_lib

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run a single query
# MAGIC
# MAGIC One invocation produces one trace. After the cell finishes, open the experiment Traces tab
# MAGIC and click into the trace to see the span tree (`answer_question` -> `retrieve` -> `generate`)
# MAGIC and the metadata block on the right.

# COMMAND ----------

result = answer_question(
    query="How are MLflow traces structured?",
    session_id="demo-session-001",
    user_id="alice@example.com",
)
print(result["answer"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Populate the experiment
# MAGIC
# MAGIC The remaining notebooks need a populated trace surface to read from. We run the agent across
# MAGIC four simulated users on a shuffled query bank, with a 0.3s sleep to be polite to the
# MAGIC Foundation Model endpoint. The deterministic seed keeps the trace mix reproducible.

# COMMAND ----------

import random
import time

USERS = [
    ("alice@example.com", "session-alice-001"),
    ("alice@example.com", "session-alice-002"),
    ("bob@example.com", "session-bob-001"),
    ("batch-job@example.com", "session-batch-001"),
]

random.seed(2026)
shuffled = DEMO_QUERIES.copy()
random.shuffle(shuffled)

ok = 0
for query in shuffled:
    user_id, session_id = random.choice(USERS)
    try:
        answer_question(query=query, session_id=session_id, user_id=user_id)
        ok += 1
    except Exception as e:
        print(f"  iter failed: {type(e).__name__}: {str(e)[:200]}")
    time.sleep(0.3)

print(f"populated {ok}/{len(shuffled)} traces")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What it looks like in the workspace
# MAGIC
# MAGIC ![Traces tab](./images/hd_02_traces_list.png)
# MAGIC
# MAGIC The Traces tab shows every `answer_question` call, with the chain / retrieve / generate span
# MAGIC tree on the left and filter chips on top. Click a chip - `user: alice@example.com` - and the
# MAGIC list narrows. Same predicate works as a SQL filter once Traces-in-UC is enabled.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to verify on the call
# MAGIC
# MAGIC 1. Open the **Experiments** left-nav, find `agent_traces`.
# MAGIC 2. Click the **Traces** tab. You should see around 24 traces named `answer_question`.
# MAGIC 3. Click any trace. The span tree shows `answer_question` (CHAIN) -> `retrieve` (RETRIEVER)
# MAGIC    -> `generate` (LLM).
# MAGIC 4. In the Metadata panel, find `mlflow.trace.user` and `mlflow.trace.session`.
# MAGIC 5. Use the **Filter** chip above the trace list to filter by user. The list narrows.
# MAGIC
# MAGIC Continue to `02_capture_assessments`.
