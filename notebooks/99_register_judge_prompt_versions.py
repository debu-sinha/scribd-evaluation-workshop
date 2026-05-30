# Databricks notebook source
# MAGIC %md
# MAGIC # 99 - Register judge prompt versions to UC Prompt Registry
# MAGIC
# MAGIC Pushes the relevance judge's instruction templates to the MLflow Prompt Registry as
# MAGIC versioned prompts. Two versions:
# MAGIC
# MAGIC 1. **Baseline** - the original instructions from notebook 03's `make_judge` call
# MAGIC 2. **Aligned** - the instructions after `judge.align(paired, optimizer=MemAlignOptimizer())`
# MAGIC
# MAGIC After this notebook runs, the Prompts tab on the experiment shows both versions with
# MAGIC commit messages and tags. You can diff versions, roll back, or pin a specific version
# MAGIC via an alias from production code.
# MAGIC
# MAGIC ## Configuration
# MAGIC
# MAGIC By default this notebook auto-detects a UC schema you own and writes the prompt there.
# MAGIC No catalog / schema is hard-coded. To pin an explicit schema (for shared team usage,
# MAGIC environment separation, etc.), set `PROMPT_SCHEMA` in the config cell below.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
from mlflow.genai.scorers import get_scorer

CURRENT_USER = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()  # noqa: F821
)
EXPERIMENT_PATH = f"/Workspace/Users/{CURRENT_USER}/genai_evals_demo/agent_traces"
mlflow.set_experiment(EXPERIMENT_PATH)
exp = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
print(f"Experiment:    {EXPERIMENT_PATH}")
print(f"Experiment id: {exp.experiment_id}")
print(f"User:          {CURRENT_USER}")

# COMMAND ----------

# --- Schema resolution ------------------------------------------------------
# Override PROMPT_SCHEMA to pin to a specific catalog.schema. Otherwise we
# auto-detect a schema the current user owns, preferring the `users` catalog
# (the Databricks convention for per-user UC schemas).

PROMPT_SCHEMA = None  # e.g. "main.genai_evals_demo_prompts"
PROMPT_BASENAME = "relevance_judge_instructions"

if PROMPT_SCHEMA is None:
    owned = spark.sql(  # noqa: F821
        f"""
        SELECT catalog_name, schema_name
        FROM system.information_schema.schemata
        WHERE schema_owner = '{CURRENT_USER}'
        ORDER BY (catalog_name = 'users') DESC, catalog_name, schema_name
        LIMIT 1
        """
    ).collect()
    if not owned:
        raise RuntimeError(
            f"No UC schema owned by {CURRENT_USER}. Set PROMPT_SCHEMA explicitly to "
            f"a catalog.schema where you have CREATE PROMPT privilege, or have your "
            f"metastore admin grant it to you."
        )
    PROMPT_SCHEMA = f"{owned[0]['catalog_name']}.{owned[0]['schema_name']}"

PROMPT_NAME = f"{PROMPT_SCHEMA}.{PROMPT_BASENAME}"
print(f"Prompt schema: {PROMPT_SCHEMA}")
print(f"Prompt name:   {PROMPT_NAME}")

# COMMAND ----------

# Pull the aligned judge that notebook 03 registered.
aligned_judge = get_scorer(experiment_id=exp.experiment_id, name="relevance")
aligned_instructions = aligned_judge.instructions
print(f"Found scorer: {aligned_judge.name}")
print(f"Aligned instructions: {len(aligned_instructions)} chars")
print("---")
print(aligned_instructions[:400])

# COMMAND ----------

# The original (pre-alignment) instructions from notebook 03's make_judge call.
BASELINE_INSTRUCTIONS = (
    "You are evaluating whether an answer is relevant to a user's question.\n\n"
    "Question: {{ inputs }}\n"
    "Answer: {{ outputs }}\n\n"
    "Reply with exactly one word from this set: yes, partial, no.\n"
    "- yes: the answer directly addresses the question.\n"
    "- partial: the answer addresses some aspect but misses key parts.\n"
    "- no: the answer does not address the question or hallucinates.\n"
    "Reply with only the single word, no punctuation, no explanation."
)

# Register baseline as version 1.
v1 = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=BASELINE_INSTRUCTIONS,
    commit_message="Baseline relevance judge instructions from notebook 03 make_judge call",
    tags={
        "stage": "baseline",
        "source": "notebook_03_initial_make_judge",
        "optimizer": "none",
    },
)
print(f"Registered BASELINE as {PROMPT_NAME} version {v1.version}")

# Register aligned as version 2.
v2 = mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=aligned_instructions,
    commit_message="Aligned via MemAlignOptimizer against SME-labeled paired traces",
    tags={
        "stage": "aligned",
        "source": "notebook_03_judge_align",
        "optimizer": "MemAlignOptimizer",
    },
)
print(f"Registered ALIGNED as {PROMPT_NAME} version {v2.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to verify
# MAGIC
# MAGIC 1. Open the experiment **Prompts** tab.
# MAGIC 2. Change the **Location** dropdown to the catalog.schema printed above (`PROMPT_SCHEMA`).
# MAGIC 3. The `relevance_judge_instructions` prompt is listed with 2 versions.
# MAGIC 4. Click into it. Version 1 has commit "Baseline..." and tag `stage=baseline`. Version 2
# MAGIC    has "Aligned via MemAlignOptimizer..." and tag `stage=aligned`.
# MAGIC 5. Use the diff control to compare the two templates side by side.
# MAGIC
# MAGIC ## How production code consumes a versioned prompt
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC from mlflow.genai.judges import make_judge
# MAGIC
# MAGIC # Load a specific version (replace <schema> with your PROMPT_SCHEMA).
# MAGIC prompt = mlflow.genai.load_prompt("prompts:/<schema>.relevance_judge_instructions/2")
# MAGIC judge = make_judge(
# MAGIC     name="relevance",
# MAGIC     instructions=prompt.template,
# MAGIC     model="databricks:/databricks-claude-sonnet-4-6",
# MAGIC )
# MAGIC
# MAGIC # Or load via alias for atomic blue/green between aligned versions.
# MAGIC prompt = mlflow.genai.load_prompt("prompts:/<schema>.relevance_judge_instructions@production")
# MAGIC ```
