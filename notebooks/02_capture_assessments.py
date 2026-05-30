# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Human assessments
# MAGIC
# MAGIC Two sources of human ground truth layered on the traces from notebook 01. First, in-app
# MAGIC thumbs feedback from end users. Second, structured Review App labeling by SMEs against
# MAGIC typed schemas. Both land as assessments attached to the trace.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why both, and how they differ
# MAGIC
# MAGIC ![Review session vs Review App](./images/hd_review_vs_session.png)
# MAGIC
# MAGIC The end-user thumbs is volume signal - cheap, dirty, biased toward the kinds of failures
# MAGIC users actually feel. The Review App label is expert signal - typed, slow, and the closest
# MAGIC thing to ground truth we have. Notebook 03 uses the expert labels to calibrate the judge.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating a labeling session in code
# MAGIC
# MAGIC ![Session creation](./images/hd_session_creation.png)
# MAGIC
# MAGIC Two real APIs from `mlflow.genai`. First, define each schema with `create_label_schema` -
# MAGIC name, type, the human-readable question shown to the SME, and the input widget
# MAGIC (`InputCategorical` for fixed options, `InputText` for free-form). Second, call
# MAGIC `create_labeling_session` with the schema names and the workspace user emails who should
# MAGIC review, then `session.add_traces(...)` to attach the queue. Both are version-controllable
# MAGIC in this bundle.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multiple SMEs on the same session
# MAGIC
# MAGIC ![Multi-reviewer](./images/hd_multi_reviewer.png)
# MAGIC
# MAGIC `assigned_users` is a flat list - every assigned SME sees every trace in the session, and
# MAGIC there is no UI-level round-robin or quota selector today. Every assessment is
# MAGIC source-attributed (the SME's email is stored on each one), so nothing overwrites and
# MAGIC who-said-what stays queryable. Disagreement across SMEs is the high-signal input that
# MAGIC `judge.align` consumes later.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-sdk databricks-agents
# MAGIC %restart_python

# COMMAND ----------

import mlflow
from mlflow.client import MlflowClient

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
# Override SME_REVIEWERS with the workspace emails that should label traces.
SME_REVIEWERS = [CURRENT_USER]
# ---------------------------------------------------------------------------

mlflow.set_experiment(EXPERIMENT_PATH)

client = MlflowClient()
experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
if experiment is None:
    raise RuntimeError(
        f"Experiment not found: {EXPERIMENT_PATH}. Run notebook 01 first."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step A. End-user thumbs feedback
# MAGIC
# MAGIC Simulating what an app would call after the user clicks thumbs-up or thumbs-down. The
# MAGIC hashed-trace-id picks roughly 70% positive to mimic real-world skew. `mlflow.log_feedback`
# MAGIC writes one assessment per trace, attributed to the end-user source.

# COMMAND ----------

candidate_traces = client.search_traces(
    experiment_ids=[experiment.experiment_id],
    filter_string="trace.name = 'answer_question'",
    max_results=50,
)
print(f"Found {len(candidate_traces)} answer_question traces")


def has_end_user_rating(trace) -> bool:
    for a in trace.info.assessments or []:
        if a.name == "end_user_rating":
            return True
    return False


to_label = [t for t in candidate_traces if not has_end_user_rating(t)]
print(f"  {len(to_label)} need feedback (rest already labeled)")

# COMMAND ----------

import hashlib
from mlflow.entities import AssessmentSource

POS_COMMENTS = [
    "Answer was clear and matched what I was looking for.",
    "Helpful, included the key term I needed.",
    "Concise and accurate.",
    "Good - linked the right concepts.",
]
NEG_COMMENTS = [
    "Missed the main point of the question.",
    "Too vague, did not answer specifically.",
    "Confused two concepts.",
    "Hallucinated content not in the source docs.",
]


def deterministic_thumbs(trace_id: str) -> tuple[bool, str]:
    h = int(hashlib.md5(trace_id.encode()).hexdigest()[:8], 16)
    is_positive = (h % 10) < 7
    pool = POS_COMMENTS if is_positive else NEG_COMMENTS
    return is_positive, pool[h % len(pool)]


for t in to_label:
    trace_id = t.info.trace_id
    is_positive, comment = deterministic_thumbs(trace_id)
    mlflow.log_feedback(
        trace_id=trace_id,
        name="end_user_rating",
        value=is_positive,
        source=AssessmentSource(
            source_type="HUMAN",
            source_id="end_user@example.com",
        ),
        rationale=comment,
    )

print(f"Logged feedback on {len(to_label)} traces")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step B. Review App labeling session
# MAGIC
# MAGIC Two cells. First define the typed schemas the SME will fill out (idempotent - existing
# MAGIC schemas are reused). Then create the session, assign the reviewer, and attach the first
# MAGIC 20 traces. The cell prints the Review App URL the SME opens to start labeling.

# COMMAND ----------

from mlflow.genai.label_schemas import (
    create_label_schema,
    InputCategorical,
    InputText,
)
from mlflow.genai.labeling import create_labeling_session


def _get_or_create_schema(name, type, title, input):
    try:
        return create_label_schema(name=name, type=type, title=title, input=input)
    except Exception as e:
        if "must be unique" in str(e) or "Duplicate" in str(e):
            return None
        raise


_get_or_create_schema(
    name="groundedness",
    type="feedback",
    title="Is the answer grounded in the retrieved documents?",
    input=InputCategorical(options=["yes", "no", "partial"]),
)

_get_or_create_schema(
    name="relevance",
    type="feedback",
    title="Is the answer relevant to the question?",
    input=InputCategorical(options=["yes", "no", "partial"]),
)

_get_or_create_schema(
    name="rationale",
    type="feedback",
    title="Optional reasoning",
    input=InputText(max_length=500),
)

# COMMAND ----------

import time

session_name = f"genai_evals_labeling_session_{int(time.time())}"
session = create_labeling_session(
    name=session_name,
    label_schemas=["groundedness", "relevance", "rationale"],
    assigned_users=SME_REVIEWERS,
)
session.add_traces(candidate_traces[:20])
print(f"Labeling session URL: {session.url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What the SME sees
# MAGIC

# MAGIC
# MAGIC Left side is the trace - question, retrieved docs, agent answer. Right side is the form
# MAGIC built from the schemas above. The SME picks a category, writes a rationale, hits save. The
# MAGIC entry posts as an `mlflow.HUMAN`-source assessment on the trace.
# MAGIC
# MAGIC **Important: the thumbs feedback from Step A does NOT show in this Review App form.**
# MAGIC The Review App form only shows the schemas you defined for the session - `groundedness`,
# MAGIC `relevance`, `rationale`. The end-user thumbs from Step A are stored as a different named
# MAGIC assessment (`end_user_rating`) on the same trace, and they surface on the trace's
# MAGIC **Assessments** panel - open the experiment, click any trace from notebook 01, and the
# MAGIC `end_user_rating: true/false` entry from `end_user@example.com` is right there alongside the
# MAGIC labels SMEs post here. Both write into the same trace's assessment list. The Review App is
# MAGIC just one of several ways to capture human signal.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Blast-radius note
# MAGIC
# MAGIC The Share button on a labeling session grants reviewers CAN_EDIT on the *parent experiment*.
# MAGIC In a single shared experiment that means SMEs can see all traces, not just the ones in the
# MAGIC session. If that matters for the workload, deploy the labeling notebook against a dedicated
# MAGIC experiment with only the trace slice the SMEs should see, and grant Share permissions only
# MAGIC on that one.

# COMMAND ----------

# MAGIC %md
# MAGIC ## External annotator workarounds
# MAGIC
# MAGIC The built-in Review App requires reviewers to be Databricks workspace users. For external
# MAGIC annotators (contractors, vendor labeling teams, anyone without a Databricks login) there
# MAGIC are two supported paths per the MLflow team:
# MAGIC
# MAGIC 1. **Account-only access (no workspace access).** Add the external annotator at the
# MAGIC    Databricks account level but not the workspace level. They get access to the Review App
# MAGIC    surface but no SQL warehouses, no notebooks, no general workspace functionality. Lowest-
# MAGIC    surface-area workaround if your account admin supports it.
# MAGIC
# MAGIC 2. **Custom Databricks App on the published Review App template.** Databricks publishes a
# MAGIC    template that you can clone into your own Databricks App. You control the auth model
# MAGIC    entirely (SSO, custom JWT, whatever). Annotations the external user submits write back
# MAGIC    to MLflow via the same API the built-in Review App uses. Recommended for PDF or other
# MAGIC    multimedia workflows where the built-in app falls short.
# MAGIC
# MAGIC With either workaround you do not need to enroll every external annotator into your
# MAGIC Databricks workspace, which is the SCIM-provisioning concern most labeling-ops teams hit.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Editing or revising assessments
# MAGIC
# MAGIC Same reviewer can change their mind. If `alice@example.com` previously marked a trace
# MAGIC `groundedness=yes` and re-opens the Review App on that trace, she can save a new value
# MAGIC (e.g. `partial`) and the latest value is what aggregation and `judge.align` will read.
# MAGIC The earlier assessment is preserved in the trace's assessment history but the latest from
# MAGIC each `source_id` wins for downstream consumers.
# MAGIC
# MAGIC A different reviewer cannot overwrite Alice's assessment - they post their own under
# MAGIC their own `source_id`. When two reviewers disagree, you pick the aggregation policy at
# MAGIC eval time (majority vote, expert override, consensus required, etc.) - the data layer
# MAGIC gives you the mechanism, your eval pipeline decides the policy.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to verify in your workspace
# MAGIC
# MAGIC 1. Open the experiment. Click the **Labeling Sessions** tab. Find the `genai_evals_labeling_session_*`
# MAGIC    row that just got created.
# MAGIC 2. Click into the session. The Configure dialog shows the three schemas selected:
# MAGIC    `groundedness`, `relevance`, `rationale`.
# MAGIC 3. Click **Open Review App**. The form on the right matches the schemas. Label one trace
# MAGIC    end-to-end.
# MAGIC 4. Go back to the experiment **Traces** tab. The trace you just labeled has a new assessment
# MAGIC    with `source_type=HUMAN` and your email as `source_id`.
# MAGIC 5. The thumbs-feedback assessments from Step A appear alongside, attributed to
# MAGIC    `end_user@example.com`. Same trace, different sources, both queryable.
# MAGIC
# MAGIC Continue to `03_offline_evals`.
