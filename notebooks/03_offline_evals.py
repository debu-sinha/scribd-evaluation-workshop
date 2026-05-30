# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Offline evals
# MAGIC
# MAGIC Run `mlflow.genai.evaluate` over a small golden dataset. Mix a code-based scorer with an
# MAGIC LLM judge so the score per row carries both deterministic signal and a model-judged
# MAGIC rationale.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What an offline eval is, and when to run one
# MAGIC
# MAGIC ![Offline evals concept](./images/hd_offline_evals_concept.png)
# MAGIC
# MAGIC The golden dataset is a fixed set of inputs with expected outputs that you trust. Run the
# MAGIC agent against it, score every row, look at the failure modes. This is the gate for shipping
# MAGIC a prompt change, a new retriever, or a different Foundation Model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The two scorer shapes you will use
# MAGIC
# MAGIC ![Scorer pattern](./images/hd_scorer_pattern.png)
# MAGIC
# MAGIC The `@scorer` decorator wraps a Python function. Inputs to the function are `outputs`,
# MAGIC `expectations`, and optionally `trace` - return a float and you have a metric. The
# MAGIC `make_judge` helper defines an LLM judge as a prompt template plus a model URI. Both produce
# MAGIC assessments on the trace, in the same table, queryable with the same SQL.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-sdk databricks-agents dspy
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import pandas as pd

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

# MAGIC %run ./_agent_lib

# COMMAND ----------

# MAGIC %md
# MAGIC ### The golden dataset
# MAGIC
# MAGIC Six rows on purpose. Four in-corpus to test whether the retriever returns the right doc
# MAGIC and the generator stays grounded. Two out-of-corpus to test refusal - the agent should say
# MAGIC "I don't know" rather than make something up.

# COMMAND ----------

eval_dataset = pd.DataFrame(
    [
        {
            "inputs": {"query": "How are MLflow traces structured?"},
            "expected_response": "MLflow traces use hierarchical spans for chain, retrieve, and generate steps.",
            "category": "in-corpus",
        },
        {
            "inputs": {"query": "Can I store OpenTelemetry traces in Unity Catalog?"},
            "expected_response": "Yes, OTel traces can be ingested into Unity Catalog Delta tables.",
            "category": "in-corpus",
        },
        {
            "inputs": {"query": "What is make_judge?"},
            "expected_response": "make_judge defines an LLM judge in MLflow.",
            "category": "in-corpus",
        },
        {
            "inputs": {
                "query": "What drift metrics does Lakehouse monitoring compute?"
            },
            "expected_response": "Lakehouse monitoring computes PSI, JS, KS, and chi-squared drift metrics.",
            "category": "in-corpus",
        },
        {
            "inputs": {"query": "What is the airspeed velocity of an unladen swallow?"},
            "expected_response": "I don't know.",
            "category": "out-of-corpus",
        },
        {
            "inputs": {"query": "What's the capital of France?"},
            "expected_response": "I don't know.",
            "category": "out-of-corpus",
        },
    ]
)
eval_dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ### Code-based scorer
# MAGIC
# MAGIC A keyword-overlap scorer that returns the fraction of expected keywords present in the
# MAGIC answer. Crude but deterministic - useful as a sanity gate before the more expensive judge
# MAGIC runs. Anything below 0.5 is a candidate for SME review.

# COMMAND ----------

from mlflow.genai.scorers import scorer


@scorer
def answer_contains_expected_keyword(outputs: dict, expectations: dict) -> float:
    answer = (outputs.get("answer") or "").lower()
    expected = (expectations.get("expected_response") or "").lower()
    if not expected:
        return 0.0
    keywords = [w for w in expected.split() if len(w) > 4]
    if not keywords:
        return 0.0 if not answer else 1.0
    matches = sum(1 for w in keywords if w in answer)
    return matches / len(keywords)


# COMMAND ----------

# MAGIC %md
# MAGIC ### LLM judge
# MAGIC
# MAGIC `make_judge` takes an instruction template with `{{ inputs }}` and `{{ outputs }}` slots and
# MAGIC a Databricks-served model. The output is a category - `yes`, `partial`, `no` - plus an
# MAGIC optional rationale. Same model that powers the agent powers the judge here, but they can
# MAGIC be different.

# COMMAND ----------

from mlflow.genai.judges import make_judge

relevance_judge = make_judge(
    name="relevance",
    instructions=(
        "You are evaluating whether an answer is relevant to a user's question.\n\n"
        "Question: {{ inputs }}\n"
        "Answer: {{ outputs }}\n\n"
        "Reply with exactly one word from this set: yes, partial, no.\n"
        "- yes: the answer directly addresses the question.\n"
        "- partial: the answer addresses some aspect but misses key parts.\n"
        "- no: the answer does not address the question or hallucinates.\n"
        "Reply with only the single word, no punctuation, no explanation."
    ),
    model="databricks:/databricks-claude-sonnet-4-6",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the eval
# MAGIC
# MAGIC `mlflow.genai.evaluate` calls the predict function once per row, scores it with each scorer,
# MAGIC and writes the whole thing as an evaluation run on the experiment. After the cell finishes,
# MAGIC the Evaluations tab shows per-row scores and the judge rationale.

# COMMAND ----------


def run_agent(query: str) -> dict:
    return answer_question(
        query=query, session_id="eval-session", user_id="eval-runner@example.com"
    )


results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=run_agent,
    scorers=[answer_contains_expected_keyword, relevance_judge],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## What it looks like in the workspace
# MAGIC
# MAGIC ![Eval runs](./images/hd_03_eval_runs.png)
# MAGIC
# MAGIC The Evaluations tab shows one run per evaluation, with each scorer as a column. Click into
# MAGIC a row to see the prompt the judge saw, the model's exact response, and the resulting score.
# MAGIC This is the gate you put in front of a prompt change before shipping.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Close the loop - judge.align()
# MAGIC
# MAGIC The judge above captures one engineer's intuition about what relevance means. SMEs often
# MAGIC have a stricter standard, or care about edge cases the prompt missed. `judge.align()` reads
# MAGIC paired HUMAN + LLM_JUDGE assessments on the same trace and rewrites the judge's instructions
# MAGIC to maximize agreement with the human labels. The original judge stays untouched; you get a
# MAGIC new aligned judge object back.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step A. Make sure judge assessments exist on the trace surface
# MAGIC
# MAGIC The offline-eval run above scored 6 golden rows. To align, we want judge scores on the wider
# MAGIC trace population from notebook 01. We run the judge across the experiment's traces here -
# MAGIC the `mlflow.log_assessment` call attaches one LLM_JUDGE-source assessment per trace.

# COMMAND ----------

from mlflow.entities import AssessmentSource, AssessmentSourceType
from mlflow.client import MlflowClient

client = MlflowClient()
exp = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
traces = client.search_traces(
    experiment_ids=[exp.experiment_id],
    filter_string="trace.name = 'answer_question'",
    max_results=50,
)

scored = 0
for t in traces:
    already = any(
        a.name == "relevance" and a.source.source_type == AssessmentSourceType.LLM_JUDGE
        for a in (t.info.assessments or [])
    )
    if already:
        continue
    inputs = t.data.request if t.data else None
    outputs = t.data.response if t.data else None
    if not inputs or not outputs:
        continue
    assessment = relevance_judge(inputs={"question": str(inputs)}, outputs=str(outputs))
    mlflow.log_assessment(trace_id=t.info.trace_id, assessment=assessment)
    scored += 1

print(f"Added LLM_JUDGE relevance assessments to {scored} traces.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step B. Make sure SME labels exist on the same trace surface
# MAGIC
# MAGIC In production, SMEs working through the labeling session from notebook 02 supply the HUMAN
# MAGIC labels. For a runnable demo without making someone label live, we fall back to a synthetic
# MAGIC SME rule when fewer than 10 real human labels exist on the `relevance` schema. The rule is
# MAGIC intentionally stricter than the judge so alignment has signal to learn from. Replace this
# MAGIC fallback with real SME labels before drawing conclusions about the agreement numbers.

# COMMAND ----------

traces = client.search_traces(
    experiment_ids=[exp.experiment_id],
    filter_string="trace.name = 'answer_question'",
    max_results=50,
)
human_count = sum(
    1
    for t in traces
    for a in (t.info.assessments or [])
    if a.name == "relevance" and a.source.source_type == AssessmentSourceType.HUMAN
)

print(f"Found {human_count} existing HUMAN relevance labels.")
print(f"Total {len(traces)} answer_question traces in the experiment.")

# Always top up to at least len(traces) by filling in synthetic HUMAN labels on traces
# that don't have one. This keeps the demo runnable end-to-end without requiring someone
# to label live via the Review App. The synthetic rule is intentionally stricter than the
# judge so alignment has signal to learn from. Replace this fallback with real SME labels
# before drawing conclusions about the agreement numbers.
synthetic = 0
for t in traces:
    if any(
        a.name == "relevance" and a.source.source_type == AssessmentSourceType.HUMAN
        for a in (t.info.assessments or [])
    ):
        continue
    question = str(t.data.request or "").lower() if t.data else ""
    answer = str(t.data.response or "").lower() if t.data else ""
    if not answer.strip() or "don't know" in answer or "don't have" in answer:
        value = "no"
    else:
        q_tokens = {tok for tok in question.split() if len(tok) > 4}
        overlap = sum(1 for tok in q_tokens if tok in answer)
        if overlap >= 2:
            value = "yes"
        elif overlap == 1:
            value = "partial"
        else:
            value = "no"
    try:
        mlflow.log_feedback(
            trace_id=t.info.trace_id,
            name="relevance",
            value=value,
            source=AssessmentSource(
                source_type=AssessmentSourceType.HUMAN,
                source_id="demo_synthetic_sme",
            ),
            rationale="Demo fallback - replace with real SME labels in production.",
        )
        synthetic += 1
    except Exception as e:
        print(f"  skipped trace {t.info.trace_id}: {e}")
print(f"Logged {synthetic} synthetic HUMAN labels.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step C. Pair the assessments and measure baseline agreement

# COMMAND ----------

import time

# Assessment writes have eventual consistency. Retry the pairing search a few times so
# fresh writes from Steps A and B have time to land on the trace surface before pairing.
paired = []
rows = []
for attempt in range(5):
    traces = client.search_traces(
        experiment_ids=[exp.experiment_id],
        filter_string="trace.name = 'answer_question'",
        max_results=50,
    )
    paired = []
    rows = []
    for t in traces:
        judge_vals = [
            a.feedback.value
            for a in (t.info.assessments or [])
            if a.name == "relevance"
            and a.source.source_type == AssessmentSourceType.LLM_JUDGE
        ]
        human_vals = [
            a.feedback.value
            for a in (t.info.assessments or [])
            if a.name == "relevance"
            and a.source.source_type == AssessmentSourceType.HUMAN
        ]
        if judge_vals and human_vals:
            paired.append(t)
            rows.append(
                {
                    "trace_id": t.info.trace_id,
                    "judge": judge_vals[-1],
                    "human": human_vals[-1],
                }
            )
    if len(paired) >= 10:
        break
    print(
        f"  attempt {attempt + 1}: {len(paired)} pairs, waiting for write propagation..."
    )
    time.sleep(8)

paired_df = pd.DataFrame(rows)
print(f"Paired traces: {len(paired)} (need at least 5 for align())")
baseline_agreement = (
    (paired_df.judge == paired_df.human).mean() if len(paired) >= 10 else None
)
if baseline_agreement is not None:
    print(f"Baseline agreement (judge vs human): {baseline_agreement:.0%}")
paired_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step D. Run alignment with MemAlign (idempotent, with rate-limit fallback)
# MAGIC
# MAGIC `judge.align(paired, optimizer=...)` runs an optimizer over the judge's instruction string
# MAGIC to maximize agreement with the human labels. Returns a new judge object; the original is
# MAGIC untouched.
# MAGIC
# MAGIC We use **MemAlign** (`MemAlignOptimizer` from `mlflow.genai.judges.optimizers`) - the
# MAGIC recommended default per the MLflow team as of 2026-05-13. It combines semantic memory
# MAGIC (general guidelines extracted from SME rationales) with episodic memory (concrete examples
# MAGIC retrieved at scoring time), and is cheaper + faster than the prior SIMBA default. The
# MAGIC reflection model is what reads SME rationales and proposes guidelines.
# MAGIC
# MAGIC SIMBA still works if you call `relevance_judge.align(paired)` without an optimizer
# MAGIC argument. **GEPA** (`GEPAAlignmentOptimizer`) is the strongest alternative when SME
# MAGIC rationales are especially rich; it needs the `gepa[confidence]` pip extra.
# MAGIC
# MAGIC Two production-readiness details baked in:
# MAGIC
# MAGIC 1. **Idempotent.** If a calibrated `relevance` judge is already registered for this experiment,
# MAGIC    we fetch it via `get_scorer` and skip the expensive alignment pass. That keeps re-running this
# MAGIC    notebook cheap and lets you re-execute Steps E and F on a saved calibrated judge.
# MAGIC 2. **Rate-limit resilient.** Alignment makes many LLM calls. On workspaces with default
# MAGIC    Foundation Model API rate limits, you may hit `REQUEST_LIMIT_EXCEEDED`. We catch that,
# MAGIC    print a clear message, and fall back to the unaligned judge for downstream steps so the
# MAGIC    notebook still completes end-to-end.

# COMMAND ----------

from mlflow.genai.scorers import get_scorer
from mlflow.genai.judges.optimizers import MemAlignOptimizer
from mlflow.exceptions import MlflowException

aligned_judge = None
alignment_ran = False
rate_limited = False

# Idempotent path - reuse a previously-registered calibrated judge if it exists.
try:
    aligned_judge = get_scorer(experiment_id=exp.experiment_id, name="relevance")
    print(
        "Found a previously-registered calibrated 'relevance' judge - skipping alignment."
    )
    print("(Delete the registered scorer from the experiment to re-run alignment.)")
except Exception:
    pass

# Graceful fallback if not enough paired traces for alignment to converge.
if aligned_judge is None and len(paired) < 5:
    print(
        f"Only {len(paired)} paired traces found. Alignment needs more signal to converge."
    )
    print(
        "Falling back to the unaligned judge so the rest of the notebook still runs. "
        "Re-run notebook 01 to produce more traces, or label more via the Review App, "
        "then re-run this cell to take the alignment path."
    )
    aligned_judge = relevance_judge

# If no cached judge and enough paired traces, run MemAlign alignment now.
# Cap the paired set to 15 to keep LLM call volume bounded.
if aligned_judge is None:
    paired_for_align = paired[:15]
    print(
        f"Running judge.align(optimizer=MemAlign) on {len(paired_for_align)} paired traces..."
    )

    # MemAlign needs a reflection model to extract semantic memory from SME rationales,
    # plus an embedding model to pick relevant episodic-memory examples at evaluation
    # time. Per the MLflow team this is the new recommended default, replacing SIMBA.
    # SIMBA still works if you call relevance_judge.align(paired) without an optimizer
    # argument. Both reflection_lm and embedding_model are pinned to Databricks
    # foundation endpoints so the optimizer never reaches out to an external provider.
    memalign_optimizer = MemAlignOptimizer(
        reflection_lm="databricks:/databricks-claude-sonnet-4-6",
        embedding_model="databricks:/databricks-bge-large-en",
    )

    try:
        aligned_judge = relevance_judge.align(
            paired_for_align, optimizer=memalign_optimizer
        )
        alignment_ran = True
        print(f"Alignment complete. Aligned judge: {aligned_judge.name}")
        print("\n--- Original instructions (head) ---")
        print(relevance_judge.instructions[:300] + "...")
        print("\n--- Aligned instructions (head) ---")
        print(aligned_judge.instructions[:300] + "...")
    except (MlflowException, Exception) as e:
        msg = str(e)
        if "REQUEST_LIMIT_EXCEEDED" in msg or "Too Many Requests" in msg:
            rate_limited = True
            print(
                "Foundation Model API rate limit hit during MemAlign alignment.\n"
                "  Falling back to the unaligned judge for Steps E and F.\n"
                "  To run alignment cleanly: re-run this cell off-peak, request a\n"
                "  higher FMAPI tier, or point MemAlignOptimizer at your own model endpoint."
            )
            aligned_judge = relevance_judge
        else:
            # Any other alignment failure (missing embedding model, API auth issue,
            # algorithm convergence error, etc.) falls back to the unaligned judge so
            # the rest of the notebook still completes. The customer sees a clear
            # diagnostic of what went wrong.
            rate_limited = True
            print(
                f"MemAlign alignment failed: {type(e).__name__}: {msg[:300]}\n"
                "  Falling back to the unaligned judge for Steps E and F.\n"
                "  This is a soft failure - the rest of the notebook still runs end to end."
            )
            aligned_judge = relevance_judge

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step E. Score with the aligned judge and measure the lift

# COMMAND ----------

if rate_limited:
    print(
        "Skipping Step E - alignment was rate-limited above, so aligned_judge equals\n"
        "the unaligned baseline. No lift to measure on this run."
    )
else:
    aligned_rows = []
    rate_limit_in_step_e = False
    for t in paired:
        inputs = t.data.request if t.data else None
        outputs = t.data.response if t.data else None
        if not inputs or not outputs:
            continue
        try:
            aligned_assessment = aligned_judge(
                inputs={"question": str(inputs)}, outputs=str(outputs)
            )
        except MlflowException as e:
            if "REQUEST_LIMIT_EXCEEDED" in str(e) or "Too Many Requests" in str(e):
                rate_limit_in_step_e = True
                break
            raise
        mlflow.log_assessment(trace_id=t.info.trace_id, assessment=aligned_assessment)
        aligned_rows.append(
            {
                "trace_id": t.info.trace_id,
                "aligned_judge": aligned_assessment.feedback.value,
            }
        )

    if rate_limit_in_step_e:
        print(
            "Foundation Model API rate limit hit while re-scoring with the aligned judge.\n"
            "  Partial results in 'aligned_rows' are still usable. Re-run this cell off-peak\n"
            "  to complete the agreement lift comparison."
        )
    elif aligned_rows:
        aligned_df = pd.DataFrame(aligned_rows).merge(paired_df, on="trace_id")
        new_agreement = (aligned_df.aligned_judge == aligned_df.human).mean()
        print(f"Baseline agreement (judge vs human):  {baseline_agreement:.0%}")
        print(f"Aligned agreement  (aligned vs human): {new_agreement:.0%}")
        print(
            f"Lift:                                 {(new_agreement - baseline_agreement) * 100:+.1f} pp"
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step F. Register the aligned judge for production use
# MAGIC
# MAGIC `aligned_judge.register(experiment_id=...)` saves the calibrated judge as a named scorer in
# MAGIC the experiment. Notebook 04's scheduled monitoring picks it up automatically.

# COMMAND ----------

if rate_limited:
    print(
        "Skipping Step F - alignment was rate-limited, so there is no calibrated judge\n"
        "to register on this run. Notebook 04 will fall back to the baseline judge until\n"
        "a successful alignment run registers a calibrated one."
    )
else:
    try:
        aligned_judge.register(experiment_id=exp.experiment_id)
        print(
            f"Registered aligned 'relevance' judge against experiment {exp.experiment_id}."
        )
    except Exception as e:
        if "already been registered" in str(e):
            print("Aligned 'relevance' judge already registered. Re-run is safe.")
        else:
            raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Before and after - what the loop produces
# MAGIC
# MAGIC ![Before/after improvement](./images/hd_before_after_improvement.png)
# MAGIC
# MAGIC Agreement goes up on the categories where the judge and SMEs disagreed most. The numbers in
# MAGIC this image are illustrative; the live numbers above are the real ones for this run.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Versioning judges via the Prompt Registry
# MAGIC
# MAGIC Managed MLflow does not have first-class versioning for registered judges today (this is a
# MAGIC known gap, surfaced live by the MLflow team). The workaround uses the **Prompt Registry**:
# MAGIC every alignment pass produces a new version of the judge's instruction string, and you
# MAGIC pin a specific version when constructing a `make_judge` to evaluate against.
# MAGIC
# MAGIC The pattern:
# MAGIC
# MAGIC ```python
# MAGIC # After each successful judge.align run, register the aligned instructions as a new
# MAGIC # version under a stable prompt name. Each align() pass = one new version.
# MAGIC mlflow.genai.register_prompt(
# MAGIC     name="relevance_judge_instructions",
# MAGIC     template=aligned_judge.instructions,
# MAGIC     commit_message="aligned 2026-05-14 against 22 SME labels, +12pp agreement",
# MAGIC )
# MAGIC
# MAGIC # Later, construct a judge that pins a specific version - useful for A/B comparing two
# MAGIC # aligned judges side by side before promoting one to production.
# MAGIC pinned_judge = make_judge(
# MAGIC     name="relevance_v3",
# MAGIC     instructions=mlflow.genai.load_prompt("prompts:/relevance_judge_instructions/3"),
# MAGIC     model="databricks:/databricks-claude-sonnet-4-6",
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC This gives you a full version history (which align pass produced which instruction
# MAGIC template, with commit message + timestamps), reproducibility across runs, and side-by-side
# MAGIC A/B testing before promoting. Until first-class judge versioning lands in Managed MLflow,
# MAGIC this is the documented pattern.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to verify on the call
# MAGIC
# MAGIC 1. Open the experiment. Click the **Evaluations** tab. Find the run that just completed.
# MAGIC 2. Per-row scores are visible for both `answer_contains_expected_keyword` and `relevance`.
# MAGIC 3. Click into the two out-of-corpus rows. The judge should mark them `yes` if the agent
# MAGIC    refused, `no` if it made something up.
# MAGIC 4. Click the **Traces** tab. The evaluation runs produced new traces tagged with
# MAGIC    `user_id=eval-runner@example.com` - same surface as production traces, different session.
# MAGIC
# MAGIC Continue to `04_production_monitoring`.
