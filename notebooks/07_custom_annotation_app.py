# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Custom Annotation App Pattern for PDF-grounded Review
# MAGIC
# MAGIC The built-in MLflow Review App has no PDF viewer. For document-parsing pipelines where
# MAGIC annotators need to see the original PDF while validating the parsed markdown or downstream
# MAGIC LLM output, the production answer is a custom Databricks App built on the Review App API.
# MAGIC
# MAGIC This notebook is the working skeleton of that App. Three pieces:
# MAGIC
# MAGIC 1. Pull a trace plus the attached source PDF from the experiment.
# MAGIC 2. Render the PDF and the parsed markdown side by side for the annotator.
# MAGIC 3. Write the annotator's feedback back to the trace via `mlflow.log_feedback`.
# MAGIC
# MAGIC ![Custom annotation app concept](./images/hd_custom_annotation_app_concept.png)
# MAGIC
# MAGIC ## How storage and references work
# MAGIC
# MAGIC MLflow uses a split architecture. The relational backend store holds the run metadata,
# MAGIC including a URI pointer to the artifact location. The artifact store is cloud blob storage
# MAGIC (DBFS, UC Volumes, or S3 / ABFS / GCS depending on configuration). `mlflow.log_artifact`
# MAGIC writes the bytes to blob storage and records the URI in the run.
# MAGIC
# MAGIC Two patterns for PDFs on a trace:
# MAGIC
# MAGIC - **Inline via `log_artifact`** for files up to 5GB (the single-PUT upload cap on
# MAGIC   Databricks-managed storage). The PDF lives next to the run, fetched by URI.
# MAGIC - **UC Volume reference** via trace metadata for larger files or files that already live in
# MAGIC   UC. Set `span.set_attribute("source_pdf_uc_volume_path", "/Volumes/...")`. The custom app
# MAGIC   reads the metadata and serves directly from the Volume. No file movement.
# MAGIC
# MAGIC ## Productionizing
# MAGIC
# MAGIC The MLflow APIs below are the stable surface. The web layer is what changes. Estimated
# MAGIC effort to ship a real annotator app is about two weeks for the team that maintains the
# MAGIC existing labeling-ops UI: Databricks Apps deployment, PDF.js viewer, OAuth-based auth (no
# MAGIC SCIM into the workspace required for external annotators), queue management, round-robin
# MAGIC reviewer assignment.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1. Pull traces with source PDF references
# MAGIC
# MAGIC `mlflow.search_traces` returns recent traces from the experiment. In a real annotator app
# MAGIC this would filter for traces that are missing assessments from the current annotator.

# COMMAND ----------

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType

# Resolve the current Databricks workspace user the same way notebooks 01-06 do.
# In production the custom app reads annotator identity from its OAuth context,
# not from the workspace user.
CURRENT_USER = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()  # noqa: F821
)
EXPERIMENT_PARENT = f"/Workspace/Users/{CURRENT_USER}/genai_evals_demo"
EXPERIMENT_PATH = f"{EXPERIMENT_PARENT}/agent_traces"

# Ensure the parent folder exists before set_experiment - same defensive call
# notebooks 01-06 make so this notebook can be run standalone after a fresh sync.
from databricks.sdk import WorkspaceClient

WorkspaceClient().workspace.mkdirs(EXPERIMENT_PARENT)
mlflow.set_experiment(EXPERIMENT_PATH)

# Pull recent traces. The skeleton uses what notebook 01 produced;
# the production app would scope to the parse-pipeline run name.
traces_df = mlflow.search_traces(max_results=5)
print(f"Loaded {len(traces_df)} traces from {EXPERIMENT_PATH}")
if len(traces_df) > 0:
    # Pick scalar columns that survive Arrow conversion. Complex columns
    # (assessments, spans, metadata, tags) hold list-of-dict values that
    # Spark's Arrow serializer cannot handle when display() converts the
    # pandas DataFrame to a Spark DataFrame.
    scalar_cols = [
        c
        for c in [
            "trace_id",
            "request_id",
            "request_time",
            "state",
            "status",
            "request_preview",
            "response_preview",
        ]
        if c in traces_df.columns
    ]
    print(f"Trace columns: {list(traces_df.columns)}")
    if scalar_cols:
        display(traces_df[scalar_cols])  # noqa: F821
else:
    print(
        "No traces found. Run notebook 01 first to populate the agent_traces experiment, "
        "then re-run this notebook."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2. Render PDF + parsed markdown side by side
# MAGIC
# MAGIC `displayHTML` renders inline. The PDF is loaded by URL (a UC Volume path served through
# MAGIC the workspace's file serving, or a presigned URL from the artifact store). The markdown is
# MAGIC the parser output extracted from the trace span. In production these are independently
# MAGIC scrollable panels with a PDF.js viewer instead of an iframe.

# COMMAND ----------


def render_annotation_view(
    trace_id, pdf_url, parsed_markdown, downstream_response=None
):
    """Render a side-by-side PDF / markdown / response view for the annotator."""
    response_panel = ""
    if downstream_response:
        response_panel = f"""
      <div style="flex: 1; border: 1px solid #ddd; padding: 10px; overflow-y: auto; max-height: 660px;">
        <h3 style="margin-top: 0;">Downstream LLM response</h3>
        <div style="font-size: 13px; line-height: 1.5;">{downstream_response}</div>
      </div>
        """
    html = f"""
    <div style="display: flex; gap: 20px; font-family: -apple-system, sans-serif;">
      <div style="flex: 1; border: 1px solid #ddd; padding: 10px;">
        <h3 style="margin-top: 0;">Source PDF</h3>
        <iframe src="{pdf_url}" style="width: 100%; height: 600px; border: 0;"></iframe>
      </div>
      <div style="flex: 1; border: 1px solid #ddd; padding: 10px; overflow-y: auto; max-height: 660px;">
        <h3 style="margin-top: 0;">Parsed markdown (downstream input)</h3>
        <pre style="white-space: pre-wrap; font-size: 12px;">{parsed_markdown}</pre>
      </div>
      {response_panel}
    </div>
    <div style="margin-top: 12px; color: #666; font-size: 11px;">Trace: {trace_id}</div>
    """
    displayHTML(html)  # noqa: F821


# Demo render. In a real run the app reads pdf_url + markdown + downstream response
# from the trace's span attributes. We use a public sample PDF here so the skeleton
# renders without workspace-specific setup.
SAMPLE_PDF_URL = (
    "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
)
SAMPLE_MARKDOWN = """# Annual Report 2025

## Executive Summary
- Revenue grew 24% year over year
- Operating margin improved to 18.5%
- Customer count crossed 50,000

## Q4 Highlights
The fourth quarter saw strong momentum across all product lines, with enterprise
deals contributing the largest share of new ARR.
"""
SAMPLE_RESPONSE = (
    "Revenue grew 24% YoY with operating margin reaching 18.5%. "
    "Q4 was the strongest quarter, driven by enterprise."
)

render_annotation_view(
    trace_id="demo-trace-001",
    pdf_url=SAMPLE_PDF_URL,
    parsed_markdown=SAMPLE_MARKDOWN,
    downstream_response=SAMPLE_RESPONSE,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3. Write annotator feedback back to the trace
# MAGIC
# MAGIC `mlflow.log_feedback` posts the annotator's assessment using the same API the built-in
# MAGIC Review App uses. The annotator's identity goes in the `source.source_id`. Many annotators
# MAGIC writing to the same trace each get their own `source_id`; the data layer preserves all
# MAGIC assessments and the downstream aggregation policy (majority vote, expert override,
# MAGIC consensus required) decides which value wins at eval time.

# COMMAND ----------


def submit_annotation(trace_id, annotator_email, assessment_name, value, rationale=""):
    """Post an annotation back to the trace using the Review App API surface."""
    mlflow.log_feedback(
        trace_id=trace_id,
        name=assessment_name,
        value=value,
        rationale=rationale or None,
        source=AssessmentSource(
            source_type=AssessmentSourceType.HUMAN,
            source_id=annotator_email,
        ),
    )
    print(f"Logged: {assessment_name}={value!r} by {annotator_email}")


# Simulated annotator submission. In the production app this is form input on the UI.
# Two assessments captured by the same annotator on the same trace:
#  - markdown_matches_pdf: did the parser capture what was in the PDF?
#  - downstream_response_correct: did the downstream LLM answer correctly?
if len(traces_df) > 0:
    # MLflow renamed request_id to trace_id; support both for forward and backward compat.
    first_row = traces_df.iloc[0]
    real_trace_id = first_row.get("trace_id") or first_row.get("request_id")
    submit_annotation(
        trace_id=real_trace_id,
        annotator_email=CURRENT_USER,
        assessment_name="markdown_matches_pdf",
        value="yes",
        rationale="Parsed markdown captured all sections present in the source PDF.",
    )
    submit_annotation(
        trace_id=real_trace_id,
        annotator_email=CURRENT_USER,
        assessment_name="downstream_response_correct",
        value="partial",
        rationale="Q1 financials correct, missed the Q4 customer-count callout.",
    )
else:
    print(
        "No traces in this experiment yet. Run notebook 01 first to populate traces, "
        "then re-run this cell to attach annotations."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## What productionizing this looks like
# MAGIC
# MAGIC The MLflow APIs above are the stable surface. The web layer is what changes.
# MAGIC
# MAGIC | Notebook skeleton | Production Databricks App |
# MAGIC | --- | --- |
# MAGIC | `displayHTML` with iframe | Real React UI with PDF.js or Adobe PDF Embed |
# MAGIC | One trace at a time, cell by cell | Queue of unannotated traces, multi-user state |
# MAGIC | Hand-coded `submit_annotation` call | Form on the UI with validation |
# MAGIC | Notebook auth (workspace user) | App OAuth with external-annotator support |
# MAGIC | No assignment policy | Round-robin reviewer assignment, reviewer queue |
# MAGIC
# MAGIC The MLflow Review App API surface stays identical. Annotations from a custom app and from
# MAGIC the built-in Review App land on the same trace fields, so downstream eval logic
# MAGIC (`judge.align`, custom scorers, aggregation policies) treats them interchangeably.
# MAGIC
# MAGIC ## Markdown visibility through a multi-stage pipeline
# MAGIC
# MAGIC When a downstream agent fails, the annotator opens the trace, clicks the parse span, and
# MAGIC sees the markdown that was fed forward. They can tell immediately whether the failure was
# MAGIC bad parse or bad LLM response. Adding `@mlflow.trace` on a parser function is the one-line
# MAGIC change that gets you this; the span tree does the rest without any custom instrumentation.
