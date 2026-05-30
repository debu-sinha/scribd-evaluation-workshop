# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Custom Annotation App Pattern for PDF-grounded Review
# MAGIC
# MAGIC The built-in MLflow Review App has no PDF viewer. For document-parsing pipelines where
# MAGIC annotators need to see the original PDF while validating the parsed markdown or downstream
# MAGIC LLM output, the production answer is a custom Databricks App built on the Review App API.
# MAGIC
# MAGIC This notebook is the working skeleton of that App. Four pieces:
# MAGIC
# MAGIC 1. Run a small parsing-agent stand-in that attaches a source PDF URI to its trace.
# MAGIC 2. Pull that trace and read the PDF URI plus the parsed markdown from it.
# MAGIC 3. Render the PDF and the parsed markdown side by side for the annotator.
# MAGIC 4. Write the annotator's feedback back to the trace via `mlflow.log_feedback`.
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
# MAGIC ## Step 1. Upstream - a parsing-agent stand-in that attaches the source PDF
# MAGIC
# MAGIC In a real pipeline this would be the document-parsing stage (Mistral OCR, Azure Document
# MAGIC Intelligence, or `ai_extract` over a Delta table of source files). For the demo it is a
# MAGIC `@mlflow.trace`-decorated function that records the source PDF URI as trace metadata and
# MAGIC returns the parsed markdown. The trace is what the annotation app downstream reads from.

# COMMAND ----------

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType, SpanType

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


# A short public PDF used as the source-of-truth artifact for the demo.
# In production this would be a UC Volume path (/Volumes/<catalog>/<schema>/<vol>/<file>.pdf)
# or a presigned URL from your artifact store. Either resolves to the same iframe `src`.
SAMPLE_PDF_URL = (
    "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
)


@mlflow.trace(span_type=SpanType.PARSER, name="parse_document")
def parse_document(source_pdf_uri: str) -> dict:
    """Stand-in for a real document-parsing agent.

    The point is the trace shape. We attach the source PDF URI as trace metadata so the
    downstream annotation app can locate the original document. The parsed markdown goes
    in the response, which `mlflow.search_traces` exposes as a queryable field.
    """
    mlflow.update_current_trace(
        metadata={
            "source_pdf_uri": source_pdf_uri,
            "mlflow.trace.user": "doc_parser_service",
        }
    )
    # A real parser (OCR, ai_extract, etc.) would produce this. We hand-write it
    # so the demo doesn't need an OCR endpoint configured.
    parsed_markdown = (
        "# Sample document (parsed)\n\n"
        "## Section 1\n"
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. The parser captured the "
        "intro paragraph cleanly.\n\n"
        "## Section 2\n"
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Numbers and tables are where parsers tend to drift, so SMEs review those carefully.\n\n"
        "## Section 3\n"
        "Conclusion paragraph. The agent uses this markdown as input to a downstream LLM."
    )
    return {
        "source_pdf_uri": source_pdf_uri,
        "parsed_markdown": parsed_markdown,
    }


# Produce a few traces so the annotator has a queue to label.
for _ in range(3):
    parse_document(SAMPLE_PDF_URL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2. Pull the parsing traces and inspect the metadata
# MAGIC
# MAGIC `mlflow.search_traces` returns the parsing traces we just produced. Each one carries the
# MAGIC source PDF URI in `request_metadata` and the parsed markdown in `response`. In a real
# MAGIC annotator app this same call would filter for traces missing assessments from the current
# MAGIC annotator.

# COMMAND ----------

from mlflow.client import MlflowClient

exp = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
client = MlflowClient()

# Pull the parse_document traces specifically. The annotator app reads these directly.
parse_traces = client.search_traces(
    experiment_ids=[exp.experiment_id],
    filter_string="trace.name = 'parse_document'",
    max_results=10,
)
print(f"Loaded {len(parse_traces)} parse_document traces.")
if parse_traces:
    sample = parse_traces[0]
    print("\nSample trace:")
    print(f"  trace_id:        {sample.info.trace_id}")
    print(f"  request_metadata: {sample.info.request_metadata}")
    if sample.data:
        resp_preview = (
            sample.data.response[:200]
            if isinstance(sample.data.response, str)
            else str(sample.data.response)[:200]
        )
        print(f"  response_preview: {resp_preview}")
else:
    print(
        "No parse_document traces found. Run Step 1 first to produce them, then re-run this cell."
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


# Read the PDF URI and parsed markdown directly from the first parse_document trace.
# This is the exact mechanism the production annotation app would use: search for traces
# that need annotation, pull the source artifact pointer and the parser output from each
# trace, render them side by side.
import json


def _extract_pdf_uri_and_markdown(trace):
    """Pull the source PDF URI from trace metadata and the parsed markdown from the response."""
    pdf_uri = (trace.info.request_metadata or {}).get("source_pdf_uri")
    markdown = "(no parsed markdown found)"
    if trace.data and trace.data.response:
        try:
            payload = (
                json.loads(trace.data.response)
                if isinstance(trace.data.response, str)
                else trace.data.response
            )
            if isinstance(payload, dict):
                markdown = payload.get("parsed_markdown", markdown)
                # Fall back to the URI inside the response payload if metadata didn't carry it.
                pdf_uri = pdf_uri or payload.get("source_pdf_uri")
        except (ValueError, TypeError):
            pass
    return pdf_uri, markdown


if parse_traces:
    target_trace = parse_traces[0]
    pdf_uri, markdown = _extract_pdf_uri_and_markdown(target_trace)
    if not pdf_uri:
        print(
            "Could not find source_pdf_uri on the trace metadata. Re-run Step 1 to produce "
            "fresh traces with the expected metadata shape."
        )
    else:
        print(f"Rendering annotation view for trace_id={target_trace.info.trace_id}")
        render_annotation_view(
            trace_id=target_trace.info.trace_id,
            pdf_url=pdf_uri,
            parsed_markdown=markdown,
        )
else:
    print(
        "No parse_document traces to render. Run Step 1 to produce them and try again."
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
# Two assessments captured by the same annotator on the same parse_document trace:
#  - markdown_matches_pdf: did the parser capture what was in the PDF?
#  - source_attribution_correct: are the section headings correctly tied to the source PDF?
if parse_traces:
    target_trace = parse_traces[0]
    real_trace_id = target_trace.info.trace_id
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
        assessment_name="source_attribution_correct",
        value="partial",
        rationale="Section 2 numbers are correct. Section 3 conclusion drifts from the PDF.",
    )
else:
    print(
        "No parse_document traces to annotate. Run Step 1 to produce them, "
        "then re-run this cell."
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
