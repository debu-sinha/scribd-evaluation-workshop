# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Traces in UC and Datadog dual-export
# MAGIC
# MAGIC Two questions this notebook answers. First, how do traces get into Unity Catalog so SQL works
# MAGIC against them. Second, how do you keep Datadog as the observability stack while running
# MAGIC evals on Databricks. This notebook answers both with the exact code path.

# COMMAND ----------

# MAGIC %md
# MAGIC ## How traces land in UC
# MAGIC
# MAGIC ![OTel + UC concept](./images/hd_otel_uc_integration_concept.png)
# MAGIC
# MAGIC Three trace surfaces, three different enablement stories:
# MAGIC
# MAGIC 1. **MLflow trace UI in the experiment** - automatic. Every `@mlflow.trace` decorator
# MAGIC    populates the managed trace store. This is what notebook 01 showed.
# MAGIC 2. **`MlflowClient.search_traces(...)` Python API** - automatic. Same managed store.
# MAGIC 3. **Traces in Unity Catalog as Delta tables, queryable with SQL** - NOT automatic.
# MAGIC    Public Preview today. A workspace admin enables two previews under the workspace
# MAGIC    **Previews** settings page:
# MAGIC    - **"OpenTelemetry on Databricks"**
# MAGIC    - **"Variant Shredding for Optimized Read Performance on Semi-Structured Data"**
# MAGIC
# MAGIC    A trace catalog and schema must also be provisioned. Documented caps:
# MAGIC    - **200 traces/second per workspace**
# MAGIC    - **100 MB/second per table**
# MAGIC
# MAGIC If a workspace has not been enabled, surfaces 1 and 2 still work. SQL access waits on
# MAGIC the preview flip.

# COMMAND ----------

# MAGIC %pip install -U -qqqq mlflow databricks-sdk
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
# ---------------------------------------------------------------------------

mlflow.set_experiment(EXPERIMENT_PATH)

client = MlflowClient()
experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Surface 2 - the Python API works today
# MAGIC
# MAGIC `client.search_traces(...)` reads from the managed store regardless of UC enablement. This
# MAGIC is the fastest way to confirm traces are landing.

# COMMAND ----------

if experiment:
    traces = client.search_traces(
        experiment_ids=[experiment.experiment_id], max_results=10
    )
    print(f"Recent traces: {len(traces)}")
    for t in traces[:5]:
        print(f"  {t.info.trace_id} | {t.info.request_time}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Surface 3 - SQL once Traces-in-UC is provisioned
# MAGIC
# MAGIC Once the workspace admin has flipped the Public Preview switch and a trace catalog plus
# MAGIC schema are configured, the same trace data is queryable as a regular Delta table. The
# MAGIC `session_id` and `user_id` from notebook 01 become first-class filter columns.
# MAGIC
# MAGIC ```sql
# MAGIC SELECT trace_id,
# MAGIC        experiment_id,
# MAGIC        timestamp_ms,
# MAGIC        tags['mlflow.trace.user']    AS user_id,
# MAGIC        tags['mlflow.trace.session'] AS session_id
# MAGIC FROM   <catalog>.<schema>.traces
# MAGIC WHERE  timestamp_ms > date_sub(current_timestamp(), 1)
# MAGIC LIMIT  20;
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Shipping your existing traces to Databricks-managed MLflow
# MAGIC
# MAGIC Two paths you can pick from on day one. Pick the one that matches where the agent runs
# MAGIC today.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Path A. Agent runs on Databricks - just decorate
# MAGIC
# MAGIC No exporter setup required. Point at an experiment and add `@mlflow.trace` to the functions
# MAGIC you want traced. Auth comes from the workspace context.
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC from mlflow.entities import SpanType
# MAGIC
# MAGIC mlflow.set_experiment("/Workspace/Users/<you>/agent_traces")
# MAGIC
# MAGIC @mlflow.trace(span_type=SpanType.CHAIN)
# MAGIC def answer_question(query, session_id, user_id):
# MAGIC     mlflow.update_current_trace(
# MAGIC         metadata={
# MAGIC             "mlflow.trace.user":    user_id,
# MAGIC             "mlflow.trace.session": session_id,
# MAGIC         }
# MAGIC     )
# MAGIC     # ... retrieve + generate ...
# MAGIC ```
# MAGIC
# MAGIC This is exactly what notebook 01 does. Traces flow into the managed store with no extra
# MAGIC infrastructure.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Path B. Agent runs outside Databricks - point MLflow tracking at the workspace
# MAGIC
# MAGIC If the agent runs in your own service (ECS, EKS, k8s, a laptop, anywhere), it needs to
# MAGIC authenticate to the Databricks workspace and write traces over the tracking API. Three
# MAGIC environment variables and the same decorator pattern.
# MAGIC
# MAGIC ```bash
# MAGIC # in the service's environment
# MAGIC export DATABRICKS_HOST="https://<customer-workspace>.cloud.databricks.com"
# MAGIC export DATABRICKS_TOKEN="<personal-or-service-principal-PAT>"
# MAGIC export MLFLOW_TRACKING_URI="databricks"
# MAGIC ```
# MAGIC
# MAGIC ```python
# MAGIC # in the service code - same decorator pattern as Path A
# MAGIC import mlflow
# MAGIC from mlflow.entities import SpanType
# MAGIC
# MAGIC mlflow.set_tracking_uri("databricks")
# MAGIC mlflow.set_experiment("/Workspace/Users/<you>/customer_prod_traces")
# MAGIC
# MAGIC @mlflow.trace(span_type=SpanType.CHAIN)
# MAGIC def answer_question(query, session_id, user_id):
# MAGIC     mlflow.update_current_trace(metadata={
# MAGIC         "mlflow.trace.user":    user_id,
# MAGIC         "mlflow.trace.session": session_id,
# MAGIC     })
# MAGIC     # ... your existing retrieve + generate ...
# MAGIC ```
# MAGIC
# MAGIC Production note: use a service-principal token from a workspace secret, not a personal PAT.
# MAGIC Grant the service principal CAN_EDIT on the experiment so it can write traces.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Datadog dual-export - how you wire this up
# MAGIC
# MAGIC ![Datadog options](./images/hd_datadog_walkback.png)
# MAGIC
# MAGIC Today your application's OTel SDK exports to Datadog. We add a second exporter on the same
# MAGIC `TracerProvider` pointing at the Databricks OTLP endpoint. One span stream, two exporters,
# MAGIC both stacks get every trace. Your Datadog pipeline does not change.
# MAGIC
# MAGIC ### Step 1. Find your existing OTel setup in your service code
# MAGIC
# MAGIC Most LangChain / OpenAI / custom-agent stacks already use the OpenTelemetry SDK. The setup
# MAGIC typically looks something like this (yours may differ in details; the important parts are
# MAGIC the `TracerProvider` and the Datadog exporter attached to it):
# MAGIC
# MAGIC ```python
# MAGIC # what your service has today
# MAGIC from opentelemetry.sdk.trace import TracerProvider
# MAGIC from opentelemetry.sdk.trace.export import BatchSpanProcessor
# MAGIC
# MAGIC provider = TracerProvider()
# MAGIC provider.add_span_processor(BatchSpanProcessor(datadog_exporter))   # already there
# MAGIC trace.set_tracer_provider(provider)
# MAGIC ```
# MAGIC
# MAGIC ### Step 2. Mint a Databricks service-principal token for ingestion
# MAGIC
# MAGIC The Databricks OTLP endpoint authenticates with a workspace bearer token. Use a service
# MAGIC principal, store the token in a workspace secret, not a personal access token. Grant the
# MAGIC service principal CAN_EDIT on the destination experiment so it can write traces.
# MAGIC
# MAGIC ### Step 3. Add the Databricks OTLP exporter alongside the Datadog one
# MAGIC
# MAGIC One import, one exporter object, one extra `add_span_processor` call. That is the entire
# MAGIC application-side change.
# MAGIC
# MAGIC ```python
# MAGIC from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# MAGIC
# MAGIC dbx_exporter = OTLPSpanExporter(
# MAGIC     endpoint=(
# MAGIC         "https://<your-workspace>.databricks.com"
# MAGIC         "/api/2.0/otel/v1/traces"
# MAGIC     ),
# MAGIC     headers={
# MAGIC         "Authorization": f"Bearer {SP_TOKEN}",
# MAGIC         "X-Databricks-UC-Table-Name": (
# MAGIC             "<catalog>.<schema>.<traces_table>"
# MAGIC         ),
# MAGIC     },
# MAGIC )
# MAGIC
# MAGIC # add it next to your Datadog exporter - do NOT remove the Datadog one
# MAGIC provider.add_span_processor(BatchSpanProcessor(dbx_exporter))
# MAGIC ```
# MAGIC
# MAGIC ### Step 4. Add the session and user metadata
# MAGIC
# MAGIC The OTel span attributes the MLflow UI uses for filtering are `mlflow.trace.user` and
# MAGIC `mlflow.trace.session`. Set them on the root span of each request - exactly the same way
# MAGIC notebook 01 did it with `mlflow.update_current_trace`. If you are emitting raw OTel, set
# MAGIC them as span attributes:
# MAGIC
# MAGIC ```python
# MAGIC with tracer.start_as_current_span("answer_question") as span:
# MAGIC     span.set_attribute("mlflow.trace.user",    request.user_id)
# MAGIC     span.set_attribute("mlflow.trace.session", request.session_id)
# MAGIC     # ... your existing retrieve + generate ...
# MAGIC ```
# MAGIC
# MAGIC ### What this gets you, in one paragraph
# MAGIC
# MAGIC Same OTel spans land in both stacks on every request. Datadog continues to power your
# MAGIC existing dashboards, alerts, and oncall workflows - nothing changes there. Databricks
# MAGIC receives the identical span stream and adds the eval-grade view on top: assessments,
# MAGIC judges, the Review App, scheduled scorers, the maturity surface. Your team does not pick
# MAGIC one or the other. They get both, and you do not maintain two trace pipelines - just two
# MAGIC exporters on the one your application already has.
# MAGIC
# MAGIC ### What is NOT supported, said directly
# MAGIC
# MAGIC There is no documented connector that reads from Datadog and writes into Databricks. We
# MAGIC do not have a "Datadog as the only forwarder" pattern. If keeping Datadog as the only
# MAGIC OTel emitter is a hard requirement on your end, that is a scoping conversation - we will
# MAGIC come back to you on it rather than improvise something live.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Multimodal trace support
# MAGIC
# MAGIC What works today (per the MLflow tracing team):
# MAGIC
# MAGIC - **Image span attributes** - supported in OSS MLflow and managed MLflow today
# MAGIC - **Audio span attributes** - supported in OSS MLflow and managed MLflow today
# MAGIC - **Text + structured JSON** - fully supported
# MAGIC
# MAGIC What is on the roadmap:
# MAGIC
# MAGIC - **Dedicated multimodal tracing UI for PDFs** - in development at Databricks, committed by
# MAGIC   Euirim Choi (tracing eng) for the next few months window. OSS MLflow already supports
# MAGIC   the underlying data model.
# MAGIC
# MAGIC The recommended pattern for source documents (PDF, image, audio) that are large or
# MAGIC change frequently is the **URL + derivative** pattern: store the source in cloud
# MAGIC storage, log the URL as one span attribute, and the agent's derivative (parsed text,
# MAGIC OCR output, extracted fields) as a second span attribute on the same span. The trace
# MAGIC stays small, the SME reviewing in the Review App can load the source on demand, and
# MAGIC the derivative is queryable as a first-class field.
# MAGIC
# MAGIC ```python
# MAGIC import mlflow
# MAGIC
# MAGIC @mlflow.trace(span_type=mlflow.entities.SpanType.PARSER)
# MAGIC def parse_document(doc_url: str) -> dict:
# MAGIC     # Pull source URL onto the span - reviewer or downstream tool can fetch on demand
# MAGIC     mlflow.update_current_trace(metadata={"source_doc_uri": doc_url})
# MAGIC     parsed = ocr_or_parser(doc_url)
# MAGIC     # The derivative also lives on the span as the function's return value
# MAGIC     return parsed
# MAGIC ```
# MAGIC
# MAGIC This keeps the trace store at single-megabyte size per trace instead of inlining PDF bytes.

# COMMAND ----------

# MAGIC %md
# MAGIC ## The unified data model these surfaces share
# MAGIC
# MAGIC ![Unified data model](./images/hd_unified_data.png)
# MAGIC
# MAGIC Once the traces and assessments land, every downstream consumer reads from the same UC
# MAGIC catalog. SQL queries, Genie spaces, scheduled scorers, the MLflow UI - same governance,
# MAGIC same permissions, same data.

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to verify in your workspace
# MAGIC
# MAGIC 1. `client.search_traces(...)` returns the traces produced by notebook 01.
# MAGIC 2. The SQL query above runs on demo-west because Traces-in-UC is enabled there. On a fresh
# MAGIC    workspace, it will return "table not found" until admin enables the Preview.
# MAGIC 3. The three paths above cover all the agent-deployment topologies your team might land on.
# MAGIC    Path C is the canonical dual-export pattern for the Datadog pillar.

# COMMAND ----------

# MAGIC %md
# MAGIC # Closing - maturity statement
# MAGIC
# MAGIC Every capability this workshop touches, called out by status. GA is shippable today. Public
# MAGIC Preview is real today with documented caps. Beta has caveats worth knowing before you depend
# MAGIC on it. Not shipped means there is no first-class surface yet, and the workaround is named.
# MAGIC
# MAGIC | Capability | Status | Notes |
# MAGIC | --- | --- | --- |
# MAGIC | MLflow GenAI core (tracing, evaluate) | **GA** | Managed MLflow 3, default in current runtimes |
# MAGIC | `make_judge` custom LLM judges | **GA** | MLflow 3.4.0+ |
# MAGIC | Custom `@scorer` decorator | **GA** | Function-based form works in both offline and production paths |
# MAGIC | Third-party scorer adapters | **GA** | RAGAS, DeepEval, Arize, TruLens, documented adapters |
# MAGIC | Review App labeling sessions | **GA** | Managed MLflow 3 |
# MAGIC | OTel + Traces in Unity Catalog | **Public Preview** | 200 traces/sec workspace cap, 100 MB/sec table cap, admin enablement required |
# MAGIC | `session.id` / `user.id` span attributes | **Public Preview** | Set as OTel span attributes, queryable as first-class trace fields once Traces-in-UC is on |
# MAGIC | Scheduled scorers in production monitoring | **Beta** | 20-scorer cap, 15 to 20 minute warm-up, function-based `@scorer` only (class-based not supported) |
# MAGIC | OTel `tasks` semantic conventions UI | **Not shipped** | Data persists and is SQL-queryable, but no first-class UI rendering yet, no public roadmap date |
# MAGIC | Datadog as upstream forwarder | **Not documented** | Dual-export from your app's OTel SDK is the supported path |
# MAGIC | `judge.align()` optimizers | **GA** | SIMBA is the no-arg default (per current docs and source). GEPA and MemAlign also selectable. MemAlign was flagged as the planned future default by the MLflow team in May 2026, but as of the workshop's verification snapshot SIMBA is still the documented default. |
# MAGIC
# MAGIC The point of this table is to be honest about what is shippable today versus what needs a
# MAGIC workspace admin enabling a Preview, versus what does not exist yet. Put the orange and red
# MAGIC rows in your deployment plan with the workspace region and SME list when you scope the rollout.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Where we go from here
# MAGIC
# MAGIC 1. You send us your target workspace region so we can confirm Traces-in-UC Preview availability.
# MAGIC 2. You send us an account on your end we can grant access to for a longer pilot run, or you
# MAGIC    clone the bundle into your own workspace via `databricks bundle deploy`.
# MAGIC 3. We follow up in writing on anything we punted on today.
# MAGIC
# MAGIC Thanks for the calibrated-honesty mandate. It made the demo sharper.
