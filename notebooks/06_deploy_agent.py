# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Deploy the agent to a serving endpoint
# MAGIC
# MAGIC Wraps the `answer_question` chain from notebook 01 as an MLflow `ChatModel`, registers it in
# MAGIC Unity Catalog, and creates a Model Serving endpoint. Once the endpoint is `READY` you can
# MAGIC open the AI Playground in the workspace, pick this endpoint, and chat with the agent
# MAGIC interactively. Every Playground call writes a trace into the same `agent_traces` experiment
# MAGIC the rest of the bundle reads from.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Why deploy it
# MAGIC
# MAGIC The four working notebooks operate on traces produced inside notebooks. Production agents
# MAGIC live behind serving endpoints. Deploying the agent here closes that gap - you get a real
# MAGIC endpoint, queryable from the Playground, from `curl`, from your service, and it generates
# MAGIC the same trace shape that notebook 02 labels, notebook 03 evaluates, and notebook 04
# MAGIC monitors.

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

# Override these with your own UC catalog / schema before deploying.
UC_CATALOG = "main"
UC_SCHEMA = "default"
MODEL_NAME = f"{UC_CATALOG}.{UC_SCHEMA}.genai_eval_demo_agent"
ENDPOINT_NAME = "genai-eval-demo-agent"
# ---------------------------------------------------------------------------

mlflow.set_experiment(EXPERIMENT_PATH)

# COMMAND ----------

# MAGIC %run ./_agent_lib

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wrap the agent as a ChatModel
# MAGIC
# MAGIC `mlflow.pyfunc.ChatModel` is the Playground-compatible serving contract. The `predict`
# MAGIC method receives `ChatCompletionRequest` messages, calls our existing `answer_question`
# MAGIC function, and returns a `ChatCompletionResponse`. The `@mlflow.trace` decorators on the
# MAGIC underlying functions still fire, so every call writes a hierarchical trace.

# COMMAND ----------

from mlflow.pyfunc import ChatModel
from mlflow.types.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage as MlflowChatMessage,
)


class GenAIEvalDemoAgent(ChatModel):
    # Captured at log-time so the deployed serving container can re-establish the
    # MLflow experiment context inside predict(). Without this, traces produced
    # by @mlflow.trace decorators inside answer_question() do not land in the
    # named experiment when the model is queried from Playground.
    _experiment_path = EXPERIMENT_PATH

    def predict(self, context, messages, params=None):
        # Bind traces to the named experiment for every Playground / serving call.
        mlflow.set_experiment(self._experiment_path)

        # Pull the last user message - same shape Playground sends.
        if isinstance(messages, ChatCompletionRequest):
            msg_list = messages.messages
        else:
            msg_list = messages

        user_message = ""
        for m in reversed(msg_list):
            role = m.role if hasattr(m, "role") else m.get("role")
            if role == "user":
                user_message = m.content if hasattr(m, "content") else m.get("content")
                break

        session_id = "playground"
        user_id = "playground-user"
        if params:
            if isinstance(params, dict):
                session_id = params.get("session_id", session_id)
                user_id = params.get("user_id", user_id)

        result = answer_question(
            query=user_message,
            session_id=session_id,
            user_id=user_id,
        )

        return ChatCompletionResponse(
            choices=[
                ChatChoice(
                    index=0,
                    message=MlflowChatMessage(
                        role="assistant", content=result["answer"]
                    ),
                )
            ],
            model=ENDPOINT_NAME,
        )


# COMMAND ----------

# MAGIC %md
# MAGIC ### Log + register the model in Unity Catalog
# MAGIC
# MAGIC The agent is registered into Unity Catalog at `MODEL_NAME` set in the config block at the
# MAGIC top of this notebook. Default is `main.default.genai_eval_demo_agent`. Update the
# MAGIC `UC_CATALOG` / `UC_SCHEMA` variables for your own deployment.

# COMMAND ----------

from mlflow.models.resources import DatabricksServingEndpoint

mlflow.set_registry_uri("databricks-uc")

# Declaring the FMAPI endpoint as a resource tells Databricks Model Serving to
# auto-inject the credentials the agent needs to call it from inside the serving
# container. Without this, the WorkspaceClient() call in _agent_lib.py fails with
# "default auth: cannot configure default credentials" when the model is queried.
with mlflow.start_run(run_name="genai_eval_demo_agent_v1") as run:
    logged = mlflow.pyfunc.log_model(
        name="agent",
        python_model=GenAIEvalDemoAgent(),
        registered_model_name=MODEL_NAME,
        pip_requirements=[
            "mlflow",
            "databricks-sdk",
        ],
        resources=[
            DatabricksServingEndpoint(endpoint_name="databricks-claude-sonnet-4-6"),
        ],
    )
    print(f"Logged: {logged.model_uri}")
    print(f"Registered as: {MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get the latest version + create the serving endpoint
# MAGIC
# MAGIC Endpoint provisioning takes 5-15 minutes for the first deployment. Status moves through
# MAGIC `NOT_READY` -> `IN_PROGRESS` -> `READY`. Once READY, the endpoint shows up in the AI
# MAGIC Playground dropdown.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

w = WorkspaceClient()

# Resolve the latest version that was just registered.
from mlflow.tracking import MlflowClient

client_uc = MlflowClient(registry_uri="databricks-uc")
versions = client_uc.search_model_versions(f"name='{MODEL_NAME}'")
latest_version = max(int(v.version) for v in versions)
print(f"Latest version: {latest_version}")

# Route Playground / serving-endpoint traces to our named experiment via
# MLFLOW_EXPERIMENT_ID. This is the canonical Databricks pattern for binding
# inference traces to a workspace experiment - the in-predict() set_experiment
# call is a secondary safety net.
exp_obj = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
exp_id = exp_obj.experiment_id if exp_obj else ""
print(f"Routing serving-endpoint traces to experiment_id={exp_id}")

config = EndpointCoreConfigInput(
    name=ENDPOINT_NAME,
    served_entities=[
        ServedEntityInput(
            entity_name=MODEL_NAME,
            entity_version=str(latest_version),
            workload_size="Small",
            scale_to_zero_enabled=True,
            environment_vars={
                # Canonical trio for serving-endpoint tracing per
                # docs.databricks.com/aws/en/mlflow3/genai/tracing/prod-tracing and
                # verified empirically on 2026-05-18 by reading serving logs:
                # - ENABLE_MLFLOW_TRACING flips the @mlflow.trace decorators ON inside
                #   the serving container.
                # - MLFLOW_TRACKING_URI=databricks points the runtime at the Databricks
                #   MLflow store. Without this it creates a local SQLite store and the
                #   trace export then fails with RESOURCE_DOES_NOT_EXIST.
                # - MLFLOW_EXPERIMENT_ID routes captured traces to the named workspace
                #   experiment so AI Playground calls land where the rest of the bundle
                #   reads from.
                "ENABLE_MLFLOW_TRACING": "true",
                "MLFLOW_TRACKING_URI": "databricks",
                "MLFLOW_EXPERIMENT_ID": exp_id,
            },
        )
    ],
)

# Create or update the endpoint
try:
    existing = w.serving_endpoints.get(name=ENDPOINT_NAME)
    print(f"Endpoint exists. Updating to version {latest_version}...")
    w.serving_endpoints.update_config(
        name=ENDPOINT_NAME, served_entities=config.served_entities
    )
except Exception:
    print(f"Creating new endpoint: {ENDPOINT_NAME}...")
    w.serving_endpoints.create(name=ENDPOINT_NAME, config=config)

print(f"Endpoint URL: https://<workspace>/ml/endpoints/{ENDPOINT_NAME}")
print(
    f"Once READY, open AI Playground -> select '{ENDPOINT_NAME}' -> chat with the agent."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to verify in your workspace
# MAGIC
# MAGIC 1. **Serving** tab in the workspace nav. Find `genai-eval-demo-agent`. State = READY.
# MAGIC 2. **AI Playground** in the left nav. Endpoint dropdown shows `genai-eval-demo-agent`.
# MAGIC 3. Ask the agent a question - e.g. "How are MLflow traces structured?" - and confirm a
# MAGIC    real answer comes back from the corpus.
# MAGIC
# MAGIC ## Playground traces routing into the named experiment
# MAGIC
# MAGIC The three env vars on `environment_vars` above are the canonical Databricks pattern for
# MAGIC routing serving-endpoint traces (including AI Playground calls) into a named workspace
# MAGIC experiment. Verified on 2026-05-18 by reading the serving-container logs and confirming
# MAGIC fresh Playground traces land in `agent_traces` within seconds of the chat response.
# MAGIC
# MAGIC | env var | what it does |
# MAGIC | --- | --- |
# MAGIC | `ENABLE_MLFLOW_TRACING=true` | Flips the `@mlflow.trace` decorators ON inside the serving container. Without this, the decorators no-op at inference time. |
# MAGIC | `MLFLOW_TRACKING_URI=databricks` | Points the runtime at the Databricks MLflow store. Without this it creates a local SQLite store inside the container and trace export fails with RESOURCE_DOES_NOT_EXIST. |
# MAGIC | `MLFLOW_EXPERIMENT_ID=<id>` | Routes captured traces to the named workspace experiment so AI Playground calls land where the rest of the bundle reads from. |
# MAGIC
# MAGIC ### How to verify after `update_config` returns READY
# MAGIC
# MAGIC 1. Open AI Playground, select `genai-eval-demo-agent`, ask a question.
# MAGIC 2. Open the experiment Traces tab on `agent_traces`. The Playground call's trace appears
# MAGIC    within seconds, with the span tree `answer_question -> retrieve -> generate`.
# MAGIC 3. If a trace does not appear, check the serving container logs for export errors. The
# MAGIC    typical failure mode is an empty `MLFLOW_EXPERIMENT_ID`, which happens when
# MAGIC    `mlflow.get_experiment_by_name` returned None earlier in this notebook.
# MAGIC
# MAGIC ### Why the in-`predict()` `set_experiment` call is still there
# MAGIC
# MAGIC The `mlflow.set_experiment(EXPERIMENT_PATH)` call inside `ChatModel.predict()` is a safety
# MAGIC net for the case where the env vars on the served entity have not yet propagated to a
# MAGIC newly-warmed container. It's redundant when the env vars are set correctly, but cheap to
# MAGIC leave in.
