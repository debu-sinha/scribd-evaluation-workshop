# Databricks notebook source
# MAGIC %md
# MAGIC # Shared agent
# MAGIC
# MAGIC `%run` from the other notebooks.

# COMMAND ----------

import mlflow
from mlflow.entities import SpanType
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

# COMMAND ----------

CORPUS = {
    "doc-001": "MLflow tracing produces hierarchical spans for chain, retrieve, and generate steps. Each span captures inputs, outputs, and timing.",
    "doc-002": "OpenTelemetry traces can be ingested into Unity Catalog Delta tables for SQL queryability and governance.",
    "doc-003": "MLflow custom judges are defined via mlflow.genai.judges.make_judge with a prompt template and a model URI.",
    "doc-004": "Custom code-based scorers in MLflow use the @scorer decorator and accept inputs, outputs, expectations, or trace as keyword arguments.",
    "doc-005": "Production scorer monitoring runs scheduled scorers against live traces with configurable sampling rates.",
    "doc-006": "Retrieval-augmented generation uses an embedding-based retriever to fetch relevant context before LLM generation.",
    "doc-007": "Vector Search on Databricks supports incremental sync from Delta source tables and exposes a low-latency similarity-search API.",
    "doc-008": "Foundation Model APIs on Databricks include Claude, Llama, Gemini, and GPT family models accessible via a unified serving interface.",
    "doc-009": "Unity Catalog provides three-level namespacing (catalog, schema, object) and ACL-based access control across all data and AI assets.",
    "doc-010": "Delta Lake supports time travel, schema evolution, ACID transactions, and Z-ORDER clustering for query performance.",
    "doc-011": "Feedback assessments on MLflow traces capture human or automated quality signals with typed value, source, and rationale.",
    "doc-012": "Review App labeling sessions assign traces to subject matter experts for batch annotation against a defined label schema.",
    "doc-013": "Lakeflow Spark Declarative Pipelines provide change data capture and incremental processing for Delta tables.",
    "doc-014": "AI/BI Genie spaces let business users query Delta tables with natural language and get charts back.",
    "doc-015": "Agent Bricks is the Databricks framework for composing and deploying GenAI agents with managed evals and monitoring.",
    "doc-016": "Mosaic AI Model Serving hosts both Foundation Models and customer-trained models behind a low-latency REST API.",
    "doc-017": "MLflow Models in Unity Catalog supports versioned aliases (e.g., production, candidate) for atomic deployment swaps.",
    "doc-018": "Lakehouse monitoring computes drift metrics on production data including PSI, JS, KS, and chi-squared tests.",
    "doc-019": "Databricks Asset Bundles wrap notebooks, jobs, and ML resources into a versioned, deployable artifact.",
    "doc-020": "Trace metadata fields like mlflow.trace.user and mlflow.trace.session drive UI filter chips and aggregation views.",
}


# COMMAND ----------


@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(query: str, k: int = 3) -> list[dict]:
    """Return documents in MLflow's canonical retriever-span schema.

    Each item has `page_content` (the chunk text) plus a `metadata` dict with
    `doc_uri` (canonical identifier MLflow's trace UI uses as the title) and the
    relevance `score`. The MLflow trace UI auto-renders this shape as document
    cards instead of raw JSON.
    """
    q_tokens = set(query.lower().split())
    scored = []
    for doc_id, text in CORPUS.items():
        d_tokens = set(text.lower().split())
        score = len(q_tokens & d_tokens)
        scored.append(
            {
                "page_content": text,
                "metadata": {"doc_uri": doc_id, "score": score},
            }
        )
    scored.sort(key=lambda x: x["metadata"]["score"], reverse=True)
    return [s for s in scored[:k] if s["metadata"]["score"] > 0]


# COMMAND ----------


@mlflow.trace(span_type=SpanType.LLM)
def generate(query: str, contexts: list[dict]) -> str:
    w = WorkspaceClient()
    if not contexts:
        return "I don't have enough information to answer that."
    context_block = "\n\n".join(c["page_content"] for c in contexts)
    prompt = (
        f"Answer the question using only the context below. Be concise. "
        f"If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {query}\nAnswer:"
    )
    response = w.serving_endpoints.query(
        name="databricks-claude-sonnet-4-6",
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
        max_tokens=200,
    )
    choices = response.choices or []
    if not choices or choices[0].message is None:
        return ""
    return choices[0].message.content or ""


# COMMAND ----------


@mlflow.trace(span_type=SpanType.CHAIN, name="answer_question")
def answer_question(
    query: str, session_id: str = "default-session", user_id: str = "default-user"
) -> dict:
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": user_id,
            "mlflow.trace.session": session_id,
        }
    )
    contexts = retrieve(query)
    answer = generate(query, contexts)
    return {
        "answer": answer,
        "contexts": [c["metadata"]["doc_uri"] for c in contexts],
    }


# COMMAND ----------

DEMO_QUERIES = [
    "How are MLflow traces structured?",
    "Can I store OpenTelemetry traces in Unity Catalog?",
    "What is make_judge and how do I use it?",
    "How do custom scorers work in MLflow?",
    "What is production scorer monitoring?",
    "Explain retrieval-augmented generation.",
    "Does Databricks support Vector Search?",
    "Which Foundation Models are available on Databricks?",
    "How does Unity Catalog organize permissions?",
    "What does Delta Lake provide?",
    "What is a feedback assessment?",
    "How do labeling sessions work?",
    "What are Lakeflow Spark Declarative Pipelines?",
    "What is AI/BI Genie?",
    "What is Agent Bricks?",
    "How does Mosaic AI Model Serving work?",
    "What aliases are supported in MLflow Models?",
    "What drift metrics does Lakehouse monitoring compute?",
    "What is a Databricks Asset Bundle?",
    "What metadata drives the MLflow UI filter chips?",
    "What is the airspeed velocity of an unladen swallow?",
    "How do I bake sourdough bread?",
    "What's the capital of France?",
]
