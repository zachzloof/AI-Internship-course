"""Minimal Streamlit UI for the Week 1 v2 RAG API: ingest + ask.

Run:
  streamlit run demo_page.py
"""

import json
import os

import httpx
import streamlit as st

WORKDIR_CMD = "ai-engineering-bootcamp-v2/week-1v2"
MODELS = ["gpt-4o-mini", "gpt-4o", "o3-mini"]


def build_ask_payload(question: str, model: str, force_bad: bool) -> dict:
    return {
        "question": question,
        "model": model,
        "force_bad": force_bad,
    }


def build_ingest_payload(text: str, document_id: str, source: str) -> dict:
    payload = {"text": text, "document_id": document_id}
    if source.strip():
        payload["source"] = source.strip()
    return payload


def render_curl(base_url: str, path: str, payload: dict) -> str:
    body = json.dumps(payload)
    return (
        f'curl -s -X POST {base_url.rstrip("/")}{path} '
        f'-H "Content-Type: application/json" '
        f"-d '{body}'"
    )


def call_json(method: str, url: str, payload: dict | None = None) -> tuple[int, dict | str]:
    try:
        if method == "POST":
            response = httpx.post(url, json=payload, timeout=120.0)
        else:
            response = httpx.get(url, timeout=5.0)

        try:
            return response.status_code, response.json()
        except json.JSONDecodeError:
            return response.status_code, response.text
    except httpx.ConnectError:
        return 0, {"error": f"Cannot reach {url}. Start the API server first."}
    except httpx.HTTPError as exc:
        return 0, {"error": str(exc)}


def render_raw(data: dict | str) -> None:
    # st.json() calls JSON.parse() client-side, which throws a cryptic error on a
    # plain-text body (e.g. FastAPI's "Method Not Allowed" or an HTML error page).
    if isinstance(data, (dict, list)):
        st.json(data)
    else:
        st.code(str(data), language="text")


def render_attempts(data: dict | str) -> None:
    if not isinstance(data, dict):
        return

    attempts = data.get("attempts", [])
    if not attempts:
        return

    st.markdown("### Attempts")
    for attempt in attempts:
        status = "passed" if attempt.get("ok") else "failed"
        title = f"Attempt {attempt.get('attempt')}: {attempt.get('step')} ({status})"
        with st.expander(title, expanded=True):
            st.write(attempt.get("message"))
            if attempt.get("raw_output"):
                st.markdown("**Raw model output**")
                st.code(attempt["raw_output"], language="json")
            if attempt.get("validation_error"):
                st.markdown("**Validation error**")
                st.code(attempt["validation_error"], language="text")


def render_citations(data: dict | str) -> None:
    """Make the RAG grounding outcome unmissable: what was cited, or that it refused."""

    if not isinstance(data, dict):
        return

    answer = data.get("answer")
    if not isinstance(answer, dict):
        return

    citations = answer.get("citations") or []
    if citations:
        st.success(f"Cited sources: {', '.join(citations)}")
    else:
        st.warning("No sources cited — the model did not find enough grounded information to answer.")

    sources = data.get("sources") or []
    if sources:
        with st.expander(f"Retrieved chunks ({len(sources)})", expanded=False):
            for chunk in sources:
                used = chunk.get("document_id") in citations
                marker = "✅ used" if used else "— not used"
                st.markdown(
                    f"**{chunk.get('document_id')}** · chunk {chunk.get('chunk_index')} "
                    f"· score {chunk.get('score'):.3f} · {marker}"
                )
                st.caption(chunk.get("text", "")[:300] + "...")


def render_response_summary(data: dict | str) -> None:
    if not isinstance(data, dict) or "error" in data:
        return

    answer = data.get("answer")
    if isinstance(answer, dict):
        st.markdown("### Answer")
        st.write(answer.get("answer", ""))
        st.caption(
            f"confidence: {answer.get('confidence')} | "
            f"sources_needed: {answer.get('sources_needed')}"
        )

    metric_cols = st.columns(4)
    metric_cols[0].metric("Model", str(data.get("model", "-")))
    metric_cols[1].metric("Tokens", str(data.get("tokens_used", "-")))
    metric_cols[2].metric("Latency", f"{data.get('latency_ms', '-')} ms")
    metric_cols[3].metric("Cost", f"${data.get('cost_usd', '-')}")


def render_ingest_result(data: dict | str) -> None:
    if not isinstance(data, dict) or "error" in data:
        return
    if "chunks_indexed" in data:
        st.success(
            f"Ingested `{data.get('document_id')}` — {data.get('chunks_indexed')} chunks indexed."
        )


st.set_page_config(page_title="Week 1 v2 RAG Demo", layout="centered")
st.title("Week 1 v2: RAG `/ingest` + `/ask` Demo")
st.caption(
    "Ingest text into the vector store, then ask questions answered only from what's been "
    "ingested — with citations, or a refusal when the docs don't cover it."
)

base_url = st.sidebar.text_input(
    "API base URL", os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
)
st.sidebar.markdown("### Start the API")
st.sidebar.code(
    f"cd {WORKDIR_CMD}\n"
    "source .venv/bin/activate\n"
    "uvicorn main:app --host 127.0.0.1 --port 8000 --reload",
    language="bash",
)
st.sidebar.markdown("### Start this page")
st.sidebar.code(
    f"cd {WORKDIR_CMD}\nsource .venv/bin/activate\nstreamlit run demo_page.py",
    language="bash",
)
if st.sidebar.button("Check API health"):
    status, data = call_json("GET", f"{base_url.rstrip('/')}/health")
    st.sidebar.markdown(f"**HTTP {status}**" if status else "**Not connected**")
    st.sidebar.json(data if isinstance(data, dict) else {"response": data})

st.header("1. Ingest a document")
with st.form("ingest_form"):
    ingest_text = st.text_area(
        "Text",
        "Remote work: up to 3 days per week with manager approval.",
        height=150,
    )
    ingest_col1, ingest_col2 = st.columns(2)
    document_id = ingest_col1.text_input("document_id", "handbook")
    source = ingest_col2.text_input("source (optional)", "")
    ingest_submitted = st.form_submit_button("Ingest", type="primary")

ingest_payload = build_ingest_payload(ingest_text, document_id, source)
st.code(render_curl(base_url, "/ingest", ingest_payload), language="bash")

if ingest_submitted:
    with st.spinner("Calling /ingest..."):
        status, data = call_json("POST", f"{base_url.rstrip('/')}/ingest", ingest_payload)
    st.markdown(f"**HTTP {status}**" if status else "**Request failed**")
    render_ingest_result(data)
    render_raw(data)

st.divider()

st.header("2. Ask a question")
with st.form("ask_form"):
    question = st.text_area(
        "Question",
        "What is the remote work policy?",
        height=100,
    )
    model = st.selectbox("Model", MODELS, index=0)
    force_bad = st.checkbox(
        "Force a bad first response to demo validation + retry",
        value=False,
    )
    ask_submitted = st.form_submit_button("Ask", type="primary")

ask_payload = build_ask_payload(question, model, force_bad)
st.code(render_curl(base_url, "/ask", ask_payload), language="bash")

if ask_submitted:
    with st.spinner("Calling /ask..."):
        status, data = call_json("POST", f"{base_url.rstrip('/')}/ask", ask_payload)
    st.markdown("### Response")
    st.markdown(f"**HTTP {status}**" if status else "**Request failed**")
    render_citations(data)
    render_response_summary(data)
    render_attempts(data)
    st.markdown("### Raw Response")
    render_raw(data)
