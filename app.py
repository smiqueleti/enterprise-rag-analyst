"""Streamlit chat client for the Atlas RAG FastAPI service."""

from __future__ import annotations

import uuid

import httpx
import streamlit as st

from ui_client import build_query_payload, query_api


st.set_page_config(page_title="Atlas AI Analyst", page_icon="🧭", layout="wide")
st.title("Atlas AI Analyst")
st.caption("Ask grounded questions across Atlas Logistics policies and procedures.")

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Session")
    st.write(f"Conversation: `{st.session_state.conversation_id[:8]}`")
    show_debug = st.toggle("Developer details", value=False)
    if st.button("New conversation", use_container_width=True):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("result"):
            result = message["result"]
            sources = result.get("sources", [])
            if sources:
                st.markdown("#### Sources")
            for source in sources:
                label = (
                    f"[{source['citation']}] {source['document']} "
                    f"- chunk {source['chunk_index']}"
                )
                with st.expander(label):
                    columns = st.columns(4)
                    columns[0].metric("Department", source["department"])
                    columns[1].metric("Type", source["document_type"])
                    columns[2].metric("Version", source["version"])
                    columns[3].metric("Effective", source["effective_date"])
                    st.code(source["chunk_id"], language=None)
                    st.markdown(source["text"])
            if show_debug:
                with st.expander("Developer trace", expanded=False):
                    st.write("Retrieval strategy", result["retrieval_strategy"])
                    st.write("Model", result["model_used"])
                    st.write("Trace ID", result["trace_id"])
                    st.write("Rewritten query", result["debug"]["rewritten_query"])
                    st.write("Atomic queries", result["debug"]["atomic_queries"])
                    st.write("Selected evidence", result["debug"]["selected_evidence"])
                    st.write("Latency (ms)", result["latency_ms"])
                    st.write("Token usage", result["token_usage"])
                    st.write("Retries", result["debug"]["retries"])

if prompt := st.chat_input("Ask about an SLA, escalation, exception, or approval"):
    prior_messages = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    payload = build_query_payload(
        prompt,
        st.session_state.conversation_id,
        prior_messages,
    )
    with st.chat_message("assistant"):
        with st.spinner("Investigating Atlas policies..."):
            try:
                result = query_api(payload)
                answer = result["answer"]
                st.markdown(answer)
            except httpx.HTTPStatusError as error:
                answer = "The Atlas RAG service could not complete this request."
                result = None
                st.error(f"Service error: HTTP {error.response.status_code}")
            except (httpx.HTTPError, ValueError) as error:
                answer = "The Atlas RAG service is currently unreachable."
                result = None
                st.error(str(error))
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "result": result}
    )
    st.rerun()
