from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# Ensure project root is on sys.path when Streamlit executes this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import settings
from src.data.database import Database
from src.ingestion.doc_loader import load_document_chunks
from src.models.rag import RAGAssistant


st.set_page_config(page_title="ERP Knowledge Assistant (LLM + RAG)", layout="wide")
st.title("ERP Knowledge Assistant")
st.caption("LLM + RAG + Vector DB workflow for enterprise document Q&A")


db = Database(settings.database_path)
chunks = load_document_chunks(settings.docs_path)
assistant = RAGAssistant(chunks)
chunk_counts_by_file: dict[str, int] = {}
for chunk in chunks:
    chunk_counts_by_file[chunk.source_file] = chunk_counts_by_file.get(chunk.source_file, 0) + 1


with st.sidebar:
    st.subheader("RAG Configuration")
    st.write(f"Provider: `{settings.llm_provider}`")
    st.write(f"Vector backend: `{assistant.backend}`")
    st.write(f"Collection: `{settings.chroma_collection}`")
    st.write(f"Docs path: `{settings.docs_path}`")
    st.write(f"Chunks loaded: `{len(chunks)}`")
    st.write(f"Retrieve candidates: `{settings.retrieve_candidates}`")
    st.write(f"Min retrieval score: `{settings.min_retrieval_score}`")
    if st.button("Reindex Documents"):
        assistant.index_documents(force_reindex=True)
        st.success("Document index rebuilt.")
        st.rerun()

tab_assistant, tab_docs, tab_insights, tab_eval = st.tabs([
    "Ask Assistant",
    "Document Management",
    "Insights",
    "Evaluation",
])

with tab_assistant:
    st.subheader("Ask ERP / policy questions")
    question = st.text_input("Question", placeholder="How should invoice approvals be routed for amounts above $10,000?")
    show_context = st.toggle("Show retrieved context", value=False)
    if st.button("Get Answer", type="primary"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            result = assistant.answer(question, top_k=settings.rag_top_k)
            st.markdown("### Response")
            st.text(result.answer)
            st.caption(f"Confidence: `{result.confidence}` | Backend: `{result.backend}`")
            st.markdown("### Citations")
            sources_df = pd.DataFrame(
                [
                    {
                        "source": hit["source"],
                        "score": hit["score"],
                        "snippet": hit["snippet"],
                    }
                    for hit in result.citations
                ]
            ) if result.citations else pd.DataFrame(columns=["source", "score", "snippet"])
            st.dataframe(sources_df, use_container_width=True, hide_index=True)
            db.log_query(
                question=question,
                answer=result.answer,
                sources=result.sources,
                retrieval_backend=result.backend,
                confidence=result.confidence,
                top_score=max(result.scores) if result.scores else None,
            )
            if show_context and result.retrieved_context:
                st.markdown("### Retrieved Context")
                for idx, ctx in enumerate(result.retrieved_context, start=1):
                    st.text(f"[{idx}] {ctx}")

with tab_docs:
    st.subheader("Knowledge Base Documents")
    docs_dir = Path(settings.docs_path)
    docs_dir.mkdir(parents=True, exist_ok=True)
    uploaded_files = st.file_uploader(
        "Upload .txt, .md, or .pdf files",
        type=["txt", "md", "pdf"],
        accept_multiple_files=True,
    )
    if uploaded_files and st.button("Save Uploaded Files"):
        for file in uploaded_files:
            target = docs_dir / file.name
            target.write_bytes(file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s). Click 'Reindex Documents' in sidebar.")
        st.rerun()

    doc_paths = sorted(
        list(docs_dir.glob("*.txt")) + list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.pdf"))
    )
    docs_table = pd.DataFrame(
        [
            {
                "file": p.name,
                "chunks_indexed": chunk_counts_by_file.get(p.name, 0),
            }
            for p in doc_paths
        ]
    )
    st.dataframe(docs_table, use_container_width=True, hide_index=True)

with tab_insights:
    st.subheader("Assistant Usage and Retrieval Quality")

    q_logs = pd.DataFrame(
        db.fetch_query_logs(),
        columns=["question", "top_sources", "retrieval_backend", "confidence", "top_score", "created_at"],
    )
    st.metric("Questions answered", len(q_logs))

    if not q_logs.empty:
        q_logs["top_score"] = pd.to_numeric(q_logs["top_score"], errors="coerce")
        avg_score = float(q_logs["top_score"].dropna().mean()) if not q_logs["top_score"].dropna().empty else 0.0
        low_conf_count = int((q_logs["confidence"].fillna("").str.lower() == "low").sum())
        col1, col2 = st.columns(2)
        col1.metric("Avg top retrieval score", f"{avg_score:.3f}")
        col2.metric("Low-confidence queries", low_conf_count)

        q_logs["date"] = pd.to_datetime(q_logs["created_at"]).dt.date
        q_counts = q_logs.groupby("date", as_index=False).size().rename(columns={"size": "count"})
        st.plotly_chart(px.line(q_counts, x="date", y="count", title="Daily Assistant Usage"), use_container_width=True)
        st.dataframe(q_logs.head(20), use_container_width=True, hide_index=True)

with tab_eval:
    st.subheader("Retrieval Benchmark")
    eval_path = Path(settings.eval_questions_path)
    st.caption(f"Evaluation file: `{eval_path}`")
    if not eval_path.exists():
        st.warning("Evaluation file not found. Add data/eval/questions.csv first.")
    else:
        eval_df = pd.read_csv(eval_path)
        st.dataframe(eval_df.head(25), use_container_width=True, hide_index=True)
        if st.button("Run Retrieval Benchmark"):
            results = []
            for row in eval_df.itertuples(index=False):
                question = str(row.question)
                expected_tag = str(row.expected_source_tag).lower()
                hits = assistant.retrieve(question, top_k=3)
                top_sources = [hit["source"] for hit in hits]
                hit = any(expected_tag in src.lower() for src in top_sources)
                top_score = float(hits[0]["score"]) if hits else 0.0
                results.append(
                    {
                        "question": question,
                        "expected_source_tag": expected_tag,
                        "top_sources": " | ".join(top_sources),
                        "match_top3": hit,
                        "top_score": top_score,
                    }
                )
            result_df = pd.DataFrame(results)
            accuracy = float(result_df["match_top3"].mean()) if not result_df.empty else 0.0
            avg_top_score = float(result_df["top_score"].mean()) if not result_df.empty else 0.0
            col1, col2 = st.columns(2)
            col1.metric("Top-3 source match rate", f"{accuracy * 100:.1f}%")
            col2.metric("Average top score", f"{avg_top_score:.3f}")
            st.dataframe(result_df, use_container_width=True, hide_index=True)
