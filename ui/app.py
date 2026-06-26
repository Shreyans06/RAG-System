import json

import requests
import streamlit as st

API_BASE = "http://localhost:8000/api"

st.set_page_config(page_title="Ask My Docs", layout="wide")
st.title("Ask My Docs")

# Sidebar — settings + upload + file management
with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Retrieval mode",
        options=["dense", "hybrid", "hybrid_rerank"],
        index=2,
        format_func=lambda x: {
            "dense": "Dense RAG",
            "hybrid": "Hybrid (BM25 + Dense)",
            "hybrid_rerank": "Hybrid + Reranker",
        }[x],
    )
    st.divider()
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "webp"],
    )
    if uploaded_file and st.button("Ingest", use_container_width=True):
        with st.spinner("Ingesting..."):
            try:
                response = requests.post(
                    f"{API_BASE}/ingest",
                    files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)},
                )
                response.raise_for_status()
                data = response.json()
                st.success(f"Done! {data['chunks_created']} chunks in {data['processing_time_seconds']}s")
            except requests.HTTPError as e:
                st.error(f"Ingest failed: {e.response.text}")
            except requests.ConnectionError:
                st.error("Cannot reach API — is the server running on port 8000?")

    # Per-file deletion
    try:
        files = requests.get(f"{API_BASE}/documents", timeout=3).json().get("files", [])
        if files:
            st.divider()
            st.header("Ingested Files")
            for fname in files:
                col1, col2 = st.columns([4, 1])
                col1.write(fname)
                if col2.button("✕", key=f"del_{fname}"):
                    try:
                        requests.delete(f"{API_BASE}/documents/{fname}").raise_for_status()
                        st.rerun()
                    except requests.HTTPError as e:
                        st.error(e.response.text)
    except Exception:
        pass

    st.divider()
    if st.button("Clear All Documents", type="secondary", use_container_width=True):
        try:
            requests.delete(f"{API_BASE}/documents").raise_for_status()
            st.success("All documents cleared.")
            st.rerun()
        except requests.ConnectionError:
            st.error("Cannot reach API.")

# Warn if no documents
try:
    status = requests.get(f"{API_BASE}/status", timeout=3).json()
    if not status.get("has_documents"):
        st.warning("No documents ingested yet. Upload a file in the sidebar to get started.")
except Exception:
    st.warning("Could not reach the API. Make sure the server is running on port 8000.")

# Tabs — Chat and Compare
chat_tab, compare_tab = st.tabs(["Chat", "Compare Retrieval Modes"])

# ── Chat tab ──────────────────────────────────────────────────────────────────
with chat_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if question := st.chat_input("Ask a question about your documents"):
        st.chat_message("user").write(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            sources = []

            try:
                with requests.post(
                    f"{API_BASE}/query",
                    json={"question": question, "mode": mode},
                    stream=True,
                    timeout=60,
                ) as r:
                    r.raise_for_status()
                    for line in r.iter_lines():
                        if not line:
                            continue
                        data = json.loads(line.decode().removeprefix("data: "))
                        if "token" in data:
                            full_response += data["token"]
                            placeholder.markdown(full_response + "▌")
                        elif data.get("done"):
                            sources = data.get("sources", [])
                            break

                placeholder.markdown(full_response)

                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            filename = s.get("filename", "unknown")
                            page = s.get("page", "")
                            chunk_id = s.get("chunk_id", "")
                            label = f"**{filename}**"
                            if page:
                                label += f", page {page}"
                            if chunk_id:
                                label += f" `[{chunk_id}]`"
                            st.markdown(label)

            except requests.ConnectionError:
                placeholder.error("Cannot reach API — is the server running on port 8000?")
            except requests.HTTPError as e:
                placeholder.error(f"Query failed: {e.response.text}")

            st.session_state.messages.append({"role": "assistant", "content": full_response})

# ── Compare tab ───────────────────────────────────────────────────────────────
with compare_tab:
    st.caption("Runs the same question through all three retrieval modes and shows results side by side.")
    compare_question = st.text_input("Question", placeholder="Ask something about your documents...")

    if st.button("Compare", type="primary", disabled=not compare_question):
        with st.spinner("Running all three modes..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/query/compare",
                    json={"question": compare_question},
                    timeout=120,
                )
                resp.raise_for_status()
                results = resp.json()

                labels = {
                    "dense": "Dense RAG",
                    "hybrid": "Hybrid (BM25 + Dense)",
                    "hybrid_rerank": "Hybrid + Reranker",
                }

                cols = st.columns(3)
                for col, (mode_key, label) in zip(cols, labels.items()):
                    result = results.get(mode_key, {})
                    with col:
                        st.subheader(label)
                        coverage = result.get("citation_coverage", 0)
                        st.metric("Citation coverage", f"{coverage:.0%}")
                        st.markdown(result.get("answer", "No answer"))
                        sources = result.get("sources", [])
                        if sources:
                            with st.expander(f"Sources ({len(sources)})"):
                                for s in sources:
                                    fname = s.get("filename", "unknown")
                                    page = s.get("page", "")
                                    cid = s.get("chunk_id", "")
                                    label_str = f"**{fname}**"
                                    if page:
                                        label_str += f", page {page}"
                                    if cid:
                                        label_str += f" `[{cid}]`"
                                    st.markdown(label_str)

            except requests.ConnectionError:
                st.error("Cannot reach API — is the server running on port 8000?")
            except requests.HTTPError as e:
                st.error(f"Compare failed: {e.response.text}")
