import json
import uuid

import requests
import streamlit as st
import os

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


API_BASE = os.environ.get("API_BASE", "http://localhost:8000/api")

st.set_page_config(page_title="Ask My Docs", layout="wide")
st.title("Ask My Docs")

# Sidebar — upload + file management
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt", "md", "htm", "html", "png", "jpg", "jpeg", "webp"],
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

# Chat
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
                json={"question": question, "session_id": st.session_state.session_id},
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
                        ticker = s.get("ticker", "")
                        filing_type = s.get("filing_type", "")
                        fiscal_year = s.get("fiscal_year", "")
                        section = s.get("section", "")
                        filename = s.get("filename", "unknown")
                        chunk_id = s.get("chunk_id", "")
                        score = s.get("relevance_score")

                        year_label = f"FY{fiscal_year}" if fiscal_year else ""
                        label_parts = [p for p in [ticker, filing_type, year_label, section] if p]
                        label = ", ".join(label_parts) if label_parts else filename
                        line_str = f"**{label}**"
                        if chunk_id:
                            line_str += f" `[{chunk_id}]`"
                        if score is not None:
                            line_str += f" — relevance {score:.2f}"
                        st.markdown(line_str)

        except requests.ConnectionError:
            placeholder.error("Cannot reach API — is the server running on port 8000?")
        except requests.HTTPError as e:
            placeholder.error(f"Query failed: {e.response.text}")

        st.session_state.messages.append({"role": "assistant", "content": full_response})
