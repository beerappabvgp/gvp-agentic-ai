# app.py
# DocChat — Multi-Document RAG Chat Interface with Conversational RAG
# Run with: streamlit run app.py

import streamlit as st
import os
from multi_doc_rag import (
    index_document,
    delete_document,
    list_documents,
    get_chunk_count,
    answer_question,
    clear_conversation,
    STATUS_NEW,
    STATUS_UPDATED,
    STATUS_CURRENT,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat",
    page_icon="📚",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# Streamlit reruns the full script on every user interaction.
# st.session_state preserves values across those reruns.
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []       # chat display history

if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = None  # filename pending delete confirmation


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.title("📚 DocChat")
    st.caption("Ask questions across your uploaded documents")
    st.divider()

    # ── SECTION 1: Upload ─────────────────────────────────────
    st.subheader("📤 Upload Documents")
    st.caption("Upload new PDFs or re-upload updated versions — changes detected automatically.")

    uploaded_files = st.file_uploader(
        label="Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        os.makedirs("./uploads", exist_ok=True)

        for uploaded_file in uploaded_files:
            save_path = f"./uploads/{uploaded_file.name}"

            # Save to disk so PyPDFLoader can read it
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())

            with st.spinner(f"Processing {uploaded_file.name}..."):
                status, count = index_document(save_path)

            # Show correct message based on sync status
            if status == STATUS_NEW:
                st.success(f"✅ **{uploaded_file.name}**\nIndexed — {count} chunks added")
            elif status == STATUS_UPDATED:
                st.warning(f"🔄 **{uploaded_file.name}**\nContent changed — re-indexed with {count} chunks")
            elif status == STATUS_CURRENT:
                st.info(f"✅ **{uploaded_file.name}**\nAlready up to date — no changes detected")

    st.divider()

    # ── SECTION 2: Document list — select + delete ────────────
    st.subheader("📂 Your Documents")

    all_docs = list_documents()

    if not all_docs:
        st.info("Upload PDF files above to get started.")
        selected_docs = []

    else:
        selected_docs = []

        for doc in all_docs:
            chunk_count = get_chunk_count(doc)

            # Two columns: checkbox on left, delete button on right
            col_check, col_del = st.columns([0.82, 0.18])

            with col_check:
                is_selected = st.checkbox(
                    label=f"📄 {doc} ({chunk_count} chunks)",
                    value=True,
                    key=f"cb_{doc}",
                )
                if is_selected:
                    selected_docs.append(doc)

            with col_del:
                # First click sets confirm_delete — does NOT delete immediately
                if st.button("🗑", key=f"del_{doc}", help=f"Delete {doc}"):
                    st.session_state.confirm_delete = doc

        st.caption(f"{len(selected_docs)} of {len(all_docs)} document(s) selected")

    # ── DELETE CONFIRMATION DIALOG ────────────────────────────
    if st.session_state.confirm_delete:
        doc_to_delete = st.session_state.confirm_delete
        st.divider()
        st.error(
            f"**Delete '{doc_to_delete}'?**\n\n"
            "This removes all indexed chunks from the database "
            "and deletes the PDF from disk. This cannot be undone."
        )

        confirm_col, cancel_col = st.columns(2)

        with confirm_col:
            if st.button("✅ Yes, delete", use_container_width=True):
                with st.spinner(f"Deleting {doc_to_delete}..."):
                    deleted = delete_document(doc_to_delete)
                st.session_state.confirm_delete = None
                st.success(f"Deleted '{doc_to_delete}' ({deleted} chunks removed).")
                st.rerun()

        with cancel_col:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.confirm_delete = None
                st.rerun()

    st.divider()

    # ── SECTION 3: Clear conversation ────────────────────────
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        clear_conversation()     # reset the backend history list too
        st.rerun()


# ─────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ─────────────────────────────────────────────────────────────
st.title("💬 Ask Your Documents")

# Show which documents are currently being searched
if selected_docs:
    st.caption(f"Searching across: {' · '.join(selected_docs)}")
else:
    st.caption("No documents selected")

st.divider()

# Guard: nothing indexed yet
if not all_docs:
    st.info("👈 Upload your PDF documents in the sidebar to get started.")
    st.stop()

# Guard: all documents deselected
if not selected_docs:
    st.warning("⚠️ Please select at least one document in the sidebar.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# DISPLAY CHAT HISTORY
# Loop through all past messages and render each as a chat bubble.
# For assistant messages, also show the rewritten question and sources.
# ─────────────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Assistant messages: show the rewritten question + source citations
        if message["role"] == "assistant":

            # Show rewritten question if it differs from the original
            # This makes Conversational RAG visible to the user —
            # they can see how their follow-up was interpreted
            if message.get("rewritten") and message["rewritten"] != message.get("original"):
                st.caption(f"🔄 Searched as: *\"{message['rewritten']}\"*")

            # Source citations in a collapsible expander
            if message.get("sources"):
                with st.expander("📄 View Sources"):
                    for label in message["sources"]:
                        st.caption(label)


# ─────────────────────────────────────────────────────────────
# CHAT INPUT
# ─────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask anything about your selected documents...")

if user_input:

    # Show user message immediately — do not wait for the LLM
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Run the Conversational RAG pipeline and display the answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            # answer_question now returns 3 values:
            #   answer    — the grounded response from Groq
            #   sources   — list of citation strings
            #   rewritten — the standalone question used for ChromaDB search
            answer, sources, rewritten = answer_question(user_input, selected_docs)

        # Show the answer
        st.write(answer)

        # Show rewritten question if it differs from the original
        # This is the key new UI element that shows Conversational RAG is working
        if rewritten and rewritten != user_input:
            st.caption(f"🔄 Searched as: *\"{rewritten}\"*")

        # Show source citations
        if sources:
            with st.expander("📄 View Sources"):
                for label in sources:
                    st.caption(label)

    # Persist the full assistant message to session state
    # Store original question, rewritten question, answer, and sources
    # so the display loop can show all of them on future reruns
    st.session_state.messages.append({
        "role":      "assistant",
        "content":   answer,
        "sources":   sources,
        "rewritten": rewritten,   # the standalone question used for search
        "original":  user_input,  # the raw question the user typed
    })