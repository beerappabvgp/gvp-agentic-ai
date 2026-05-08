# multi_doc_rag.py
# Multi-Document RAG Backend — with Conversational RAG
#
# KEY FEATURES:
#   - Content hashing (MD5) to detect when a PDF has changed
#   - Smart sync: NEW / UP-TO-DATE / UPDATED states on every upload
#   - Delete document: removes all chunks from ChromaDB + file from disk
#   - Conversational RAG: rewrites follow-up questions before searching
#     so pronouns and short questions work correctly across turns

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import hashlib
import os

load_dotenv()

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
PERSIST_DIR     = "./multi_doc_db"
COLLECTION_NAME = "documents"
UPLOADS_DIR     = "./uploads"

# How many past conversation turns to use when rewriting questions.
# 3 turns = the last 3 user questions and their answers.
HISTORY_TURNS   = 3

# Status constants returned by index_document() — used by the UI
STATUS_NEW     = "new"       # first time this file is indexed
STATUS_UPDATED = "updated"   # same filename, different content → re-indexed
STATUS_CURRENT = "current"   # same filename, same content → skipped


# ─────────────────────────────────────────────────────────────
# SHARED COMPONENTS
# ─────────────────────────────────────────────────────────────

# Local embedding model — no API key, no cost, ~90MB download on first run
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to ChromaDB on disk — creates the folder if it does not exist
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# Splits PDF pages into overlapping 500-char chunks
# chunk_overlap=100 ensures sentences at chunk boundaries are not lost
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Groq LLM — used for BOTH question rewriting and final answering
# temperature=0 for deterministic factual answers
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)


# ─────────────────────────────────────────────────────────────
# HELPER — COMPUTE FILE HASH
# ─────────────────────────────────────────────────────────────
def compute_file_hash(pdf_path: str) -> str:
    """
    Compute the MD5 fingerprint of a PDF file's raw bytes.

    If even one byte in the PDF changes the hash changes completely.
    This is how we detect whether an uploaded file is truly new or
    just the same file uploaded again.

    Reads in 8KB chunks so large files never load fully into RAM.

    Returns:
        str — 32-character hex digest e.g. "a3f8c2d1e9b74501..."
    """
    hasher = hashlib.md5()
    with open(pdf_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            hasher.update(block)
    return hasher.hexdigest()


# ─────────────────────────────────────────────────────────────
# HELPER — GET STORED HASH FROM CHROMADB
# ─────────────────────────────────────────────────────────────
def get_stored_hash(filename: str) -> str | None:
    """
    Read the MD5 hash that was stored in chunk metadata when this
    document was last indexed.

    Every chunk from the same PDF carries the same file_hash value.
    We only need to fetch one chunk to get it.

    Returns:
        str  — the stored hash string if the file exists in ChromaDB
        None — if this filename has never been indexed
    """
    result = vectorstore.get(
        where={"source": {"$eq": filename}},
        limit=1,
        include=["metadatas"]
    )
    if not result["ids"]:
        return None
    return result["metadatas"][0].get("file_hash", None)


# ─────────────────────────────────────────────────────────────
# HELPER — DELETE CHUNKS FROM CHROMADB (private)
# ─────────────────────────────────────────────────────────────
def _delete_chunks_from_db(filename: str) -> int:
    """
    Remove every chunk belonging to a filename from ChromaDB.

    Process:
        1. Query all chunk IDs where source == filename
        2. Call vectorstore.delete(ids=...) to remove them

    Used internally by both index_document() (during updates)
    and delete_document() (when user deletes a file).

    Returns:
        int — number of chunks deleted
    """
    existing  = vectorstore.get(where={"source": {"$eq": filename}})
    chunk_ids = existing["ids"]

    if not chunk_ids:
        return 0

    vectorstore.delete(ids=chunk_ids)
    print(f"  Deleted {len(chunk_ids)} chunks for '{filename}' from ChromaDB.")
    return len(chunk_ids)


# ─────────────────────────────────────────────────────────────
# FUNCTION 1 — INDEX DOCUMENT (smart sync with hash comparison)
# ─────────────────────────────────────────────────────────────
def index_document(pdf_path: str) -> tuple:
    """
    Load, chunk, embed, and store a PDF into ChromaDB.

    Before indexing, compares the MD5 hash of the uploaded file
    against the hash stored during the last index to decide what to do:

        NEW     — filename not seen before → index fresh
        CURRENT — filename seen, hash matches → skip (already up to date)
        UPDATED — filename seen, hash differs → delete old, index new

    Every chunk is tagged with:
        source    : filename (used for metadata filtering in search)
        file_hash : MD5 hash (used for future sync detection)

    Args:
        pdf_path : full path to the PDF on disk

    Returns:
        tuple — (status: str, chunk_count: int)
    """
    filename     = os.path.basename(pdf_path)
    current_hash = compute_file_hash(pdf_path)
    stored_hash  = get_stored_hash(filename)

    # Already indexed with identical content → skip
    if stored_hash is not None and stored_hash == current_hash:
        print(f"  '{filename}' is up to date. No changes detected.")
        return STATUS_CURRENT, get_chunk_count(filename)

    # Same filename but content has changed → delete old first
    if stored_hash is not None and stored_hash != current_hash:
        print(f"  '{filename}' content changed. Removing old version...")
        _delete_chunks_from_db(filename)
        status = STATUS_UPDATED
    else:
        # Brand new file → index for the first time
        print(f"  '{filename}' is new. Indexing...")
        status = STATUS_NEW

    # Load PDF pages
    loader    = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"  Loaded {len(documents)} pages.")

    # Chunk pages into overlapping 500-char pieces
    chunks = splitter.split_documents(documents)

    # Tag every chunk with source filename and content hash
    for chunk in chunks:
        chunk.metadata["source"]    = filename
        chunk.metadata["file_hash"] = current_hash

    # Embed and persist to ChromaDB
    vectorstore.add_documents(chunks)
    print(f"  Stored {len(chunks)} chunks. Status: {status}.")

    return status, len(chunks)


# ─────────────────────────────────────────────────────────────
# FUNCTION 2 — DELETE DOCUMENT
# ─────────────────────────────────────────────────────────────
def delete_document(filename: str) -> int:
    """
    Completely remove a document from the system:
        Step 1 — Delete all its chunks from ChromaDB
        Step 2 — Delete the PDF file from the uploads folder on disk

    After this the document is gone from both the vector index
    and the filesystem. The user must re-upload to use it again.

    Args:
        filename : e.g. "health_policy.pdf"

    Returns:
        int — number of chunks deleted from ChromaDB
    """
    deleted   = _delete_chunks_from_db(filename)
    file_path = os.path.join(UPLOADS_DIR, filename)

    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"  Removed file from disk: {file_path}")

    return deleted


# ─────────────────────────────────────────────────────────────
# FUNCTION 3 — LIST INDEXED DOCUMENTS
# ─────────────────────────────────────────────────────────────
def list_documents() -> list:
    """
    Return a sorted list of all unique PDF filenames currently indexed.
    Used by the UI for checkboxes and delete buttons.
    """
    all_data = vectorstore.get(include=["metadatas"])

    if not all_data["metadatas"]:
        return []

    filenames = list(set(
        m["source"] for m in all_data["metadatas"] if "source" in m
    ))
    return sorted(filenames)


# ─────────────────────────────────────────────────────────────
# FUNCTION 4 — GET CHUNK COUNT PER DOCUMENT
# ─────────────────────────────────────────────────────────────
def get_chunk_count(filename: str) -> int:
    """
    Return how many chunks are stored for a specific document.
    Shown in the UI next to each document checkbox.
    """
    result = vectorstore.get(where={"source": {"$eq": filename}})
    return len(result["ids"])


# ─────────────────────────────────────────────────────────────
# FUNCTION 5 — SEARCH DOCUMENTS WITH METADATA FILTER
# ─────────────────────────────────────────────────────────────
def search_documents(query: str, selected_files: list, k: int = 4) -> list:
    """
    Semantic search restricted to only the user-selected documents.

    IMPORTANT: This function always receives the REWRITTEN question
    from rewrite_question() — never the raw user input.
    That is what makes conversational follow-ups work correctly.

    Args:
        query          : the rewritten standalone question
        selected_files : filenames to search within
        k              : number of top results to return

    Returns:
        list of (Document, distance) tuples
        distance is cosine distance — lower means more similar
    """
    if not selected_files:
        return []

    if len(selected_files) == 1:
        doc_filter = {"source": {"$eq": selected_files[0]}}
    else:
        doc_filter = {"source": {"$in": selected_files}}

    return vectorstore.similarity_search_with_score(
        query, k=k, filter=doc_filter
    )


# ─────────────────────────────────────────────────────────────
# FUNCTION 6 — REWRITE QUESTION  ← THE KEY CONVERSATIONAL STEP
# ─────────────────────────────────────────────────────────────
def rewrite_question(question: str) -> str:
    """
    THE HEART OF CONVERSATIONAL RAG.

    Rewrites the user's latest question into a complete standalone
    question using the recent conversation history so that ChromaDB
    can search with full context — not just a vague fragment.

    WHY THIS IS NEEDED:
        Without rewriting, follow-up questions like:
            "Is it waived for anyone?"
            "What about for NRIs?"
            "And the documents needed?"
        are searched literally in ChromaDB. The pronouns and short
        references have no meaning to the vector search engine.

        With rewriting, the LLM resolves the context:
            "Is the prepayment penalty waived for any borrower category?"
            "What is the maximum loan tenure for NRI applicants?"
            "What documents are required for a BankEase home loan?"

        These complete questions return the right chunks every time.

    HOW IT WORKS:
        Makes one small, fast LLM call that reads the last 3 turns
        and produces a rewritten question. This rewritten question
        is passed to search_documents() instead of the raw input.

    FIRST QUESTION SHORTCUT:
        If there is no conversation history yet, the question is
        already standalone — return it unchanged, skip the LLM call.

    Args:
        question : the raw question the user just typed

    Returns:
        str — the rewritten standalone question
              (or the original question if no history exists)
    """
    # No history yet → first question is already standalone, no rewriting needed
    if not conversation_history:
        return question

    # Build readable history from the last HISTORY_TURNS exchanges
    # conversation_history stores alternating "User: ..." / "Assistant: ..." strings
    # Take the last HISTORY_TURNS * 2 items (each turn = 1 user + 1 assistant)
    recent = conversation_history[-(HISTORY_TURNS * 2):]
    history_text = "\n".join(recent)

    rewrite_prompt = f"""You are a search query rewriter for a document Q&A system.

Given the conversation history below and a follow-up question,
rewrite the follow-up question as a complete, standalone question
that can be fully understood without reading the conversation history.

The rewritten question will be used to search a document database,
so it must include all the context needed to find the right answer.

Rules:
- Return ONLY the rewritten question — no explanation, no prefix
- If the question is already complete and standalone, return it unchanged
- Preserve the original intent exactly — do not add assumptions

Conversation history:
{history_text}

Follow-up question: {question}
Standalone question:"""

    response  = llm.invoke([HumanMessage(content=rewrite_prompt)])
    rewritten = response.content.strip()
    return rewritten


# ─────────────────────────────────────────────────────────────
# RAG ANSWER PROMPT
# ─────────────────────────────────────────────────────────────
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful document assistant.

Answer the user's question using ONLY the context extracted from their
uploaded documents. Do not use any outside knowledge.

Rules:
1. Answer ONLY from the context below. Never guess or assume.
2. If the answer is not in the context say:
   "I could not find this in the selected documents."
3. Always cite the source document name and page number.
4. Be specific and factual. Short clear answers beat long vague ones.
5. Use conversation history to write natural, contextually aware responses.

CONTEXT FROM SELECTED DOCUMENTS:
{context}

CONVERSATION HISTORY (last 3 exchanges):
{history}"""),
    ("human", "{question}"),
])

# Conversation history — list of alternating "User: ..." / "Assistant: ..." strings
# Persists for the session lifetime, reset via clear_conversation()
conversation_history = []


# ─────────────────────────────────────────────────────────────
# FUNCTION 7 — ANSWER QUESTION (full Conversational RAG pipeline)
# ─────────────────────────────────────────────────────────────
def answer_question(question: str, selected_files: list) -> tuple:
    """
    Full Conversational RAG pipeline for one user question.

    Pipeline (4 steps):
        STEP 1 — REWRITE
            Call rewrite_question() to convert the raw user question
            into a complete standalone question using conversation history.
            Example: "Is it waived?" → "Is the prepayment penalty waived for women?"

        STEP 2 — SEARCH
            Use the REWRITTEN question (not raw) to search ChromaDB.
            This finds the right chunks even for vague follow-up questions.

        STEP 3 — ANSWER
            Inject retrieved chunks + conversation history into the LLM prompt.
            Groq generates a grounded, cited answer.

        STEP 4 — REMEMBER
            Append this turn to conversation_history so future questions
            can reference it in the rewrite step.

    Args:
        question       : raw question typed by the user
        selected_files : filenames to restrict the search to

    Returns:
        tuple — (answer: str, sources: list[str], rewritten: str)
        rewritten is the standalone question — shown in the UI for transparency
    """
    # STEP 1 — Rewrite the question using conversation history
    rewritten_question = rewrite_question(question)

    # STEP 2 — Search ChromaDB using the REWRITTEN question
    # This is the key difference from regular RAG — we search with full context
    results = search_documents(rewritten_question, selected_files, k=4)

    if not results:
        return "Please select at least one document to search.", [], rewritten_question

    # Format retrieved chunks as grounded context with source citations
    context_parts = []
    source_labels = []

    for doc, distance in results:
        source    = doc.metadata.get("source", "Unknown")
        page      = doc.metadata.get("page", "?")
        relevance = 1 - distance   # cosine distance → similarity score

        context_parts.append(
            f"[Source: {source} | Page: {page} | Relevance: {relevance:.0%}]\n"
            f"{doc.page_content}"
        )
        source_labels.append(
            f"📄 {source} — Page {page} — {relevance:.0%} relevant"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Format the last 3 turns as history string for the answer prompt
    history_str = (
        "\n".join(conversation_history[-6:])
        if conversation_history
        else "No previous conversation."
    )

    # STEP 3 — Call Groq to generate the grounded answer
    # Pass the ORIGINAL question (not rewritten) so the response sounds natural
    chain  = rag_prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "question": question,          # original for natural-sounding answer
        "context":  context,
        "history":  history_str,
    })

    # STEP 4 — Save this turn for future rewrites
    conversation_history.append(f"User: {question}")
    conversation_history.append(f"Assistant: {answer}")

    # Return answer, sources, and the rewritten question
    # The rewritten question is shown in the UI so users can see what happened
    return answer, source_labels, rewritten_question


# ─────────────────────────────────────────────────────────────
# FUNCTION 8 — CLEAR CONVERSATION HISTORY
# ─────────────────────────────────────────────────────────────
def clear_conversation():
    """Reset conversation history. Called by the Clear Chat button in the UI."""
    global conversation_history
    conversation_history = []


# ─────────────────────────────────────────────────────────────
# QUICK TEST — python3 multi_doc_rag.py
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== DocChat Conversational RAG Test ===\n")
    print("Indexed documents:", list_documents() or "None yet")

    test_pdf = "BankEase_Home_Loan_Policy.pdf"
    if os.path.exists(test_pdf):
        status, count = index_document(test_pdf)
        print(f"Result: {status} | {count} chunks")
    else:
        print("Place a PDF in this folder to test indexing.")

    docs = list_documents()
    if docs:
        # Test 1 — standalone question (no rewriting needed)
        print("\n--- Test 1: Standalone question ---")
        answer, sources, rewritten = answer_question(
            "What is the prepayment penalty?", docs
        )
        print(f"Rewritten : {rewritten}")
        print(f"Answer    : {answer[:150]}")

        # Test 2 — follow-up question (rewriting kicks in)
        print("\n--- Test 2: Follow-up with pronoun ---")
        answer, sources, rewritten = answer_question(
            "Is it waived for anyone?", docs
        )
        print(f"Rewritten : {rewritten}")
        print(f"Answer    : {answer[:150]}")