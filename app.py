import streamlit as st
from PyPDF2 import PdfReader
import os
import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------
# API KEY HANDLER (IMPORTANT FIX)
# ---------------------------
def get_api_key():
    # 1. Streamlit Cloud (BEST)
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]

    # 2. Local .env fallback
    return os.getenv("GROQ_API_KEY")


# ---------------------------
# SESSION STATE
# ---------------------------
defaults = {
    "chat_history": [],
    "vector_store": None,
    "processed": False,
    "doc_stats": {},
    "pending_prompt": None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------
# PDF PROCESSING
# ---------------------------
def get_pdf_text(pdfs):
    text = ""
    total_pages = 0

    for pdf in pdfs:
        reader = PdfReader(pdf)
        total_pages += len(reader.pages)

        for i, page in enumerate(reader.pages):
            t = page.extract_text()
            if t:
                text += f"\n[Page {i+1}]\n{t}"

    return text, total_pages


def get_chunks(text):
    return RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    ).split_text(text)


def build_store(chunks):
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embedding=emb)


# ---------------------------
# GROQ LLM (FIXED)
# ---------------------------
def get_llm():
    api_key = get_api_key()

    if not api_key:
        st.error("❌ GROQ_API_KEY is missing. Add it to Streamlit Secrets or .env")
        st.stop()

    return ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=api_key,
        temperature=0.3,
    )


# ---------------------------
# UTILITIES
# ---------------------------
def now_time():
    return datetime.datetime.now().strftime("%I:%M %p")


# ---------------------------
# UI HEADER
# ---------------------------
st.markdown(
    "<h1>🧠 DocuMind</h1>"
    "<p>Chat with your PDFs using Groq + LangChain</p>",
    unsafe_allow_html=True,
)


# ---------------------------
# UPLOAD SECTION (NO API KEY HERE)
# ---------------------------
with st.expander("⚙️ Setup — Upload PDFs", expanded=not st.session_state.processed):

    pdf_docs = st.file_uploader(
        "Upload PDFs",
        accept_multiple_files=True,
        type=["pdf"]
    )

    process_btn = st.button("⚡ Process Documents")

    if process_btn:

        if not pdf_docs:
            st.warning("Upload PDFs")
            st.stop()

        with st.spinner("Processing PDFs..."):
            raw, pages = get_pdf_text(pdf_docs)
            chunks = get_chunks(raw)
            st.session_state.vector_store = build_store(chunks)

            st.session_state.processed = True
            st.session_state.chat_history = []

            st.session_state.doc_stats = {
                "files": len(pdf_docs),
                "pages": pages,
                "chunks": len(chunks),
            }

        st.rerun()


# ---------------------------
# DOC STATS
# ---------------------------
if st.session_state.processed and st.session_state.doc_stats:
    ds = st.session_state.doc_stats

    st.info(
        f"📁 {ds['files']} files | "
        f"📄 {ds['pages']} pages | "
        f"🧩 {ds['chunks']} chunks"
    )


if not st.session_state.processed:
    st.warning("Upload PDFs and process them to start chatting.")


# ---------------------------
# CHAT HISTORY
# ---------------------------
for sender, msg in st.session_state.chat_history:
    if sender == "user":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**DocuMind:** {msg}")


# ---------------------------
# SUGGESTIONS
# ---------------------------
SUGGESTIONS = [
    "Summarize this document",
    "Key points",
    "What is this about?",
    "Any data mentioned?"
]

if st.session_state.processed and len(st.session_state.chat_history) == 0:
    st.write("Try asking:")

    for s in SUGGESTIONS:
        if st.button(s):
            st.session_state.pending_prompt = s
            st.rerun()


# ---------------------------
# INPUT
# ---------------------------
prompt = st.chat_input("Ask DocuMind...")

if st.session_state.pending_prompt and not prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None


# ---------------------------
# CHAT ENGINE
# ---------------------------
if prompt:

    if not st.session_state.processed:
        st.warning("Process PDFs first")
        st.stop()

    st.session_state.chat_history.append(("user", prompt))

    docs = st.session_state.vector_store.similarity_search(prompt)
    context = "\n".join(d.page_content for d in docs)

    final_prompt = f"""
You are DocuMind, an AI that answers only from the PDF context.

Context:
{context}

Question:~
{prompt}
"""

    with st.spinner("Thinking..."):
        try:
            resp = get_llm().invoke(final_prompt)
            st.session_state.chat_history.append(("assistant", resp.content))
            st.rerun()
        except Exception as e:
            st.error(str(e))
