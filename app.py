import streamlit as st
from PyPDF2 import PdfReader
import os
import datetime
from dotenv import load_dotenv

load_dotenv("api.env", override=True)

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


def get_api_key():
    return os.getenv("GROQ_API_KEY")


def get_llm():
    api_key = get_api_key()

    if not api_key:
        st.error("❌ GROQ_API_KEY is missing. Please check api.env.")
        st.stop()

    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=api_key,
        temperature=0.3,
    )


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


def get_pdf_text(pdfs):
    text = ""
    total_pages = 0

    for pdf in pdfs:
        reader = PdfReader(pdf)
        total_pages += len(reader.pages)

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()

            if page_text:
                text += f"\n[Page {i + 1}]\n{page_text}"

    return text, total_pages


def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    return splitter.split_text(text)


def build_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_texts(
        chunks,
        embedding=embeddings
    )


def now_time():
    return datetime.datetime.now().strftime("%I:%M %p")


st.markdown(
    "<h1>🧠 DocuMind</h1>"
    "<p>Chat with your PDFs using Groq + LangChain</p>",
    unsafe_allow_html=True,
)


with st.expander(
    "⚙️ Setup — Upload PDFs",
    expanded=not st.session_state.processed
):

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

            if not raw.strip():
                st.error("❌ No readable text was found in the uploaded PDF.")
                st.stop()

            chunks = get_chunks(raw)

            if not chunks:
                st.error("❌ Could not create document chunks.")
                st.stop()

            st.session_state.vector_store = build_store(chunks)

            st.session_state.processed = True
            st.session_state.chat_history = []

            st.session_state.doc_stats = {
                "files": len(pdf_docs),
                "pages": pages,
                "chunks": len(chunks),
            }

        st.rerun()


if st.session_state.processed and st.session_state.doc_stats:

    ds = st.session_state.doc_stats

    st.info(
        f"📁 {ds['files']} files | "
        f"📄 {ds['pages']} pages | "
        f"🧩 {ds['chunks']} chunks"
    )


if not st.session_state.processed:
    st.warning("Upload PDFs and process them to start chatting.")


for sender, msg in st.session_state.chat_history:

    if sender == "user":
        st.markdown(f"**You:** {msg}")

    else:
        st.markdown(f"**DocuMind:** {msg}")


SUGGESTIONS = [
    "Summarize this document",
    "Key points",
    "What is this about?",
    "Any data mentioned?"
]


if st.session_state.processed and len(st.session_state.chat_history) == 0:

    st.write("Try asking:")

    for suggestion in SUGGESTIONS:

        if st.button(suggestion):
            st.session_state.pending_prompt = suggestion
            st.rerun()


prompt = st.chat_input("Ask DocuMind...")


if st.session_state.pending_prompt and not prompt:

    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None


if prompt:

    if not st.session_state.processed:
        st.warning("Process PDFs first")
        st.stop()

    st.session_state.chat_history.append(
        ("user", prompt)
    )

    docs = st.session_state.vector_store.similarity_search(
        prompt,
        k=4
    )

    context = "\n\n".join(
        d.page_content
        for d in docs
    )

    final_prompt = f"""
You are DocuMind, an AI assistant that answers questions using only the provided PDF context.

Rules:
- Answer only using information from the PDF context.
- If the answer is not present in the context, clearly say that the information is not available in the uploaded document.
- Do not make up information.
- Keep the answer clear and easy to understand.
- When possible, mention the relevant page number from the context.

PDF Context:
{context}

Question:
{prompt}

Answer:
"""

    with st.spinner("Thinking..."):

        try:

            response = get_llm().invoke(final_prompt)

            answer = response.content

            st.session_state.chat_history.append(
                ("assistant", answer)
            )

            st.rerun()

        except Exception as e:

            st.session_state.chat_history.pop()

            st.error(
                f"❌ Unable to generate a response: {str(e)}"
            )

