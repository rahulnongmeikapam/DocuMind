import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import datetime, re

load_dotenv()

st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0B1120; color: #E2E8F0; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 7rem !important;
    max-width: 760px !important;
}

h1, h2, h3 { color: #F1F5F9 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #111827 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    color: #94A3B8 !important;
    font-size: 0.9rem !important;
}

/* ── Inputs ── */
.stTextInput > label {
    color: #64748B !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
}
.stTextInput input {
    background: #0B1120 !important;
    color: #E2E8F0 !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    padding: 0.7rem 1rem !important;
}
.stTextInput input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.22) !important;
}
.stTextInput input::placeholder { color: #374151 !important; }

/* ── File uploader ── */
[data-testid="stFileUploaderDropzone"] {
    background: #0B1120 !important;
    border: 1.5px dashed rgba(99,102,241,0.35) !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span { color: #475569 !important; font-size: 0.85rem !important; }
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(99,102,241,0.15) !important;
    color: #A5B4FC !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 8px !important;
}

/* ── All buttons default ── */
.stButton > button {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 0.9rem !important; width: 100% !important;
    padding: 0.68rem 1rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

/* ── Suggestion chips (ghost buttons in a row) ── */
div[data-testid="column"] .stButton > button {
    background: rgba(99,102,241,0.08) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    color: #A5B4FC !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    border-radius: 20px !important;
    padding: 0.4rem 0.9rem !important;
    width: auto !important;
    white-space: nowrap !important;
}
div[data-testid="column"] .stButton > button:hover {
    background: rgba(99,102,241,0.18) !important;
    transform: none !important;
}

/* ── Chat bubbles ── */
.bubble-user {
    background: rgba(99,102,241,0.1);
    border: 1px solid rgba(99,102,241,0.22);
    border-radius: 18px 18px 4px 18px;
    padding: 13px 17px;
    margin: 6px 0 6px 12%;
    color: #C7D2FE;
    font-size: 0.91rem;
    line-height: 1.7;
}
.bubble-ai {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.05);
    border-left: 3px solid #6366F1;
    border-radius: 4px 18px 18px 18px;
    padding: 15px 17px;
    margin: 6px 12% 6px 0;
    color: #CBD5E1;
    font-size: 0.91rem;
    line-height: 1.85;
}
.blabel {
    font-size: 0.64rem; font-weight: 700;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 5px;
}
.blabel-you { color: #818CF8; }
.blabel-ai  { color: #6366F1; }
.msg-time {
    font-size: 0.62rem; color: #374151;
    margin-top: 6px; text-align: right;
}

/* ── Suggested questions label ── */
.suggest-label {
    font-size: 0.72rem; font-weight: 600;
    color: #475569; letter-spacing: 0.08em;
    text-transform: uppercase; margin-bottom: 8px;
    margin-top: 4px;
}

/* ── Stats bar ── */
.stats-bar {
    display: flex; gap: 10px; flex-wrap: wrap;
    margin: 12px 0 4px;
}
.stat-chip {
    background: #111827;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.75rem;
    color: #64748B;
    display: flex; align-items: center; gap: 5px;
}
.stat-chip span { color: #A5B4FC; font-weight: 600; }

/* ── Typing indicator ── */
.typing-dot {
    display:inline-block; width:7px; height:7px;
    background:#6366F1; border-radius:50%; margin:0 2px;
    animation: bounce 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay:.2s; }
.typing-dot:nth-child(3) { animation-delay:.4s; }
@keyframes bounce {
    0%,80%,100%{ transform:translateY(0); opacity:.4; }
    40%{ transform:translateY(-5px); opacity:1; }
}

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    background: #111827 !important; color: #E2E8F0 !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 14px !important; font-size: 0.92rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #374151 !important; }

/* ── Section headings ── */
.section-head {
    font-size: 0.7rem; font-weight: 700;
    color: #374151; letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 1.4rem 0 0.7rem;
    display: flex; align-items: center; gap: 8px;
}
.section-head::after {
    content: ''; flex: 1;
    height: 1px; background: rgba(255,255,255,0.05);
}

/* ── Summary card ── */
.summary-card {
    background: #0d1424;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    color: #94A3B8;
    font-size: 0.88rem;
    line-height: 1.75;
    margin-bottom: 1rem;
}
.summary-card b { color: #A5B4FC; }

/* ── Footer ── */
.dm-footer {
    position: fixed; bottom: 0; left: 0; width: 100%;
    background: rgba(11,17,32,0.97);
    border-top: 1px solid rgba(255,255,255,0.04);
    text-align: center; padding: 9px;
    color: #1E293B; font-size: 0.7rem; z-index: 999;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0B1120; }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 4px; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
defaults = {
    "chat_history": [],
    "vector_store": None,
    "processed": False,
    "api_key": "",
    "doc_stats": {},
    "pending_prompt": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Helpers ───────────────────────────────────────────────────────────────────
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
        chunk_size=800, chunk_overlap=100
    ).split_text(text)

def build_store(chunks):
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embedding=emb)

def get_llm():
    return ChatOpenAI(
        model="llama-3.1-8b-instant",
        api_key=st.session_state.api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.3,
    )

def safe_html(text):
    return (text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>"))

def now_time():
    return datetime.datetime.now().strftime("%I:%M %p")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:2.3rem;font-weight:700;letter-spacing:-0.03em;"
    "margin-bottom:2px'>🧠 DocuMind</h1>"
    "<p style='color:#475569;font-size:0.95rem;margin-bottom:1.4rem'>"
    "Chat intelligently with your PDFs — powered by Groq &amp; LangChain</p>",
    unsafe_allow_html=True,
)

# ── Setup expander ────────────────────────────────────────────────────────────
expander_label = (
    "⚙️  Setup — API Key & PDF Upload"
    if not st.session_state.processed
    else "✅  Documents ready — expand to change"
)
with st.expander(expander_label, expanded=not st.session_state.processed):
    api_input = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_••••••••••••••••",
        value=st.session_state.api_key,
    )
    pdf_docs = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True,
        type=["pdf"],
    )
    if pdf_docs:
        for pdf in pdf_docs:
            st.markdown(
                f"<div style='background:#1A2236;border-radius:8px;padding:7px 11px;"
                f"margin-bottom:4px;font-size:0.81rem;color:#94A3B8;"
                f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>"
                f"📄 {pdf.name}</div>",
                unsafe_allow_html=True,
            )

    process_btn = st.button("⚡  Process Documents")
    if process_btn:
        if not api_input:
            st.warning("Please enter your Groq API key.")
            st.stop()
        if not pdf_docs:
            st.warning("Please upload at least one PDF.")
            st.stop()
        st.session_state.api_key = api_input
        with st.spinner("Indexing your documents…"):
            raw, total_pages = get_pdf_text(pdf_docs)
            chunks = get_chunks(raw)
            st.session_state.vector_store = build_store(chunks)
            st.session_state.processed = True
            st.session_state.chat_history = []
            st.session_state.doc_stats = {
                "files": len(pdf_docs),
                "pages": total_pages,
                "chunks": len(chunks),
                "names": [p.name for p in pdf_docs],
            }
        st.rerun()

# ── Doc stats bar ─────────────────────────────────────────────────────────────
if st.session_state.processed and st.session_state.doc_stats:
    ds = st.session_state.doc_stats
    names_str = ", ".join(ds["names"]) if len(ds["names"]) <= 2 else f"{ds['names'][0]} +{len(ds['names'])-1} more"
    st.markdown(
        f"<div class='stats-bar'>"
        f"<div class='stat-chip'>📁 <span>{ds['files']}</span> file{'s' if ds['files']>1 else ''}</div>"
        f"<div class='stat-chip'>📄 <span>{ds['pages']}</span> pages</div>"
        f"<div class='stat-chip'>🧩 <span>{ds['chunks']}</span> chunks indexed</div>"
        f"<div class='stat-chip'>💬 <span>{len(st.session_state.chat_history)//2}</span> messages</div>"
        f"</div>"
        f"<div style='font-size:0.75rem;color:#1E293B;margin-bottom:0.8rem'>{names_str}</div>",
        unsafe_allow_html=True,
    )

# ── Divider ───────────────────────────────────────────────────────────────────
st.markdown(
    "<hr style='border:none;border-top:1px solid rgba(255,255,255,0.05);margin:0.8rem 0'>",
    unsafe_allow_html=True,
)

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.processed and not st.session_state.chat_history:
    st.markdown(
        "<div style='text-align:center;padding:3.5rem 1rem 2rem'>"
        "<div style='font-size:3.2rem;margin-bottom:1rem'>📚</div>"
        "<div style='font-size:1.15rem;font-weight:600;color:#334155;margin-bottom:0.35rem'>"
        "No documents loaded yet</div>"
        "<div style='color:#374151;font-size:0.88rem;line-height:1.7'>"
        "Enter your Groq API key, upload your PDFs,<br>and click <b style='color:#6366F1'>⚡ Process Documents</b> to begin.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Suggested questions (shown after processing, before first message) ────────
SUGGESTIONS = [
    "📋 Summarize this document",
    "🔑 What are the key points?",
    "❓ What questions does this answer?",
    "📊 Any data or statistics mentioned?",
]

if st.session_state.processed and len(st.session_state.chat_history) == 0:
    st.markdown("<div class='suggest-label'>✨ Try asking</div>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            if st.button(suggestion, key=f"sug_{i}"):
                st.session_state.pending_prompt = suggestion.split(" ", 1)[1]
                st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown("<div class='section-head'>Conversation</div>", unsafe_allow_html=True)

for i, (sender, message) in enumerate(st.session_state.chat_history):
    ts = now_time()
    if sender == "user":
        st.markdown(
            f"<div class='bubble-user'>"
            f"<div class='blabel blabel-you'>You</div>"
            f"{safe_html(message)}"
            f"<div class='msg-time'>{ts}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='bubble-ai'>"
            f"<div class='blabel blabel-ai'>🧠 DocuMind</div>"
            f"{safe_html(message)}"
            f"<div class='msg-time'>{ts}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ── Quick action buttons (after conversation starts) ─────────────────────────
if st.session_state.processed and len(st.session_state.chat_history) > 0:
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    qa1, qa2, qa3 = st.columns(3)
    with qa1:
        if st.button("📋 Summarize", key="qa_sum"):
            st.session_state.pending_prompt = "Give me a detailed summary of this document"
            st.rerun()
    with qa2:
        if st.button("🔑 Key Points", key="qa_key"):
            st.session_state.pending_prompt = "What are the most important key points in this document?"
            st.rerun()
    with qa3:
        if st.button("🗑️ New Chat", key="qa_clear"):
            st.session_state.chat_history = []
            st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input(
    "Ask DocuMind anything about your PDF…"
    if st.session_state.processed
    else "Process your PDFs above to start chatting…"
)

# Handle suggestion / quick-action clicks
if st.session_state.pending_prompt and not prompt:
    prompt = st.session_state.pending_prompt
    st.session_state.pending_prompt = None

if prompt:
    if not st.session_state.processed:
        st.warning("Complete setup above — enter your API key, upload PDFs, and click Process.")
        st.stop()

    st.session_state.chat_history.append(("user", prompt))

    docs = st.session_state.vector_store.similarity_search(prompt)
    context = "\n".join(d.page_content for d in docs)

    final_prompt = (
        "You are DocuMind, an expert AI assistant for reading and analysing PDF documents.\n"
        "Answer using ONLY the context below. Be clear, structured, and helpful.\n"
        "If the answer is not in the context, say: "
        "'This information is not available in the uploaded PDF.'\n"
        "Always reference page numbers where available.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{prompt}"
    )

    with st.spinner("DocuMind is thinking…"):
        try:
            resp = get_llm().invoke(final_prompt)
            st.session_state.chat_history.append(("assistant", resp.content))
            st.rerun()
        except Exception as e:
            st.error("Error generating response.")
            st.code(str(e))

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='dm-footer'>"
    "🧠 DocuMind &nbsp;·&nbsp; Streamlit &nbsp;·&nbsp; Groq &nbsp;·&nbsp; "
    "LangChain &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; HuggingFace"
    "</div>",
    unsafe_allow_html=True,
)
