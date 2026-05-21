# AI PDF Chatbot

A simple Streamlit app that lets you upload PDF files, create embeddings, and chat with your documents.

## Live Demo
https://docu-mind-pdf.streamlit.app/

🚀 Quick Start

Create and activate a Python virtual environment:

python -m venv .venv
.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt
🔐 Setup API Key (Required for Deployment)

This app uses Groq API, and the key is managed securely via Streamlit Secrets.

👉 For Local Development (.env file)

Create a .env file:

GROQ_API_KEY=your_api_key_here
👉 For Streamlit Cloud Deployment

Go to:

App → Settings → Secrets

Add:

GROQ_API_KEY = "your_api_key_here"
▶️ Run the App
streamlit run app.py
⚙️ Features
📄 Upload multiple PDF documents
🧠 AI-powered document understanding using Groq LLM
🔍 Semantic search using FAISS vector database
📚 Chunk-based document indexing for better accuracy
💬 Interactive chat interface for Q&A over PDFs
⚡ Suggested prompts for quick interaction
🎨 Clean modern UI built with Streamlit
🧠 How It Works
Upload one or more PDF files
Documents are split into chunks and embedded using HuggingFace embeddings
FAISS vector database stores document embeddings
User queries are matched against relevant chunks
Groq LLM generates responses using only retrieved context
📌 Usage
Upload PDF files
Click Process Documents
Ask any question about your document
Get AI-generated answers based only on PDF content
🔐 Supported Backend
🟢 Groq API (llama-3.1-8b-instant)
🧾 Notes
No API key is required in the UI anymore
API key is securely stored using environment variables or Streamlit Secrets
Works locally and on Streamlit Cloud
