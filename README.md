# AI PDF Chatbot

A simple Streamlit app that lets you upload PDF files, create embeddings, and chat with your documents.

## Live Demo
https://docu-mind-pdf.streamlit.app/

##  Features
📄 Upload multiple PDF documents

🧠 AI-powered document understanding using Groq LLM

🔍 Semantic search using FAISS vector database

📚 Chunk-based document indexing for better accuracy

💬 Interactive chat interface for Q&A over PDFs

⚡ Suggested prompts for quick interaction

🎨 Clean modern UI built with Streamlit


## How It Works
Upload one or more PDF files

Documents are split into chunks and embedded using HuggingFace embeddings

FAISS vector database stores document embeddings

User queries are matched against relevant chunks

Groq LLM generates responses using only retrieved context


## Usage

Upload PDF files

Click Process Documents

Ask any question about your document

Get AI-generated answers based only on PDF content
🔐 Supported Backend

🟢 Groq API (llama-3.1-8b-instant)

## Notes
No API key is required in the UI anymore

API key is securely stored using environment variables or Streamlit Secrets

Works locally and on Streamlit Cloud
