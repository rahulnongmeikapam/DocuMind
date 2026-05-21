# AI PDF Chatbot

A simple Streamlit app that lets you upload PDF files, create embeddings, and chat with your documents.

## Quick start

1. Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Add your API key to `api.env` or enter it in the app sidebar.

4. Run the app:

```powershell
streamlit run app.py
```

## Supported models

- `Google AI` using the Gemini API key (`GOOGLE_API_KEY` or `GEMINI_API_KEY`)
- `OpenAI` using `OPENAI_API_KEY`

## Usage

- Upload one or more PDF files
- Click `Process PDFs`
- Enter a question and see the answer returned from the indexed document content
- Download the conversation history as a CSV file
