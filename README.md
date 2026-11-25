# Intelligent RAG-Based Chatbot

DataMites AI Engineering Track — Client Project 5

## Problem domain

This project provides question answering over a focused knowledge base of three arXiv research papers about Retrieval-Augmented Generation (RAG). The use case is to help a learner ask questions about RAG methods, architecture and evaluation while receiving answers grounded in retrieved passages rather than unsupported general model knowledge.

## Technology stack

- Python and Jupyter Notebook
- LangChain for document loading, splitting and LLM orchestration
- Hugging Face `sentence-transformers/all-MiniLM-L6-v2` for embeddings
- ChromaDB for persistent vector storage
- BM25 keyword retrieval combined with vector retrieval using Reciprocal Rank Fusion
- Groq API for answer generation
- RAGAS for Faithfulness and Answer Relevancy evaluation
- Streamlit for the web interface

## Architecture

```text
PDF/TXT/Markdown documents
          |
    LangChain loaders
          |
  Cleaning and chunking
          |
 Sentence Transformer embeddings
          |
   +------+------+
   |             |
 ChromaDB       BM25
 vector        keyword
 search        search
   |             |
   +------+------+
          |
 Reciprocal Rank Fusion
          |
 Retrieved context + question
          |
 Grounded prompt -> Groq LLM
          |
 Answer + source/page labels
          |
     CLI / Streamlit

10 reference Q&A pairs -> RAGAS -> Faithfulness + Answer Relevancy
```

## Setup

1. Install Python and Anaconda, then open this notebook in Jupyter.
2. Run the notebook cells from top to bottom. The installation cell installs the required packages.
3. The data cell downloads the three selected arXiv PDFs into `data/` when they are not already present. You can also place your own PDF, TXT or Markdown files in that folder.
4. When prompted, enter a Groq API key. The key is requested through hidden input and is not written into the notebook.
5. Continue through document loading, chunking, embedding and ChromaDB indexing.

## Run the chatbot

After the notebook has created `vectorstore/`, `rag_core.py` and `app.py`, open a terminal in the same project directory and set your API key.

Windows Command Prompt:

```bat
set GROQ_API_KEY=your_groq_key
python -m streamlit run app.py
```

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_key"
python -m streamlit run app.py
```

Open the local Streamlit URL printed in the terminal (normally `http://localhost:8501`). The app provides retrieval-mode and top-k controls and displays retrieved source/page labels.

For an interactive notebook command-line chat, call `chat_cli()` after the RAG pipeline has been initialized.

## Evaluation

The notebook contains ten reference Q&A pairs and evaluates generated answers using RAGAS Faithfulness and Answer Relevancy. It also tests out-of-scope questions and compares vector, keyword and hybrid retrieval settings. Scores are saved under `results/`. RAGAS values depend on the LLM judge and may vary between runs; report only completed scores.

## Generated files

- `data/` — source documents
- `rag_core.py` — shared retrieval and answer-generation functions
- `vectorstore/` — persistent ChromaDB index
- `app.py` — Streamlit application
- `results/` — QA, RAGAS, retrieval and environment reports
- `rag_artifacts.zip` — packaged vector store and results (created by the final packaging cell)

## Notes

Keep the notebook and generated project files in the same directory. Do not commit or share your Groq API key. The PDF loader extracts selectable text; scanned pages, complex tables, equations and figures may not be represented accurately. The chatbot is document-grounded and is not a general web-search system.
