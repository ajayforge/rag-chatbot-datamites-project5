"""Streamlit interface for the local RAG project.

Run from the project directory:
    streamlit run app.py
Set GROQ_API_KEY as an environment variable or in .streamlit/secrets.toml.
The notebook must have been run once so that the vectorstore/ folder exists.
"""
import os
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from rag_core import (
    ABSTAIN_TEXT, EmptyAnswerError, ModelRateLimitError, HybridRetriever, build_prompt, docs_from_chroma,
    generate_answer, page_label,
)

PROJECT_DIR = Path(__file__).resolve().parent
VECTOR_DIR = PROJECT_DIR / "vectorstore"
COLLECTION = "knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = os.getenv("ANSWER_MODEL", "openai/gpt-oss-20b")

st.set_page_config(page_title="Intelligent RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 Intelligent RAG-Based Chatbot")
st.caption("DataMites AI Client Project 5 · Answers are grounded in your indexed research papers.")


def get_api_key():
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        try:
            key = str(st.secrets.get("GROQ_API_KEY", "")).strip()
        except Exception:
            key = ""
    return key


@st.cache_resource(show_spinner="Loading embedding model and ChromaDB index...")
def load_retriever():
    if not VECTOR_DIR.exists():
        raise FileNotFoundError(
            "The vectorstore folder is missing. Run the notebook through the "
            "Embeddings + ChromaDB indexing section first."
        )
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    store = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DIR),
    )
    if store._collection.count() == 0:
        raise ValueError("The ChromaDB collection is empty. Rebuild the index in the notebook.")
    # The keyword (BM25) index is rebuilt from the chunks stored in Chroma, so it always matches the vectors.
    return HybridRetriever(store, docs_from_chroma(store))


with st.sidebar:
    st.header("Settings")
    k = st.slider("Retrieved chunks (top-k)", min_value=2, max_value=10, value=8)
    mode = st.selectbox("Retrieval mode", HybridRetriever.MODES, index=0,
                        help="hybrid = vector search + BM25 keyword search combined")
    st.caption("The vector database is loaded from the local `vectorstore/` folder.")
    st.divider()
    st.markdown("**Knowledge base**")
    st.write("Three RAG research papers (PDFs)")
    st.markdown("**Model**")
    st.write(MODEL_NAME)

api_key = get_api_key()
if not api_key:
    st.warning(
        "Groq API key not found. Set the GROQ_API_KEY environment variable, "
        "or add GROQ_API_KEY = \"your_key\" to .streamlit/secrets.toml. "
        "Never share or commit your API key."
    )
    st.stop()

try:
    retriever = load_retriever()
except Exception as exc:
    st.error(str(exc))
    st.stop()

llm = ChatGroq(model=MODEL_NAME, temperature=0, api_key=api_key)
prompt = build_prompt()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved sources"):
                for source in message["sources"]:
                    st.write("•", source)

question = st.chat_input("Ask a question about the RAG research papers...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base and generating an answer..."):
            try:
                found = retriever.search(question, k=k, mode=mode)
                if not found:
                    text, sources = ABSTAIN_TEXT, []
                else:
                    text = generate_answer(llm, prompt, question, found)
                    sources = sorted({page_label(doc) for doc in found})
                st.markdown(text)
                if sources:
                    with st.expander("Retrieved sources"):
                        for source in sources:
                            st.write("•", source)
                st.session_state.messages.append(
                    {"role": "assistant", "content": text, "sources": sources}
                )
            except EmptyAnswerError:
                st.error("The model returned an empty answer. Please ask again.")
            except ModelRateLimitError as exc:
                st.warning(f"Groq quota/rate limit reached: {exc}")
            except Exception as exc:
                st.error(f"Request failed: {exc}")
