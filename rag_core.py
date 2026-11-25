"""Retrieval and answer helpers shared by the notebook and the Streamlit app.

Why this module exists
----------------------
* Hybrid retrieval: dense vector search finds passages with similar meaning, BM25 keyword search
  finds passages that share the question's exact terms. Scientific text contains many specific
  terms (method names, numbers), so combining both with reciprocal rank fusion is more robust
  than either one alone.
* Reliable answers: a model call that returns an empty string is retried and never stored as a result.
* One prompt and one set of helpers for the notebook and the app, so both behave the same way.
"""
from __future__ import annotations

import hashlib
import json
import re
import time

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from rank_bm25 import BM25Okapi

ABSTAIN_TEXT = "I don't know based on the provided documents."

SYSTEM_PROMPT = (
    "You are a document-grounded assistant. Answer the question using ONLY the retrieved context. "
    "Do not use outside knowledge or invent facts. If the answer is not supported by the context, "
    "reply exactly: " + ABSTAIN_TEXT + " Keep the answer clear and concise. "
    "When useful, cite the source and page label shown in the context.\n\nContext:\n{context}"
)

STOPWORDS = frozenset(
    "the a an of in to and for is are was were be by with on as that this it its from or at "
    "what which how does do did used use between about into their there".split()
)


class EmptyAnswerError(RuntimeError):
    """The language model kept returning an empty answer."""


class ModelRateLimitError(RuntimeError):
    """The model provider rejected a request because its rate/token quota was reached."""


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens without stop words (used for BM25)."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


def page_label(doc: Document) -> str:
    """Human-friendly source label. PyPDFLoader stores zero-based page numbers."""
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page")
    return f"{source}, p. {int(page) + 1}" if page is not None else source


def doc_key(doc: Document) -> tuple:
    """Identity of a chunk: source, page and a hash of its text."""
    digest = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()
    return (doc.metadata.get("source", ""), doc.metadata.get("page", ""), digest)


def reciprocal_rank_fusion(rankings: list[list], k: int = 60) -> list[tuple]:
    """Fuse several ranked lists of keys. Higher score = better. Returns [(key, score)]."""
    scores: dict = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: -item[1])


def docs_from_chroma(vectorstore) -> list[Document]:
    """Rebuild the chunk list from a persisted Chroma collection (used by the Streamlit app)."""
    data = vectorstore.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]


class HybridRetriever:
    """Vector, keyword (BM25) or hybrid retrieval over the same chunk list."""

    MODES = ("hybrid", "vector", "keyword")

    def __init__(self, vectorstore, docs, candidates: int = 20, rrf_k: int = 60):
        self.vectorstore = vectorstore
        self.docs = list(docs)
        self.candidates = candidates
        self.rrf_k = rrf_k
        self._by_key = {doc_key(d): d for d in self.docs}
        self._bm25 = BM25Okapi([tokenize(d.page_content) or [""] for d in self.docs])

    def vector_search(self, query: str, n: int) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=n)

    def keyword_search(self, query: str, n: int) -> list[Document]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return [self.docs[i] for i in order if scores[i] > 0]

    def search(self, query: str, k: int = 6, mode: str = "hybrid") -> list[Document]:
        if mode == "vector":
            return self.vector_search(query, k)
        if mode == "keyword":
            return self.keyword_search(query, k)
        if mode != "hybrid":
            raise ValueError(f"Unknown retrieval mode: {mode!r}. Use one of {self.MODES}.")
        vector_docs = self.vector_search(query, self.candidates)
        keyword_docs = self.keyword_search(query, self.candidates)
        fused = reciprocal_rank_fusion(
            [[doc_key(d) for d in vector_docs], [doc_key(d) for d in keyword_docs]], k=self.rrf_k
        )
        lookup = {doc_key(d): d for d in vector_docs + keyword_docs}
        return [self._by_key.get(key) or lookup[key] for key, _ in fused[:k]]


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])


def format_context(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(f"[Source: {page_label(d)}]\n{d.page_content}" for d in docs)


def _as_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):  # some chat models return a list of content parts
        parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
        return "".join(parts).strip()
    return str(content or "").strip()


def generate_answer(llm, prompt, question: str, docs: list[Document],
                    retries: int = 3, wait_seconds: float = 3.0) -> str:
    """Ask the LLM for an answer. Empty replies may be retried; provider 429s fail fast."""
    messages = prompt.format_messages(context=format_context(docs), question=question)
    for attempt in range(1, retries + 1):
        try:
            response = llm.invoke(messages)
        except Exception as exc:
            # Do not retry a daily-token/rate-limit error: another immediate call cannot help.
            message = str(exc).lower()
            status = getattr(exc, "status_code", None)
            code = getattr(exc, "code", None)
            if status == 429 or "rate_limit_exceeded" in message or "tokens per day" in message:
                raise ModelRateLimitError(
                    "The LLM provider quota is exhausted or rate-limited. "
                    "Wait for the reset or switch to another model enabled for your API key."
                ) from exc
            raise

        text = _as_text(response.content)
        if text:
            return text
        if attempt < retries:
            time.sleep(wait_seconds)
    raise EmptyAnswerError(f"The model returned an empty answer {retries} times for: {question[:80]}")


def stable_signature(payload: dict) -> str:
    """Short hash that changes whenever the configuration or the questions change."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
