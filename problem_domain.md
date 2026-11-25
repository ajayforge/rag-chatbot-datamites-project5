# Problem Domain Description

### Domain: Research-paper question answering for Retrieval-Augmented Generation

This project addresses the difficulty of finding reliable, document-grounded answers across technical research papers about Retrieval-Augmented Generation (RAG). RAG literature contains specialized terminology, related methods, evaluation criteria and implementation details. Reading multiple papers manually to answer a focused question can be time-consuming, and a general-purpose language model may provide an answer that is not supported by the selected papers.

The chatbot uses three arXiv research papers placed in the local `data/` directory as its knowledge base. A user asks a question about RAG. The system searches the indexed paper passages, selects the most relevant chunks (hybrid retrieval: vector similarity plus BM25 keyword search) and supplies those passages to a Groq-hosted language model. The model is instructed to answer only from the retrieved context and to say that it does not know when the documents do not support an answer. Source filenames and page numbers are retained so that users can trace answers back to the papers.

The intended users are students, researchers and developers who need to explore RAG concepts such as retrieval, generation, indexing, RAG architectures and evaluation. The system is a learning and research assistant for the supplied papers; it is not a general web search engine and cannot answer reliably about information outside its local document collection.

The implementation uses LangChain for document loading and orchestration, Hugging Face `all-MiniLM-L6-v2` for embeddings, ChromaDB for persistent vector storage, and Groq `openai/gpt-oss-20b` for answer generation. RAGAS evaluates faithfulness and answer relevancy using ten question/reference pairs. The notebook selects a separate available Groq judge model when the account permits, keeping answer-generation and evaluation quotas separate where possible. A separate retrieval experiment compares vector, keyword and hybrid retrieval across chunk sizes and top-k settings. The Streamlit interface in `app.py` provides a user-facing chat experience over the same persisted vector index.

**Scope and limitations:** the knowledge base is limited to the three supplied papers. PDF text extraction may not preserve complex equations, tables or figures, and no OCR is included for scanned pages. Retrieval quality depends on chunking, embedding similarity and question wording. RAGAS scores are estimates from an LLM judge and must be reported from an actual successful run rather than entered manually.

**Success criteria (measurable):**
- RAGAS faithfulness of at least 0.85 and answer relevancy of at least 0.80 on the ten reference questions.
- Questions that are not covered by the papers must return the abstention sentence instead of an invented answer.
- Every answer shows the source file and page labels of the retrieved passages.
