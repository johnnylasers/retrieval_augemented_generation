# RAG (Retrieval-Augmented Generation)

A from-scratch implementation of a RAG pipeline in Python, with an alternative LangChain-based version.

## What it does

Takes a user query, finds the most relevant passages from a document, and generates a grounded answer using an LLM — without hallucinating beyond the provided context.

## Pipeline

```
doc.md → chunking → embedding → vector DB → retrieval → reranking → generate → answer
```

| Step | File | Description |
|---|---|---|
| Chunking | `chunking.py` | Splits a markdown document into chunks by blank lines |
| Embedding | `embedding.py` | Encodes chunks into vectors using `shibing624/text2vec-base-chinese` (HuggingFace) |
| Vector DB | `vector_db.py` | Stores and retrieves embeddings using ChromaDB (in-memory) |
| Retrieval | `retrieval.py` | Queries ChromaDB with the user prompt to fetch top-K semantically similar chunks |
| Reranking | `reranking.py` | Re-scores retrieved chunks using a CrossEncoder for higher precision |
| Generation | `generate.py` | Sends reranked chunks + query to Gemini 2.5 Flash and returns the answer |

## LangChain version

`generate_byLangChain.py` reimplements the `generate` step using LangChain's `ChatGoogleGenerativeAI`, `PromptTemplate`, and `StrOutputParser`, while keeping the same custom chunking/embedding/retrieval/reranking pipeline.

## Setup

```bash
uv sync
cp .env.example .env  # add your GOOGLE_API_KEY
```

## Run

```bash
uv run python generate.py
# or
uv run python generate_byLangChain.py
```

## Dependencies

- `sentence-transformers` — local embeddings (pinned to 2.7.0 for Intel Mac compatibility)
- `chromadb` — vector store
- `google-genai` — Gemini LLM
- `langchain-google-genai` — LangChain wrapper for Gemini (used in LangChain version)
- `torch==2.2.2`, `numpy<2` — pinned for Intel Mac compatibility
