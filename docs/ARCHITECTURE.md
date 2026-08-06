# Retrieval Pipeline

```
Query
  -> Sentence Transformer embed
  -> FAISS ANN search (dense)
  +
  -> BM25 keyword index (sparse)
  ->
  Reciprocal Rank Fusion
  -> Cross-Encoder Reranker (top-5)
  -> LLM Generation (Groq)
```

## Module responsibilities
- `rag/fusion.py` — RRF score combination
- `rag/reranker.py` — cross-encoder reranking
- `rag/retriever.py` — FAISS + BM25 search + deduplication
- `rag/embedder.py` — batched sentence-transformer embeddings
