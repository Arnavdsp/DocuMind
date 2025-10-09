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
