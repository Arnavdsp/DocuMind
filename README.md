# DocuMind

### Document Intelligence & Grounded RAG System

**Upload a document. Ask a question. Get an answer grounded in the document — with source citations and an explicit abstention path when the evidence is insufficient.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow?logo=huggingface)](https://huggingface.co/spaces/ADP123456/DocuMind)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/Frontend-React%2019-61DAFB?logo=react)](https://react.dev/)
[![RAG](https://img.shields.io/badge/Architecture-RAG-blueviolet)](https://github.com/Arnavdsp/DocuMind)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **DocuMind is not just a document chatbot.**
>
> It is an end-to-end document intelligence system designed around **reliable ingestion, persistent retrieval, reranking, grounded generation, source attribution, and measurable inference stages.**

---

## Live Demo

**Try DocuMind:**  
https://huggingface.co/spaces/ADP123456/DocuMind

The project is also structured as a deployable application rather than only a notebook/demo: the repository contains a dedicated backend, frontend, Docker deployment configuration, tests, tooling, and a persistent data layer. citeturn0view0turn0view1

---

# What Problem Does DocuMind Solve?

Traditional document QA systems often hide several engineering problems behind a simple chat interface:

- PDFs may contain both native text and scanned pages.
- OCR can be expensive and unnecessary when a usable text layer already exists.
- Re-embedding an entire document for every question wastes computation.
- Pure vector similarity can retrieve semantically related but imprecise passages.
- LLMs can generate plausible answers that are not actually supported by the document.
- Users need to know **where an answer came from**.
- A system should know when the retrieved evidence is insufficient instead of confidently guessing.

DocuMind addresses these problems as an integrated pipeline.

---

# System Architecture

```text
                         ┌──────────────────────┐
                         │       Frontend       │
                         │   React + TypeScript │
                         │      + Vite          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      FastAPI API     │
                         │  Documents / QA /    │
                         │  Jobs / Health / ... │
                         └──────────┬───────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   │                                 │
                   ▼                                 ▼
        ┌──────────────────────┐          ┌──────────────────────┐
        │ Document Ingestion   │          │     RAG Pipeline     │
        │                      │          │                      │
        │ PDF / Image / TXT    │          │ Query Embedding      │
        │ Native Extraction    │          │        ↓             │
        │ OCR Fallback         │          │ Vector Retrieval     │
        │ Validation            │          │        ↓             │
        │ Page Metadata         │          │ Reranking            │
        └──────────┬───────────┘          │        ↓             │
                   │                      │ Grounding Gate       │
                   ▼                      │        ↓             │
        ┌──────────────────────┐          │ Grounded Generation  │
        │ Persistent Storage   │          │        ↓             │
        │                      │          │ Citations / Abstain  │
        │ Document Metadata    │          └──────────────────────┘
        │ Raw Documents        │
        │ Page Text            │
        │ Embeddings            │
        └──────────────────────┘
```

---

# Core Engineering Pipeline

## 1. Document Ingestion

DocuMind supports multiple document inputs:

- PDF
- JPG / JPEG
- PNG
- TXT

PDF processing is **page-aware** rather than treating the entire document as either text or OCR.

For each PDF page:

```text
                PDF Page
                   │
                   ▼
          Native text extraction
                   │
             ┌─────┴─────┐
             │           │
       Sufficient       Sparse /
          text           empty
             │           │
             ▼           ▼
        Use native     Render page
          text             │
                           ▼
                          OCR
                           │
                           ▼
                    Combine metadata
```

Pages with sufficient native text use the extracted text directly. Sparse or empty pages fall back to OCR, allowing scanned pages to be processed without unnecessarily OCR-ing every page. 

This design also records:

- page number
- extraction method
- OCR confidence
- low-quality status

That metadata later becomes useful for source inspection and debugging.

---

# 2. Content-Addressed Document Handling

Documents are identified from their content rather than relying only on filenames.

This enables a useful optimization:

```text
Upload
  │
  ▼
Compute document ID
  │
  ▼
Already ingested?
  │
 ┌┴───────────────┐
 │                │
Yes              No
 │                │
 ▼                ▼
Reuse existing    Run ingestion
index             pipeline
```

The API therefore avoids unnecessarily repeating ingestion for an identical document that has already reached the ready state. 

---

# 3. Persistent Vector Retrieval

A major design decision in DocuMind is that document chunks are **embedded once during ingestion**.

The query path does **not** re-embed every document chunk.

Instead:

```text
Document ingestion

Document
   ↓
Chunking
   ↓
Embedding
   ↓
Persistent Vector Store
```

Then:

```text
User Question
      ↓
ONE query embedding
      ↓
Vector similarity search
      ↓
Top-K candidates
```

The current implementation provides a `VectorStore` abstraction with a `NumpyVectorStore` implementation.

Embeddings are normalized and searched using cosine similarity. Per-document embeddings and chunk metadata are persisted as compressed `.npz` files. 

### Why this matters

The architecture deliberately separates the vector-store interface from its implementation.

The same interface can support a future migration to:

- FAISS
- Qdrant
- pgvector
- another production vector database

without forcing changes throughout the retrieval or API layers. 

---

# 4. Retrieval + Reranking

Vector retrieval provides the initial candidate set.

DocuMind then applies a second-stage reranking step.

```text
Question
   │
   ▼
Query Embedding
   │
   ▼
Vector Search
   │
   ▼
Candidate Chunks
   │
   ▼
Reranking
   │
   ▼
Best Evidence
```

Two reranking strategies are supported behind the same interface:

### Lexical overlap reranking

The default lightweight implementation combines:

- semantic retrieval score
- query-token overlap

with semantic similarity receiving the larger weight.

### Cross-encoder reranking

A configurable `CrossEncoderReranker` can use a Sentence Transformers cross-encoder when a reranker model is configured.

This gives the system a path from a lightweight demo configuration toward a stronger neural reranking configuration without changing the surrounding retrieval interface. 

---

# 5. Grounding-Aware Generation

DocuMind does not simply retrieve text and send the entire document to an LLM.

The generator receives **only the retrieved evidence passages**.

The generation prompt explicitly instructs the model to:

1. use only the supplied evidence,
2. avoid outside knowledge,
3. answer concisely,
4. abstain when the evidence is insufficient,
5. reference evidence passages when appropriate.

If retrieval does not provide sufficient evidence, the system can return:

> "I don't have enough information in this document to answer that."

This is an explicit **abstention path**, rather than relying on the model to decide on its own whether it knows the answer. 

---

# 6. Grounding Levels

Retrieval is converted into a grounding classification:

```text
             Retrieved evidence
                    │
                    ▼
             Relevance score
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
       NONE       WEAK       MODERATE
                                │
                                ▼
                              STRONG
```

The API exposes the resulting grounding level and relevance score alongside the generated answer. 

This makes retrieval quality part of the application state rather than an invisible internal operation.

---

# 7. Source Citations

Every retrieved chunk retains document provenance such as:

- chunk ID
- page number
- section
- snippet
- relevance score

The answer layer can therefore associate generated responses with the underlying source passages.

Conceptually:

```text
Answer
  │
  ├── Source → Page 4
  ├── Source → Page 7
  └── Source → Page 8
```

This allows the frontend to present document-grounded evidence rather than an unsupported text response. 

---

# 8. Measurable Retrieval Pipeline

DocuMind also instruments individual retrieval stages.

The retrieval pipeline tracks:

- embedding latency
- vector-search latency
- reranking latency
- generation latency
- total request latency

The implementation intentionally distinguishes between a stage that **did not execute** and a stage that executed in approximately zero milliseconds, returning `None` where appropriate rather than misleading `0 ms` measurements. 

This makes the application easier to profile and optimize.

---

# Supported Capabilities

| Capability | Implementation |
|---|---|
| PDF ingestion | Yes |
| Native PDF text extraction | Yes |
| Page-level OCR fallback | Yes |
| Image OCR | Yes |
| TXT ingestion | Yes |
| Document validation | Yes |
| Background ingestion jobs | Yes |
| Persistent document metadata | Yes |
| Persistent embeddings | Yes |
| Semantic retrieval | Yes |
| Lightweight reranking | Yes |
| Cross-encoder reranking | Configurable |
| Grounding classification | Yes |
| Grounded generation | Yes |
| Abstention | Yes |
| Source citations | Yes |
| Retrieval latency instrumentation | Yes |
| REST API | FastAPI |
| Frontend | React + TypeScript |
| Docker deployment | Yes |
| Hugging Face deployment | Yes |

---

# Technology Stack

## Backend

- Python 3.11+
- FastAPI
- Pydantic
- NumPy
- pdfplumber
- PyTesseract
- Pillow
- HTTPX
- deep-translator
- py3langid

The backend pins its core dependencies and separately defines its ML/RAG dependencies. 

## ML / RAG

- Hugging Face Transformers
- Sentence Transformers
- Accelerate
- bitsandbytes
- embedding-based retrieval
- optional cross-encoder reranking
- configurable generation backend

The repository intentionally avoids pinning PyTorch in the ML requirements because the Colab deployment can provide a CUDA-compatible build. 

## Frontend

- React 19
- TypeScript
- Vite
- Tailwind CSS
- PostCSS
- Playwright
- oxlint

The frontend is configured as a modern Vite/React application with dedicated build, lint, preview, and API-test scripts. 

## Deployment

- Docker
- Docker Compose
- persistent application volume
- health checks
- Hugging Face Spaces

The Docker deployment uses a persistent volume for documents, SQLite metadata, and per-document vector indexes. 

---

# Repository Structure

```text
DocuMind/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── documents.py
│   │   │       ├── health.py
│   │   │       ├── jobs.py
│   │   │       ├── qa.py
│   │   │       ├── summarize.py
│   │   │       └── translate.py
│   │   │
│   │   ├── ingestion/
│   │   │   ├── extractors.py
│   │   │   ├── ocr.py
│   │   │   └── validation.py
│   │   │
│   │   ├── rag/
│   │   │   ├── chunking.py
│   │   │   ├── generation.py
│   │   │   ├── reranker.py
│   │   │   ├── retrieval.py
│   │   │   └── vector_store.py
│   │   │
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── storage/
│   │   └── utils/
│   │
│   ├── tests/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── requirements.txt
│   └── requirements-ml.txt
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   ├── vite.config.ts
│   └── tsconfig.json
│
├── space/
├── tools/
├── docs/
├── docker-compose.yml
├── MIGRATION.md
├── bench.py
├── .env.example
└── LICENSE
```

The repository is organized as a full application rather than a single notebook, with separate backend, frontend, Space, documentation, tooling, deployment, and migration components. 

---

# API Surface

The backend exposes dedicated API routes for the major document workflows.

### Documents

```http
POST   /api/documents
GET    /api/documents
GET    /api/documents/{document_id}
DELETE /api/documents/{document_id}
GET    /api/documents/{document_id}/pages
```

### Document QA

```http
POST /api/documents/{document_id}/ask
```

The document endpoint also creates background ingestion jobs, while the QA endpoint validates document readiness before executing retrieval and generation.  

---

# Running Locally

## 1. Clone

```bash
git clone https://github.com/Arnavdsp/DocuMind.git
cd DocuMind
```

## 2. Configure environment

```bash
cp .env.example .env
```

Configure the model/provider settings required by your deployment.

---

## 3. Backend

```bash
cd backend

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-ml.txt
```

Start the API:

```bash
uvicorn app.main:app --reload --port 8000
```

---

## 4. Frontend

```bash
cd frontend

npm install
npm run dev
```

For a production frontend build:

```bash
npm run build
```

---

# Docker Deployment

DocuMind includes a Docker Compose deployment path:

```bash
docker compose up --build
```

The deployment exposes the application on:

```text
http://localhost:8000
```

The Compose configuration mounts a persistent volume for:

- documents
- SQLite metadata
- vector indexes

and includes an application health check. 

---

# Design Decisions

## Why page-level OCR?

A document does not necessarily have one extraction mode.

A single PDF can contain:

```text
Page 1 → native text
Page 2 → scanned image
Page 3 → native text
Page 4 → scanned table
```

Applying OCR globally would waste computation.

Applying native extraction globally could silently lose scanned content.

DocuMind evaluates pages independently and selectively falls back to OCR. 

---

## Why persistent embeddings?

Re-embedding every document chunk for every user question creates unnecessary computation.

DocuMind instead follows:

```text
INGESTION

chunks → embeddings → persistent index


QUERY

question → ONE embedding → search → rerank
```

This turns document embeddings into reusable state rather than recomputing them for every request. 

---

## Why reranking?

Vector retrieval is useful for finding semantically related material, but the top semantic results are not necessarily the most precise evidence for a particular question.

The second-stage reranker provides another signal before generation.

The architecture also keeps reranking behind an interface, allowing lightweight lexical reranking and neural cross-encoder reranking to coexist. 

---

## Why abstention?

A document QA system should not treat "generate something plausible" as success.

DocuMind introduces an explicit retrieval/grounding gate:

```text
Question
   ↓
Retrieve
   ↓
Is evidence sufficiently relevant?
   │
   ├── No  → Abstain
   │
   └── Yes → Generate grounded answer
```

The generator is additionally instructed to answer only from the supplied evidence. 

---

# Engineering Focus

DocuMind was designed around a few principles:

### 1. Grounding over fluency

A fluent answer that cannot be traced to the document is not sufficient.

### 2. Compute reuse

Expensive document-side computation should happen during ingestion and be reused during querying.

### 3. Explicit interfaces

Vector storage and reranking are abstracted so individual components can be replaced without rewriting the application.

### 4. Observable inference

Retrieval stages expose latency information rather than treating the RAG pipeline as a black box.

### 5. Graceful failure

The system has explicit handling for:

- invalid documents
- oversized documents
- extraction failures
- low-quality OCR
- documents that are not ready
- insufficient retrieval evidence

---

# Testing & Development

The backend includes a dedicated test structure and development dependencies.

The frontend also exposes development commands for:

```bash
npm run build
npm run lint
npm run test:api
```

The project therefore separates development, ML dependencies, application dependencies, and deployment concerns rather than relying on a single monolithic environment.  

---

# Current Deployment Model

DocuMind is available as a Hugging Face Space:

**https://huggingface.co/spaces/ADP123456/DocuMind**

The repository also contains a Docker-based deployment path designed around persistent application storage, providing a path beyond the ephemeral demo environment. citeturn0view1turn15file0

---

# Future Engineering Directions

The current abstractions make several extensions straightforward:

- Replace NumPy vector storage with FAISS/Qdrant/pgvector
- Add hybrid lexical + semantic retrieval
- Expand reranking models
- Add streaming generation
- Add richer document formats
- Introduce distributed background workers
- Add automated retrieval-quality benchmarks
- Add observability/tracing
- Add authentication and multi-user document isolation
- Add evaluation datasets for grounded QA
- Optimize GPU inference and batching

These are natural extensions of the existing architecture rather than changes to the project's core design.

---

# Why This Project Matters

DocuMind demonstrates a broader engineering idea:

> **Building useful AI systems is not only about choosing an LLM. It is about designing the system around the model.**

The project combines:

**Document Processing**  
→ **OCR**  
→ **Chunking**  
→ **Embeddings**  
→ **Persistent Retrieval**  
→ **Reranking**  
→ **Grounding**  
→ **Generation**  
→ **Citations**  
→ **Abstention**  
→ **API + UI + Deployment**

That makes DocuMind an example of **end-to-end AI engineering**, rather than an isolated LLM demo.

---

# License

This project is licensed under the **MIT License**.

See [LICENSE](LICENSE) for details.

---

## Author

**Arnav Deshpande**

B.Tech — Space Science & Engineering, IIT Indore

Interested in:

`AI Engineering` · `Machine Learning` · `Deep Learning` · `Computer Vision` · `Generative AI` · `RAG Systems` · `Document Intelligence`

GitHub: **https://github.com/Arnavdsp**

---

<p align="center">
  <strong>DocuMind — From documents to grounded intelligence.</strong>
</p>
