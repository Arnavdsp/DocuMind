# DocuMind

Document question answering with retrieval, citations and abstention.

Upload a PDF, image or text file and ask questions about it. Answers cite the pages they came from, and when the retrieved passages don't support an answer, it says so instead of guessing.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow?logo=huggingface)](https://huggingface.co/spaces/ADP123456/DocuMind)
[![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Frontend](https://img.shields.io/badge/Frontend-React%2019-61DAFB?logo=react)](https://react.dev/)
[![RAG](https://img.shields.io/badge/Architecture-RAG-blueviolet)](https://github.com/Arnavdsp/DocuMind)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

The chat box is the small part. Most of the work is in ingestion, persistent retrieval, reranking, grounded generation, source attribution and timing each stage.

---

## Live demo

https://huggingface.co/spaces/ADP123456/DocuMind

The repo is a full application: backend, frontend, Docker deployment, tests, tooling and a persistent data layer.

---

## The problem

A chat box over a document hides several problems:

- PDFs may contain both native text and scanned pages.
- OCR can be expensive and unnecessary when a usable text layer already exists.
- Re-embedding an entire document for every question wastes computation.
- Pure vector similarity can retrieve semantically related but imprecise passages.
- LLMs can generate plausible answers that are not actually supported by the document.
- Users need to know where an answer came from.
- A system should know when the retrieved evidence is insufficient instead of confidently guessing.

The pipeline below handles each of these.

---

## Architecture

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

## Pipeline

### 1. Ingestion

DocuMind supports multiple document inputs:

- PDF
- JPG / JPEG
- PNG
- TXT

PDFs are handled page by page, so one document can mix native text and OCR.

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

### 2. Content-addressed documents

A document's ID is a hash of its content, not its filename.

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

### 3. Persistent vector retrieval

Chunks are embedded once, at ingestion.

A question only embeds the query, never the document chunks again.

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

#### Why the interface is separate

The vector-store interface is separate from its implementation.

The same interface can support a future migration to:

- FAISS
- Qdrant
- pgvector
- another production vector database

without forcing changes throughout the retrieval or API layers. 

---

### 4. Retrieval and reranking

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

#### Lexical overlap reranking

The default lightweight implementation combines:

- semantic retrieval score
- query-token overlap

with semantic similarity receiving the larger weight.

#### Cross-encoder reranking

A configurable `CrossEncoderReranker` can use a Sentence Transformers cross-encoder when a reranker model is configured.

This gives the system a path from a lightweight demo configuration toward a stronger neural reranking configuration without changing the surrounding retrieval interface. 

---

### 5. Grounded generation

DocuMind does not simply retrieve text and send the entire document to an LLM.

The generator only sees the retrieved passages.

The generation prompt explicitly instructs the model to:

1. use only the supplied evidence,
2. avoid outside knowledge,
3. answer concisely,
4. abstain when the evidence is insufficient,
5. reference evidence passages when appropriate.

If retrieval does not provide sufficient evidence, the system can return:

> "I don't have enough information in this document to answer that."

So abstaining is a code path, not something left to the model.

---

### 6. Grounding levels

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

So retrieval quality is visible in the API response and the UI.

---

### 7. Source citations

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

The frontend shows these next to the answer.

---

### 8. Stage timings

DocuMind also instruments individual retrieval stages.

The retrieval pipeline tracks:

- embedding latency
- vector-search latency
- reranking latency
- generation latency
- total request latency

A stage that didn't run reports `null` in the API response, not `0 ms`, so a skipped stage can't be mistaken for a fast one.

This makes the application easier to profile and optimize.

---

## What it supports

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

## Stack

### Backend

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

### ML / RAG

- Hugging Face Transformers
- Sentence Transformers
- Accelerate
- bitsandbytes
- embedding-based retrieval
- optional cross-encoder reranking
- configurable generation backend

The repository intentionally avoids pinning PyTorch in the ML requirements because the Colab deployment can provide a CUDA-compatible build. 

### Frontend

- React 19
- TypeScript
- Vite
- Tailwind CSS
- PostCSS
- Playwright
- oxlint

The frontend is configured as a modern Vite/React application with dedicated build, lint, preview, and API-test scripts. 

### Deployment

- Docker
- Docker Compose
- persistent application volume
- health checks
- Hugging Face Spaces

The Docker deployment uses a persistent volume for documents, SQLite metadata, and per-document vector indexes. 

---

## Repository layout

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

Backend, frontend, the Space, docs, tooling, deployment and migrations each have their own folder. 

---

## API

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

## Running locally

### 1. Clone

```bash
git clone https://github.com/Arnavdsp/DocuMind.git
cd DocuMind
```

### 2. Configure the environment

```bash
cp .env.example .env
```

Configure the model/provider settings required by your deployment.

---

### 3. Backend

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

### 4. Frontend

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

## Docker

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

## Design decisions

### Why page-level OCR?

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

### Why persistent embeddings?

Re-embedding every document chunk for every user question creates unnecessary computation.

DocuMind instead follows:

```text
INGESTION

chunks → embeddings → persistent index


QUERY

question → ONE embedding → search → rerank
```

Embeddings are stored once and reused by every question.

---

### Why reranking?

Vector retrieval is useful for finding semantically related material, but the top semantic results are not necessarily the most precise evidence for a particular question.

The second-stage reranker provides another signal before generation.

The architecture also keeps reranking behind an interface, allowing lightweight lexical reranking and neural cross-encoder reranking to coexist. 

---

### Why abstention?

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

## Principles

DocuMind was designed around a few principles:

### 1. Grounding over fluency

A fluent answer that cannot be traced to the document is not sufficient.

### 2. Compute reuse

Expensive document-side computation should happen during ingestion and be reused during querying.

### 3. Explicit interfaces

Vector storage and reranking are abstracted so individual components can be replaced without rewriting the application.

### 4. Observable inference

Each retrieval stage reports its latency.

### 5. Graceful failure

The system has explicit handling for:

- invalid documents
- oversized documents
- extraction failures
- low-quality OCR
- documents that are not ready
- insufficient retrieval evidence

---

## Testing and development

The backend includes a dedicated test structure and development dependencies.

The frontend also exposes development commands for:

```bash
npm run build
npm run lint
npm run test:api
```

Development, ML, application and deployment dependencies are kept in separate files.

---

## Deployment today

DocuMind is available as a Hugging Face Space:

https://huggingface.co/spaces/ADP123456/DocuMind

The Docker setup keeps application storage on a persistent volume, for running it outside the Space, where storage is ephemeral.

---

## Next

What I'd add next:

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

None of these needs a redesign; they fit behind the existing interfaces.

---

## Why I built it

Picking the model was the easy part. Most of the effort went into the system around it:

document processing → OCR → chunking → embeddings → persistent retrieval → reranking → grounding → generation → citations → abstention → API, UI and deployment.

I wanted to build all of it end to end, not only the LLM call.

---

## License

MIT.

See [LICENSE](LICENSE) for details.

---

## Author

Arnav Deshpande

B.Tech — Space Science & Engineering, IIT Indore

Interested in:

`AI Engineering` · `Machine Learning` · `Deep Learning` · `Computer Vision` · `Generative AI` · `RAG Systems` · `Document Intelligence`

GitHub: https://github.com/Arnavdsp

---

<p align="center">
  DocuMind: answers from your documents, with citations.
</p>
