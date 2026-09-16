# GARGANTUA

**Document intelligence rendered as a Schwarzschild raytracer.**

Upload a document. It collapses into a singularity. Its chunks become the
accretion disk. When you ask a question, null geodesics are traced from the
observer toward the mass — most fall past the horizon, a few strike the disk
and light up. Those hot spots are your citations.

This is not a black hole with a chatbot in the sidebar. It is one continuous
simulation whose every parameter is driven by real retrieval state, and a
grounded RAG system whose every on-screen number is a measured value.

<p align="center">
  <img src="docs/poster.png" alt="The GARGANTUA render: an accretion disk lensed over the shadow of a black hole, with the photon ring visible and citation hot spots on the disk." width="820">
</p>

---

## Run it

### Colab (recommended — real models, one URL, ~5 minutes)

Open `notebooks/gargantua.ipynb` in Google Colab, set the runtime to a **GPU**
(Runtime → Change runtime type → T4 or better), and run all cells.

The notebook regenerates the entire project from an embedded archive, installs
dependencies without disturbing Colab's CUDA-matched `torch`, builds the
frontend, starts the API with the built UI mounted at `/`, and opens a public
tunnel. You get **one URL** serving both the interface and the API.

It does not need this repository, a clone, or any credentials. Cloudflare's
quick tunnel requires no account; ngrok is available as an alternative if you
have a token.

### Locally

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-ml.txt
cp ../.env.example .env
uvicorn app.main:app --reload           # http://127.0.0.1:8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev                             # http://localhost:5173, proxies /api
```

For a single-origin build the way Colab serves it:

```bash
cd frontend && npm run build            # emits frontend/dist
cd ../backend && uvicorn app.main:app   # serves the SPA at / and the API at /api
```

### Tests

```bash
MODEL_BACKEND=mock pytest backend/tests -q      # 48 tests, no GPU, no downloads
```

The mock backend is deterministic and dependency-free, which is what lets the
suite run in CI with no network. See *A warning about the mock backend* below.

---

## The mapping

The physics is not decoration laid over a dashboard. Each element of the
simulation is bound to a specific value from the API, and to nothing else.

| Physical object | RAG reality | Source |
|---|---|---|
| The singularity | the uploaded document | `DocumentRecord` |
| Schwarzschild radius | document size, page count, word count | `metrics` |
| Accretion disk particles | chunks in the vector index | estimated from `character_count` |
| Disk luminosity | index density (chunks per page) | ingestion result |
| Disk gaps / mottling | pages flagged `is_low_quality` | `pages[]` |
| Photon geodesics bending inward | retrieval — the question's path to evidence | `/ask` |
| Null geodesics past the horizon | retrieved-then-discarded candidates | `retrieval_top_k=8` → `rerank_top_k=4` |
| Lensed hot spots on the disk | surviving citations, positioned by page number | `citations[]` |
| Signal strength | `grounding` + `relevance_score` | `AskResponse` |
| **SIGNAL LOST** | `abstained === true` | `AskResponse.abstained` |
| Mission elapsed | time since ingestion completed | job timestamps |
| Inference backend | `model_used` / `backend_name` | `AskResponse.model_used` |
| Frequency shift | translation target language | `/translate` |

The whole table lives in one file — [`frontend/src/sim/mapping.ts`](frontend/src/sim/mapping.ts).
Nothing downstream of it invents a value.

**Ingestion is the collapse sequence.** The nine `ProcessingStage` values
(`uploading → validating → extracting → ocr → chunking → embedding → indexing →
ready`) drive the disk igniting from the inside out, from the real
`/api/jobs/{id}` poll at 900 ms. There is no timer filling in the gap between
polls and no easing toward the next stage. If the backend sits on `embedding`
for two minutes, the collapse sits there too.

**Grounding drives colour, exclusively.** `strong` → white-hot, `moderate` →
orange, `weak` → dim amber, `none` → dark. The `--color-grounding-*` tokens are
reserved: no hover state, progress bar or loading shimmer may read them, and a
grounding-bound element may not fall back to the chrome accent when the level is
unknown. Unknown renders as an em-dash, not as a colour.

---

## Architecture

```
gargantua/
├── backend/                  FastAPI — ingestion, RAG, API
│   ├── app/
│   │   ├── main.py           CORS, request-ID middleware, AppError handler,
│   │   │                     mounts frontend/dist LAST so it never shadows /api
│   │   ├── config.py         Pydantic Settings — every model name and limit
│   │   ├── api/routes/       health, documents, jobs, qa, summarize, translate
│   │   ├── rag/              chunking, vector_store, retrieval, reranker, generation
│   │   ├── services/         model_service, ingestion_pipeline, summarization, translation
│   │   ├── ingestion/        validation, extractors, ocr
│   │   └── storage/          repository (SQLite), blob_store (local, content-addressed)
│   └── tests/                11 modules, 48 tests, mock backend
├── frontend/
│   └── src/
│       ├── design-tokens.css The palette, sampled from the reference
│       ├── sim/              mapping.ts, camera.ts, quality.ts, renderer.ts,
│       │                     schwarzschild.frag.glsl  ← the physics
│       ├── scenes/           gargantua-canvas.tsx, static-fallback.tsx
│       ├── hud/              title-block, telemetry, navigation-panel
│       └── panels/           collapse, query, briefing, frequency, integrity
└── notebooks/gargantua.ipynb Self-contained Colab bootstrap
```

### Two abstractions everything else is built on

**`ModelService`** (ABC) — `embed`, `extractive_qa`, `generate`,
`get_cross_encoder`, `backend_name`, `device_info`. Implementations:
`HFModelService` (real models, lazy-loaded, detects CUDA and picks
dtype/quantization from the GPU's actual compute capability rather than assuming
bf16) and `MockModelService` (deterministic, no downloads). Selected by
`MODEL_BACKEND ∈ {auto, hf, mock}`. **Nothing outside `model_service.py` imports
`torch` or `transformers`.**

**`VectorStore`** (ABC) — `add`, `search`, `get`, `delete`, `exists`.
Implementation: `NumpyVectorStore`, cosine similarity over one `.npz` per
document on disk. Swapping in FAISS, Qdrant or pgvector means implementing five
methods; retrieval and the API layer do not change.

### API

```
GET    /health
GET    /readiness
POST   /api/documents                     -> { document, job_id }
GET    /api/documents
GET    /api/documents/{id}
DELETE /api/documents/{id}
GET    /api/documents/{id}/pages
GET    /api/jobs/{job_id}                 -> { status, stage, progress, error_message }
POST   /api/documents/{id}/ask            -> AskResponse
POST   /api/documents/{id}/summarize      -> SummarizeResponse
POST   /api/documents/{id}/translate      -> TranslateResponse
```

`frontend/src/types/api.ts` is the source of truth for response shapes.

---

## The rules this build holds itself to

**Every number on screen is a measured value.** Where a value is not measured,
the HUD renders an em-dash — exactly as the reference interface does. There is
no code path that turns a null into a zero, a placeholder or a plausible-looking
default.

This matters because of what it replaces. The version of this product that
preceded the rebuild displayed an **uncalibrated start/end logit as a
"confidence: NN%"** — a number that looked like a probability and was not one.
It also concatenated retrieved chunks into a 512-token extractive model and
silently truncated them, and re-embedded every chunk on every question.
All three are gone, and the first one is why there is no confidence percentage
anywhere in this interface.

- `FRAME RATE` is null until the first 1-second measurement window closes, so
  the HUD shows an em-dash on startup rather than a fabricated 60.
- `RENDER PROFILE` and `GEODESIC STEPS` are the resolution scale and shader loop
  bound actually in use this frame, not a preference the renderer ignores.
- `model_used` is reported truthfully. If it is `mock`, the HUD says so.
- Abstention, low OCR confidence and `is_low_quality` pages are surfaced, never
  hidden.

**Abstention is a first-class visual state, not an error.** When the retrieval
layer finds nothing above `min_relevance_score = 0.18`, the disk goes dark and
the HUD reads *NO GROUNDED SIGNAL — the document does not support this query*.
It gets the same visual weight as a successful answer, because the system
declining to answer from evidence it does not have is the system working.

**Embeddings are computed once, at ingestion.** A query embeds the question and
nothing else.

### Accessibility

This is where cinematic interfaces usually fail, so it is stated explicitly.

- `prefers-reduced-motion` is honoured: no orbit, no cinematic auto-sequence,
  static render. The camera holds a fixed pose and the geodesic trace completes
  instantly. Nothing is gated behind an animation.
- **Every visual-only state has a text equivalent.** The canvas is
  `aria-hidden`; the same values it depicts are rendered in parallel in the DOM
  from the same source, live-announced. Answers, citations, page numbers and
  scores are readable, selectable and screen-reader accessible with the canvas
  switched off entirely.
- Full keyboard operation (`1-4` views, `C` cine, `R` orbit, `P` params,
  `M` sound, `H` HUD), real focus rings, and no focus trap — the canvas is not
  focusable.
- Uppercase wide-tracked type is applied with `text-transform`, so accessible
  names stay in normal case. A screen reader announces "Summary", not
  "S-U-M-M-A-R-Y".
- WebGL unsupported or context lost → a genuinely usable non-3D fallback, not a
  dead end. Every panel is DOM, so losing WebGL costs the render and nothing
  else; a CSS-drawn schematic stands in, coloured by the same grounding level.

### Performance

- The scene is `React.lazy` + `Suspense`. The shader and WebGL host are in their
  own chunk and never sit on the critical path — verified by checking that
  `traceGeodesic` appears only in the `gargantua-canvas` chunk after a
  production build.
- `requestAnimationFrame` is paused when the tab is hidden **or** the canvas is
  offscreen. Both halves are needed; either alone leaves a raymarcher burning a
  core.
- Adaptive quality moves one tier at a time on measured FPS, with hysteresis and
  a cooldown so it cannot oscillate. Downshifts are fast (2 bad windows),
  upshifts slow (6 good windows). `LOWER QUALITY` pins the tier — an explicit
  user choice outranks the controller.
- Mobile and low-core devices start at a reduced tier rather than starting high
  and stuttering down.

---

## A warning about the mock backend

`MODEL_BACKEND=mock` exists so the test suite runs without a GPU or a multi-GB
download. Its embeddings are a hashed bag-of-words projection: good enough to
exercise retrieval logic, **not** good enough for its scores to mean anything.

On the mock backend specifically:

- `relevance_score` and `grounding` are not meaningful quantities.
- Abstention will not trigger reliably — an unrelated question can still score
  above the relevance floor.
- `generate` returns leading sentences of its input rather than a generation, so
  summaries come back sparse and answers read as echoes.

The UI states the backend on screen whenever it is `mock`. Judge the system's
grounding behaviour on the HF backend, which is what the Colab notebook runs.

## Known limits

- **Chunk count is estimated.** No endpoint returns the vector store's true
  count, so `DISK PARTICLES` is derived from `character_count` against the
  configured chunk size and is labelled with a `~`. It is the one number on
  screen that is an estimate, and it says so.
- **The displayed mass is theatre over a real number.** `4.2 × 10³ M☉` is the
  document's word count restated in solar-mass units. Divide it back out and you
  get the word count. The units are a conceit; the quantity is not invented.
- **Colab is a demo environment, not hosting.** The runtime recycles, the
  tunnel URL is ephemeral, and there is no real persistence or TLS story.
  `docker-compose.yml` and `backend/Dockerfile` are the deployment path.
- **OCR quality is Tesseract's.** Scanned pages surface `ocr_confidence` and
  `is_low_quality` honestly rather than being silently accepted.
- The reranker falls back to embedding-similarity ranking unless
  `RERANKER_MODEL` names a cross-encoder.

## Licence

MIT — see [LICENSE](LICENSE).
