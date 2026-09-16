# Phase log

One entry per phase. Records what was verified against real source, what
changed, which files were touched, test counts before and after, the eval
result file produced, the smoke-test result, and every deviation from the
upgrade plan. The plan documents (`00_CURRENT_STATE_AUDIT` through
`07_IMPLEMENTATION_ROADMAP`) are held outside this repository; every
deviation from them is restated here in full, so this log is readable on
its own.

---

## Phase 0.0 — Repository reconstitution

**Status:** done
**Tests:** 48 passed before, 48 passed after (`MODEL_BACKEND=mock pytest backend/tests -q`)

### Why this step exists (deviation from the plan)

The plan documents assume the application source is a committed tree at
`backend/app/...` and `frontend/src/...`. It was not.

The upstream repository `Arnavdsp/Gargantua-end-to-end-RAG-pipeline` contains
**11 files on a single branch (`main`), no tags**:

```
LICENSE  MIGRATION.md  README.md  bench.py  build_notebook.py
docker-compose.yml  pytest.ini  gargantua.ipynb
document_intelligence_suite_final.ipynb
document_summarizer_iitisoc_ml32.ipynb
gargantua  demo vid_final.mp4
```

The entire 113-file application tree existed only as a base64 gzipped tarball
embedded in cell 5 of `gargantua.ipynb`, produced by `build_notebook.py`.
No file the plan names was under version control, so no file the plan names
could be opened, diffed or edited.

**Resolution:** the archive was extracted and committed to this repository as
a real tree. The upstream repository is left untouched.

### What was copied

| From the archive | Notes |
|---|---|
| `backend/` | `app/`, `tests/`, `Dockerfile`, `pyproject.toml`, three requirements files |
| `frontend/` | `src/`, `public/`, `tools/`, all tsconfig/vite/oxlint configs |
| `README.md`, `MIGRATION.md` | verbatim |
| `docker-compose.yml`, `pytest.ini`, `.env.example`, `.gitignore` | verbatim |

| From the upstream root | Notes |
|---|---|
| `bench.py` | verbatim |
| `build_notebook.py` → `tools/build_notebook.py` | **relocated.** The script computes `ROOT = Path(__file__).resolve().parent.parent` and writes `notebooks/gargantua.ipynb`, so it was written to live one directory below the repository root. At the upstream root its `ROOT` resolved outside the repository. Contents unchanged; only the path is corrected. |

### What was deliberately not copied

| Excluded | Reason |
|---|---|
| `gargantua.ipynb` | It embeds a snapshot of the source tree. Committing it beside the real tree creates two sources of truth that diverge on the first edit. It is fully regenerable from the committed tree by `tools/build_notebook.py`, and will be regenerated in Phase 5 rather than carried stale. |
| `gargantua  demo vid_final.mp4` | 3.8 MB binary, not referenced by the build |
| `document_intelligence_suite_final.ipynb`, `document_summarizer_iitisoc_ml32.ipynb` | Superseded predecessors; `MIGRATION.md` already records what changed |
| `LICENSE` | This repository's existing MIT `LICENSE` was kept. The upstream copy differs only in the copyright holder line. |

---

## Phase 0.1 — Structural verification

**Status:** done, read-only
**Tests:** 48 passed (47 without the `tesseract` binary installed — environmental,
not a code defect; `test_run_ocr_on_blank_image_does_not_crash` requires the
Tesseract executable on `PATH`)

### Hypotheses from `00_CURRENT_STATE_AUDIT.md` §0

| Hypothesis | Verdict | Evidence |
|---|---|---|
| `ModelService` ABC with `embed`, `extractive_qa`, `generate`, `get_cross_encoder`, `backend_name`, `device_info` | **confirmed, one signature correction** | `backend/app/services/model_service.py:45` |
| `VectorStore` ABC with `add`, `search`, `get`, `delete`, `exists` | **confirmed, signatures differ from the plan's assumption** | `backend/app/rag/vector_store.py:30` |
| `NumpyVectorStore` = cosine over one `.npz` per document | **confirmed** | `backend/app/rag/vector_store.py:47` |
| `retrieval_top_k=8`, `rerank_top_k=4`, `min_relevance_score=0.18` | **confirmed** | `backend/app/config.py:48-50` |
| Nine `ProcessingStage` values, ninth inferred as `failed` or `queued` | **confirmed — the ninth is `FAILED`** | `backend/app/schemas/documents.py:9-18` |
| `AskResponse` carries `grounding`, `relevance_score`, `abstained`, `model_used`, `citations[]` | **confirmed, plus `conversation_id` and `question`** | `backend/app/schemas/qa.py:37` |
| Retrieval is a single synchronous dense pass | **confirmed** | `backend/app/rag/retrieval.py:38` |
| `rag/reranker.py` exists with a fallback path | **confirmed, but the fallback is not what the plan describes** | `backend/app/rag/reranker.py:35` |
| Summarization/translation token budgeting — unknown, high risk | **resolved — see GAP-8 below** | |
| 48 tests across 11 modules, mock backend, no network | **confirmed exactly** | 48 collected, 48 passed |
| Pydantic Settings holds every model name and limit | **confirmed** | `backend/app/config.py:19` |
| `frontend/src/types/api.ts` is the response-shape source of truth | **confirmed** (129 lines) | |
| `frontend/src/sim/mapping.ts` is the single binding table | **confirmed** (413 lines) | |

### Real signatures, extracted verbatim

```python
# backend/app/services/model_service.py:45
class ModelService(ABC):
    def embed(self, texts: list[str]) -> np.ndarray: ...
    def extractive_qa(self, question: str, context: str) -> tuple[str, float, float]: ...
    def generate(self, system_prompt: str, user_prompt: str, *,
                 max_new_tokens: int, temperature: float) -> str: ...
    def get_cross_encoder(self, model_name: str): ...
    @property
    def backend_name(self) -> str: ...
    @property
    def device_info(self) -> dict[str, str]: ...
```

```python
# backend/app/rag/vector_store.py:30
class VectorStore(ABC):
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None: ...
    def search(self, document_id: str, query_embedding: np.ndarray,
               top_k: int) -> list[ScoredChunk]: ...
    def get(self, document_id: str) -> list[Chunk]: ...
    def delete(self, document_id: str) -> None: ...
    def exists(self, document_id: str) -> bool: ...
```

**Correction to `02_ARCHITECTURE_UPGRADE_SPEC.md` §B.2.** The spec expects
`get_cross_encoder(self) -> CrossEncoderProtocol | None`. The real method
**takes a `model_name` argument** and returns the loaded encoder. The spec's
own instruction applies: adapt `reranker.py` to the real signature, do not
change `ModelService`.

### Response shapes, extracted verbatim

```python
# backend/app/schemas/qa.py
class Citation(BaseModel):
    chunk_id: str
    page_number: int | None
    section: str | None
    snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)

class AskResponse(BaseModel):
    conversation_id: str
    question: str
    answer: str
    abstained: bool
    grounding: GroundingLevel          # none | weak | moderate | strong
    relevance_score: float = Field(ge=0.0, le=1.0)
    citations: list[Citation]
    model_used: str
```

```python
# backend/app/schemas/summary.py
class StructuredSummary(BaseModel):
    executive_summary: str
    key_findings: list[str]
    important_numbers: list[str]
    methodology: str | None
    limitations: str | None

class SummarizeResponse(BaseModel):
    document_id: str
    summary: StructuredSummary         # a nested object, NOT a string
    strategy: str                      # "direct" | "map_reduce"
    model_used: str
    cached: bool
```

```python
# backend/app/schemas/translate.py
class TranslateResponse(BaseModel):
    document_id: str
    source_language: str
    target_language: str
    translated_text: str
    provider: str                      # NOT model_used
    truncated: bool
```

**Corrections to `04_DATA_SCHEMA_API_DELTA.md` §2–§3.** The plan assumes
`Citation.page` / `Citation.text`; the real names are `page_number` /
`snippet`. It assumes `SummarizeResponse.summary: str`; it is a nested
`StructuredSummary`. It assumes `TranslateResponse.model_used`; the field is
`provider`. It assumes `relevance_score: float | None` on both `Citation` and
`AskResponse`; both are **non-nullable `float` constrained to `[0.0, 1.0]`**.

### Torch containment (R6 / NFR-3)

```
$ grep -rln "import torch\|from transformers" backend/app/
backend/app/services/model_service.py
```

**Clean — the invariant holds.** The broader grep in `00` §5
(`"import torch\|from torch\|transformers"`) additionally matches
`backend/app/config.py:53` and `backend/app/rag/reranker.py`, but both are
false positives: the first is the string `sentence-transformers/all-MiniLM-L6-v2`
in a model-name default, the second is the words "sentence-transformers" in a
docstring. No import exists in either.

### Config keys

Settings are **snake_case**, not the `UPPER_CASE` used throughout
`04_DATA_SCHEMA_API_DELTA.md` §5. New keys follow the existing convention:
`retrieval_top_k`, `rerank_top_k`, `min_relevance_score`, `reranker_model`,
`embedding_model`, `generation_model`, `qa_model`, `translation_provider`,
`model_backend`, `generation_max_new_tokens`, `generation_temperature`.

### Gap register, re-checked against real code

| Gap | Verdict |
|---|---|
| GAP-1 dense-only retrieval | **confirmed.** `retrieval.py:49` is a single `vector_store.search`. No lexical channel. |
| GAP-2 reranker scaffolded, not active | **confirmed, but mis-described — see below** |
| GAP-3 no eval harness | **confirmed.** No `eval/` anywhere. |
| GAP-4 no adaptive layer | **confirmed.** No `router.py`. |
| GAP-5 no persistent public demo | **confirmed.** |
| GAP-6 unmeasured abstention threshold | **confirmed, and worse than described — see below** |
| GAP-7 chunk count estimated | **confirmed.** |
| GAP-8 summarize/translate integrity | **one real defect found, narrower than feared — see below** |
| GAP-9 no latency instrumentation | **confirmed.** No timing anywhere in the ask path. |
| GAP-10 mock cannot exercise rerank | **confirmed, and stronger than described — see below** |
| GAP-11 chunking unmeasured | **confirmed.** |
| GAP-12 single-document scope | **confirmed.** Out of scope by instruction. |

### Deviations that change the plan

**D1 — The abstain gate already reads the reranker channel, not dense.**

`rag/retrieval.py:53-55`:

```python
reranked = reranker.rerank(question, initial, top_k=settings.rerank_top_k)
top_score = reranked[0].score if reranked else 0.0
grounding = classify_grounding(top_score, min_relevance=settings.min_relevance_score)
```

The default reranker is `LexicalOverlapReranker`, which returns
`0.75 * cosine + 0.25 * token_overlap` (`rag/reranker.py:48`). So
`min_relevance_score = 0.18` is **already** compared against a blended score,
not a dense cosine.

This contradicts `02` §0's migration path, which states the current abstain
channel is `dense`, and `04` §2's compatibility note, which states
`relevance_score` reports the dense cosine. Neither is true today. The
score-channel doctrine still holds — but the starting point is one channel
further along than the plan assumes, and the Phase 3 recalibration is
therefore mandatory rather than merely prudent.

**D2 — `relevance_score` is clamped, and will fabricate values under a cross-encoder.**

`rag/generation.py:50` and `api/routes/qa.py:63` both apply:

```python
round(min(max(score, 0.0), 1.0), 4)
```

Both `Citation.relevance_score` and `AskResponse.relevance_score` are Pydantic
`float` fields constrained `ge=0.0, le=1.0` and are **not nullable**. A
cross-encoder logit ranges roughly −11…+11, so once Phase 2 puts that score on
this channel, every positive logit clamps to `1.0` and every negative to `0.0`.

That is a fabricated value of exactly the kind NFR-5 and RISK-2 exist to
prevent, and `04` does not flag it because it assumed these fields were
nullable. Phase 2 must carry the cross-encoder score on a **new, separate,
kind-labelled field** and leave `relevance_score` on its existing channel.

**D3 — GAP-2 is overstated: the fallback is not a no-op.**

`00` §2 and `02` §B.3 describe the fallback as "embedding-similarity
re-sorting", i.e. a no-op reordering of an already-ranked list. It is not.
`LexicalOverlapReranker` blends in token overlap, which is real lexical
signal. `CrossEncoderReranker` is fully implemented and correct; it is simply
never constructed, because `config.reranker_model` defaults to `None` and
`build_reranker` branches on it (`rag/reranker.py:75`).

Consequence: some lexical signal is **already present in the baseline**, so
the Phase 1 hybrid gain will be smaller than the plan projects, and the
before/after table must not be read as "dense-only → hybrid".

**D4 — GAP-8: map-reduce already exists; the defect is an unbounded reduce step.**

`services/summarization_service.py` (note: the file is `summarization_service.py`
and `translation_service.py`, **not** `summarization.py` / `translation.py` as
the plan names them) already implements map-reduce and already reports
`strategy` on the response. Three real defects remain:

1. `_MAP_STEP_WORD_BUDGET = 1500` (line 22) is a hardcoded **word** count, not
   a token budget derived from the tokenizer's `model_max_length`. FR-21 and
   `02` §6 require the latter.
2. **The reduce step is single-pass and unbounded** (lines 111–117):
   `synthesis_input` concatenates *every* group summary into one `generate`
   call with no cap and no recursion. A 60–100 page document produces roughly
   50 group summaries at ~220 tokens each — around 11k tokens into one prompt,
   truncated inside the model, silently. **This is the GAP-8 defect.**
3. `_parse_structured_output` hard-slices at `cleaned[:800]` (line 64) and
   `cleaned[:1500]` (line 74) on the JSON-parse-failure path — silent content
   loss in the degraded path, which is the path the mock backend always takes.

**D5 — Translation is a network call to Google, not a local model.**

`services/translation_service.py` uses `deep_translator`'s unofficial Google
Translate endpoint. It is free, so NFR-1 holds, but it breaks NFR-8 (no
runtime network for the pipeline) and has no SLA. All of `05` §1.4's licence
analysis (m2m100 vs NLLB vs opus-mt) is moot unless the provider is replaced.

Sentence-aware segmentation already exists (`_sentence_aware_chunks`, line 37),
so FR-22's core requirement is largely met.

Separately: `detect_language` (line 56) calls
`single_detection(text[:500], api_key=None)`. That function requires a
detectlanguage.com API key, so the call always raises and the
`except Exception: return "en"` always fires. **Source-language detection is
silently hardcoded to English.** No error, no warning, and `TranslateResponse`
reports the fabricated `"en"` as a measured fact.

**D6 — GAP-10 is stronger than described: the mock raises rather than returning `None`.**

`MockModelService.get_cross_encoder` (`model_service.py:316`) raises
`ModelUnavailable`. `CrossEncoderReranker.rerank` (`reranker.py:67`) calls it
with no `try`/`except`. So setting `reranker_model` while on the mock backend
produces an uncaught exception and a 500, not a graceful fallback. FR-10's
deterministic mock cross-encoder is required before Phase 2, not optional.

**D7 — `extractive_qa` is dead code.**

Declared on the ABC and implemented in both `HFModelService` and
`MockModelService`, but **called from nowhere** in the application. The
`qa_model` setting (`deepset/roberta-base-squad2`) is therefore loaded by
nothing. Relevant to the generation-backend change: a new backend can raise
`ModelUnavailable` from `extractive_qa` without affecting any live pipeline.

**D8 — RISK-7 (the embedding prefix trap) does not currently apply.**

The configured embedding model is `sentence-transformers/all-MiniLM-L6-v2`,
which is symmetric and takes no query/passage prefixes. RISK-7 becomes live
only if the model is switched to a bge- or e5-family model per `05` §1.1.

### Baseline health

```
$ MODEL_BACKEND=mock pytest backend/tests -q
48 passed
```

Network-free and GPU-free. Test count recorded at **48** across 11 modules:
`test_api_documents` 9, `test_document_validation` 8, `test_retrieval` 5,
`test_translation` 5, `test_api_qa` 4, `test_api_summary` 4, `test_chunking` 4,
`test_rag` 4, `test_pdf_extraction` 3, `test_ocr` 2.
