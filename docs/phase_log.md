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

---

## Phase 0.2 — Groq generation backend

**Status:** done
**Tests:** 48 before → **63 after** (15 added, zero existing tests modified)

### Why

Generation moves from a locally-loaded `microsoft/Phi-3-mini-128k-instruct`
to the Groq API. Groq serves chat completions only — no embeddings, no
cross-encoders — so this is a composition, not a replacement: `generate` goes
over the wire, `embed` and `get_cross_encoder` stay local on CPU.

`HFModelService.generate` already built a `messages` list from a system and a
user prompt, so the call shape maps onto Groq's chat completions endpoint
without touching `rag/generation.py` or any call site.

### Consequence worth recording

With generation off-box, **nothing in the pipeline needs a GPU.** The
embedding model (`all-MiniLM-L6-v2`, ~90 MB) and the cross-encoder
(`ms-marco-MiniLM-L-6-v2`, ~90 MB) both run acceptably on CPU. The eval
harness can therefore run on any CPU box, with no Colab session and no daily
GPU quota — which removes the constraint that shapes most of the deployment
plan's resource budgeting.

### Free-tier facts, measured not assumed

Against `https://api.groq.com/openai/v1` on 2026-09-16:

| Observation | Value |
|---|---|
| Models available | 13 |
| Candidates with a usable context window | `openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b` — all 131k |
| Rate limit, requests | 1000 / min |
| Rate limit, **tokens** | **8000 / min** — the binding constraint |
| Round trip, grounded 2-passage question | ~223 ms |

**Chosen default: `openai/gpt-oss-20b`.** `groq/compound` and
`compound-mini` were rejected despite fitting the context budget: they are
agentic systems with built-in tool access, and a generator that can reach the
open web cannot honour "answer using ONLY the numbered evidence passages".
That would silently break groundedness, which is the property this project is
built to defend.

**`reasoning_effort` is set to `low`.** gpt-oss models emit internal reasoning
tokens billed against the 8000 TPM ceiling. Measured on one grounded
extraction question:

| effort | reasoning tokens | total tokens | answer |
|---|---|---|---|
| low | 31 | 169 | identical |
| medium | 54 | 191 | identical |
| high | 91 | 228 | identical |

Same answer, 26% fewer tokens at `low`. On a token-per-minute budget that is
throughput, so `low` is the default and the setting is exposed rather than
hardcoded.

### Design decisions

**`auto` never selects `groq`.** Routing generation off-box is an explicit
choice; a key appearing in the environment must not silently change which
backend answers, because `model_used` and every eval number are attributed to
it.

**`backend_name` reports both halves.** When the local delegate is the mock,
it returns `groq:<model> (embeddings: mock)` rather than plain `groq:<model>`.
Generation being real while retrieval scores are meaningless is exactly the
half-working state the existing mock-backend warning exists for, and the UI
renders this string directly.

**A missing key raises; it does not fall back.** Degrading to another backend
on a missing credential would make `model_used` a lie.

**`extractive_qa` raises on this backend.** Per D7 it has no call site
anywhere in the application, so standing up a span-extraction model to satisfy
the ABC would download ~500 MB that nothing consumes. It raises rather than
returning a fabricated span with a fabricated score.

**Retry policy distinguishes retryable from not.** 429 and 5xx back off and
retry (honouring `Retry-After` when present); other 4xx raise immediately,
because a bad key or a bad model name will not resolve on retry and retrying
burns the rate-limit budget.

### Files touched

| File | Change |
|---|---|
| `backend/app/config.py` | `model_backend` gains `"groq"`; new `groq_base_url`, `groq_model`, `groq_reasoning_effort`, `groq_timeout_seconds`, `groq_max_retries`; `groq_api_key` added to the existing Secrets section |
| `backend/app/services/model_service.py` | new `GroqModelService`, `_sleep_backoff`; `get_model_service` routes `"groq"` |
| `backend/requirements.txt` | `httpx==0.28.1` |
| `.env.example` | Groq section; credential documented by **name only**, with the four places it is supplied |
| `backend/tests/test_groq_backend.py` | new, 15 tests, network-free |

### Verification

```
$ MODEL_BACKEND=mock pytest backend/tests -q
63 passed

$ grep -rln "import torch\|from transformers" backend/app/
backend/app/services/model_service.py
```

Torch containment (R6) holds — the new backend imports neither, because the
local delegate owns that.

Live check through `GroqModelService` with the project's real
`_SYSTEM_PROMPT`, two evidence passages:

- grounded question → `Recall@4 for the hybrid configuration was 71.2 percent [1].`
- unanswerable question → `I don't have enough information in this document to answer that.`

Abstention behaviour survives the backend change.

### Credential handling

No key is committed. `groq_api_key` is read from the environment by the
existing pydantic-settings mechanism, exactly as `ngrok_authtoken` already
was. `.gitignore` already covers `.env`, `.env.local` and `backend/.env`.
A test asserts the key does not appear in the text of a raised
`ModelUnavailable`, since those messages reach logs and error handlers.

### Deviation

`03_EVAL_BENCHMARK_PLAN.md` §4.2 forbids "LLM-as-judge on a paid API". Groq's
free tier is not a paid API, but an optional LLM judge running on the same
endpoint as the generator would be judging its own family's output. If a judge
is used at all it must be disclosed under the §4.2 constraints, and it remains
a secondary signal — no claim rests on it alone.

---

## Phase 0.3 — Deterministic mock cross-encoder (FR-10, GAP-10, D6)

**Status:** done
**Tests:** 63 → **77** (14 added, zero existing tests modified)

### The defect

`MockModelService.get_cross_encoder` **raised** `ModelUnavailable`, and
`CrossEncoderReranker.rerank` calls it with no `try`/`except`. So setting
`reranker_model` while on the mock backend produced an uncaught exception and
a 500 — not a graceful fallback.

`00_CURRENT_STATE_AUDIT.md` describes the mock as one that "presumably returns
`None` or a stub". It did neither. The practical consequence was that no
assertion about rerank ordering could be written without a GPU and a
multi-gigabyte download, so none were.

### What was added

`_MockCrossEncoder` with a `predict(pairs) -> list[float]` shape matching
`sentence_transformers.CrossEncoder`. Scores are token overlap weighted toward
longer tokens (standing in for the rarer, more discriminative terms a real
cross-encoder keys on), plus a SHA-256-derived tie-break so equal-overlap
pairs have a total order that does not depend on list position or dict
iteration.

`score_kind` is `"mock_cross_encoder"`, so a mock score can never be mistaken
for a real one.

**Deliberately bounded to [0, 1].** A real cross-encoder emits unbounded
logits, but `Citation.relevance_score` is currently clamped to `[0, 1]`
(D2), so emitting logits here would fabricate values through that clamp.
Phase 2 moves the cross-encoder onto its own kind-labelled field; until then
the mock stays inside the existing range rather than making D2 worse.

### What did not change

`config.reranker_model` still defaults to `None`, so `build_reranker` still
returns `LexicalOverlapReranker` and default behaviour is untouched. A test
asserts this.

### Not fixed here, deliberately

`CrossEncoderReranker` still has no `try`/`except` around `get_cross_encoder`,
so a genuine load failure on the HF backend still raises rather than
degrading visibly. That is FR-8's job — the fallback must be *reported*
(`applied: false` plus a `fallback_reason`), not merely caught — and it needs
`RerankerInfo` on the response, which is Phase 2. Catching it here without
somewhere to report it would convert a loud failure into a silent downgrade,
which is worse.

---

## Phase 0.4 — Offline source-language detection (D5)

**Status:** done
**Tests:** 77 → **85** (8 added, zero existing tests modified)

### The defect

`translation_service.py` called:

```python
return single_detection(text[:500], api_key=None) or "en"
```

`deep_translator.single_detection` requires a detectlanguage.com API key.
Called with `api_key=None` it raises on **every** invocation, and the
surrounding `except Exception: return "en"` caught it. Source-language
detection was therefore hardcoded to English — no error, no warning — and
`TranslateResponse.source_language` reported `"en"` as a measured fact for
every document in every language.

### The fix

`detect_language_offline()` using **py3langid**, which carries its model as
bundled package data: no network call, no API key, no runtime download, so
NFR-8 holds. Measured cold: import plus four classifications in 0.57 s,
correct on English, French, German and Hindi.

It returns `str | None`. **None is a real answer** — returning a plausible
default is what caused the original defect.

### Schema change (additive)

`TranslateResponse` gains `source_language_detected: bool = False`. The three
cases are now distinguishable by a client:

| Case | `source_language` | `source_language_detected` |
|---|---|---|
| Caller declared it | the declared code | `false` |
| Detector identified it | the detected code | `true` |
| Neither | `"auto"` | `false` |

`"auto"` is the value actually passed to the provider, so it reports what
happened rather than asserting a language nobody determined.
`source_language` keeps its existing type (`str`, non-nullable), so R2 holds
and existing clients are unaffected.

### New finding

**D9 — `TranslateResponse.truncated` is hardcoded to `False`.**
`api/routes/translate.py` passes `truncated=False` unconditionally. The
provider does segment long input via `_sentence_aware_chunks`, so the field
reports a constant rather than an observation. It happens to be *accurate*
today (segmentation is not truncation, and nothing is dropped), but it is a
fabricated value in the NFR-5 sense: nothing measures it.

Not fixed here. Reporting it truthfully requires `translate()` to return
segment counts rather than a bare string, which is the FR-22 work in Phase
0.6 along with `segments_total` / `segments_translated` / `content_dropped`.
Recorded so it is not lost.

---

## Phase 0.5 — DocuMind Space (out of plan order, by request)

**Status:** built and verified locally; deploy pending authentication
**Tests:** 85 → **93** (8 added, zero existing tests modified)

### Ordering deviation

The roadmap puts deployment in Phase 5, after the eval harness and the frozen
baseline. This was built at Arnav's request while Phase 0 is still in
progress. The consequence is recorded rather than hidden: **the Evidence tab
has no numbers**, because no evaluation has been run yet. It renders an
all-em-dash table and says explicitly that the harness has not run. That is
the project's own rule applied to itself — an unmeasured value is an em-dash,
not a plausible placeholder.

### Free-tier rules, re-verified (RISK-5, `05` §7)

Checked against the official `huggingface-spaces` skill on 2026-09-16:

| Claim in `05` §1 | Verdict |
|---|---|
| Free accounts cannot create a Docker or plain CPU Gradio Space | **confirmed** — Gradio and Docker Spaces need a paid plan; `cpu-basic` is gated too |
| Free accounts get Static Spaces and ZeroGPU Gradio Spaces | **confirmed** |
| ZeroGPU cap of 2 Spaces per free account | **confirmed** |
| "~5 minutes of shared GPU time per day" for the account | **corrected** — the Space creator is not charged; each *visitor* consumes their own daily quota (~5 min free tier). The plan's arithmetic of "40–60 visitor interactions per day" understated capacity. |

**One fact changes the design.** The skill documents that a CPU-bound or
API-proxy Space on a free account should use `zero-a10g` with a single no-op
`@spaces.GPU` function — ZeroGPU requires at least one — while keeping the
real work outside it, so no quota is ever consumed.

That fits exactly. Generation is a Groq HTTP call, and both local models
(MiniLM embedder ~90 MB, cross-encoder ~90 MB) run acceptably on CPU. So
DocuMind schedules on ZeroGPU without ever requesting a GPU. The quota-anxiety
that shapes most of `05` §3.2 does not apply.

### Metaphor decision

The black-hole framing is not just naming: `schwarzschild.frag.glsl` is a real
null-geodesic raytracer and `sim/mapping.ts` binds RAG state to physics.
Renaming it "a mind" while the shader still renders a black hole would put a
label on screen that contradicts the pixels — the same failure the project
defines itself against.

**Resolved:** DocuMind is its own surface with its own visual language — a
document becomes memory nodes, a question becomes a signal, retrieved nodes
fire, and abstention is "no pathway activated". The raytracer keeps the
Gargantua identity and stays the Colab surface. The Space README states plainly
that neither is a cut-down version of the other.

### Structure

| Path | Role |
|---|---|
| `space/app.py` | Gradio Blocks UI, DocuMind identity. Presentation only. |
| `space/pipeline.py` | Thin adapter. Imports `app.rag.*` and `app.services.*`; defines no retrieval logic. |
| `space/README.md` | Space card with frontmatter |
| `space/requirements.txt` | Space runtime |
| `tools/build_space.py` | Assembles a flat deploy bundle from the real tree |

A Space repo must be flat — `app.py` at the root with its imports resolvable —
but this repository keeps the application under `backend/app/`. The bundle is
therefore **generated**, mirroring what `tools/build_notebook.py` already does
for the Colab notebook, rather than maintaining a second hand-kept copy of
`backend/app/`. `build/` is gitignored.

`backend/tests/test_space_bundle.py` enforces the rule by AST inspection:
neither `space/` file may define `retrieve`, `rerank`, `chunk_document`,
`classify_grounding`, `generate_grounded_answer` or the fusion functions, and
`pipeline.py` must import them from `app.*`.

### Verified locally

Bundle built (47 files), then run end to end against the real Groq API on a
document containing known figures:

| Question | Result |
|---|---|
| "What was the recall@4 for the hybrid configuration?" | answered `71.2 percent` with citation, grounding `strong`, top score 0.6807 |
| "What was the p95 latency on the fast path?" | answered `980 milliseconds` with citation |
| "What is the airspeed velocity of an unladen swallow?" | **abstained** |

The run also exercised the disclosure path: `torch` was absent from the local
environment, so the delegate fell back to mock embeddings and `backend_name`
reported `groq:openai/gpt-oss-20b (embeddings: mock)`, which the UI renders as
a banner. Real generation over meaningless retrieval scores was visible rather
than silent — the property Phase 0.2 added, working.

### Not done

- Not deployed. `hf auth login` is a device flow requiring Arnav to authorise
  in a browser; per R8 no credential is generated or requested inline.
- No preloaded corpus yet (`05` §2.1 wants 2–3 eval documents indexed at build
  time). That depends on the eval corpus, which is Phase 0.5 of the roadmap.
- Summarize and translate are not exposed as tabs yet; `pipeline.summarize`
  exists but has no UI. R4 requires all three pipelines at a phase boundary,
  so this is incomplete against that gate and is recorded as such.
