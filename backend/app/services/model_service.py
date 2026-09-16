"""Model-service abstraction.

All model access goes through `ModelService` so the rest of the app never
imports `torch`/`transformers`/`sentence_transformers` directly, never
hard-codes a model name, and can be tested without downloading gigabytes of
weights. Two implementations are provided:

  * `HFModelService`  — real Hugging Face models, used in Colab/production.
    Loads lazily (first call, not import time), detects CUDA/CPU, and picks
    dtype/quantization based on what's actually available rather than
    assuming every GPU supports bf16.
  * `MockModelService` — deterministic, dependency-free stand-ins used in
    automated tests and as a CPU-only local-dev fallback. This is what lets
    `test_rag.py`/`test_api_qa.py` etc. run without a GPU or a multi-GB
    download, per the testing requirements.

`get_model_service()` selects an implementation based on
`settings.model_backend` ("auto" probes for torch/transformers being
importABLE — not downloading weights — and falls back to mock if absent).
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Protocol

import numpy as np

from app.config import Settings, get_settings
from app.logging import get_logger, log_event
from app.utils.errors import ModelUnavailable

logger = get_logger(__name__)


class QAResult(Protocol):
    answer: str
    score: float
    no_answer_score: float


class ModelService(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def extractive_qa(self, question: str, context: str) -> tuple[str, float, float]:
        """Returns (answer, answer_score, no_answer_score). An empty answer
        with high no_answer_score signals the model itself found no
        supporting span, distinct from the retrieval layer's abstention."""
        ...

    @abstractmethod
    def generate(
        self, system_prompt: str, user_prompt: str, *, max_new_tokens: int, temperature: float
    ) -> str: ...

    @abstractmethod
    def get_cross_encoder(self, model_name: str): ...

    @property
    @abstractmethod
    def backend_name(self) -> str: ...

    @property
    @abstractmethod
    def device_info(self) -> dict[str, str]: ...


# ---------------------------------------------------------------------------
# Hugging Face backend
# ---------------------------------------------------------------------------


class HFModelService(ModelService):
    def __init__(self, settings: Settings):
        self._settings = settings
        self._embedder = None
        self._qa_tokenizer = None
        self._qa_model = None
        self._gen_tokenizer = None
        self._gen_model = None
        self._cross_encoders: dict[str, object] = {}
        self._device = None

    # -- hardware detection --------------------------------------------------
    def _detect_device(self) -> str:
        if self._device:
            return self._device

        configured = getattr(self._settings, "model_device", "auto")
        if configured != "auto":
            # Honoured without consulting torch at all. On ZeroGPU,
            # torch.cuda.is_available() answers True under CUDA emulation, so
            # merely asking is not enough — the caller has to be able to say
            # "CPU" and have that be final.
            self._device = configured
            return self._device

        import torch

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._device

    def _select_compute_dtype(self):
        import torch

        if not torch.cuda.is_available():
            return torch.float32
        # bf16 requires Ampere+ (compute capability 8.x); older GPUs (e.g.
        # T4, compute 7.5) silently produce garbage or error with bf16, so
        # detect capability instead of hard-coding bf16 for "any GPU".
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16

    @property
    def device_info(self) -> dict[str, str]:
        import torch

        device = self._detect_device()
        info = {"device": device}
        if device == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(0)
        return info

    @property
    def backend_name(self) -> str:
        return "huggingface"

    # -- embeddings -----------------------------------------------------------
    def _load_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            device = self._detect_device()
            log_event(logger, "loading_model", model=self._settings.embedding_model, device=device)
            self._embedder = SentenceTransformer(self._settings.embedding_model, device=device)
        return self._embedder

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        try:
            embedder = self._load_embedder()
            return np.asarray(
                embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32),
                dtype=np.float32,
            )
        except Exception as exc:
            raise ModelUnavailable(internal_detail=str(exc)) from exc

    # -- extractive QA (kept as an optional path; generation is preferred) ---
    def _load_qa_model(self):
        if self._qa_model is None:
            from transformers import AutoModelForQuestionAnswering, AutoTokenizer

            device = self._detect_device()
            self._qa_tokenizer = AutoTokenizer.from_pretrained(self._settings.qa_model)
            self._qa_model = AutoModelForQuestionAnswering.from_pretrained(self._settings.qa_model).to(device)
            self._qa_model.eval()
        return self._qa_tokenizer, self._qa_model

    def extractive_qa(self, question: str, context: str) -> tuple[str, float, float]:
        import torch

        try:
            tokenizer, model = self._load_qa_model()
            device = self._detect_device()
            inputs = tokenizer(
                question, context, return_tensors="pt", truncation="only_second", max_length=384
            ).to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            start_logits, end_logits = outputs.start_logits[0], outputs.end_logits[0]
            # SQuAD2-style no-answer score: logit mass on the [CLS] position
            # (index 0) versus the best real span, so "no answer" is an
            # explicit, modeled outcome rather than an afterthought.
            no_answer_score = (start_logits[0] + end_logits[0]).item()

            start_probs = torch.softmax(start_logits, dim=0)
            end_probs = torch.softmax(end_logits, dim=0)

            max_answer_len = 60
            best_score, best_start, best_end = float("-inf"), 0, 0
            top_starts = torch.topk(start_probs, k=min(10, len(start_probs))).indices
            top_ends = torch.topk(end_probs, k=min(10, len(end_probs))).indices
            for s in top_starts.tolist():
                for e in top_ends.tolist():
                    if e < s or e - s > max_answer_len:
                        continue  # enforce valid, length-bounded spans
                    score = (start_probs[s] * end_probs[e]).item()
                    if score > best_score:
                        best_score, best_start, best_end = score, s, e

            if best_score == float("-inf"):
                return "", 0.0, 1.0

            answer_ids = inputs["input_ids"][0][best_start : best_end + 1]
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            return answer, float(best_score), float(torch.sigmoid(torch.tensor(no_answer_score)))
        except Exception as exc:
            raise ModelUnavailable(internal_detail=str(exc)) from exc

    # -- generation -------------------------------------------------------------
    def _load_generation_model(self):
        if self._gen_model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self._detect_device()
            model_kwargs: dict = {}
            if device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig

                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=self._select_compute_dtype(),
                    )
                    model_kwargs["device_map"] = "auto"
                except ImportError:
                    model_kwargs["torch_dtype"] = self._select_compute_dtype()
            else:
                model_kwargs["torch_dtype"] = torch.float32

            self._gen_tokenizer = AutoTokenizer.from_pretrained(self._settings.generation_model)
            self._gen_model = AutoModelForCausalLM.from_pretrained(
                self._settings.generation_model, **model_kwargs
            )
        return self._gen_tokenizer, self._gen_model

    def generate(
        self, system_prompt: str, user_prompt: str, *, max_new_tokens: int, temperature: float
    ) -> str:
        try:
            from transformers import pipeline

            tokenizer, model = self._load_generation_model()
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            result = pipe(
                messages,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                return_full_text=False,
            )
            return result[0]["generated_text"].strip()
        except Exception as exc:
            raise ModelUnavailable(internal_detail=str(exc)) from exc

    def get_cross_encoder(self, model_name: str):
        if model_name not in self._cross_encoders:
            from sentence_transformers import CrossEncoder

            self._cross_encoders[model_name] = CrossEncoder(model_name, device=self._detect_device())
        return self._cross_encoders[model_name]


# ---------------------------------------------------------------------------
# Mock backend — deterministic, no downloads, used in tests / CPU-only dev
# ---------------------------------------------------------------------------


class MockModelService(ModelService):
    """Deterministic stand-in with no model downloads. Embeddings are a
    stable hashed bag-of-words projection, so semantically similar text
    (shared vocabulary) still scores higher than unrelated text — enough to
    exercise retrieval logic meaningfully in tests without a real model.
    """

    _DIM = 128

    @property
    def backend_name(self) -> str:
        return "mock"

    @property
    def device_info(self) -> dict[str, str]:
        return {"device": "cpu (mock backend)"}

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self._DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                idx = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self._DIM
                vectors[i, idx] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def extractive_qa(self, question: str, context: str) -> tuple[str, float, float]:
        sentences = re.split(r"(?<=[.!?])\s+", context)
        question_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
        if not sentences or not question_tokens:
            return "", 0.0, 1.0
        best_sentence, best_overlap = "", 0.0
        for sentence in sentences:
            tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            if not tokens:
                continue
            overlap = len(tokens & question_tokens) / len(question_tokens)
            if overlap > best_overlap:
                best_overlap, best_sentence = overlap, sentence
        if best_overlap == 0:
            return "", 0.0, 1.0
        return best_sentence.strip(), min(best_overlap, 0.99), max(0.0, 1 - best_overlap)

    def generate(
        self, system_prompt: str, user_prompt: str, *, max_new_tokens: int, temperature: float
    ) -> str:
        # Extractive fallback: return the leading sentences of the input,
        # clearly usable for smoke tests without inventing content.
        sentences = re.split(r"(?<=[.!?])\s+", user_prompt.strip())
        return " ".join(sentences[:4]).strip() or "No content was available to summarize."

    def get_cross_encoder(self, model_name: str):
        return _MockCrossEncoder(model_name)


class _MockCrossEncoder:
    """Deterministic stand-in for a cross-encoder, for tests and CPU-only dev.

    Real cross-encoders score a (query, passage) pair jointly. This imitates
    the *shape* of that — a single float per pair, higher meaning more
    relevant — using token overlap weighted toward rarer, longer tokens, plus
    a stable hash tie-break so equal-overlap pairs order identically on every
    run and every machine. No GPU, no network, no download.

    The scores are explicitly meaningless as a quality signal. They exist so
    rerank *ordering*, candidate counts and response shapes are testable; the
    existing mock-backend warning covers them exactly as it covers embeddings.

    Deliberately bounded to [0, 1]. A real cross-encoder emits unbounded
    logits, but `Citation.relevance_score` is currently clamped to [0, 1]
    (see D2 in docs/phase_log.md), so emitting logits here would fabricate
    values through that clamp. Phase 2 moves the cross-encoder onto its own
    kind-labelled field; until then the mock stays inside the existing range.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.score_kind = "mock_cross_encoder"

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for query, passage in pairs:
            q_tokens = self._tokens(query)
            p_tokens = self._tokens(passage)
            if not q_tokens or not p_tokens:
                scores.append(0.0)
                continue

            shared = q_tokens & p_tokens
            # Weight longer tokens higher: they stand in for the rarer,
            # more discriminative terms a real cross-encoder keys on.
            weight = sum(len(token) for token in shared)
            ceiling = sum(len(token) for token in q_tokens)
            overlap = weight / ceiling if ceiling else 0.0

            # Stable, tiny tie-break so equal-overlap pairs have a total order
            # that does not depend on dict iteration or list position.
            digest = hashlib.sha256(f"{query}\x00{passage}".encode()).hexdigest()
            jitter = int(digest[:8], 16) / 0xFFFFFFFF * 0.01

            scores.append(round(min(overlap * 0.99 + jitter, 1.0), 6))
        return scores


class GroqModelService(ModelService):
    """Generation via the Groq API; embedding and reranking stay local.

    Groq serves chat completions only — no embeddings, no cross-encoders — so
    this is a composition rather than a replacement: `generate` goes over the
    wire, everything else is delegated to a local `ModelService`. The delegate
    is held explicitly so `backend_name` can report *both* halves truthfully.
    A deployment generating with Groq but embedding with the mock is not a
    Groq deployment, and the response says so rather than implying parity.

    No `torch`/`transformers` import appears here: the delegate owns that, so
    the containment invariant survives this backend existing.
    """

    def __init__(self, settings: Settings, local: ModelService):
        self._settings = settings
        self._local = local

    # -- capabilities delegated to local models --------------------------------
    def embed(self, texts: list[str]) -> np.ndarray:
        return self._local.embed(texts)

    def get_cross_encoder(self, model_name: str):
        return self._local.get_cross_encoder(model_name)

    def extractive_qa(self, question: str, context: str) -> tuple[str, float, float]:
        # Not implemented deliberately. No call site in the application reaches
        # this method, and standing up a span-extraction model purely to
        # satisfy the ABC would download ~500 MB that nothing consumes. If a
        # caller ever appears, this raises rather than returning a fabricated
        # span with a fabricated score.
        raise ModelUnavailable(
            internal_detail="extractive_qa is not served by the Groq backend; "
            "use MODEL_BACKEND=hf if a span-extraction model is required."
        )

    # -- identity ---------------------------------------------------------------
    @property
    def backend_name(self) -> str:
        local = self._local.backend_name
        if local == "mock":
            # Surfaced verbatim on AskResponse.model_used. Retrieval scores are
            # meaningless under mock embeddings, so this must never read as a
            # plain "groq" deployment.
            return f"groq:{self._settings.groq_model} (embeddings: mock)"
        return f"groq:{self._settings.groq_model}"

    @property
    def device_info(self) -> dict[str, str]:
        info = dict(self._local.device_info)
        info["generation"] = f"groq api ({self._settings.groq_model})"
        return info

    # -- generation -------------------------------------------------------------
    def generate(
        self, system_prompt: str, user_prompt: str, *, max_new_tokens: int, temperature: float
    ) -> str:
        import httpx

        key = self._settings.groq_api_key
        if not key:
            # Loud, not a fallback. Silently degrading to another backend would
            # make model_used a lie and the eval numbers unattributable.
            raise ModelUnavailable(
                internal_detail="GROQ_API_KEY is not set. Export it in the environment "
                "(or set it as a Space secret); it is never read from a file in this repo."
            )

        payload: dict = {
            "model": self._settings.groq_model,
            "temperature": temperature,
            "max_completion_tokens": max_new_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._settings.groq_reasoning_effort:
            payload["reasoning_effort"] = self._settings.groq_reasoning_effort

        url = f"{self._settings.groq_base_url.rstrip('/')}/chat/completions"
        last_detail = "no attempt was made"

        for attempt in range(self._settings.groq_max_retries):
            try:
                response = httpx.post(
                    url,
                    json=payload,
                    headers={"Authorization": f"Bearer {key}"},
                    timeout=self._settings.groq_timeout_seconds,
                )
            except httpx.HTTPError as exc:
                last_detail = f"transport error: {exc}"
                _sleep_backoff(attempt)
                continue

            # 429 is the free tier's tokens-per-minute ceiling, not an outage.
            # Retry-After is authoritative when present; back off otherwise.
            if response.status_code == 429 or response.status_code >= 500:
                last_detail = f"http {response.status_code}: {response.text[:200]}"
                _sleep_backoff(attempt, retry_after=response.headers.get("retry-after"))
                continue

            if response.status_code != 200:
                # 4xx other than 429 will not change on retry (bad key, bad model).
                raise ModelUnavailable(
                    internal_detail=f"groq http {response.status_code}: {response.text[:200]}"
                )

            body = response.json()
            try:
                return body["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError, AttributeError) as exc:
                raise ModelUnavailable(internal_detail=f"unexpected groq response shape: {exc}") from exc

        raise ModelUnavailable(
            internal_detail=f"groq unavailable after {self._settings.groq_max_retries} attempts: {last_detail}"
        )


def _sleep_backoff(attempt: int, *, retry_after: str | None = None) -> None:
    import time

    if retry_after:
        try:
            time.sleep(min(float(retry_after), 60.0))
            return
        except ValueError:
            pass
    time.sleep(min(0.5 * (2**attempt), 16.0))


def _hf_backend_importable() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache
def get_model_service() -> ModelService:
    settings = get_settings()
    backend = settings.model_backend
    if backend == "auto":
        backend = "hf" if _hf_backend_importable() else "mock"

    log_event(logger, "model_service_selected", backend=backend)
    if backend == "groq":
        # Embedding and reranking still need local models. Fall back to mock
        # only when the HF stack is genuinely absent, and let backend_name
        # carry that fact to the client rather than hiding it here.
        local: ModelService = HFModelService(settings) if _hf_backend_importable() else MockModelService()
        if isinstance(local, MockModelService):
            log_event(
                logger,
                "groq_local_delegate_is_mock",
                level=30,
                detail="generation is real; embeddings and rerank scores are not",
            )
        return GroqModelService(settings, local)
    if backend == "hf":
        return HFModelService(settings)
    return MockModelService()
