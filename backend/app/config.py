"""Centralized, environment-driven application configuration.

Every model name, limit, and secret lives here (sourced from environment
variables / `.env`) instead of being scattered as string literals across
the codebase. This is what lets the model layer be swapped without
touching business logic, and keeps secrets out of source control.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    # --- Service identity -------------------------------------------------
    app_name: str = "Document Intelligence Suite"
    environment: Literal["development", "colab", "production"] = "development"
    log_level: str = "INFO"

    # --- Storage ------------------------------------------------------------
    data_dir: Path = Path("./data")

    # --- Upload / security limits -------------------------------------------
    max_upload_bytes: int = 25 * 1024 * 1024  # 25 MB
    max_pages_per_document: int = 300
    max_image_pixels: int = 40_000_000  # guards decompression-bomb style images
    allowed_extensions: tuple[str, ...] = (".pdf", ".txt", ".jpg", ".jpeg", ".png")
    allowed_mime_types: tuple[str, ...] = (
        "application/pdf",
        "text/plain",
        "image/jpeg",
        "image/png",
    )
    request_timeout_seconds: int = 120

    # --- Chunking -------------------------------------------------------------
    chunk_target_tokens: int = 220
    chunk_overlap_tokens: int = 40

    # --- Retrieval -------------------------------------------------------------
    retrieval_top_k: int = 8
    rerank_top_k: int = 4
    min_relevance_score: float = 0.18  # below this, the system abstains

    # --- Model configuration (names only — loading lives in services/models) ---
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    generation_model: str = Field(default="microsoft/Phi-3-mini-128k-instruct")
    qa_model: str = Field(default="deepset/roberta-base-squad2")
    reranker_model: str | None = Field(default=None)  # optional cross-encoder
    # "groq" reuses the generation model. "google" uses deep_translator's
    # unofficial endpoint, which rate-limits by source IP and is therefore
    # unreliable on shared hosts like HF Spaces.
    translation_provider: Literal["google", "groq", "none"] = "google"

    # --- Model runtime behavior ---
    # "groq" routes generation to the Groq API and keeps embedding and
    # cross-encoder work local; "auto" never selects it, so a missing key can
    # never silently change which backend answers.
    model_backend: Literal["auto", "hf", "mock", "groq"] = "auto"
    # "auto" picks cuda when torch reports it available. Forcing "cpu" matters
    # on ZeroGPU: torch.cuda.is_available() returns True there under CUDA
    # emulation, but any real CUDA init outside a @spaces.GPU function is
    # rejected outright. A Space doing its embedding on CPU must say so.
    model_device: Literal["auto", "cpu", "cuda"] = "auto"
    generation_max_new_tokens: int = 500
    # Structured summarization emits a JSON object carrying up to 10 findings
    # and 15 numbers. At 500 tokens that object is cut mid-string, json.loads
    # fails, and the summary silently degrades to raw JSON text. Budgeted
    # separately rather than raising the answer budget, which does not need it.
    summarize_max_new_tokens: int = 1600
    generation_temperature: float = 0.0

    # --- Groq generation backend ---
    # Only the model name and transport settings live here. The credential is
    # read from the environment (see the Secrets section) and is never written
    # to a file in this repository.
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-20b"
    # gpt-oss models emit internal reasoning tokens that count against the
    # free tier's tokens-per-minute budget. On grounded extraction the answer
    # is unchanged between efforts, so the cheapest one is the default.
    groq_reasoning_effort: Literal["low", "medium", "high"] | None = "low"
    groq_timeout_seconds: float = 60.0
    groq_max_retries: int = 4

    # --- Secrets (never logged, never sent to the frontend) ---
    ngrok_authtoken: str | None = None
    groq_api_key: str | None = None

    # --- CORS ---
    cors_allow_origins: tuple[str, ...] = ("http://localhost:5173", "http://localhost:8000")

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.documents_dir, self.index_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
