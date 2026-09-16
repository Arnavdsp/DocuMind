# Production image. The Colab notebook is a reproducible demo environment;
# this is the real deployment path (see README.md).
#
# Build from the REPOSITORY ROOT, not from backend/:
#   docker build -f backend/Dockerfile -t gargantua .
# The frontend stage needs frontend/ in the build context.

# ---------------------------------------------------------------------------
# Stage 1 — build the SPA
# ---------------------------------------------------------------------------
FROM node:22-slim AS frontend

WORKDIR /build

# Lockfile first so a source-only change doesn't reinstall the whole tree.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ---------------------------------------------------------------------------
# Stage 2 — the application
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# tesseract-ocr is required for scanned-page extraction. Without it the
# ingestion pipeline reports a clear error rather than silently producing
# empty text — but scanned documents will not work.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt backend/requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

COPY backend/app ./app

# main.py mounts this directory LAST, so it never shadows /api/*.
COPY --from=frontend /build/dist ./frontend/dist

ENV DATA_DIR=/data \
    PYTHONUNBUFFERED=1
VOLUME ["/data"]

# Non-root. /data must be writable by this user.
RUN useradd --create-home --uid 10001 gargantua \
    && mkdir -p /data \
    && chown -R gargantua:gargantua /data /app
USER gargantua

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
