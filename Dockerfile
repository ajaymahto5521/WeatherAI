FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch (~200 MB) instead of the default CUDA build (~2 GB).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY app ./app
COPY training ./training
COPY scripts ./scripts
COPY data/reference ./data/reference

# Trained models are NOT baked in (they are git-ignored); compose mounts ./models.

RUN useradd --create-home appuser && mkdir -p data/processed && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Build the RAG index into Postgres, then serve.
CMD ["sh", "-c", "python scripts/ingest_documents.py pgvector && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
