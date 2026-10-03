"""Ingest data/reference/* into the RAG index. Run from project root."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from app.config import setup_logging
from app.rag.ingestion import ingest

if __name__ == "__main__":
    setup_logging()
    backend = sys.argv[1] if len(sys.argv) > 1 else "auto"  # auto | local | pgvector
    print(ingest(backend=backend))
