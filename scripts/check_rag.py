"""Verify the RAG pipeline end-to-end using the local store. Run from project root."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from app.rag.ingestion import ingest
from app.rag.retrieval import Retriever

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"  [OK]   {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}  {detail}")
        failed += 1


with tempfile.TemporaryDirectory() as tmp:
    res = ingest(index_dir=Path(tmp), backend="local", embedding_backend="tfidf")
    check("chunks ingested", res.n_chunks > 20, str(res))
    check("all 3 documents ingested", res.n_documents == 3, str(res))

    r = Retriever(index_dir=Path(tmp), backend="local")
    cases = [
        ("What is the difference between a tornado watch and a tornado warning?",
         "weather_alerts_definitions"),
        ("When does the southwest monsoon arrive in India?", "india_weather_guide"),
        ("What is relative humidity and dew point?", "meteorological_terms"),
    ]
    for q, doc in cases:
        hits = r.retrieve(q)
        top = [h.chunk.document_name for h in hits]
        check(f"top-{len(hits)} contains {doc} for '{q[:40]}...'",
              any(doc in t for t in top), str(top))
    check("sources shape", set(r.to_sources(hits)[0]) == {"document", "section"})

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
