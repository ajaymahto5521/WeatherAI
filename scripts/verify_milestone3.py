"""Verify the answer service and /ask endpoint. Run from project root."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from app.rag.ingestion import ingest
from app.rag.retrieval import Retriever
from app.services import answer_service
from app.services.answer_service import AnswerService, NO_CONTEXT_ANSWER

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
    ingest(index_dir=Path(tmp), backend="local", embedding_backend="tfidf")
    retriever = Retriever(index_dir=Path(tmp), backend="local")

    seen = {}

    def fake_llm(system, user):
        seen["user"] = user
        return "A tornado warning means a tornado is occurring or imminent [1]."

    q = "What is the difference between a tornado watch and a tornado warning?"

    r = AnswerService(retriever, fake_llm).answer(q)
    check("LLM answer returned", r.answer.startswith("A tornado warning"))
    check("prompt carries numbered context", "[1]" in seen["user"] and q in seen["user"])
    check("sources populated + deduped",
          len(r.sources) >= 1
          and len({(s.document, s.section) for s in r.sources}) == len(r.sources))
    check("tools_used", r.tools_used == ["rag_retriever"])

    r = AnswerService(retriever, None).answer(q)
    check("extractive fallback without LLM", "LLM unavailable" in r.answer and r.sources)

    def boom(s, u):
        raise RuntimeError("down")

    r = AnswerService(retriever, boom).answer(q)
    check("fallback when LLM raises", "LLM unavailable" in r.answer)

    r = AnswerService(retriever, fake_llm).answer("zzzqqq xxyyzz")
    check("no-context answer for gibberish", r.answer == NO_CONTEXT_ANSWER and not r.sources)

    # HTTP layer, with the service swapped for the test one
    try:
        from fastapi.testclient import TestClient
        from app.main import app

        answer_service._service = AnswerService(retriever, fake_llm)
        client = TestClient(app)
        resp = client.post("/ask", json={"question": q})
        body = resp.json()
        check("POST /ask 200", resp.status_code == 200, resp.text[:200])
        check("response shape", {"answer", "sources", "tools_used"} <= set(body))
        check("blank question -> 422", client.post("/ask", json={"question": "  "}).status_code == 422)
    except ImportError as exc:
        print(f"  [SKIP] HTTP tests ({exc})")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
