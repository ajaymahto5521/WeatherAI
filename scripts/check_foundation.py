"""Smoke-check the API skeleton (/, /health, /ask validation). Run from project root."""
import json
import sys

from fastapi.testclient import TestClient

try:
    from app.main import app
except Exception as e:
    print(f"FATAL: could not import app.main: {e}")
    sys.exit(1)

client = TestClient(app, raise_server_exceptions=False)

passed = failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        print(f"  [OK]   {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}  {detail}")
        failed += 1


print("GET /")
r = client.get("/")
check("status 200", r.status_code == 200, f"got {r.status_code}")
body = r.json()
for key in ("message", "docs", "health"):
    check(f"has '{key}' key", key in body)

print("GET /health")
r = client.get("/health")
check("status 200", r.status_code == 200, f"got {r.status_code}")
body = r.json()
check("status == 'ok'", body.get("status") == "ok", f"got '{body.get('status')}'")
check("environment present", "environment" in body)
check("version present", "version" in body)
print(f"  body: {json.dumps(body)}")

print("POST /ask (valid)")
# 200 once the RAG index is built, 503 with an ingest hint before that
r = client.post("/ask", json={"question": "What does a tornado watch mean?"})
check("200 or 503", r.status_code in (200, 503), f"got {r.status_code}")
body = r.json()
if r.status_code == 200:
    check("has answer", bool(body.get("answer")))
    check("sources is a list", isinstance(body.get("sources"), list))
    check("tools_used is a list", isinstance(body.get("tools_used"), list))
else:
    check("503 explains how to fix it", "ingest" in body.get("detail", ""))

print("POST /ask (invalid)")
for label, payload in [
    ("empty string", {"question": ""}),
    ("whitespace only", {"question": "   "}),
    ("missing field", {}),
    ("over 2000 chars", {"question": "x" * 2001}),
]:
    r = client.post("/ask", json=payload)
    check(f"422 for {label}", r.status_code == 422, f"got {r.status_code}")

print(f"\n{passed}/{passed + failed} checks passed")
sys.exit(1 if failed else 0)
