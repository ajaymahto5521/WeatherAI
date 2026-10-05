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
r = client.post("/ask", json={"question": "What does a tornado watch mean?"})
check("status 200", r.status_code == 200, f"got {r.status_code}")
body = r.json()
check("has 'answer' key", "answer" in body)
check("answer is non-empty", bool(body.get("answer", "")))
check("has 'sources' list", isinstance(body.get("sources"), list))
check("has 'tools_used' list", isinstance(body.get("tools_used"), list))
print(f"  answer: {body.get('answer', '')[:100]}...")

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
