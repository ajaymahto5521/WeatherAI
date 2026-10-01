"""
scripts/verify_milestone0.py
=============================
Verification script for Milestone 0.

This is NOT a pytest test file — it is a manual verification script
that prints a clear pass/fail report. Run it to confirm the foundation
is solid before committing.

Usage:
    python scripts/verify_milestone0.py

What it tests:
    1. app.main imports without errors
    2. GET /        returns 200
    3. GET /health  returns 200 with correct JSON shape
    4. POST /ask    returns 200 with stub answer (valid input)
    5. POST /ask    returns 422 for empty question (validation working)
    6. POST /ask    returns 422 for whitespace-only question
    7. POST /ask    returns 422 for missing 'question' field
"""

import sys
import json
from fastapi.testclient import TestClient

# ── import the app ─────────────────────────────────────────────────────────
try:
    from app.main import app
    print("✓ app.main imported successfully\n")
except Exception as e:
    print(f"✗ FATAL: Could not import app.main: {e}")
    sys.exit(1)

client = TestClient(app, raise_server_exceptions=False)

# ── helpers ────────────────────────────────────────────────────────────────
passed = 0
failed = 0

def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        print(f"  ✓  {name}")
        passed += 1
    else:
        print(f"  ✗  {name}")
        if detail:
            print(f"       detail: {detail}")
        failed += 1

# ── Test 1: GET / ───────────────────────────────────────────────────────────
print("── GET / ──────────────────────────────────────────────────────")
r = client.get("/")
check("Status 200", r.status_code == 200, f"got {r.status_code}")
body = r.json()
check("Has 'message' key",   "message" in body)
check("Has 'docs' key",      "docs"    in body)
check("Has 'health' key",    "health"  in body)
print()

# ── Test 2: GET /health ──────────────────────────────────────────────────────
print("── GET /health ────────────────────────────────────────────────")
r = client.get("/health")
check("Status 200",             r.status_code == 200, f"got {r.status_code}")
body = r.json()
check("status == 'ok'",         body.get("status") == "ok",         f"got '{body.get('status')}'")
check("environment present",    "environment" in body)
check("version present",        "version"     in body)
print(f"  Response body: {json.dumps(body, indent=2)}")
print()

# ── Test 3: POST /ask — valid question ─────────────────────────────────────
print("── POST /ask  (valid question) ────────────────────────────────")
r = client.post("/ask", json={"question": "What does a tornado watch mean?"})
check("Status 200",               r.status_code == 200, f"got {r.status_code}")
body = r.json()
check("Has 'answer' key",         "answer"       in body)
check("answer is non-empty",      bool(body.get("answer", "")))
check("Has 'sources' list",       isinstance(body.get("sources"),    list))
check("Has 'tools_used' list",    isinstance(body.get("tools_used"), list))
print(f"  Answer snippet: {body.get('answer','')[:100]}...")
print()

# ── Test 4: POST /ask — empty string ────────────────────────────────────────
print("── POST /ask  (empty string → 422) ────────────────────────────")
r = client.post("/ask", json={"question": ""})
check("Status 422 for empty string", r.status_code == 422, f"got {r.status_code}")
print()

# ── Test 5: POST /ask — whitespace only ─────────────────────────────────────
print("── POST /ask  (whitespace only → 422) ─────────────────────────")
r = client.post("/ask", json={"question": "   "})
check("Status 422 for whitespace", r.status_code == 422, f"got {r.status_code}")
print()

# ── Test 6: POST /ask — missing question field ──────────────────────────────
print("── POST /ask  (missing field → 422) ───────────────────────────")
r = client.post("/ask", json={})
check("Status 422 for missing field", r.status_code == 422, f"got {r.status_code}")
print()

# ── Test 7: POST /ask — question too long ───────────────────────────────────
print("── POST /ask  (question > 2000 chars → 422) ───────────────────")
r = client.post("/ask", json={"question": "x" * 2001})
check("Status 422 for question > 2000 chars", r.status_code == 422, f"got {r.status_code}")
print()

# ── Summary ──────────────────────────────────────────────────────────────────
print("=" * 60)
total = passed + failed
print(f"Results: {passed}/{total} checks passed")
if failed == 0:
    print("✓ Milestone 0 verification PASSED — ready to commit.")
else:
    print(f"✗ {failed} check(s) FAILED — fix before committing.")
    sys.exit(1)
